"""
天然度正则化损失: 二肽频率 + k-mer 频率匹配
在训练中直接惩罚不自然的残基共现模式
"""
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from config import AMINO_ACIDS, AA_TO_IDX


class NaturalnessRegularizer:
    """
    可微的天然度正则化器，在训练时直接优化 k-mer 和二肽频率分布。

    用法:
        reg = NaturalnessRegularizer(train_fasta='data/train_sequences.fasta')
        loss_dipep = reg.dipeptide_kl_loss(logits, targets, mask)
        loss_kmer  = reg.kmer_penalty(logits, targets, mask, k=5)
    """

    def __init__(self, train_fasta=None, max_seqs=10000):
        # 预计算的天然频率矩阵
        self.nat_dipep = None   # [20, 20]
        self.nat_tripep = None  # [20, 20, 20]
        self.kmer_index = None  # set of k-mers in nature (k=5,7)

        if train_fasta is not None:
            self._load_natural_frequencies(train_fasta, max_seqs)

    def _load_natural_frequencies(self, fasta_path, max_seqs):
        """从训练数据预计算天然二肽/三肽/k-mer频率"""
        from dataset import load_fasta
        seqs = load_fasta(fasta_path, max_seqs=max_seqs)
        valid_aa = set(AMINO_ACIDS)

        # 二肽频率 [20, 20]
        dipep_counter = Counter()
        # 三肽频率 [20, 20, 20]
        tripep_counter = Counter()
        # k-mer 集合 (k=5,7)
        kmer5_set = set()
        kmer7_set = set()
        total_dipep = 0
        total_tripep = 0

        for seq in seqs:
            clean = ''.join(c for c in seq if c in valid_aa)
            if len(clean) < 3:
                continue

            # 二肽
            for i in range(len(clean) - 1):
                dp = clean[i:i+2]
                if dp[0] in AA_TO_IDX and dp[1] in AA_TO_IDX:
                    dipep_counter[dp] += 1
                    total_dipep += 1

            # 三肽
            for i in range(len(clean) - 2):
                tp = clean[i:i+3]
                if all(c in AA_TO_IDX for c in tp):
                    tripep_counter[tp] += 1
                    total_tripep += 1

            # k-mer (存储为整数元组，O(1)查找)
            for i in range(len(clean) - 4):
                kmer5 = tuple(AA_TO_IDX[c] for c in clean[i:i+5] if c in AA_TO_IDX)
                if len(kmer5) == 5:
                    kmer5_set.add(kmer5)
            for i in range(len(clean) - 6):
                kmer7 = tuple(AA_TO_IDX[c] for c in clean[i:i+7] if c in AA_TO_IDX)
                if len(kmer7) == 7:
                    kmer7_set.add(kmer7)

        # 构建频率矩阵
        nat_dipep = torch.ones(20, 20) * 1e-6  # 拉普拉斯平滑
        for dp, count in dipep_counter.items():
            i, j = AA_TO_IDX[dp[0]], AA_TO_IDX[dp[1]]
            nat_dipep[i, j] += count
        nat_dipep = nat_dipep / nat_dipep.sum()

        nat_tripep = torch.ones(20, 20, 20) * 1e-8
        for tp, count in tripep_counter.items():
            i, j, k = AA_TO_IDX[tp[0]], AA_TO_IDX[tp[1]], AA_TO_IDX[tp[2]]
            nat_tripep[i, j, k] += count
        nat_tripep = nat_tripep / nat_tripep.sum()

        self.nat_dipep = nat_dipep
        self.nat_tripep = nat_tripep
        self.kmer5_set = kmer5_set
        self.kmer7_set = kmer7_set

        # 高频二肽的 log 值 (用于 KL)
        self.log_nat_dipep = torch.log(nat_dipep + 1e-8)
        self.log_nat_tripep = torch.log(nat_tripep + 1e-8)

        print(f'[NatLoss] 加载完成: {total_dipep} 二肽, {total_tripep} 三肽, '
              f'{len(kmer5_set)} 五肽, {len(kmer7_set)} 七肽')

    def dipeptide_kl_loss(self, logits, targets, mask):
        """
        二肽频率 KL 散度损失。
        令模型输出的二肽分布接近天然分布。

        Args:
            logits:  [B, L, V] raw logits
            targets: [B, L] target token ids
            mask:    [B, L] valid token mask

        Returns:
            scalar loss
        """
        if self.nat_dipep is None:
            return torch.tensor(0.0, device=logits.device)

        B, L, V = logits.shape
        # 只取 20 种标准氨基酸 (索引 0-19)，排除 PAD/BOS/EOS/MASK
        probs = F.softmax(logits[:, :, :20], dim=-1)  # [B, L, 20]

        # valid dipeptide positions: both t-1 and t are valid
        valid_pair = (mask[:, :-1].float() * mask[:, 1:].float()).bool()  # [B, L-1]
        if valid_pair.sum() < 10:
            return torch.tensor(0.0, device=logits.device)

        prev_aa = targets[:, :-1].clamp(0, 19)  # [B, L-1], clamp 特殊token
        next_probs = probs[:, 1:, :]              # [B, L-1, 20]

        # 聚合: 对于每个前驱氨基酸 i，收集所有后随位置的概率分布，取平均
        model_dipep = torch.zeros(20, 20, device=logits.device)
        for i in range(20):
            prev_mask = ((prev_aa == i) & valid_pair).unsqueeze(-1)  # [B, L-1, 1]
            count = prev_mask.sum()
            if count > 0:
                model_dipep[i] = (next_probs * prev_mask).sum(dim=(0, 1)) / count

        # 归一化
        model_dipep = model_dipep / model_dipep.sum().clamp(min=1e-8)

        # KL(天然 || 模型)
        nat = self.nat_dipep.to(logits.device)
        log_nat = self.log_nat_dipep.to(logits.device)
        log_model = torch.log(model_dipep + 1e-8)
        kl = (nat * (log_nat - log_model)).sum()
        kl = torch.clamp(kl, 0, 10)

        return kl

    def tripeptide_kl_loss(self, logits, targets, mask):
        """
        三肽频率 KL 散度损失 (20×20×20 矩阵)。
        注意: 矩阵有 8000 个元素，天然数据中大部分为 0。
        """
        if self.nat_tripep is None:
            return torch.tensor(0.0, device=logits.device)

        B, L, V = logits.shape
        probs = F.softmax(logits[:, :, :20], dim=-1)  # [B, L, 20]

        # valid tripeptide: 连续 3 个位置有效
        valid_tri = mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]
        if valid_tri.float().sum() < 10:
            return torch.tensor(0.0, device=logits.device)

        # 只用 100 个最高频三肽做稀疏 KL (避免 8000 维稀疏矩阵的噪声)
        nat_flat = self.nat_tripep.flatten()
        top_indices = nat_flat.topk(100).indices  # 最高频的 100 个三肽
        top_nat = nat_flat[top_indices]
        top_nat = top_nat / top_nat.sum()

        model_tripep = torch.zeros(20, 20, 20, device=logits.device)
        prev2_aa = targets[:, :-2].clamp(0, 19)
        prev1_aa = targets[:, 1:-1].clamp(0, 19)
        next_probs = probs[:, 2:, :]

        for i in range(20):
            for j in range(20):
                pair_mask = ((prev2_aa == i) & (prev1_aa == j) & valid_tri).unsqueeze(-1)
                count = pair_mask.sum()
                if count > 0:
                    model_tripep[i, j] = (next_probs * pair_mask).sum(dim=(0, 1)) / count

        model_tripep = model_tripep / model_tripep.sum().clamp(min=1e-8)

        # 只对高频三肽计算 KL
        top_log_nat = torch.log(top_nat + 1e-8).to(logits.device)
        top_log_model = torch.log(model_tripep.flatten()[top_indices] + 1e-8)
        kl = (top_nat.to(logits.device) * (top_log_nat - top_log_model)).sum()
        kl = torch.clamp(kl, 0, 10)

        return kl

    def kmer_penalty(self, logits, targets, mask, k=5):
        """
        K-mer 新颖性惩罚 (优化版: 整数元组查找 + 抽样, O(1)每个 k-mer)。
        只对每个batch中随机采样的 ~50 个 k-mer 位置进行检查，避免 O(B*L) 遍历。
        """
        kmer_set = self.kmer5_set if k == 5 else self.kmer7_set
        if kmer_set is None or len(kmer_set) == 0:
            return torch.tensor(0.0, device=logits.device)

        pred_tokens = logits[:, :, :20].argmax(dim=-1)  # [B, L]
        B, L = pred_tokens.shape

        # 有效 k-mer 起始位置
        valid_k = mask[:, :L-k+1]
        for shift in range(1, k):
            valid_k = valid_k & mask[:, shift:shift+L-k+1]

        valid_indices = valid_k.nonzero(as_tuple=False)  # [N, 2] (batch_idx, pos_idx)
        if valid_indices.size(0) < 1:
            return torch.tensor(0.0, device=logits.device)

        # 抽样: 最多检查 50 个位置
        n_sample = min(50, valid_indices.size(0))
        sample_idx = torch.randperm(valid_indices.size(0), device=logits.device)[:n_sample]
        sample_positions = valid_indices[sample_idx]  # [n_sample, 2]

        penalty = torch.tensor(0.0, device=logits.device)
        count = 0

        for idx in range(sample_positions.size(0)):
            b = sample_positions[idx, 0].item()
            pos = sample_positions[idx, 1].item()

            # 构建整数元组 (无需字符串)
            kmer_tuple = tuple(pred_tokens[b, pos + i].item() for i in range(k))

            if kmer_tuple not in kmer_set:
                # 该 k-mer 不存在于天然序列 → 惩罚
                penalty = penalty + logits[b, pos:pos+k, :20].softmax(dim=-1).max(dim=-1).values.mean()
                count += 1

        if count > 0:
            return penalty / count
        return torch.tensor(0.0, device=logits.device)
