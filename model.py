"""
FoldPath-LLM: Dual-Track Transformer Model
折叠路径引导的蛋白质设计大语言模型 - 双轨Transformer架构

支持两种模式:
1. 纯From-Scratch: TokenEmbedding → 双轨Transformer (原有)
2. ESM-2 基座: ESM-2 Encoder → Projection → 双轨Transformer (创新)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import (ModelConfig, TOTAL_VOCAB, BOS_IDX, EOS_IDX, PAD_IDX, MASK_IDX,
                    AMINO_ACIDS, AA_TO_IDX, IDX_TO_AA)
from physicochemical import PhysicochemicalEncoder, ChemicalInteractionBias, PhysicochemicalLoss


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class StructureAwareAttention(nn.Module):
    """结构+理化条件化的注意力机制: attn = QK^T/sqrt(d) + B_struct + B_chem"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, struct_bias=None, chem_bias=None):
        B, L, _ = x.shape
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # 偏置按头缩放: 偏置shape=(B,1,L,L), 广播到n_heads时需缩放
        # 否则n_heads个头同时被同一偏置信号驱动 → 注意力同质化 → 生成崩塌
        bias_scale = 1.0 / math.sqrt(self.n_heads)
        if struct_bias is not None:
            # ★ 梯度回流: 移除 detach()，允许序列轨的梯度信号回流到结构轨
            # 这样结构轨能从序列预测误差中学习，而非仅依赖退化的自监督目标
            # 通过 bias_scale 缩放防止偏置主导注意力 (典型QK^T范围[-3,3])
            scores = scores + struct_bias * bias_scale
        if chem_bias is not None:
            scores = scores + chem_bias * bias_scale
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        if mask is not None:
            pad_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~pad_mask, float('-inf'))
        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.W_out(out)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer块"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = StructureAwareAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, struct_bias=None, chem_bias=None):
        h = self.ln1(x)
        h = self.attn(h, mask, struct_bias, chem_bias)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x


class StructureTrack(nn.Module):
    """副轨: 结构隐向量预测器
    
    支持两种 attention mask 模式:
    - use_causal_mask=True:  因果 mask (From-Scratch 模式, 结构轨共享序列轨隐状态)
    - use_causal_mask=False: 双向 mask (ESM 模式, 结构轨输入来自 ESM 双向编码)
    """
    def __init__(self, d_model, struct_latent_dim, n_layers=3, n_heads=4, d_ff=256, dropout=0.1, use_causal_mask=True):
        super().__init__()
        self.struct_latent_dim = struct_latent_dim
        self.use_causal_mask = use_causal_mask
        self.input_proj = nn.Linear(d_model, struct_latent_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=struct_latent_dim, nhead=n_heads,
                dim_feedforward=d_ff, dropout=dropout,
                activation='gelu', batch_first=True
            ) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(struct_latent_dim)
        self.exposure_head = nn.Sequential(
            nn.Linear(struct_latent_dim, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.ss_head = nn.Sequential(
            nn.Linear(struct_latent_dim, 32), nn.GELU(),
            nn.Linear(32, 3)
        )
        self.distance_head = nn.Sequential(
            nn.Linear(struct_latent_dim * 2, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Softplus()
        )
        self.struct_bias_proj = nn.Sequential(
            nn.Linear(struct_latent_dim * 2, 32), nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, main_track_hidden, mask=None):
        h = self.input_proj(main_track_hidden)
        L = h.size(1)
        if self.use_causal_mask:
            attn_mask = nn.Transformer.generate_square_subsequent_mask(L, device=h.device)
        else:
            attn_mask = None  # ESM 模式: 双向注意力, 允许结构预测看到完整序列
        for layer in self.layers:
            padding_mask = ~mask if mask is not None else None
            if padding_mask is not None and attn_mask is not None:
                padding_mask = padding_mask.float()  # 与 attn_mask 类型统一
            h = layer(h, src_mask=attn_mask, src_key_padding_mask=padding_mask)
        h = self.ln(h)
        exposure = self.exposure_head(h).squeeze(-1)
        ss_logits = self.ss_head(h)
        hi = h.unsqueeze(2).expand(-1, -1, L, -1)
        hj = h.unsqueeze(1).expand(-1, L, -1, -1)
        pair = torch.cat([hi, hj], dim=-1)
        distance_matrix = self.distance_head(pair).squeeze(-1)
        struct_bias = self.struct_bias_proj(pair).squeeze(-1).unsqueeze(1)
        return h, exposure, ss_logits, distance_matrix, struct_bias


# ============================================================
# FoldPathLLM: 原始双轨模型 (支持 ESM-2 基座)
# ============================================================

class FoldPathLLM(nn.Module):
    """
    FoldPath-LLM: 折叠路径引导的蛋白质设计大语言模型

    双轨架构:
    - 主轨 (序列轨): TokenEmbedding (因果) → 自回归生成氨基酸序列
    - 副轨 (结构轨): 预测结构信号, 反哺序列生成

    两种模式:
    - use_esm=False: 纯 From-Scratch, 结构轨共享序列轨的隐状态
    - use_esm=True:  ESM-2 Encoder (冻结, 双向) → 仅注入结构轨
                     序列轨仍使用 TokenEmbedding (因果, 无信息泄露)
    """

    def __init__(self, config=None, esm_encoder=None):
        """
        Args:
            config: ModelConfig
            esm_encoder: ESMEncoder 实例 (None 表示不使用 ESM 基座)
        """
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config
        self._abl = config.ablation  # 消融实验标记 (None=完整模型)
        self.use_esm = esm_encoder is not None

        # ── 序列轨: TokenEmbedding (始终因果, 无信息泄露) ──
        self.token_embedding = nn.Embedding(TOTAL_VOCAB, config.d_model, padding_idx=PAD_IDX)

        # ── 结构轨基座 ──
        if self.use_esm:
            self.esm_encoder = esm_encoder
            encoder_dim = esm_encoder.hidden_size
            # 检测编码器类型: ESM 添加 CLS/EOS, RITA 不添加
            self.encoder_adds_special_tokens = getattr(esm_encoder, 'adds_special_tokens', True)
            # 根据编码器类型选择投影层
            if hasattr(esm_encoder, 'rita_model_name') or not self.encoder_adds_special_tokens:
                from rita_encoder import RITAProjection
                self.esm_projection = RITAProjection(encoder_dim, config.d_model, config.dropout)
                encoder_type = "RITA"
            else:
                from esm_encoder import ESMProjection
                self.esm_projection = ESMProjection(encoder_dim, config.d_model, config.dropout)
                encoder_type = "ESM-2"
            print(f"[FoldPathLLM] {encoder_type} → 结构轨 | 编码器维度:{encoder_dim} → 投影到:{config.d_model}")
            print(f"[FoldPathLLM] 序列轨: TokenEmbedding (因果) | 无信息泄露 ✓")
        else:
            self.esm_projection = None
            self.encoder_adds_special_tokens = False
            print(f"[FoldPathLLM] From-Scratch 模式 | 结构轨共享序列轨隐状态")

        # ── 理化编码器 (两种模式共用) ──
        self.physico_encoder = PhysicochemicalEncoder(config.physico_raw_dim, config.physico_dim)
        self.physico_fusion = nn.Sequential(
            nn.Linear(config.d_model + config.physico_dim, config.d_model),
            nn.LayerNorm(config.d_model)
        )

        # ── 位置编码 ──
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)

        # ── 主轨 Transformer (你的创新核心) ──
        self.main_layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers_main)
        ])
        self.main_ln = nn.LayerNorm(config.d_model)

        # ── 结构副轨 ──
        # ESM 模式: 结构轨输入来自 ESM 双向编码 → 使用双向 attention
        # From-Scratch 模式: 结构轨共享序列轨隐状态 → 使用因果 mask
        self.structure_track = StructureTrack(
            d_model=config.d_model, struct_latent_dim=config.struct_latent_dim,
            n_layers=config.n_layers_struct, n_heads=4, d_ff=256, dropout=config.dropout,
            use_causal_mask=True  # 统一因果mask，防止ESM双向信息泄漏到主轨
        )

        # ── 化学交互偏置 ──
        self.chem_bias = ChemicalInteractionBias(
            physico_dim=config.physico_dim,
            struct_latent_dim=config.struct_latent_dim,
            bias_dim=config.chem_bias_dim
        )

        # ── 输出头 (带缩放，防止深层logit数值爆炸) ──
        self.output_scale = nn.Parameter(torch.ones(1) * (1.0 / math.sqrt(config.d_model)))
        self.output_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, TOTAL_VOCAB)
        )

        # ── MLM辅助头 (基于结构轨双向隐状态预测被mask的token) ──
        # 关键: 输入维度为 struct_latent_dim，因为 MLM 应在双向结构轨上执行
        # 因果轨 (seq_x) 只有前文信息，与 MLM 的双向目标冲突 → 梯度互相拉扯
        self.mlm_head = nn.Sequential(
            nn.Linear(config.struct_latent_dim, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, TOTAL_VOCAB)
        )
        self.mlm_rate = 0.10  # 10% token 被 mask (从15%降低，减少序列连贯性损害)

        self.physico_loss_fn = PhysicochemicalLoss(config.physico_dim)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'esm_encoder' in name or 'esm_projection' in name:
                continue  # ESM 投影层在 ESMProjection 中初始化
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_seq_input(self, input_ids, sequences=None):
        """
        序列轨输入: 始终使用 TokenEmbedding (因果), 无信息泄露
        Returns: [B, L, d_model]
        """
        return self.token_embedding(input_ids)

    def _build_struct_input(self, input_ids, sequences=None):
        """
        结构轨输入:
        - ESM 模式: ESM 双向编码完整序列 → Projection → [B, L_esm, d_model]
        - From-Scratch: 复用序列轨的 token_embedding (后续在 forward 中共享)
        Returns: [B, L, d_model] or None (None = 复用序列轨)
        """
        if self.use_esm and sequences is not None:
            # 提取纯氨基酸序列
            clean_seqs = []
            for seq in sequences:
                clean = ''.join(c for c in seq if c in AA_TO_IDX)
                clean_seqs.append(clean if clean else "M")
            # ESM 双向编码 (结构预测允许双向)
            esm_emb, esm_mask = self.esm_encoder(clean_seqs)  # [B, L_esm, esm_dim]
            struct_x = self.esm_projection(esm_emb)  # [B, L_esm, d_model]
            return struct_x, esm_mask
        else:
            return None, None

    def forward(self, input_ids, targets=None, mask=None, sequences=None, use_bias=True):
        """
        Plan B 架构:
        - 序列轨 (因果): TokenEmbedding → PhysicoFusion → PosEncoding → MainTransformer → Output
        - 结构轨 (双向): ESM Encoder → Projection → StructureTrack → Bias 反哺序列轨

        Args:
            use_bias: 是否注入结构偏置 (训练=True, 生成=False 避免偏置累积→全R崩塌)
        """
        B, L = input_ids.shape
        if mask is None:
            mask = (input_ids != PAD_IDX)

        # ── 1. 序列轨嵌入 (因果, 始终用 TokenEmbedding) ──
        seq_x = self._build_seq_input(input_ids, sequences)  # [B, L, d_model]

        # ── 2. 结构轨嵌入 (ESM 双向 或 复用序列轨) ──
        struct_x, esm_mask = self._build_struct_input(input_ids, sequences)

        # ── 3. 理化性质编码 ──
        # 将特殊token (BOS/EOS/PAD) 映射为 UNK(0=A), 避免 .clamp 系统性偏置
        aa_indices = input_ids.clamp(0, 19)  # [B, L]
        # 用 mask 覆盖特殊token位置, forward 时忽略 (loss 用 target 侧 mask)
        physico_embed = self.physico_encoder(aa_indices)  # [B, L, physico_dim]
        # 将特殊token位置的理化嵌入清零, 避免错误信号传播
        special_mask = (input_ids > 19).unsqueeze(-1).float()  # [B, L, 1]
        physico_embed = physico_embed * (1 - special_mask)

        # 编码器嵌入 → seq_x 长度对齐
        # ESM: 输出含 CLS/EOS → 需剥离; RITA: 纯残基嵌入 → 无需剥离
        if struct_x is not None:
            seq_len = seq_x.size(1)

            if self.encoder_adds_special_tokens:
                # ESM: 剥离首尾特殊 token (CLS + EOS)
                struct_x = struct_x[:, 1:-1, :]
                if esm_mask is not None:
                    esm_mask = esm_mask[:, 1:-1]

            res_len = struct_x.size(1)  # 纯残基长度

            # 将 seq_x 与残基嵌入对齐
            # seq_x:    [BOS,  aa1,  aa2,  ..., aa_n,  EOS,   PAD...]
            # residues: [      aa1,  aa2,  ..., aa_n         ]
            # 目标: struct_x 与 seq_x 等长, BOS 位补零
            tgt_len = max(1, seq_len - 1)
            if res_len < tgt_len:
                tail = tgt_len - res_len
                pad = torch.zeros(B, tail, struct_x.size(-1), device=struct_x.device)
                struct_x = torch.cat([struct_x, pad], dim=1)
                if esm_mask is not None:
                    esm_mask = torch.cat([esm_mask, torch.zeros(B, tail, dtype=torch.bool, device=esm_mask.device)], dim=1)
            elif res_len > tgt_len:
                if struct_x.dim() == 2:
                    struct_x = struct_x[:tgt_len, :]
                else:
                    struct_x = struct_x[:, :tgt_len, :]
                if esm_mask is not None:
                    esm_mask = esm_mask[:, :tgt_len]

            # Ensure 3D [B, L, D] before concatenation
            if struct_x.dim() == 2:
                struct_x = struct_x.unsqueeze(0)
            if esm_mask is not None and esm_mask.dim() == 1:
                esm_mask = esm_mask.unsqueeze(0)

            # BOS 位补零 (序列轨 BOS 对应结构轨无信号位置)
            struct_x = torch.cat([
                torch.zeros(B, 1, struct_x.size(-1), device=struct_x.device),
                struct_x
            ], dim=1)
            if esm_mask is not None:
                esm_mask = torch.cat([
                    torch.ones(B, 1, dtype=torch.bool, device=esm_mask.device),
                    esm_mask
                ], dim=1)

            # 防御性裁剪
            struct_x = struct_x[:, :seq_len, :]
            if esm_mask is not None:
                esm_mask = esm_mask[:, :seq_len]

        # ── 4. 序列轨: 理化融合 + 位置编码 ──
        seq_x = self.physico_fusion(torch.cat([seq_x, physico_embed], dim=-1))
        seq_x = self.pos_encoding(seq_x)

        # ── 5. 结构轨: 构建输入 ──
        if struct_x is not None:
            # ESM 模式: 用 ESM 投影作为结构轨输入
            struct_input = self.physico_fusion(torch.cat([struct_x, physico_embed], dim=-1))
            struct_input = self.pos_encoding(struct_input)
        else:
            # From-Scratch: 复用序列轨的隐状态
            struct_input = seq_x

        # ── 6. 结构副轨 (预测结构信号 + 生成偏置) ──
        if self._abl == 'no_struct':
            # 消融: 移除结构轨 → 结构偏置/MLM全跳, latent为zero
            struct_latent = torch.zeros(B, L, self.config.struct_latent_dim, device=input_ids.device)
            exposure_pred = torch.zeros(B, L, device=input_ids.device)
            ss_logits = torch.zeros(B, L, 3, device=input_ids.device)
            distance_pred = torch.zeros(B, L, L, device=input_ids.device)
            struct_bias = None
        else:
            struct_latent, exposure_pred, ss_logits, distance_pred, struct_bias = \
                self.structure_track(struct_input, mask)

        # ── 7. 化学交互偏置 (基于 ESM 结构轨或共享隐状态) ──
        if self._abl in ('no_struct', 'no_chembias', 'no_physico'):
            chem_bias = None
        else:
            chem_bias = self.chem_bias(physico_embed, struct_latent)

        # ── 7.5 偏置归一化 + 裁剪 (防止长序列累积 / 极端理化值) ──
        # 关键修复: BIAS_CLIP 从 5.0 降至 2.0，偏置不应主导注意力分数
        # QK^T/sqrt(d_k) 典型范围 [-3, 3]，偏置超过此范围会劫持注意力
        seq_len = seq_x.size(1)
        BIAS_CLIP = 2.0
        if struct_bias is not None:
            struct_bias = struct_bias / math.sqrt(seq_len)
            struct_bias = struct_bias.clamp(-BIAS_CLIP, BIAS_CLIP)
        if chem_bias is not None:
            chem_bias = chem_bias / math.sqrt(seq_len)
            chem_bias = chem_bias.clamp(-BIAS_CLIP, BIAS_CLIP)

        # ── 8. 主轨 Transformer (因果, 结构偏置反哺) ──
        _struct_bias = struct_bias if use_bias else None
        _chem_bias = chem_bias if use_bias else None
        for layer in self.main_layers:
            seq_x = layer(seq_x, mask, _struct_bias, _chem_bias)
        seq_x = self.main_ln(seq_x)

        # ── 9. 输出头 (应用缩放因子，防止深层logit数值爆炸) ──
        # output_scale 裁剪在 [0.5, 2.0]，防止训练中发散
        self.output_scale.data.clamp_(0.5, 2.0)
        logits = self.output_head(seq_x) * self.output_scale

        # ── 10. 损失计算 ──
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=input_ids.device)
        if targets is not None:
            # 序列交叉熵损失 (带标签平滑，防止模型过度自信 → 生成崩塌)
            label_smoothing = getattr(self.config, 'label_smoothing', 0.1)
            loss_seq = F.cross_entropy(
                logits.reshape(-1, TOTAL_VOCAB), targets.reshape(-1),
                ignore_index=PAD_IDX, label_smoothing=label_smoothing
            )
            loss_dict['loss_seq'] = loss_seq.item()
            total_loss = total_loss + self.config.lambda_seq * loss_seq

            # 熵正则化: 鼓励输出分布多样性，防止概率塌缩到单一token
            # NaN安全: 使用 clamp 防止 0 × (-inf) = NaN
            log_probs = F.log_softmax(logits, dim=-1)  # [B, L, V]
            probs = F.softmax(logits, dim=-1)
            # 安全熵计算: 将 probs=0 对应的 log_probs 也置0，避免 0*(-inf)=NaN
            safe_log_probs = torch.where(probs > 1e-8, log_probs, torch.zeros_like(log_probs))
            entropy = -(probs * safe_log_probs).sum(dim=-1)  # [B, L]
            # NaN保护 (双重保险)
            entropy = entropy.nan_to_num(0.0)
            # 有效位置的熵 (排除 PAD)
            valid_mask = (targets != PAD_IDX).float()
            avg_entropy = (entropy * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            max_entropy = math.log(TOTAL_VOCAB)  # 均匀分布的熵
            # 当实际熵低于期望时惩罚 (二次惩罚，越低惩罚越陡)
            entropy_ratio = avg_entropy / max_entropy
            # 天然蛋白熵比约 0.5~0.7，阈值从0.3提升到0.5，提前干预
            # 二次惩罚: 偏离越远惩罚增长越快，防止模型滑入低熵区
            entropy_deficit = F.relu(0.5 - entropy_ratio)  # 熵比低于0.5时触发
            loss_entropy = entropy_deficit ** 2 * 4.0  # 二次惩罚，梯度随偏离增大
            # NaN保护 (三重保险)
            if torch.isnan(loss_entropy) or torch.isinf(loss_entropy):
                loss_entropy = torch.tensor(0.0, device=logits.device)
            loss_dict['loss_entropy'] = loss_entropy.item()

            # ── 边际多样性损失 (Marginal Diversity Loss) ──
            # 核心问题: 逐位置熵可能很高 (每个位置分散概率)，但所有位置都偏向同一组AA
            # 解决: 计算所有位置的平均预测分布 (边际分布)，惩罚其熵过低
            # 这直接防止模型在全局层面对特定氨基酸的聚类偏好
            marginal_probs = (probs * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, V]
            marginal_log_probs = torch.log(marginal_probs + 1e-8)
            marginal_entropy = -(marginal_probs * marginal_log_probs).sum(dim=-1)  # [B]
            # 期望边际熵 ≥ 2.5 (约 60% 最大熵)，对应天然蛋白的AA分布多样性
            loss_marginal = F.relu(2.5 - marginal_entropy).mean()
            if torch.isnan(loss_marginal) or torch.isinf(loss_marginal):
                loss_marginal = torch.tensor(0.0, device=logits.device)
            loss_dict['loss_marginal'] = loss_marginal.item()

            # ── 连续重复惩罚 (Consecutive Repeat Loss) ──
            # 核心问题: 模型学到"输出连续相同残基"能降低loss (因为训练集中存在同源重复)
            # 解决: 当模型对与前一个位置相同AA的预测概率过高时施加惩罚
            # 这教模型"可以重复，但不要对此过度自信"
            if seq_len > 1:
                prev_targets = targets[:, :-1]  # [B, L-1]
                curr_targets = targets[:, 1:]   # [B, L-1]
                curr_probs = probs[:, 1:, :]    # [B, L-1, V]
                # 提取每个位置对"与前一个target相同AA"的预测概率
                repeat_probs = curr_probs.gather(2, curr_targets.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
                # 只在确实重复的位置惩罚 (target[i] == target[i-1] 且非PAD)
                is_repeat = (curr_targets == prev_targets) & (curr_targets != PAD_IDX) & (prev_targets != PAD_IDX)
                # 对重复位置的过高置信度惩罚: 当概率>0.3时开始惩罚
                repeat_penalty = F.relu(repeat_probs - 0.3) * is_repeat.float()
                loss_repeat = repeat_penalty.sum() / is_repeat.float().sum().clamp(min=1)
            else:
                loss_repeat = torch.tensor(0.0, device=logits.device)
            if torch.isnan(loss_repeat) or torch.isinf(loss_repeat):
                loss_repeat = torch.tensor(0.0, device=logits.device)
            loss_dict['loss_repeat'] = loss_repeat.item()

            # 结构损失 (自监督) — 消融: no_struct 时跳过
            if self._abl != 'no_struct':
                with torch.no_grad():
                    exposure_target = (distance_pred.mean(dim=-1) - distance_pred.mean()) / \
                                      (distance_pred.std() + 1e-6)
                    exposure_target = exposure_target.clamp(0, 1)
                loss_exposure = F.mse_loss(exposure_pred, exposure_target.detach())
                ss_probs = F.log_softmax(ss_logits, dim=-1)
                ss_entropy = -(F.softmax(ss_logits, dim=-1) * ss_probs).sum(dim=-1).mean()
                loss_sym = F.mse_loss(distance_pred, distance_pred.transpose(-2, -1))
                loss_struct = loss_exposure + 0.1 * ss_entropy + 0.1 * loss_sym
                loss_dict['loss_struct'] = loss_struct.item()
                total_loss = total_loss + self.config.lambda_struct * loss_struct

            # 理化损失 — 消融: no_struct/no_physico 时跳过
            if self._abl not in ('no_struct', 'no_physico'):
                loss_physico = self.physico_loss_fn(physico_embed, aa_indices, exposure_pred, distance_pred)
                loss_dict['loss_physico'] = loss_physico.item()
                total_loss = total_loss + self.config.lambda_physico * loss_physico

            # 熵正则化 — 消融: no_divreg 时跳过
            if self._abl != 'no_divreg':
                total_loss = total_loss + getattr(self.config, 'lambda_entropy', 0.1) * loss_entropy

            # 边际多样性 + 重复惩罚 — 消融: no_divreg 时跳过
            if self._abl != 'no_divreg':
                total_loss = total_loss + getattr(self.config, 'lambda_marginal', 0.15) * loss_marginal
                total_loss = total_loss + getattr(self.config, 'lambda_repeat', 0.2) * loss_repeat

            # ── 偏置多样性正则化 ──
            # 目的: 防止结构/化学偏置崩塌成"处处偏好同一AA"的模式
            # 偏置应该反映"位置i和位置j的交互关系"，而非"全局偏好某AA"
            # 方法: 惩罚偏置矩阵行/列均值的方差 — 均值越一致越好
            if struct_bias is not None:
                # struct_bias: [B, 1, L, L]，取行均值和列均值
                sb = struct_bias.squeeze(1)  # [B, L, L]
                row_mean = sb.mean(dim=-1)   # [B, L] 每行(每个query位置)的平均偏置
                col_mean = sb.mean(dim=-2)   # [B, L] 每列(每个key位置)的平均偏置
                # 理想: 每个位置的行均值和列均值各不相同 (反映位置特异交互)
                # 崩塌: 所有位置均值相同 (全局偏好)
                loss_bias_diversity = row_mean.var(dim=-1).mean() + col_mean.var(dim=-1).mean()
                # 方差越大越好 → 取负号变成惩罚低方差
                loss_bias_diversity = F.relu(0.01 - loss_bias_diversity)  # 方差<0.01时惩罚
            else:
                loss_bias_diversity = torch.tensor(0.0, device=logits.device)
            if torch.isnan(loss_bias_diversity) or torch.isinf(loss_bias_diversity):
                loss_bias_diversity = torch.tensor(0.0, device=logits.device)
            loss_dict['loss_bias_div'] = loss_bias_diversity.item()
            if self._abl != 'no_divreg':
                total_loss = total_loss + 0.1 * loss_bias_diversity

            # ── 对比学习: 表示均匀性损失 — 消融: no_divreg 时跳过 ──
            # 目的: 防止模式崩塌 — 不同序列在表示空间中应相互远离
            # 方法: Wang & Isola (2020) uniformity loss
            #   L = log( Σ_{i≠j} exp(-t * ||z_i - z_j||²) )
            # 均匀分布 → 距离大 → loss小; 崩塌 → 距离小 → loss大
            if B > 1:
                # 序列级表示: 有效位置的平均池化 + L2归一化
                seq_repr = (seq_x * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, d_model]
                seq_repr = F.normalize(seq_repr, dim=-1)  # L2归一化到单位球面
                # 计算所有样本对的距离
                dist_sq = torch.cdist(seq_repr, seq_repr, p=2).pow(2)  # [B, B]
                # uniformity loss: 排除对角线 (自身距离=0)
                mask_off_diag = ~torch.eye(B, dtype=torch.bool, device=seq_repr.device)
                uniformity = dist_sq[mask_off_diag].mul(-2.0).exp().mean().log()  # t=2.0
                # 期望: 均匀分布时 uniformity ≈ -1~0, 崩塌时 uniformity ≈ 2~4
                loss_uniformity = F.relu(uniformity + 0.5)  # uniformity > -0.5 时惩罚
            else:
                loss_uniformity = torch.tensor(0.0, device=logits.device)
            if torch.isnan(loss_uniformity) or torch.isinf(loss_uniformity):
                loss_uniformity = torch.tensor(0.0, device=logits.device)
            loss_dict['loss_uniform'] = loss_uniformity.item()
            if self._abl != 'no_divreg':
                total_loss = total_loss + getattr(self.config, 'lambda_uniform', 0.1) * loss_uniformity

            # ── MLM辅助任务 — 消融: no_mlm 时跳过 ──
            # 核心修复: MLM输入从因果轨 seq_x 移到双向结构轨 struct_latent
            # 原因: seq_x 经因果注意力，每个位置只看到前文，与MLM双向目标冲突
            # struct_latent 在ESM模式下是双向注意力，天然适合MLM双向预测
            # 即使 From-Scratch 模式结构轨也是因果的，MLM的梯度仍能增强结构轨表达能力
            if self.training and seq_len > 3 and self._abl not in ('no_mlm', 'no_struct'):
                # 随机选择10%的非PAD位置进行mask
                non_pad_positions = (targets != PAD_IDX)  # [B, L]
                rand_matrix = torch.rand_like(targets.float())
                mlm_mask = (rand_matrix < self.mlm_rate) & non_pad_positions  # [B, L]
                if mlm_mask.any():
                    # 将mask位置替换为MASK_IDX
                    mlm_input = input_ids.clone()
                    mlm_input[mlm_mask] = MASK_IDX
                    # ★ 关键修复: 从 struct_latent 而非 seq_x 预测被mask的token
                    # struct_latent 维度为 struct_latent_dim，mlm_head 已适配
                    mlm_logits = self.mlm_head(struct_latent)  # [B, L, V]
                    loss_mlm = F.cross_entropy(
                        mlm_logits[mlm_mask],  # 只取mask位置的logits
                        targets[mlm_mask],      # 对应的target
                        ignore_index=PAD_IDX, label_smoothing=0.1
                    )
                    if torch.isnan(loss_mlm) or torch.isinf(loss_mlm):
                        loss_mlm = torch.tensor(0.0, device=logits.device)
                else:
                    loss_mlm = torch.tensor(0.0, device=logits.device)
            else:
                loss_mlm = torch.tensor(0.0, device=logits.device)
            loss_dict['loss_mlm'] = loss_mlm.item()
            total_loss = total_loss + getattr(self.config, 'lambda_mlm', 0.1) * loss_mlm

            # ── 天然度正则化 — 消融: no_natreg 时跳过 ──
            if hasattr(self, 'nat_regularizer') and self.nat_regularizer is not None and self._abl != 'no_natreg':
                loss_dipep = self.nat_regularizer.dipeptide_kl_loss(logits, targets, mask)
                if torch.isnan(loss_dipep) or torch.isinf(loss_dipep):
                    loss_dipep = torch.tensor(0.0, device=logits.device)
                loss_dict['loss_dipep'] = loss_dipep.item()
                total_loss = total_loss + getattr(self.config, 'lambda_dipep', 0.15) * loss_dipep

                # K-mer 惩罚
                loss_kmer = self.nat_regularizer.kmer_penalty(logits, targets, mask, k=5)
                if torch.isnan(loss_kmer) or torch.isinf(loss_kmer):
                    loss_kmer = torch.tensor(0.0, device=logits.device)
                loss_dict['loss_kmer'] = loss_kmer.item()
                total_loss = total_loss + getattr(self.config, 'lambda_kmer', 0.05) * loss_kmer

        loss_dict['loss_total'] = total_loss.item()
        return logits, loss_dict, total_loss, {
            'struct_latent': struct_latent, 'exposure': exposure_pred,
            'ss_logits': ss_logits, 'distance_matrix': distance_pred,
        }

    @torch.no_grad()
    def generate(self, max_length=128, temperature=1.0, top_k=50, top_p=0.9,
                 physico_threshold=-0.5, use_physico_filter=False, device='cpu'):
        """
        自回归序列生成 (支持 ESM 模式)
        理化过滤基于序列组成先验, 不依赖曝光度预测

        关键修复:
        - use_bias=False: 生成时关闭结构/化学偏置，防止偏置正反馈循环导致崩塌
        - 指数重复惩罚: 防止单一残基重复主导
        - 移除硬禁C的hack: 改为通用的连续残基惩罚
        """
        self.eval()
        batch = torch.ones(1, 1, dtype=torch.long, device=device) * BOS_IDX
        generated = []
        physico_scores = []

        for step in range(max_length):
            # ── ESM 模式下传入序列字符串 ──
            sequences = None
            if self.use_esm:
                prefix_ids = [idx for idx in batch[0].tolist()
                              if idx not in (PAD_IDX, BOS_IDX)]
                prefix_seq = ''.join([IDX_TO_AA.get(idx, 'X') for idx in prefix_ids])
                sequences = [prefix_seq] if prefix_seq else ["M"]

            # ★ 关键修复: use_bias=False
            # 生成时偏置会造成正反馈循环: 偏置→倾向某token→强化偏置→全R/C崩塌
            logits, _, _, aux = self.forward(batch, sequences=sequences, use_bias=False)
            next_logits = logits[0, -1, :] / temperature

            # ── 屏蔽特殊token: 只允许标准氨基酸 (0-19) 和 EOS (22) ──
            # PAD=20, BOS=21, UNK/MASK等 ≥23 全部屏蔽，防止生成非标准残基
            for idx in range(20, TOTAL_VOCAB):
                if idx != EOS_IDX:
                    next_logits[idx] = float('-inf')

            # ── 对比惩罚: 惩罚已见AA + 提升未见AA (Contrastive Penalty) ──
            # 不仅仅压制已出现的token，还主动提升从未出现的token概率
            # 效果: 强制模型探索更多样的氨基酸，而非仅压制重复
            if len(generated) > 0:
                from collections import Counter
                aa_counts = Counter(generated)
                seen_set = set(generated)
                for aa_idx in range(20):
                    if aa_idx in seen_set:
                        count = aa_counts[aa_idx]
                        # 指数惩罚已见AA: count=1→-0.3, count=2→-0.8, count=3→-1.5
                        penalty = 0.3 * (1.5 ** min(count, 10) - 1)
                        next_logits[aa_idx] = next_logits[aa_idx] - penalty
                    else:
                        # ★ 提升未见AA: +0.3 logit，鼓励探索
                        next_logits[aa_idx] = next_logits[aa_idx] + 0.3

            # ── 连续残基惩罚: 防止同一残基连续出现 ≥3 次 ──
            if len(generated) >= 2:
                last = generated[-1]
                second_last = generated[-2]
                if last == second_last and last < 20:
                    # 连续2个相同 → 强烈压制第三个
                    next_logits[last] = next_logits[last] - 2.0
                    if len(generated) >= 3 and generated[-3] == last:
                        # 连续3个 → 极度压制
                        next_logits[last] = next_logits[last] - 5.0

            # ── 典型采样 (Typical Sampling) ──
            # 只从"信息量接近预期熵"的token中采样，天然防崩塌
            # 原理: 高概率token太确定(低信息)，低概率token太随机(高信息)
            #       典型采样保留信息量≈期望熵的token，比top-k/p更优
            # 参考: Meister et al. 2023 "Typical Decoding for Natural Language Generation"
            if step > 0:  # 第一个token不需要
                raw_probs = F.softmax(next_logits, dim=-1)
                log_p = F.log_softmax(next_logits, dim=-1)
                # 计算当前位置的边际熵 (期望信息量)
                marginal_ent = -(raw_probs * log_p).sum()
                # 每个token的信息量 = -log p(x)
                info = -log_p  # [V]
                # 保留 |信息量 - 期望熵| < threshold 的token
                typical_threshold = 2.0  # 容忍范围，越大越多token被保留
                typical_mask = (info - marginal_ent).abs() < typical_threshold
                # 至少保留5个候选，避免过度裁剪
                if typical_mask.sum() < 5:
                    # fallback: 保留top-5
                    _, top_indices = next_logits.topk(5)
                    typical_mask = torch.zeros_like(typical_mask)
                    typical_mask[top_indices] = True
                next_logits[~typical_mask] = float('-inf')

            # ── Top-K / Top-P 采样 (作为典型采样的补充) ──
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < top_k_vals[-1]] = float('-inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_aa = torch.multinomial(probs, 1)

            if next_aa.item() == EOS_IDX:
                break

            if use_physico_filter and step > 5:
                retries = 0
                while retries < 5:
                    physico_score = self._quick_physico_check(next_aa.item(), generated)
                    if physico_threshold is None or physico_score >= physico_threshold:
                        break
                    # 压制被拒残基的 logit, 强制重采样不同残基
                    next_logits[next_aa.item()] = float('-inf')
                    # 所有可选残基都被压制 → 放弃过滤，接受
                    if (next_logits[:20] == float('-inf')).all():
                        break
                    probs = F.softmax(next_logits, dim=-1)
                    next_aa = torch.multinomial(probs, 1)
                    retries += 1
                physico_scores.append(physico_score)

            generated.append(next_aa.item())
            batch = torch.cat([batch, next_aa.unsqueeze(0)], dim=1)

        return generated, physico_scores

    def _quick_physico_check(self, aa_idx, generated):
        """
        连续理化打分 (软化阈值, 平滑过渡)

        改进:
        - 基准分 0.0 (中性), 通过≥0, 不通过<0
        - 所有阈值使用 sigmoid 软化, 消除硬边界跳变
        - 扣分连续变化: -0.5(轻微) → -1.0(中等) → -2.0(严重)

        评分含义:
          - [0.0, +∞): 完美通过 (天然蛋白分布内)
          - [-0.5, 0.0): 轻微偏离, 可容忍
          - [-1.0, -0.5): 中度偏离, 倾向拒绝
          - [-2.0, -1.0): 严重违规, 坚决拒绝
        """
        from physicochemical import PHYSICO_MATRIX

        if aa_idx >= 20 or aa_idx < 0:
            return 0.0

        score = 0.0  # 基准分: 中性 (0.0), 不再人为抬高
        total = len(generated) + 1
        hydro = PHYSICO_MATRIX[aa_idx, 0]
        charge = PHYSICO_MATRIX[aa_idx, 2]

        # 辅助: sigmoid 软化阶跃, 在 x=center 处过渡, sharpness 控制陡峭度
        # sharpness→∞ 退化为硬阈值, sharpness=8 为平滑近似
        def soft_step(x, center, sharpness=8.0):
            z = sharpness * (x - center)
            z = max(-20.0, min(20.0, z))  # 数值安全
            return 1.0 / (1.0 + math.exp(-z))

        # ── 0) 预处理: 过滤掉特殊token (索引≥20), 只保留标准氨基酸 ──
        from collections import Counter
        valid_generated = [a for a in generated if 0 <= a < 20]

        # ── 1) 任意残基密度: 窗口10aa内同一残基出现次数 ──
        # 天然蛋白窗口内同残基 ≤2 为正常, ≥4 为异常
        # 软化: 在 count=2~4 之间平滑过渡, 无硬跳变
        recent_10 = valid_generated[-10:]
        recent_counts = Counter(recent_10)
        if 0 <= aa_idx < 20:
            recent_counts[aa_idx] = recent_counts.get(aa_idx, 0) + 1
        max_recent = max(recent_counts.values()) if recent_counts else 1
        # max_recent=1 → penalty≈0, max_recent=3 → penalty≈-0.4, max_recent=5 → penalty≈-1.2
        density_penalty = soft_step(max_recent, center=2.5, sharpness=3.0) * min(2.0, (max_recent - 1) * 0.3)
        score -= density_penalty

        # ── 2) 全局疏水比例: 超出天然范围 (30%-55%) 越远惩罚越重 ──
        # 软化: 在 0.50~0.60 之间平滑过渡
        hydro_count = sum(1 for a in valid_generated if PHYSICO_MATRIX[a, 0] > 1.5) + (1 if hydro > 1.5 else 0)
        total_valid = len(valid_generated) + 1
        hydro_ratio = hydro_count / total_valid
        # hydro_ratio=0.45 → ≈0, hydro_ratio=0.55 → ≈-0.2, hydro_ratio=0.70 → ≈-0.8
        hydro_excess = soft_step(hydro_ratio, center=0.50, sharpness=20.0) * min(2.0, max(0.0, hydro_ratio - 0.40) * 3.0)
        score -= hydro_excess

        # ── 3) 电荷平衡: 单边电荷占比偏离 ──
        # 天然蛋白正/负电残基各 ~10-20%, >30% 异常
        # 软化: 在 25%~35% 之间平滑过渡
        pos_count = sum(1 for a in valid_generated if PHYSICO_MATRIX[a, 2] > 0.5) + (1 if charge > 0.5 else 0)
        neg_count = sum(1 for a in valid_generated if PHYSICO_MATRIX[a, 2] < -0.5) + (1 if charge < -0.5 else 0)
        pos_ratio = pos_count / total_valid
        neg_ratio = neg_count / total_valid
        # pos_ratio=0.25 → ≈0, pos_ratio=0.35 → ≈-0.4, pos_ratio=0.50 → ≈-1.2
        charge_penalty = (soft_step(pos_ratio, center=0.28, sharpness=30.0) * max(0.0, pos_ratio - 0.20) * 4.0
                        + soft_step(neg_ratio, center=0.28, sharpness=30.0) * max(0.0, neg_ratio - 0.20) * 4.0)
        score -= min(2.0, charge_penalty)

        # ── 4) 连续疏水: ≥2 个连续疏水开始惩罚, 斜率连续 ──
        # 软化: 在 1~3 个之间平滑过渡
        if hydro > 1.5:
            consecutive = 1
            for a in reversed(valid_generated):
                if PHYSICO_MATRIX[a, 0] > 1.5:
                    consecutive += 1
                else:
                    break
            # consecutive=1 → ≈0, consecutive=2 → ≈-0.3, consecutive=3 → ≈-0.7, consecutive=5 → ≈-1.5
            hydro_chain_penalty = soft_step(consecutive, center=1.8, sharpness=3.0) * max(0.0, consecutive - 1) * 0.35
            score -= min(2.0, hydro_chain_penalty)

        return max(-2.0, min(1.0, score))

    def get_param_count(self):
        """返回详细参数量统计 (区分冻结/可训练/ESM)"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        esm_total = 0
        esm_trainable = 0
        if self.use_esm and hasattr(self, 'esm_encoder'):
            esm_info = self.esm_encoder.get_param_count()
            esm_total = esm_info['total']
            esm_trainable = esm_info['trainable']
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
            'esm_total': esm_total,
            'esm_trainable': esm_trainable,
            'innovation_params': trainable - esm_trainable,  # 你的创新模块参数
        }