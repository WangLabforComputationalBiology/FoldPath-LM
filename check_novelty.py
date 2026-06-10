"""
序列新颖性检测: 生成序列 vs 训练集的最近邻 identity
用法: python check_novelty.py --checkpoint esmpro/foldpath_best.pt --num-samples 50
"""
import torch
import sys
import os
import json
import argparse
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True, help='模型路径')
parser.add_argument('--data-dir', type=str, default='./data')
parser.add_argument('--num-samples', type=int, default=50)
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--output', type=str, default='novelty_result.json')
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import IDX_TO_AA, AA_TO_IDX
from dataset import load_fasta
from generate import ProteinGenerator
from config import GenerateConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 1. 生成序列 ──
print(f'加载模型: {args.checkpoint}')
gen = ProteinGenerator(checkpoint_path=args.checkpoint, device=device)
gc = GenerateConfig()
gc.num_samples = args.num_samples
gc.max_length = 256
gc.temperature = args.temperature
gc.top_k = 50
gc.top_p = 0.92
gc.use_physico_filter = True

print(f'生成 {args.num_samples} 条序列...')
seqs, _ = gen.generate(gc)
seqs = [s for s in seqs if len(s) >= 20]
print(f'有效序列: {len(seqs)}')

# ── 2. 加载训练集 ──
train_path = os.path.join(args.data_dir, 'train_sequences.fasta')
print(f'加载训练集: {train_path}')
train_seqs = load_fasta(train_path, max_seqs=50000)
train_seqs = [s for s in train_seqs if len(s) >= 20]
print(f'训练集序列数: {len(train_seqs)}')

# ── 3. 最近邻 identity ──
print('\n计算最近邻 identity...')
identities = []
nn_info = []

# 先对训练集做k-mer索引加速 (5-mer)
print('  构建训练集5-mer索引...')
kmer_index = {}
for idx, tseq in enumerate(train_seqs):
    for i in range(len(tseq) - 4):
        kmer = tseq[i:i+5]
        if kmer not in kmer_index:
            kmer_index[kmer] = set()
        kmer_index[kmer].add(idx)

for gi, gseq in enumerate(seqs):
    # 用5-mer快速候选 → 只对候选做精确比对
    candidates = set()
    for i in range(len(gseq) - 4):
        kmer = gseq[i:i+5]
        if kmer in kmer_index:
            candidates.update(kmer_index[kmer])
            if len(candidates) > 200:
                break  # 限制候选数

    if not candidates:
        # 无共享5-mer → 极低 identity
        max_id = 0.0
        best_seq = train_seqs[0]
    else:
        max_id = 0.0
        best_idx = 0
        for cidx in list(candidates)[:100]:  # 最多比100条
            tseq = train_seqs[cidx]
            min_len = min(len(gseq), len(tseq))
            if min_len < 10:
                continue
            matches = 0
            for i in range(min_len):
                if gseq[i] == tseq[i]:
                    matches += 1
            # 滑动窗口比对(取最大): 简单版本用起始对齐
            # 更精确: 滑动窗口
            best_local = 0
            for offset in range(max(0, len(tseq) - len(gseq) + 1)):
                m = sum(1 for j in range(min_len) if gseq[j] == tseq[offset + j])
                best_local = max(best_local, m / min_len)
            if best_local > max_id:
                max_id = best_local
                best_seq = tseq

    identities.append(max_id)
    nn_info.append({
        'gen_idx': gi,
        'gen_seq': gseq[:60],
        'gen_len': len(gseq),
        'max_identity': round(max_id, 4),
        'nn_seq': best_seq[:60],
        'nn_len': len(best_seq),
    })

    if (gi + 1) % 10 == 0:
        print(f'  [{gi+1}/{len(seqs)}]')

# ── 4. 统计 ──
identities = np.array(identities)
print('\n' + '=' * 60)
print('  序列新颖性检测结果')
print('=' * 60)
print(f'  生成序列数: {len(seqs)}')
print(f'  训练集大小: {len(train_seqs)}')
print(f'')
print(f'  最近邻 identity 分布:')
print(f'    均值 ± 标准差: {identities.mean():.4f} ± {identities.std():.4f}')
print(f'    中位数:         {np.median(identities):.4f}')
print(f'    最小值:         {identities.min():.4f}')
print(f'    最大值:         {identities.max():.4f}')
print(f'')
# 分布统计
bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hist, _ = np.histogram(identities, bins=bins)
print(f'  分布直方图:')
for i in range(len(hist)):
    label = f'{bins[i]:.1f}-{bins[i+1]:.1f}'
    bar = '█' * hist[i]
    print(f'    {label:>8}: {bar} ({hist[i]})')

# 判断
low = (identities < 0.4).mean()
mid = ((0.4 <= identities) & (identities < 0.7)).mean()
high = (identities >= 0.7).mean()
print(f'')
print(f'  低相似度 (<0.40): {low:.0%}  — 高新颖度')
print(f'  中等相似度 (0.40-0.70): {mid:.0%}  — 同源蛋白水平')
print(f'  高相似度 (≥0.70): {high:.0%}  — 可能记忆')

if high < 0.05:
    print(f'  ✅ 无明显记忆现象 (高相似度 <5%)')
elif high < 0.15:
    print(f'  ⚠ 少量高相似序列 ({high:.0%})，建议人工检查')
else:
    print(f'  ❌ 高相似序列偏多 ({high:.0%})，可能过拟合训练集')
print('=' * 60)

# 保存
output = {
    'timestamp': datetime.now().isoformat(),
    'checkpoint': args.checkpoint,
    'num_generated': len(seqs),
    'num_train': len(train_seqs),
    'identity_mean': round(float(identities.mean()), 4),
    'identity_std': round(float(identities.std()), 4),
    'identity_median': round(float(np.median(identities)), 4),
    'identity_min': round(float(identities.min()), 4),
    'identity_max': round(float(identities.max()), 4),
    'low_novelty_pct': round(float(low), 4),
    'mid_novelty_pct': round(float(mid), 4),
    'high_novelty_pct': round(float(high), 4),
    'nn_details': nn_info,
}
with open(args.output, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f'\n结果已保存: {args.output}')
