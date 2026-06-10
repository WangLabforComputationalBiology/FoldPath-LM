"""检查 evaluation.py batch_score 的方差计算"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import NaturalnessComparator

comp = NaturalnessComparator(reference_fasta='data/train_sequences.fasta')

test_seqs = [
    "MALVGHMQRTTAESDIAATAR",           # 21
    "MALVGHMQRTTAESDIAATAR" * 2,       # 42
    "MALVGHMQRTTAESDIAATAR" * 5,       # 105
    "MALVGHMQRTTAESDIAATAR" * 10,      # 210 (模拟RITA按200步)
]

for seq in test_seqs:
    result = comp.score(seq)
    print(f'长度{len(seq):>4}: total={result["total"]:.4f}  '
          f'aa_js={result["detail"].get("aa_js",0):.4f}  '
          f'dipep={result["detail"].get("dipep_corr",0):.4f}  '
          f'kmer={result["detail"].get("kmer_recall",0):.4f}  '
          f'len={result["detail"].get("length_naturalness",0):.4f}  '
          f'helix={result["detail"].get("helix_periodicity",0):.4f}')

# 批量
batch = comp.batch_score(test_seqs * 12)  # 48条，模拟50条
print(f'\n批量: mean={batch["mean"]:.4f} std={batch["std"]:.4f} '
      f'min={batch["min"]:.4f} max={batch["max"]:.4f}')

# 查看个体值
individual = [s['total'] for s in batch['individual']]
print(f'唯一值: {sorted(set([round(x,4) for x in individual]))}')
