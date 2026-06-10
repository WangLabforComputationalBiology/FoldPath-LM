"""调试天然度评测——看单条序列的子项分数"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import FoldPathBenchmark, NaturalnessComparator

bench = FoldPathBenchmark(reference_fasta='data/train_sequences.fasta')

# 构造三条差异明显的测试序列
short = "MALVGHMQRTTAESDIAATAR"       # 21 AA，RITA典型
medium = "M" * 50                       # 50 AA，全甲硫氨酸（极端不天然）
long_real = "MALVGHMQRTTAESDIAATAR" * 5 # 105 AA，重复短序列

for label, seq in [("RITA典型(21AA)", short), ("全M(50AA)", medium), ("重复片段(105AA)", long_real)]:
    result = bench.naturalness.score(seq)
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  Total: {result['total']:.4f}")
    for k, v in result['detail'].items():
        print(f"    {k}: {v:.4f}")

# 批量评测——看方差
seqs_batch = [short] * 10 + [medium] * 5 + [long_real] * 5
batch_result = bench.naturalness.batch_score(seqs_batch)
print(f"\n{'='*50}")
print(f"  批量评测 (20条混合):")
print(f"  Mean: {batch_result['mean']:.4f}")
print(f"  Std:  {batch_result['std']:.4f}")
print(f"  Min:  {batch_result['min']:.4f}")
print(f"  Max:  {batch_result['max']:.4f}")
