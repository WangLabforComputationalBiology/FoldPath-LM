"""诊断 k-mer 索引为什么全返回0"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import load_fasta

# 加载参考序列
seqs = load_fasta('data/train_sequences.fasta', max_seqs=100)
print(f'加载 {len(seqs)} 条参考序列')
print(f'首条前30字符: {seqs[0][:30]}')
print(f'首条长度: {len(seqs[0])}')

# 构建 k-mer 索引
kmer_idx = {}
for seq in seqs:
    for i in range(len(seq) - 6):
        kmer = seq[i:i+7]
        kmer_idx[kmer] = kmer_idx.get(kmer, 0) + 1

print(f'K-mer 索引大小: {len(kmer_idx)}')

# 测试
test = "MALVGHMQRTTAESDIAATAR"
print(f'\n测试序列: {test} (长度={len(test)})')
found = 0
for i in range(len(test) - 6):
    kmer = test[i:i+7]
    in_idx = kmer in kmer_idx
    if in_idx:
        found += 1
    print(f'  [{i}] {kmer} → {"✓" if in_idx else "✗"}')
print(f'找到: {found}/{len(test)-6}')
