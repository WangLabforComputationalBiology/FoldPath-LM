"""检查训练集和验证集的数据质量和潜在泄露"""
import os
from collections import Counter

DATA_DIR = r"D:\proteinllm\data"

def load_fasta(filepath, max_seqs=None):
    sequences = []
    headers = []
    current_seq = []
    current_header = ""
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    seq = ''.join(current_seq)
                    if len(seq) >= 10:
                        sequences.append(seq)
                        headers.append(current_header)
                    current_seq = []
                current_header = line[1:]
            else:
                current_seq.append(line)
        if current_seq:
            seq = ''.join(current_seq)
            if len(seq) >= 10:
                sequences.append(seq)
                headers.append(current_header)
    return sequences, headers

def check_aa_composition(sequences, name):
    aa_counter = Counter()
    total = 0
    for seq in sequences:
        for aa in seq:
            if aa in "ACDEFGHIKLMNPQRSTVWY":
                aa_counter[aa] += 1
                total += 1
    print(f"\n{name} 氨基酸组成 (共{total}个残基):")
    for aa in sorted(aa_counter.keys()):
        pct = aa_counter[aa] / total * 100
        bar = "█" * int(pct / 0.5)
        print(f"  {aa}: {pct:5.2f}%  {bar}")

def check_length_distribution(sequences, name):
    lengths = [len(s) for s in sequences]
    print(f"\n{name} 长度分布:")
    print(f"  总数: {len(lengths)}")
    print(f"  最短: {min(lengths)}, 最长: {max(lengths)}")
    print(f"  平均: {sum(lengths)/len(lengths):.1f}")
    print(f"  中位: {sorted(lengths)[len(lengths)//2]}")
    # 分桶
    buckets = [0]*7
    for l in lengths:
        if l < 50: buckets[0] += 1
        elif l < 100: buckets[1] += 1
        elif l < 200: buckets[2] += 1
        elif l < 300: buckets[3] += 1
        elif l < 500: buckets[4] += 1
        elif l < 1000: buckets[5] += 1
        else: buckets[6] += 1
    labels = ["<50", "50-100", "100-200", "200-300", "300-500", "500-1000", ">1000"]
    for label, count in zip(labels, buckets):
        pct = count / len(lengths) * 100
        print(f"  {label:>8s}: {count:7d} ({pct:5.1f}%)")

def check_overlap(train_seqs, val_seqs, train_headers, val_headers):
    """检查训练集和验证集之间的重叠"""
    print("\n=== 数据泄露检查 ===")
    
    # 1. 完全相同的序列
    train_set = set(train_seqs)
    val_set = set(val_seqs)
    overlap = train_set & val_set
    print(f"\n1. 完全相同的序列数: {len(overlap)} / {len(val_set)} ({len(overlap)/len(val_set)*100:.2f}%)")
    if len(overlap) > 0 and len(overlap) <= 20:
        for seq in list(overlap)[:5]:
            print(f"   示例: {seq[:60]}...")
    
    # 2. Header中的ID重叠
    train_ids = set()
    for h in train_headers:
        parts = h.split()
        if parts:
            train_ids.add(parts[0])
    val_ids = set()
    for h in val_headers:
        parts = h.split()
        if parts:
            val_ids.add(parts[0])
    id_overlap = train_ids & val_ids
    print(f"\n2. 相同序列ID数: {len(id_overlap)} / {len(val_ids)} ({len(id_overlap)/len(val_ids)*100:.2f}%)")
    if len(id_overlap) > 0 and len(id_overlap) <= 20:
        for pid in list(id_overlap)[:5]:
            print(f"   示例ID: {pid}")
    
    # 3. 短序列前缀重叠（快速近似同源检测）
    # 取前20个残基作为指纹，检测高度相似序列
    train_prefixes = {}
    for i, seq in enumerate(train_seqs):
        prefix = seq[:20] if len(seq) >= 20 else seq
        if prefix not in train_prefixes:
            train_prefixes[prefix] = 0
        train_prefixes[prefix] += 1
    
    val_prefix_matches = 0
    for seq in val_seqs:
        prefix = seq[:20] if len(seq) >= 20 else seq
        if prefix in train_prefixes:
            val_prefix_matches += 1
    
    print(f"\n3. 前20aa完全匹配的验证序列数: {val_prefix_matches} / {len(val_seqs)} ({val_prefix_matches/len(val_seqs)*100:.2f}%)")
    
    # 4. 训练集内部冗余
    print(f"\n4. 训练集去重前后: {len(train_seqs)} → {len(train_set)} (冗余: {len(train_seqs)-len(train_set)})")
    print(f"   验证集去重前后: {len(val_seqs)} → {len(val_set)} (冗余: {len(val_seqs)-len(val_set)})")

if __name__ == "__main__":
    print("加载训练集...")
    train_seqs, train_headers = load_fasta(os.path.join(DATA_DIR, "train_sequences.fasta"))
    print(f"训练集: {len(train_seqs)} 条序列")
    
    print("\n加载验证集...")
    val_seqs, val_headers = load_fasta(os.path.join(DATA_DIR, "val_sequences.fasta"))
    print(f"验证集: {len(val_seqs)} 条序列")
    
    # 基本统计
    check_length_distribution(train_seqs, "训练集")
    check_length_distribution(val_seqs, "验证集")
    
    # 氨基酸组成 (跳过，太慢)
    # check_aa_composition(train_seqs, "训练集")
    # check_aa_composition(val_seqs, "验证集")
    
    # 重叠检查
    check_overlap(train_seqs, val_seqs, train_headers, val_headers)
    
    print("\n=== 检查完成 ===")
