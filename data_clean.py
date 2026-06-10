"""
数据集清洗: 去除训练集-验证集泄露 + 内部冗余

使用方法:
  python data_clean.py

输出:
  data/train_sequences_clean.fasta
  data/val_sequences_clean.fasta
"""

import os
import hashlib
from collections import defaultdict

DATA_DIR = r"D:\proteinllm\data"

def load_fasta(filepath):
    """加载FASTA，返回 [(header, sequence), ...]"""
    entries = []
    current_header = None
    current_seq = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    seq = ''.join(current_seq)
                    if len(seq) >= 10:
                        entries.append((current_header, seq))
                current_header = line
                current_seq = []
            else:
                current_seq.append(line)
        if current_header is not None:
            seq = ''.join(current_seq)
            if len(seq) >= 10:
                entries.append((current_header, seq))
    return entries

def save_fasta(filepath, entries):
    """保存FASTA"""
    with open(filepath, 'w') as f:
        for header, seq in entries:
            f.write(f"{header}\n")
            # 每80字符换行
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

def seq_hash(seq):
    """序列哈希 (用于快速去重)"""
    return hashlib.md5(seq.encode()).hexdigest()

def prefix_key(seq, prefix_len=20):
    """前缀键 (用于近似同源检测)"""
    return seq[:prefix_len] if len(seq) >= prefix_len else seq

def main():
    print("=" * 60)
    print("FoldPath-LLM 数据集清洗")
    print("=" * 60)

    # 1. 加载
    print("\n[1/5] 加载FASTA文件...")
    train_entries = load_fasta(os.path.join(DATA_DIR, "train_sequences.fasta"))
    val_entries = load_fasta(os.path.join(DATA_DIR, "val_sequences.fasta"))
    print(f"  训练集: {len(train_entries)} 条")
    print(f"  验证集: {len(val_entries)} 条")

    # 2. 训练集内部去重 (保留首次出现的)
    print("\n[2/5] 训练集内部去重...")
    seen_train = set()
    train_dedup = []
    dup_count = 0
    for header, seq in train_entries:
        h = seq_hash(seq)
        if h not in seen_train:
            seen_train.add(h)
            train_dedup.append((header, seq))
        else:
            dup_count += 1
    print(f"  去重: {len(train_entries)} → {len(train_dedup)} (去除 {dup_count} 条重复)")

    # 3. 训练集前缀索引 (用于同源检测)
    print("\n[3/5] 构建训练集前缀索引...")
    train_prefix_set = set()
    train_seq_set = seen_train  # 已经是去重后的hash集合
    for _, seq in train_dedup:
        train_prefix_set.add(prefix_key(seq, 20))

    # 4. 验证集清洗
    print("\n[4/5] 清洗验证集...")
    val_clean = []
    removed_exact = 0
    removed_homolog = 0

    # 先去重
    seen_val = set()
    val_dedup = []
    for header, seq in val_entries:
        h = seq_hash(seq)
        if h not in seen_val:
            seen_val.add(h)
            val_dedup.append((header, seq))

    for header, seq in val_dedup:
        h = seq_hash(seq)
        # 4a. 完全相同的序列 → 删除
        if h in train_seq_set:
            removed_exact += 1
            continue
        # 4b. 前20aa完全匹配训练集 → 标记为同源，删除
        if prefix_key(seq, 20) in train_prefix_set:
            removed_homolog += 1
            continue
        val_clean.append((header, seq))

    print(f"  验证集去重: {len(val_entries)} → {len(val_dedup)}")
    print(f"  去除与训练集完全相同: {removed_exact} 条")
    print(f"  去除与训练集同源(前20aa匹配): {removed_homolog} 条")
    print(f"  清洗后验证集: {len(val_clean)} 条")

    # 5. 保存
    print("\n[5/5] 保存清洗后数据...")
    train_path = os.path.join(DATA_DIR, "train_sequences_clean.fasta")
    val_path = os.path.join(DATA_DIR, "val_sequences_clean.fasta")
    save_fasta(train_path, train_dedup)
    save_fasta(val_path, val_clean)
    print(f"  训练集: {train_path} ({len(train_dedup)} 条)")
    print(f"  验证集: {val_path} ({len(val_clean)} 条)")

    # 6. 验证
    print("\n=== 清洗后验证 ===")
    train_hashes = set(seq_hash(s) for _, s in train_dedup)
    val_hashes = set(seq_hash(s) for _, s in val_clean)
    overlap = train_hashes & val_hashes
    print(f"  训练集-验证集重叠: {len(overlap)} 条")

    train_prefixes = set(prefix_key(s, 20) for _, s in train_dedup)
    val_prefix_matches = sum(1 for _, s in val_clean if prefix_key(s, 20) in train_prefixes)
    print(f"  前20aa匹配: {val_prefix_matches} / {len(val_clean)} ({val_prefix_matches/len(val_clean)*100:.2f}%)")

    if len(overlap) == 0 and val_prefix_matches == 0:
        print("\n✅ 数据泄露已完全消除!")
    else:
        print(f"\n⚠️ 仍有残留泄露，可能需要更严格的CD-HIT聚类 (40%阈值)")

    print("\n=== 完成 ===")
    print(f"\n使用方法: 修改 config.py 中 DataConfig.train_file 和 val_file 为:")
    print(f"  train_file = 'train_sequences_clean.fasta'")
    print(f"  val_file = 'val_sequences_clean.fasta'")

if __name__ == "__main__":
    main()
