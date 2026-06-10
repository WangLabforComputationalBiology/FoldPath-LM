"""
从 UniProt Swiss-Prot 原始数据重新划分训练集/验证集
核心: 基于聚类的划分，确保同源序列不会跨集泄露

使用方法:
  python data_split.py

输出:
  D:\proteinllm\data\train_sequences.fasta
  D:\proteinllm\data\val_sequences.fasta
"""

import os
import hashlib
import random
from collections import defaultdict

random.seed(42)

SRC_PATH = r"D:\proteinllm\ceck\uniprot_sprot.fasta"
OUT_DIR = r"D:\proteinllm\data"
TRAIN_FILE = os.path.join(OUT_DIR, "train_sequences.fasta")
VAL_FILE = os.path.join(OUT_DIR, "val_sequences.fasta")
VAL_RATIO = 0.10          # 10% 验证集
MIN_LEN = 10              # 最短序列
MAX_LEN = 35213           # 最长序列 (保留所有)

# ============================================================
# 1. 加载 FASTA
# ============================================================
def load_fasta(filepath):
    """加载FASTA，返回 [(header, sequence), ...]"""
    entries = []
    current_header = None
    current_seq = []
    count = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    seq = ''.join(current_seq)
                    if len(seq) >= MIN_LEN:
                        entries.append((current_header, seq))
                        count += 1
                        if count % 100000 == 0:
                            print(f"  已加载 {count} 条...")
                current_header = line
                current_seq = []
            else:
                current_seq.append(line)
        if current_header is not None:
            seq = ''.join(current_seq)
            if len(seq) >= MIN_LEN:
                entries.append((current_header, seq))
                count += 1
    return entries

def save_fasta(filepath, entries):
    """保存FASTA"""
    with open(filepath, 'w') as f:
        for header, seq in entries:
            f.write(f"{header}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

# ============================================================
# 2. 精确去重
# ============================================================
def dedup_exact(entries):
    """去除完全相同的序列，保留首次出现"""
    seen = set()
    deduped = []
    dup_count = 0
    for header, seq in entries:
        h = hashlib.md5(seq.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append((header, seq))
        else:
            dup_count += 1
    return deduped, dup_count

# ============================================================
# 3. 同源聚类 (无需CD-HIT的近似方法)
# ============================================================
def build_clusters(entries):
    """
    基于多重签名聚类:
    - 签名1: 前20aa前缀 (捕获N端同源)
    - 签名2: 后20aa后缀 (捕获C端同源)  
    - 签名3: 长度+前5aa+后5aa (捕获片段同源)
    - 签名4: 3-mer频谱分桶 (捕获全局相似性)
    
    共享任一签名的序列归入同一簇 (Union-Find)
    """
    print("  构建同源聚类 (Union-Find)...")
    n = len(entries)
    parent = list(range(n))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    
    # 签名 → 首次出现的索引
    sig_maps = [dict() for _ in range(4)]
    
    for idx, (header, seq) in enumerate(entries):
        # 签名1: 前20aa
        if len(seq) >= 20:
            sig1 = seq[:20]
            if sig1 in sig_maps[0]:
                union(idx, sig_maps[0][sig1])
            else:
                sig_maps[0][sig1] = idx
        
        # 签名2: 后20aa
        if len(seq) >= 20:
            sig2 = seq[-20:]
            if sig2 in sig_maps[1]:
                union(idx, sig_maps[1][sig2])
            else:
                sig_maps[1][sig2] = idx
        
        # 签名3: 长度桶 + 前5aa + 后5aa
        len_bucket = len(seq) // 50  # 50aa为一个桶
        prefix5 = seq[:5] if len(seq) >= 5 else seq
        suffix5 = seq[-5:] if len(seq) >= 5 else seq
        sig3 = f"{len_bucket}|{prefix5}|{suffix5}"
        if sig3 in sig_maps[2]:
            union(idx, sig_maps[2][sig3])
        else:
            sig_maps[2][sig3] = idx
        
        # 签名4: 3-mer频谱 (每10个位置采样一个3-mer)
        kmers = []
        for i in range(0, min(len(seq) - 2, 200), 10):
            kmers.append(seq[i:i+3])
        sig4 = '|'.join(sorted(set(kmers)))
        if sig4 in sig_maps[3]:
            union(idx, sig_maps[3][sig4])
        else:
            sig_maps[3][sig4] = idx
        
        if (idx + 1) % 100000 == 0:
            print(f"  已处理 {idx+1}/{n} 条...")
    
    # 收集簇
    clusters = defaultdict(list)
    for idx in range(n):
        root = find(idx)
        clusters[root].append(idx)
    
    return dict(clusters)

# ============================================================
# 4. 聚类感知划分
# ============================================================
def cluster_aware_split(entries, clusters, val_ratio=0.10):
    """
    同一簇的所有序列进入同一集合 (train 或 val)
    按簇大小加权随机划分，确保两集比例合理
    """
    print("  聚类感知划分...")
    cluster_list = list(clusters.values())
    
    # 按簇大小降序排列
    cluster_list.sort(key=lambda c: len(c), reverse=True)
    
    train_indices = []
    val_indices = []
    train_count = 0
    val_count = 0
    total = len(entries)
    target_val = int(total * val_ratio)
    
    # 大簇优先分配 (大簇更可能包含同源序列，必须整体分配)
    for cluster in cluster_list:
        cluster_size = len(cluster)
        
        # 如果验证集已满，剩余全进训练集
        if val_count >= target_val:
            train_indices.extend(cluster)
            train_count += cluster_size
            continue
        
        # 按比例决定当前簇的去向
        # 当前验证集还差多少
        val_need = target_val - val_count
        # 剩余待分配总量
        remaining = total - train_count - val_count
        
        if remaining <= 0:
            train_indices.extend(cluster)
            train_count += cluster_size
            continue
        
        # 当前簇去val的概率 = 还需要分配给val的比例
        val_prob = val_need / remaining if remaining > 0 else 0
        
        if random.random() < val_prob:
            val_indices.extend(cluster)
            val_count += cluster_size
        else:
            train_indices.extend(cluster)
            train_count += cluster_size
    
    train_entries = [entries[i] for i in train_indices]
    val_entries = [entries[i] for i in val_indices]
    
    return train_entries, val_entries

# ============================================================
# 5. 验证无泄露
# ============================================================
def verify_no_leakage(train_entries, val_entries):
    """快速验证无泄露"""
    print("\n=== 泄露验证 ===")
    
    # 前20aa匹配检查
    train_prefixes = set()
    for _, seq in train_entries:
        if len(seq) >= 20:
            train_prefixes.add(seq[:20])
    
    val_prefix_matches = 0
    for _, seq in val_entries:
        if len(seq) >= 20 and seq[:20] in train_prefixes:
            val_prefix_matches += 1
    
    # 精确匹配检查 (采样)
    train_hashes = set()
    for _, seq in train_entries:
        train_hashes.add(hashlib.md5(seq.encode()).hexdigest())
    
    exact_overlap = 0
    for _, seq in val_entries:
        if hashlib.md5(seq.encode()).hexdigest() in train_hashes:
            exact_overlap += 1
    
    print(f"  训练集: {len(train_entries)} 条")
    print(f"  验证集: {len(val_entries)} 条")
    print(f"  精确重叠: {exact_overlap} 条")
    print(f"  前20aa匹配: {val_prefix_matches} / {len(val_entries)} ({val_prefix_matches/len(val_entries)*100:.2f}%)")
    
    if exact_overlap == 0 and val_prefix_matches == 0:
        print("  ✅ 无数据泄露!")
    else:
        print("  ⚠️ 仍有泄露，需要更严格的聚类")
    
    return exact_overlap, val_prefix_matches

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("FoldPath-LLM 数据集重新划分")
    print("=" * 60)
    
    # Step 1: 加载
    print("\n[1/5] 加载原始FASTA...")
    entries = load_fasta(SRC_PATH)
    print(f"  原始序列: {len(entries)} 条")
    
    # Step 2: 精确去重
    print("\n[2/5] 精确去重...")
    entries, dup_count = dedup_exact(entries)
    print(f"  去重后: {len(entries)} 条 (去除 {dup_count} 条重复)")
    
    # Step 3: 同源聚类
    print("\n[3/5] 同源聚类...")
    clusters = build_clusters(entries)
    num_clusters = len(clusters)
    max_cluster = max(len(c) for c in clusters.values())
    avg_cluster = sum(len(c) for c in clusters.values()) / num_clusters
    print(f"  簇数: {num_clusters}")
    print(f"  最大簇: {max_cluster} 条序列")
    print(f"  平均簇大小: {avg_cluster:.2f}")
    
    # Step 4: 聚类感知划分
    print("\n[4/5] 聚类感知划分 (90% train / 10% val)...")
    train_entries, val_entries = cluster_aware_split(entries, clusters, VAL_RATIO)
    print(f"  训练集: {len(train_entries)} 条")
    print(f"  验证集: {len(val_entries)} 条")
    
    # Step 5: 保存
    print("\n[5/5] 保存...")
    os.makedirs(OUT_DIR, exist_ok=True)
    save_fasta(TRAIN_FILE, train_entries)
    save_fasta(VAL_FILE, val_entries)
    print(f"  训练集: {TRAIN_FILE}")
    print(f"  验证集: {VAL_FILE}")
    
    # 验证
    verify_no_leakage(train_entries, val_entries)
    
    # 基本统计
    train_lens = [len(s) for _, s in train_entries]
    val_lens = [len(s) for _, s in val_entries]
    print(f"\n=== 数据统计 ===")
    print(f"  训练集长度: avg={sum(train_lens)/len(train_lens):.0f}, median={sorted(train_lens)[len(train_lens)//2]}")
    print(f"  验证集长度: avg={sum(val_lens)/len(val_lens):.0f}, median={sorted(val_lens)[len(val_lens)//2]}")
    
    print("\n=== 完成 ===")

if __name__ == "__main__":
    main()
