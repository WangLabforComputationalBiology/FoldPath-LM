"""
FoldPath-LLM: PDB序列文件转FASTA格式
将PDB下载的txt序列文件或标准FASTA文件转换为训练用的FASTA格式

Usage:
    python convert_to_fasta.py --input_dir /path/to/pdb_txt_files
    python convert_to_fasta.py --input_file /path/to/single_file.txt
    python convert_to_fasta.py --input_dir /path/to/files --filter protein
    python convert_to_fasta.py --input_dir /path/to/files --filter protein --min_len 30 --max_len 256
"""

import os
import sys
import argparse
import random
from collections import Counter

# 标准氨基酸
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
VAL_RATIO = 0.1
RANDOM_SEED = 42


def parse_pdb_txt_file(filepath):
    """
    解析PDB格式的txt文件
    
    文件格式示例:
    100d_A mol:na length:10  DNA/RNA (5'-R(*CP*)-D(*CP*GP*GP*CP*GP*CP*CP*GP*)-R(*G)-3')
    CCGGCGCCGG
    
    返回: list of (header, sequence, molecule_type)
    """
    entries = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # 检测header行: 包含 mol: 标记
        if 'mol:' in line:
            header = line
            # 提取分子类型
            mol_type = "unknown"
            if "mol:protein" in line:
                mol_type = "protein"
            elif "mol:na" in line or "mol:DNA" in line or "mol:RNA" in line:
                mol_type = "na"
            
            # 下一行应该是序列
            seq = ""
            if i + 1 < len(lines):
                seq_line = lines[i + 1].strip()
                if seq_line and 'mol:' not in seq_line:
                    seq = seq_line
                    i += 2
                else:
                    i += 1
            else:
                i += 1
            
            if seq:
                entries.append((header, seq, mol_type))
        else:
            i += 1
    
    return entries


def parse_fasta_file(filepath):
    """
    解析标准FASTA文件
    
    返回: list of (header, sequence, molecule_type)
    """
    entries = []
    current_header = None
    current_seq = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_header is not None and current_seq:
                    seq = ''.join(current_seq)
                    entries.append((current_header, seq, "protein"))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
    
    if current_header is not None and current_seq:
        seq = ''.join(current_seq)
        entries.append((current_header, seq, "protein"))
    
    return entries


def clean_sequence(seq, remove_non_standard=True):
    """清洗序列: 转大写, 移除非标准氨基酸"""
    cleaned = ""
    for c in seq.upper():
        if c in STANDARD_AA:
            cleaned += c
        elif c in ('X', 'B', 'Z', 'J', 'U', 'O'):
            if not remove_non_standard:
                cleaned += 'X'
    return cleaned


def extract_id_from_header(header):
    """从header中提取序列ID"""
    parts = header.split()
    if parts:
        return parts[0].replace('|', '_').replace(':', '_')
    return "unknown"


def process_files(input_path, filter_type=None, min_len=1, max_len=99999,
                  remove_non_standard=True, val_ratio=VAL_RATIO):
    """处理输入文件，返回训练集和验证集"""
    all_entries = []
    stats = {
        'total_files': 0, 'total_entries': 0,
        'protein': 0, 'na': 0, 'unknown': 0,
        'filtered_by_type': 0, 'filtered_by_length': 0,
        'filtered_by_composition': 0, 'final_count': 0,
        'aa_composition': Counter(), 'length_distribution': Counter(),
    }
    
    # 收集所有文件
    files = []
    if os.path.isfile(input_path):
        files = [input_path]
    elif os.path.isdir(input_path):
        for fname in os.listdir(input_path):
            fpath = os.path.join(input_path, fname)
            if os.path.isfile(fpath):
                files.append(fpath)
    else:
        print(f"[ERROR] 路径不存在: {input_path}")
        return [], [], stats
    
    print(f"[INFO] 找到 {len(files)} 个文件")
    
    for fpath in files:
        stats['total_files'] += 1
        fname = os.path.basename(fpath)
        
        if fname.endswith('.fasta') or fname.endswith('.fa') or fname.endswith('.faa'):
            entries = parse_fasta_file(fpath)
        else:
            entries = parse_pdb_txt_file(fpath)
        
        all_entries.extend(entries)
        stats['total_entries'] += len(entries)
        if entries:
            print(f"  {fname}: {len(entries)} 条序列")
    
    print(f"\n[INFO] 共解析 {stats['total_entries']} 条序列")
    
    for _, _, mol_type in all_entries:
        stats[mol_type] = stats.get(mol_type, 0) + 1
    print(f"  蛋白质: {stats.get('protein', 0)}")
    print(f"  核酸: {stats.get('na', 0)}")
    print(f"  未知: {stats.get('unknown', 0)}")
    
    # 过滤
    filtered_entries = []
    for header, seq, mol_type in all_entries:
        if filter_type and mol_type != filter_type:
            stats['filtered_by_type'] += 1
            continue
        
        cleaned = clean_sequence(seq, remove_non_standard=remove_non_standard)
        
        if len(cleaned) == 0:
            stats['filtered_by_composition'] += 1
            continue
        
        standard_count = sum(1 for c in cleaned if c in STANDARD_AA)
        if standard_count / len(cleaned) < 0.8:
            stats['filtered_by_composition'] += 1
            continue
        
        if len(cleaned) < min_len or len(cleaned) > max_len:
            stats['filtered_by_length'] += 1
            continue
        
        seq_id = extract_id_from_header(header)
        stats['aa_composition'].update(cleaned)
        length_bucket = (len(cleaned) // 50) * 50
        stats['length_distribution'][f"{length_bucket}-{length_bucket+49}"] += 1
        
        filtered_entries.append((seq_id, header, cleaned))
    
    stats['final_count'] = len(filtered_entries)
    
    print(f"\n[INFO] 过滤结果:")
    print(f"  按类型过滤: {stats['filtered_by_type']}")
    print(f"  按长度过滤: {stats['filtered_by_length']}")
    print(f"  按组成过滤: {stats['filtered_by_composition']}")
    print(f"  最终保留: {stats['final_count']}")
    
    # 划分训练集和验证集
    random.seed(RANDOM_SEED)
    random.shuffle(filtered_entries)
    val_count = max(1, int(len(filtered_entries) * val_ratio))
    val_entries = filtered_entries[:val_count]
    train_entries = filtered_entries[val_count:]
    
    print(f"\n[INFO] 数据划分:")
    print(f"  训练集: {len(train_entries)}")
    print(f"  验证集: {len(val_entries)}")
    
    return train_entries, val_entries, stats


def write_fasta(entries, output_path):
    """写入FASTA文件"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for seq_id, header, seq in entries:
            f.write(f">{seq_id} {header}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")
    file_size = os.path.getsize(output_path) / 1e6
    print(f"  已保存: {output_path} ({file_size:.1f} MB)")


def print_stats(stats):
    """打印统计信息"""
    print("\n" + "=" * 60)
    print("  数据统计")
    print("=" * 60)
    print(f"  处理文件数: {stats['total_files']}")
    print(f"  原始序列数: {stats['total_entries']}")
    print(f"  最终序列数: {stats['final_count']}")
    
    if stats['length_distribution']:
        print(f"\n  长度分布:")
        for bucket in sorted(stats['length_distribution'].keys()):
            count = stats['length_distribution'][bucket]
            bar = "█" * min(count // max(1, stats['final_count'] // 50), 50)
            print(f"    {bucket}: {count:>6} {bar}")
    
    if stats['aa_composition']:
        print(f"\n  氨基酸组成 (Top 10):")
        total_aa = sum(stats['aa_composition'].values())
        for aa, count in stats['aa_composition'].most_common(10):
            pct = count / total_aa * 100
            print(f"    {aa}: {count:>8} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="PDB序列文件转FASTA格式")
    parser.add_argument("--input_dir", type=str, help="输入目录（包含txt文件）")
    parser.add_argument("--input_file", type=str, help="输入单个文件")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（默认: data/）")
    parser.add_argument("--filter", type=str, choices=["protein", "na", "all"], default="protein",
                        help="过滤分子类型 (protein/na/all)")
    parser.add_argument("--min_len", type=int, default=1, help="最小序列长度 (默认: 30)")
    parser.add_argument("--max_len", type=int, default=99999, help="最大序列长度 (默认: 256)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例 (默认: 0.1)")
    parser.add_argument("--keep_non_standard", action="store_true",
                        help="保留非标准氨基酸（替换为X）")
    
    args = parser.parse_args()
    
    if args.input_file:
        input_path = args.input_file
    elif args.input_dir:
        input_path = args.input_dir
    else:
        print("[ERROR] 请指定 --input_dir 或 --input_file")
        print("\n示例:")
        print("  python convert_to_fasta.py --input_dir /path/to/pdb_files")
        print("  python convert_to_fasta.py --input_file /path/to/uniprot.fasta")
        print("  python convert_to_fasta.py --input_dir /path/to/files --filter protein --min_len 30 --max_len 256")
        return
    
    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    filter_type = None if args.filter == "all" else args.filter
    
    print("=" * 60)
    print("  FoldPath-LLM 数据转换工具")
    print("=" * 60)
    print(f"  输入: {input_path}")
    print(f"  输出: {output_dir}")
    print(f"  过滤: {args.filter}")
    print(f"  长度: {args.min_len}-{args.max_len}")
    print()
    
    train_entries, val_entries, stats = process_files(
        input_path=input_path,
        filter_type=filter_type,
        min_len=args.min_len,
        max_len=args.max_len,
        remove_non_standard=not args.keep_non_standard,
        val_ratio=args.val_ratio,
    )
    
    if not train_entries:
        print("\n[WARNING] 没有有效的训练数据！")
        print("  可能原因:")
        print("  1. 输入文件格式不正确")
        print("  2. 过滤条件太严格（尝试 --filter all 或调整 --min_len/--max_len）")
        print("  3. 数据中不包含蛋白质序列（检查 mol: 标记）")
        return
    
    train_path = os.path.join(output_dir, "train.fasta")
    val_path = os.path.join(output_dir, "val.fasta")
    
    print(f"\n[INFO] 写入FASTA文件:")
    write_fasta(train_entries, train_path)
    write_fasta(val_entries, val_path)
    
    print_stats(stats)
    
    print(f"\n✅ 转换完成！")
    print(f"  训练集: {train_path}")
    print(f"  验证集: {val_path}")
    print(f"\n  使用方式:")
    print(f"    python train.py --use_synthetic False")
    print(f"    或在可视化界面中取消勾选'合成数据'")


if __name__ == "__main__":
    main()