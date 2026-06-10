"""
FoldPath-LLM: Dataset Loading
折叠路径引导的蛋白质设计大语言模型 - 数据集加载模块
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import random
import numpy as np
from config import AA_TO_IDX, BOS_IDX, EOS_IDX, PAD_IDX, DataConfig


class ProteinSequenceDataset(Dataset):
    """蛋白质序列数据集"""

    def __init__(self, sequences, max_len=128):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) > self.max_len - 2:
            seq = seq[:self.max_len - 2]
        token_ids = [BOS_IDX]
        for aa in seq:
            token_ids.append(AA_TO_IDX.get(aa, 0))
        token_ids.append(EOS_IDX)
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]
        pad_len = self.max_len - len(input_ids)
        input_ids = input_ids + [PAD_IDX] * pad_len
        target_ids = target_ids + [PAD_IDX] * pad_len
        mask = [1] * (len(token_ids) - 1) + [0] * pad_len
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'sequence': seq,  # 原始氨基酸序列 (ESM-2 编码需要)
        }


def load_fasta(filepath, max_seqs=None, min_len=10, max_len=None):
    """从FASTA文件加载蛋白质序列 (长序列自动截断而非丢弃)"""
    if max_len is None:
        from config import ModelConfig
        max_len = ModelConfig.max_seq_len
    sequences = []
    current_seq = []
    if not os.path.exists(filepath):
        print(f"[WARNING] FASTA文件不存在: {filepath}")
        return None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    seq = ''.join(current_seq)
                    seq = ''.join(c for c in seq if c in AA_TO_IDX)
                    if len(seq) >= min_len:
                        sequences.append(seq[:max_len] if len(seq) > max_len else seq)
                    current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            seq = ''.join(current_seq)
            seq = ''.join(c for c in seq if c in AA_TO_IDX)
            if len(seq) >= min_len:
                sequences.append(seq[:max_len] if len(seq) > max_len else seq)
    if max_seqs and len(sequences) > max_seqs:
        sequences = random.sample(sequences, max_seqs)
    print(f"[INFO] 从 {filepath} 加载了 {len(sequences)} 条序列 (max_len={max_len})")
    return sequences


def generate_synthetic_data(num_seqs=5000, min_len=30, max_len=120):
    """生成合成蛋白质序列数据 (原型测试用)"""
    aa_freq = {
        'A': 0.074, 'C': 0.025, 'D': 0.054, 'E': 0.054, 'F': 0.047,
        'G': 0.074, 'H': 0.026, 'I': 0.068, 'K': 0.058, 'L': 0.099,
        'M': 0.025, 'N': 0.045, 'P': 0.039, 'Q': 0.034, 'R': 0.052,
        'S': 0.057, 'T': 0.051, 'V': 0.073, 'W': 0.013, 'Y': 0.032,
    }
    aas = list(aa_freq.keys())
    probs = list(aa_freq.values())
    sequences = []
    for _ in range(num_seqs):
        length = random.randint(min_len, max_len)
        seq = ''.join(random.choices(aas, weights=probs, k=length))
        sequences.append(seq)
    print(f"[INFO] 生成了 {num_seqs} 条合成蛋白质序列")
    return sequences


def create_dataloaders(config=None, use_synthetic=False, batch_size=None):
    """创建训练和验证数据加载器"""
    if config is None:
        config = DataConfig()
    if batch_size is None:
        batch_size = 8
    train_path = os.path.join(config.data_dir, config.train_file)
    val_path = os.path.join(config.data_dir, config.val_file)
    train_seqs = load_fasta(train_path) if not use_synthetic else None
    val_seqs = load_fasta(val_path) if not use_synthetic else None
    if not train_seqs:
        train_seqs = generate_synthetic_data(num_seqs=5000, min_len=30, max_len=120)
    if not val_seqs:
        val_seqs = generate_synthetic_data(num_seqs=500, min_len=30, max_len=120)
    from config import ModelConfig
    max_len = ModelConfig.max_seq_len
    train_dataset = ProteinSequenceDataset(train_seqs, max_len=max_len)
    val_dataset = ProteinSequenceDataset(val_seqs, max_len=max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    print(f"[INFO] 训练集: {len(train_dataset)} 条, 验证集: {len(val_dataset)} 条")
    return train_loader, val_loader