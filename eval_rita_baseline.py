"""
RITA_m 原生基线评估 — 服务器版本 (A10 24GB)
用法: python eval_rita_baseline.py --rita-dir path/to/RITA_m --data-dir ./data
"""
import torch
import sys
import os
import argparse
import json

# ── 参数 ──
parser = argparse.ArgumentParser()
parser.add_argument('--rita-dir', type=str, default='C:/Users/13380/Desktop/RITA_m',
                    help='RITA_m 模型目录')
parser.add_argument('--data-dir', type=str, default='./data',
                    help='数据目录 (含 val_sequences.fasta)')
parser.add_argument('--max-seqs', type=int, default=500,
                    help='最多评估序列数 (0=全部)')
parser.add_argument('--batch-size', type=int, default=4,
                    help='验证批次大小')
parser.add_argument('--output', type=str, default='rita_baseline_result.json',
                    help='结果保存路径')
args = parser.parse_args()

# ── 设备 ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')
if device.type == 'cuda':
    free, total = torch.cuda.mem_get_info()
    print(f'显存: 空闲 {free/1e9:.1f} GB / 总计 {total/1e9:.1f} GB')

# ── 导入项目模块 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import AA_TO_IDX, PAD_IDX
from dataset import load_fasta

# ══════════════════════════════════════════
# 加载 RITA_m
# ══════════════════════════════════════════
print(f'\n加载 RITA_m: {args.rita_dir}')
save_dir = os.getcwd()
os.chdir(args.rita_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

# 兼容新版 transformers: 打补丁
_orig_init = PreTrainedModel.__init__
def _patched_init(self, config, *a, **kw):
    self._tied_weights_keys = []
    self.all_tied_weights_keys = {}
    _orig_init(self, config, *a, **kw)
PreTrainedModel.__init__ = _patched_init

tokenizer = AutoTokenizer.from_pretrained('.', trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = '[PAD]'
    tokenizer.pad_token_id = 1  # RITA 固定
model = AutoModelForCausalLM.from_pretrained('.', trust_remote_code=True)
model.to(device)
model.eval()
os.chdir(save_dir)
print('RITA_m 加载完成')

# RITA 是 CausalLM，直接用 model() 获取 logits
# 不需要手动分离 backbone + lm_head

# ══════════════════════════════════════════
# AA → RITA token 映射
# ══════════════════════════════════════════
aa_to_rita = {}
for aa in 'ACDEFGHIKLMNPQRSTVWY':
    ids = tokenizer.encode(aa)
    aa_to_rita[aa] = ids[0] if ids else None

rita_to_aa = {v: AA_TO_IDX[k] for k, v in aa_to_rita.items() if v is not None}
valid_rita_ids = set(rita_to_aa.keys())
print(f'AA→RITA token 映射: {len(rita_to_aa)}/20')
for aa, rid in sorted(aa_to_rita.items()):
    print(f'  {aa} → {rid}', end='')
print()

# ══════════════════════════════════════════
# 加载验证集 (纯序列)
# ══════════════════════════════════════════
val_path = os.path.join(args.data_dir, 'val_sequences.fasta')
print(f'\n加载验证集: {val_path}')
val_seqs = load_fasta(val_path)
print(f'验证序列数: {len(val_seqs)}')

if args.max_seqs > 0:
    val_seqs = val_seqs[:args.max_seqs]
    print(f'截取前 {args.max_seqs} 条')

# ══════════════════════════════════════════
# 评估: Next-Token P/R
# ══════════════════════════════════════════
total_correct = 0
total_tokens = 0
total_pred = 0
n_seqs = 0

print(f'\n评估中 (批次大小={args.batch_size})...')
with torch.no_grad():
    for i in range(0, len(val_seqs), args.batch_size):
        batch_seqs = val_seqs[i:i + args.batch_size]
        # 清洗序列
        clean_seqs = []
        for s in batch_seqs:
            c = ''.join(ch for ch in s if ch in 'ACDEFGHIKLMNPQRSTVWY')
            clean_seqs.append(c if len(c) >= 2 else None)

        # 过滤太短的
        valid_seqs = [(j, s) for j, s in enumerate(clean_seqs) if s is not None]
        if not valid_seqs:
            continue

        seqs = [s for _, s in valid_seqs]

        inputs = tokenizer(seqs, padding=True, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits  # [B, L, V]

        # 逐序列计算 P/R
        for k in range(len(seqs)):
            input_ids = inputs['input_ids'][k]
            attn_mask = inputs['attention_mask'][k]

            # 有效 token 范围
            valid_len = attn_mask.sum().item()
            if valid_len < 2:
                continue

            tgt = input_ids[1:valid_len]
            log = logits[k, :valid_len-1, :]

            # 屏蔽非 AA token
            aa_mask = torch.zeros(log.size(-1), device=log.device)
            for rid in valid_rita_ids:
                aa_mask[rid] = 1
            log[:, aa_mask == 0] = float('-inf')

            pred_ids = log.argmax(dim=-1)
            pred_aa = torch.full_like(pred_ids, -1)
            target_aa = torch.full_like(tgt, -1)
            for rid, aidx in rita_to_aa.items():
                pred_aa[pred_ids == rid] = aidx
                target_aa[tgt == rid] = aidx

            valid_mask = pred_aa >= 0
            total_correct += ((pred_aa == target_aa) & valid_mask).sum().item()
            total_tokens += tgt.size(0)
            total_pred += valid_mask.sum().item()
            n_seqs += 1

        if (i // args.batch_size) % 20 == 0:
            p = total_correct / max(total_pred, 1)
            r = total_correct / max(total_tokens, 1)
            print(f'  [{n_seqs}/{len(val_seqs)}] P={p:.4f} R={r:.4f}')

# ══════════════════════════════════════════
# 输出
# ══════════════════════════════════════════
p = total_correct / max(total_pred, 1)
r = total_correct / max(total_tokens, 1)

result = {
    'model': 'RITA_m (原生, 无创新模块)',
    'num_sequences': n_seqs,
    'total_tokens': total_tokens,
    'total_correct': total_correct,
    'total_predicted': total_pred,
    'precision': round(p, 4),
    'recall': round(r, 4),
}

print()
print('=' * 55)
print(f'  RITA_m 原生基线')
print(f'  序列数 : {n_seqs}')
print(f'  Tokens : {total_tokens}')
print(f'  Correct: {total_correct}')
print(f'  ─────────────────')
print(f'  Precision : {p:.4f}')
print(f'  Recall    : {r:.4f}')
print('=' * 55)

with open(args.output, 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f'\n结果已保存: {args.output}')
