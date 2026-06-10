"""
RITA_m 原生生成 + 天然度评测 (50条)
用法: python eval_rita_gen.py --rita-dir ./pretrained/RITA_m --num-samples 50
"""
import torch
import sys
import os
import json
import argparse
import time
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--rita-dir', type=str, default='./pretrained/RITA_m')
parser.add_argument('--data-dir', type=str, default='./data')
parser.add_argument('--num-samples', type=int, default=50)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--top-k', type=int, default=50)
parser.add_argument('--top-p', type=float, default=0.92)
parser.add_argument('--max-length', type=int, default=256)
parser.add_argument('--output', type=str, default='rita_gen_result.json')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')
if device.type == 'cuda':
    free, total = torch.cuda.mem_get_info()
    print(f'显存: 空闲 {free/1e9:.1f} GB / 总计 {total/1e9:.1f} GB')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import AA_TO_IDX, IDX_TO_AA
from evaluation import FoldPathBenchmark

# ══════════════════════════════════════════
# 加载 RITA_m
# ══════════════════════════════════════════
print(f'\n加载 RITA_m: {args.rita_dir}')
save_dir = os.getcwd()
os.chdir(args.rita_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

_orig_init = PreTrainedModel.__init__
def _patched_init(self, config, *a, **kw):
    self._tied_weights_keys = []
    self.all_tied_weights_keys = {}
    _orig_init(self, config, *a, **kw)
PreTrainedModel.__init__ = _patched_init

tokenizer = AutoTokenizer.from_pretrained('.', trust_remote_code=True)
tokenizer.pad_token = '[PAD]'
tokenizer.pad_token_id = 1
model = AutoModelForCausalLM.from_pretrained('.', trust_remote_code=True)
model.to(device)
model.eval()
os.chdir(save_dir)
print('RITA_m 加载完成')

# AA → RITA token 映射
aa_to_rita = {}
rita_to_aa = {}
for aa in 'ACDEFGHIKLMNPQRSTVWY':
    ids = tokenizer.encode(aa)
    if ids:
        rid = ids[0]
        aa_to_rita[aa] = rid
        rita_to_aa[rid] = AA_TO_IDX[aa]
valid_rita_ids = torch.tensor(sorted(rita_to_aa.keys()), device=device)

# ══════════════════════════════════════════
# 自回归生成
# ══════════════════════════════════════════
print(f'\n生成 {args.num_samples} 条序列 (temp={args.temperature})...')
generated_seqs = []
t0 = time.time()

with torch.no_grad():
    for i in range(args.num_samples):
        generated = []
        start_token = aa_to_rita.get('M', list(valid_rita_ids)[0].item())
        input_ids = torch.tensor([[start_token]], device=device)

        for _ in range(args.max_length):
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :] / args.temperature

            # 屏蔽非 AA token
            mask = torch.full_like(logits, float('-inf'))
            mask[valid_rita_ids] = logits[valid_rita_ids]

            # 简单 top-k
            k = min(args.top_k, (mask > float('-inf')).sum().item())
            if k > 0:
                tk_vals, _ = torch.topk(mask, k)
                mask[mask < tk_vals[-1]] = float('-inf')

            probs = torch.softmax(mask, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if next_token in rita_to_aa:
                generated.append(rita_to_aa[next_token])
            else:
                break

            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

        seq = ''.join([IDX_TO_AA.get(a, 'X') for a in generated])
        if len(seq) >= 20:
            generated_seqs.append(seq)

        if (i + 1) % 10 == 0:
            print(f'  [{i+1}/{args.num_samples}] 有效: {len(generated_seqs)}')

gen_time = time.time() - t0
print(f'有效序列: {len(generated_seqs)}/{args.num_samples}, 耗时 {gen_time:.1f}s')

if len(generated_seqs) == 0:
    print('[ERROR] 无有效序列生成，退出')
    sys.exit(1)

# ══════════════════════════════════════════
# 评测
# ══════════════════════════════════════════
ref_fasta = os.path.join(args.data_dir, 'train_sequences.fasta')
bench = FoldPathBenchmark(reference_fasta=ref_fasta)
result = bench.evaluate(generated_seqs, verbose=True)

output = {
    'timestamp': datetime.now().isoformat(),
    'model': 'RITA_m (原生)',
    'config': {
        'num_samples': args.num_samples,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'max_length': args.max_length,
        'valid_sequences': len(generated_seqs),
        'generation_time_s': round(gen_time, 1),
    },
    'physico_mean': round(result['physico']['mean'], 4),
    'physico_std': round(result['physico']['std'], 4),
    'diversity': round(result['diversity']['total'], 4),
    'naturalness_mean': round(result['naturalness']['mean'], 4),
    'naturalness_std': round(result['naturalness']['std'], 4),
    'composite': round(result['composite'], 4),
    'grade': result['grade'],
}
with open(args.output, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f'\n结果已保存: {args.output}')
