"""
模型对比评测: FoldPathLLM vs RITA_m 原生
用法: python benchmark_compare.py --foldpath-checkpoint esmpro/foldpath_best.pt --rita-dir ./pretrained/RITA_m
"""
import torch
import sys
import os
import json
import argparse
import time
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--foldpath-checkpoint', type=str, required=True, help='FoldPathLLM 模型路径')
parser.add_argument('--rita-dir', type=str, default='./pretrained/RITA_m', help='RITA_m 目录')
parser.add_argument('--data-dir', type=str, default='./data', help='数据目录')
parser.add_argument('--num-samples', type=int, default=100, help='各模型生成序列数')
parser.add_argument('--max-length', type=int, default=256, help='生成最大长度')
parser.add_argument('--temperature', type=float, default=1.0, help='生成温度')
parser.add_argument('--device', type=str, default='cuda', help='设备')

args = parser.parse_args()
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')
if device.type == 'cuda':
    free, total = torch.cuda.mem_get_info()
    print(f'显存: 空闲 {free/1e9:.1f} GB / 总计 {total/1e9:.1f} GB')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE as CFG_DEVICE  # noqa: E402
# override config device
import config
config.DEVICE = device

from evaluation import FoldPathBenchmark
from generate import ProteinGenerator
from config import GenerateConfig

results = {}
results['timestamp'] = datetime.now().isoformat()
results['config'] = {
    'num_samples': args.num_samples,
    'max_length': args.max_length,
    'temperature': args.temperature,
}

# ═══════════════════════════════════════════════════
# 1. FoldPathLLM 生成 + 评测
# ═══════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  [1/2] FoldPathLLM 生成 & 评测')
print('=' * 60)

gen = ProteinGenerator(checkpoint_path=args.foldpath_checkpoint, device=device)

gc = GenerateConfig()
gc.num_samples = args.num_samples
gc.max_length = args.max_length
gc.temperature = args.temperature
gc.top_k = 50
gc.top_p = 0.92
gc.use_physico_filter = True

t0 = time.time()
sequences, _ = gen.generate(gc)
sequences = [s for s in sequences if len(s) >= 20]
gen_time = time.time() - t0
print(f'生成 {len(sequences)} 条有效序列 ({len(sequences)}/{args.num_samples}), 耗时 {gen_time:.1f}s')

ref_fasta = os.path.join(args.data_dir, 'train_sequences.fasta')
benchmark = FoldPathBenchmark(reference_fasta=ref_fasta)
print('评测中...')
fp_result = benchmark.evaluate(sequences, verbose=False)
results['foldpath'] = {
    'model': 'FoldPathLLM',
    'num_valid': len(sequences),
    'generation_time_s': round(gen_time, 1),
    'physico_mean': round(fp_result['physico']['mean'], 4),
    'physico_std': round(fp_result['physico']['std'], 4),
    'diversity': round(fp_result['diversity']['total'], 4),
    'naturalness_mean': round(fp_result['naturalness']['mean'], 4),
    'naturalness_std': round(fp_result['naturalness']['std'], 4),
    'composite': round(fp_result['composite'], 4),
    'grade': fp_result['grade'],
    'sample_seqs': sequences[:3],
}

# ═══════════════════════════════════════════════════
# 2. RITA_m 原生生成 + 评测
# ═══════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  [2/2] RITA_m 原生生成 & 评测')
print('=' * 60)

# 加载 RITA
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

rita_tokenizer = AutoTokenizer.from_pretrained('.', trust_remote_code=True)
rita_tokenizer.pad_token = '[PAD]'
rita_tokenizer.pad_token_id = 1
rita_model = AutoModelForCausalLM.from_pretrained('.', trust_remote_code=True)
rita_model.to(device)
rita_model.eval()
os.chdir(save_dir)

# AA → RITA token 映射 (用于生成)
from config import AA_TO_IDX, IDX_TO_AA, BOS_IDX, EOS_IDX

aa_to_rita = {}
rita_to_aa = {}
for aa in 'ACDEFGHIKLMNPQRSTVWY':
    ids = rita_tokenizer.encode(aa)
    if ids:
        rid = ids[0]
        aa_to_rita[aa] = rid
        rita_to_aa[rid] = AA_TO_IDX[aa]

valid_rita_ids = torch.tensor(sorted(rita_to_aa.keys()), device=device)

# RITA 原生自回归生成
print(f'生成 {args.num_samples} 条序列 (温度={args.temperature})...')
rita_seqs = []
t0 = time.time()

with torch.no_grad():
    for i in range(args.num_samples):
        # 从空序列开始, RITA 用换行符或空字符串
        generated = []
        input_ids = torch.tensor([[0]], device=device)  # BOS or start token

        for _ in range(args.max_length):
            outputs = rita_model(input_ids)
            logits = outputs.logits[0, -1, :] / args.temperature

            # 只允许标准 AA token
            aa_only = torch.full_like(logits, float('-inf'))
            aa_only[valid_rita_ids] = logits[valid_rita_ids]
            logits = aa_only

            # top-k + top-p
            top_k = 50
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[-1]] = float('-inf')

            top_p = 0.92
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_logits[cumprobs > top_p] = float('-inf')
                logits = torch.full_like(logits, float('-inf'))
                logits[sorted_idx] = sorted_logits

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if next_token in rita_to_aa:
                generated.append(rita_to_aa[next_token])
            else:
                break  # 遇到非 AA token 则停止

            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

        seq = ''.join([IDX_TO_AA.get(a, 'X') for a in generated])
        if len(seq) >= 20:
            rita_seqs.append(seq)

        if (i + 1) % 20 == 0:
            print(f'  [{i+1}/{args.num_samples}] 有效序列: {len(rita_seqs)}')

rita_time = time.time() - t0
print(f'生成 {len(rita_seqs)} 条有效序列, 耗时 {rita_time:.1f}s')

print('评测中...')
rita_result = benchmark.evaluate(rita_seqs, verbose=False)
results['rita'] = {
    'model': 'RITA_m (原生)',
    'num_valid': len(rita_seqs),
    'generation_time_s': round(rita_time, 1),
    'physico_mean': round(rita_result['physico']['mean'], 4),
    'physico_std': round(rita_result['physico']['std'], 4),
    'diversity': round(rita_result['diversity']['total'], 4),
    'naturalness_mean': round(rita_result['naturalness']['mean'], 4),
    'naturalness_std': round(rita_result['naturalness']['std'], 4),
    'composite': round(rita_result['composite'], 4),
    'grade': rita_result['grade'],
    'sample_seqs': rita_seqs[:3],
}

# ═══════════════════════════════════════════════════
# 3. 对比报告
# ═══════════════════════════════════════════════════
print('\n' + '=' * 62)
print('  ★ 对比报告')
print('=' * 62)

headers = ['指标', 'FoldPathLLM', 'RITA_m 原生', '差异']
rows = [
    ('理化合理性', results['foldpath']['physico_mean'], results['rita']['physico_mean']),
    ('序列多样性', results['foldpath']['diversity'], results['rita']['diversity']),
    ('天然相似度', results['foldpath']['naturalness_mean'], results['rita']['naturalness_mean']),
    ('★ 综合分', results['foldpath']['composite'], results['rita']['composite']),
    ('有效序列数', results['foldpath']['num_valid'], results['rita']['num_valid']),
    ('生成耗时(s)', results['foldpath']['generation_time_s'], results['rita']['generation_time_s']),
]

print()
print(f'  {"指标":<14} {"FoldPathLLM":>12} {"RITA_m原生":>12} {"差异":>10}')
print(f'  {"─"*14} {"─"*12} {"─"*12} {"─"*10}')
for name, fp, rita in rows:
    if isinstance(fp, float) and isinstance(rita, float):
        diff = fp - rita
        sign = '+' if diff > 0 else ''
        diff_str = f'{sign}{diff:.4f}'
        arrow = '>' if diff > 0 else ('<' if diff < 0 else '=')
        print(f'  {name:<14} {fp:>12.4f} {rita:>12.4f} {arrow} {diff_str:>8}')
    else:
        print(f'  {name:<14} {str(fp):>12} {str(rita):>12} {"─":>10}')

print()
print(f'  FoldPathLLM 等级: {results["foldpath"]["grade"]}')
print(f'  RITA_m 原生 等级: {results["rita"]["grade"]}')

# 显示样例
print('\n' + '─' * 62)
print('  FoldPathLLM 生成样例:')
for i, s in enumerate(results['foldpath']['sample_seqs']):
    print(f'  [{i+1}] {s[:80]}{"..." if len(s)>80 else ""}')
print()
print('  RITA_m 原生生成样例:')
for i, s in enumerate(results['rita']['sample_seqs']):
    print(f'  [{i+1}] {s[:80]}{"..." if len(s)>80 else ""}')

# 保存
output_path = 'benchmark_comparison.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'\n结果已保存: {output_path}')
