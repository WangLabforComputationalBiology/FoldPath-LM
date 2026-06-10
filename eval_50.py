"""
50条序列评测 — 天然度准确评估
用法:
  终端: python eval_50.py --checkpoint esmpro/foldpath_best.pt --num-samples 50
  输出: eval_50_result.json + 详细报告
"""
import torch
import sys
import os
import json
import argparse
import time
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True, help='模型路径')
parser.add_argument('--data-dir', type=str, default='./data')
parser.add_argument('--num-samples', type=int, default=50)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--top-k', type=int, default=50)
parser.add_argument('--top-p', type=float, default=0.92)
parser.add_argument('--output', type=str, default='eval_50_result.json')
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

from evaluation import FoldPathBenchmark
from generate import ProteinGenerator
from config import GenerateConfig

from config import ModelConfig, TrainConfig  # noqa: E402

# ── 加载模型 ──
print(f'加载: {args.checkpoint}')
gen = ProteinGenerator(checkpoint_path=args.checkpoint, device=device)

# ── 生成 ──
gc = GenerateConfig()
gc.num_samples = args.num_samples
gc.max_length = 256
gc.temperature = args.temperature
gc.top_k = args.top_k
gc.top_p = args.top_p
gc.use_physico_filter = True

print(f'生成 {args.num_samples} 条序列 (temp={args.temperature})...')
t0 = time.time()
seqs, _ = gen.generate(gc)
seqs = [s for s in seqs if len(s) >= 20]
gen_time = time.time() - t0
print(f'有效序列: {len(seqs)}/{args.num_samples}, 耗时 {gen_time:.1f}s')

# ── 评测 ──
ref_fasta = os.path.join(args.data_dir, 'train_sequences.fasta')
bench = FoldPathBenchmark(reference_fasta=ref_fasta)
result = bench.evaluate(seqs, verbose=True)

# ── 保存 ──
output = {
    'timestamp': datetime.now().isoformat(),
    'checkpoint': args.checkpoint,
    'config': {
        'num_samples': args.num_samples,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'valid_sequences': len(seqs),
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
