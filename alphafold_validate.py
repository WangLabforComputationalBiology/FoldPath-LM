"""
ColabFold 结构验证 — 为论文生成 AlphaFold2 预测结构
Step 1 (服务器): 生成序列并保存 FASTA
Step 2 (浏览器): 上传到 ColabFold 预测结构
Step 3 (任意): 下载结果，运行 pLDDT 分析

用法:
  python alphafold_validate.py --checkpoint esmpro/foldpath_best_eopch10.pt --num-seqs 10
  然后按提示上传到 ColabFold
"""
import torch, sys, os, json, argparse, numpy as np
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--num-seqs', type=int, default=10)
parser.add_argument('--output', type=str, default='colabfold_input')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import ProteinGenerator
from config import GenerateConfig

# ── 生成序列 ──
print(f'Loading: {args.checkpoint}')
gen = ProteinGenerator(checkpoint_path=args.checkpoint, device=device)

gc = GenerateConfig()
gc.num_samples = args.num_seqs * 3  # Generate more, filter to best
gc.max_length = 256; gc.temperature = 0.24
gc.top_k = 50; gc.top_p = 0.92; gc.use_physico_filter = True

print(f'Generating {args.num_seqs * 3} sequences...')
seqs, scores = gen.generate(gc)
seqs = [s for i, s in enumerate(seqs) if len(s) >= 50]

# ── 选最优 ──
from evaluation import FoldPathBenchmark
import os as _os
bench = FoldPathBenchmark(reference_fasta=_os.path.join('data', 'train_sequences.fasta'))

# Score each sequence individually for naturalness proxy
seq_scores = []
for seq in seqs:
    L = len(seq)
    # Quick scoring: prefer longer sequences with hydrophobic content 30-50%
    from physicochemical import PHYSICO_MATRIX
    from config import AA_TO_IDX
    hydro = [PHYSICO_MATRIX[AA_TO_IDX.get(aa, 0), 0] for aa in seq if aa in AA_TO_IDX]
    hydro_pct = np.mean([h > 0.5 for h in hydro]) if hydro else 0
    score = np.clip(L / 200, 0, 1) * 0.4 + (1 - abs(0.4 - hydro_pct)) * 0.6  # prefer ~40% hydrophobic
    seq_scores.append((score, seq))

seq_scores.sort(key=lambda x: x[0], reverse=True)
best_seqs = [s for _, s in seq_scores[:args.num_seqs]]

# ── 保存 FASTA ──
fasta_path = f'{args.output}.fasta'
with open(fasta_path, 'w') as f:
    for i, seq in enumerate(best_seqs):
        f.write(f'>FoldPathLLM_seq{i+1}_len{len(seq)}\n')
        f.write(f'{seq}\n')

print(f'\nSaved {len(best_seqs)} sequences to: {fasta_path}')
print(f'Lengths: {[len(s) for s in best_seqs]}')

# ── 保存元数据 ──
meta = {
    'timestamp': datetime.now().isoformat(),
    'checkpoint': args.checkpoint,
    'num_sequences': len(best_seqs),
    'sequences': [{'id': f'FoldPathLLM_seq{i+1}', 'length': len(s), 'seq': s[:80]+'...'}
                  for i, s in enumerate(best_seqs)],
}
with open(f'{args.output}_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f'\n{"="*60}')
print(f'  NEXT: ColabFold 结构预测')
print(f'{"="*60}')
print(f'''
  1. 打开浏览器访问: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb
  2. 上传文件: {fasta_path}
  3. 设置参数:
     - template_mode: none (你想要 de novo 预测)
     - num_relax: 1 (结构优化)
     - num_models: 1 (用最快的)
  4. 点击 Runtime → Run all (~2-5 分钟/序列)
  5. 下载结果 ZIP，解压到当前目录
  6. 运行分析: python analyze_colabfold.py
''')
