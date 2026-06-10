"""
筛选高质量生成序列 — 为 ColabFold 准备 Top-N 序列
用法: python select_best_seqs.py --checkpoint esmpro/foldpath_best_eopch10.pt --num-gen 30 --num-best 10
"""
import torch, sys, os, argparse, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--num-gen', type=int, default=30)
parser.add_argument('--num-best', type=int, default=10)
parser.add_argument('--output', type=str, default='colabfold_input.fasta')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate import ProteinGenerator
from config import GenerateConfig
from physicochemical import PHYSICO_MATRIX
from config import AA_TO_IDX

# ── 生成 ──
print(f'Loading: {args.checkpoint}')
gen = ProteinGenerator(checkpoint_path=args.checkpoint, device=device)
gc = GenerateConfig()
gc.num_samples = args.num_gen
gc.max_length = 256
gc.temperature = 0.24
gc.top_k = 50
gc.top_p = 0.92
gc.use_physico_filter = True

print(f'Generating {args.num_gen} sequences...')
seqs, _ = gen.generate(gc)
seqs = [s for s in seqs if len(s) >= 50]
print(f'Valid (len>=50): {len(seqs)}/{args.num_gen}')

# ── 多维度评分 ──
scored = []
for seq in seqs:
    L = len(seq)
    hydro = np.array([PHYSICO_MATRIX[AA_TO_IDX.get(aa, 0), 0] for aa in seq])
    charge = np.array([PHYSICO_MATRIX[AA_TO_IDX.get(aa, 0), 2] for aa in seq])
    volume = np.array([PHYSICO_MATRIX[AA_TO_IDX.get(aa, 0), 1] for aa in seq])
    hbond = np.array([PHYSICO_MATRIX[AA_TO_IDX.get(aa, 0), 4] for aa in seq])

    # 1. 半胱氨酸含量 (<3% 为优, >5% 严重惩罚)
    c_count = seq.count('C')
    c_ratio = c_count / L
    c_score = max(0, 1 - c_ratio * 25)

    # 2. 疏水残基比例 (目标 30-45%, TM蛋白特征)
    hydro_pct = np.mean(hydro > 0.5)
    h_score = max(0, 1 - abs(0.38 - hydro_pct) * 3)

    # 3. 电荷平衡 (|正电荷 - 负电荷| 占比 < 15%)
    pos = np.mean(charge > 0.5)
    neg = np.mean(charge < -0.5)
    ch_score = max(0, 1 - abs(pos - neg) * 5)

    # 4. 连续重复惩罚 (连续3+相同残基)
    max_run = 1
    cur_run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    rep_score = max(0, 1 - (max_run - 2) * 0.3)

    # 5. 长度奖励 (100-200 为优)
    len_score = min(1.0, L / 160) if L < 160 else max(0, 1 - (L - 200) / 200)

    # 6. 疏水模式多样性 (交替疏水/亲水)
    hydro_runs = 0
    prev_state = hydro[0] > 0.5
    for h in hydro[1:]:
        state = h > 0.5
        if state != prev_state:
            hydro_runs += 1
            prev_state = state
    diversity_score = min(1.0, hydro_runs / (L / 15))

    total = (c_score * 0.25 + h_score * 0.20 + ch_score * 0.15 +
             rep_score * 0.15 + len_score * 0.15 + diversity_score * 0.10)

    scored.append((total, seq, {
        'c_count': c_count, 'hydro_pct': round(hydro_pct * 100, 1),
        'pos_pct': round(pos * 100, 1), 'neg_pct': round(neg * 100, 1),
        'max_run': max_run, 'length': L,
        'hydro_runs': hydro_runs,
        'c_score': round(c_score, 3), 'h_score': round(h_score, 3),
        'ch_score': round(ch_score, 3), 'rep_score': round(rep_score, 3),
        'len_score': round(len_score, 3), 'div_score': round(diversity_score, 3),
    }))

scored.sort(key=lambda x: x[0], reverse=True)

# ── 输出 ──
print(f'\n{"="*65}')
print(f'  Top {args.num_best} Sequences (from {len(scored)} valid)')
print(f'{"="*65}')
print(f'  {"#":<3} {"Score":<7} {"Len":<5} {"C":<4} {"Hydro%":<7} {"Pos%":<5} {"Neg%":<5} {"MaxRun":<7} {"Preview":<30}')
print(f'  {"-"*65}')

best = scored[:args.num_best]
with open(args.output, 'w') as f:
    for i, (score, seq, meta) in enumerate(best):
        preview = seq[:40]
        f.write(f'>seq{i+1}_score{score:.3f}_len{meta["length"]}\n{seq}\n')
        print(f'  {i+1:<3} {score:<7.3f} {meta["length"]:<5} {meta["c_count"]:<4} '
              f'{meta["hydro_pct"]:<7.1f} {meta["pos_pct"]:<5.1f} {meta["neg_pct"]:<5.1f} '
              f'{meta["max_run"]:<7} {preview}...')

print(f'\n  Saved: {args.output}')
print(f'\n  Next: upload {args.output} to ColabFold')
print(f'  https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb')
