"""
ESM-2 8M 独立打分 — Embedding Naturalness Score
对比 FoldPathLLM vs RITA_m vs NoStruct vs Natural
用法: python esm_score.py
"""
import torch, sys, os, json, numpy as np
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esm_encoder import create_esm_encoder
from config import AA_TO_IDX
from dataset import load_fasta as lf

# ════════ Load ESM-2 8M ════════
print('Loading ESM-2 8M...')
encoder = create_esm_encoder(model_name='esm2_t6_8M_UR50D', device=device, freeze=True, local_dir='pretrained')
print(f'ESM-2 8M loaded. Dim={encoder.hidden_size}')

# ════════ Load sequences ════════
def load_fa(path):
    seqs = []; cur = ''
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                if cur: seqs.append(cur); cur = ''
            else: cur += line.strip()
        if cur: seqs.append(cur)
    return seqs

fp_seqs = load_fa('experiment/foldpath_13.fasta') if os.path.exists('experiment/foldpath_13.fasta') else []
ns_seqs = load_fa('experiment/nostruct_13.fasta') if os.path.exists('experiment/nostruct_13.fasta') else []

# Natural reference: sample from validation set
val_seqs = lf('data/val_sequences.fasta')
np.random.seed(42)
nat_seqs = list(np.random.choice(val_seqs, min(50, len(val_seqs)), replace=False))

# Fallback
if not fp_seqs:
    print('WARNING: No FP sequences. Using defaults.')
    fp_seqs = ['MAKKVVTDLDLKEKKVLVRVDFNSLYNPTSVMEEAFDNLRQASDFDETFI']
    ns_seqs = ['MDIILASSTSKGKLAIEIAVNILLSVTFLSLLLLWVLAFIPSSKTVFLHPI']
print(f'FP={len(fp_seqs)} NS={len(ns_seqs)} Natural={len(nat_seqs)}')

# ════════ ESM Embedding Score ════════
# 原理: ESM-2 的嵌入空间捕获了结构/功能信息 (Rives et al. 2021).
# 天然蛋白的残基间嵌入相似度更高、方差更小。用两个指标:
#   coherence = avg pairwise cosine sim (越高越天然)
#   consistency = 1 - std of per-residue embedding norm (越高越天然)

@torch.no_grad()
def compute_esm_scores(seqs, label=''):
    coherence_vals = []
    consistency_vals = []
    n_ok = 0
    for seq in seqs:
        clean = ''.join(c for c in seq if c in AA_TO_IDX)
        if len(clean) < 5: continue
        clean = clean[:200]
        emb, _ = encoder([clean])
        L = min(emb.size(1), len(clean))
        e = emb[0, :L, :]
        # Coherence: avg pairwise cosine similarity
        normed = e / (e.norm(dim=1, keepdim=True) + 1e-8)
        sim = torch.mm(normed, normed.t())
        n = sim.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        coherence = sim[mask].mean().item()
        # Consistency: 1 - normalized std of embedding norms
        norms = e.norm(dim=1)
        consistency = 1.0 - (norms.std() / (norms.mean() + 1e-8)).item()
        coherence_vals.append(coherence)
        consistency_vals.append(consistency)
        n_ok += 1

    return {
        'coherence_mean': round(np.mean(coherence_vals), 4) if coherence_vals else 0,
        'coherence_std': round(np.std(coherence_vals), 4) if coherence_vals else 0,
        'consistency_mean': round(np.mean(consistency_vals), 4) if consistency_vals else 0,
        'consistency_std': round(np.std(consistency_vals), 4) if consistency_vals else 0,
        'n': n_ok,
        'raw_coherence': coherence_vals,
        'raw_consistency': consistency_vals,
    }

# ════════ Compute ════════
print('\nComputing ESM-2 embedding naturalness scores...')
fp_scores = compute_esm_scores(fp_seqs, 'FoldPathLLM')
ns_scores = compute_esm_scores(ns_seqs, 'NoStruct')
nat_scores = compute_esm_scores(nat_seqs, 'Natural')

print(f'\n{"="*60}')
print(f'  ESM-2 8M Embedding Naturalness Score')
print(f'{"="*60}')
print(f'  {"Model":<20} {"N":>5} {"Coherence":>12} {"Consistency":>13}')
print(f'  {"-"*20} {"-"*5} {"-"*12} {"-"*13}')
for name, sc in [('FoldPath-LLM', fp_scores), ('NoStruct', ns_scores), ('Natural', nat_scores)]:
    print(f'  {name:<20} {sc["n"]:>5} {sc["coherence_mean"]:>12.4f} {sc["consistency_mean"]:>13.4f}')
print(f'{"="*60}')
print(f'  Higher coherence = more natural-like residue interactions')
print(f'  Higher consistency = more uniform embedding quality')
print(f'  Reference: Rives et al. (2021) PNAS')

# Save
result = {
    'esm_model': 'esm2_t6_8M_UR50D',
    'method': 'Per-residue embedding coherence and consistency',
    'reference': 'Rives et al. (2021) PNAS',
    'foldpath': fp_scores, 'nostruct': ns_scores, 'natural': nat_scores,
}
for k in ['raw_coherence', 'raw_consistency']:
    del result['foldpath'][k]
    del result['nostruct'][k]
    del result['natural'][k]
with open('esm_score_result.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f'\nSaved: esm_score_result.json')
