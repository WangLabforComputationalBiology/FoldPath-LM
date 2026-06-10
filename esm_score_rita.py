"""
ESM-2 8M 评分 RITA_m 验证集序列
用法: python esm_score_rita.py
"""
import torch, sys, os, json, numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esm_encoder import create_esm_encoder
from config import AA_TO_IDX
from dataset import load_fasta

print('Loading ESM-2 8M...')
enc = create_esm_encoder(model_name='esm2_t6_8M_UR50D', device=device, freeze=True, local_dir='pretrained')

val_seqs = load_fasta('data/val_sequences.fasta')
np.random.seed(42)
selected = list(np.random.choice(val_seqs, min(50, len(val_seqs)), replace=False))

@torch.no_grad()
def esm_score(seqs):
    coh, cons = [], []
    for s in seqs:
        clean = ''.join(c for c in s if c in AA_TO_IDX)
        if len(clean) < 5: continue
        clean = clean[:200]
        e = enc([clean])[0][0, :enc([clean])[0].size(1), :]
        L = min(enc([clean])[0].size(1), len(clean))
        e = enc([clean])[0][0, :L, :]
        n = e / (e.norm(dim=1, keepdim=True) + 1e-8)
        sim = torch.mm(n, n.t())
        mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=device)
        coh.append(sim[mask].mean().item())
        norms = e.norm(dim=1)
        cons.append(1.0 - (norms.std() / (norms.mean() + 1e-8)).item())
    return round(np.mean(coh), 4), round(np.mean(cons), 4), len(coh)

c, s, n = esm_score(selected)
print(f'\nRITA_m (validation, n={n}):')
print(f'  Coherence:   {c}')
print(f'  Consistency: {s}')

with open('esm_score_rita.json', 'w') as f:
    json.dump({'model': 'RITA_m (val set)', 'n': n, 'coherence': c, 'consistency': s}, f)
print('Saved: esm_score_rita.json')
