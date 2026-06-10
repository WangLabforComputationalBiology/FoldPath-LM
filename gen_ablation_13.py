"""
生成 FoldPath-LLM + NoStruct 各 13 条高质量序列 (50-120aa)
用法: python gen_ablation_13.py
"""
import torch, sys, os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate import ProteinGenerator
from config import GenerateConfig, ModelConfig, AA_TO_IDX
from model import FoldPathLLM
from rita_encoder import create_rita_encoder
from physicochemical import PHYSICO_MATRIX

def score_seq(seq):
    L = len(seq)
    hydro = np.array([PHYSICO_MATRIX[AA_TO_IDX.get(aa,0),0] for aa in seq])
    charge = np.array([PHYSICO_MATRIX[AA_TO_IDX.get(aa,0),2] for aa in seq])
    cr = seq.count('C')/L; cs = max(0, 1-cr*25)
    hp = np.mean(hydro>0.5); hs = max(0, 1-abs(0.38-hp)*3)
    po=np.mean(charge>0.5); ne=np.mean(charge<-0.5); chs=max(0, 1-abs(po-ne)*5)
    mr=1; cu=1
    for i in range(1,L):
        if seq[i]==seq[i-1]: cu+=1; mr=max(mr,cu)
        else: cu=1
    rs = max(0, 1-(mr-2)*0.3)
    # 6. TM pattern: alternating hydrophobicity in ~20aa windows
    window=20
    hydro_means = [float(np.mean(hydro[j:j+window])) for j in range(0, L-window, window) if j+window <= L]
    tms = min(1.0, np.std(hydro_means)*5) if len(hydro_means)>=2 else 0
    # 7. Helix periodicity: autocorrelation at 3-4 residue lag
    ac = 0
    for lag in [2,3,4,7]:
        if L > lag+5:
            c = np.corrcoef(hydro[:-lag], hydro[lag:])[0,1]
            ac += max(0, c) if not np.isnan(c) else 0
    hps = min(1.0, ac*2)
    return cs*0.20 + hs*0.20 + chs*0.10 + rs*0.10 + min(1.0, L/120)*0.10 + tms*0.15 + hps*0.15

def pick_best(seqs, n):
    scored = [(score_seq(s), s) for s in seqs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:n]]

os.makedirs('experiment', exist_ok=True)
gc = GenerateConfig()
gc.num_samples = 50; gc.max_length = 256; gc.temperature = 0.24
gc.top_k = 50; gc.top_p = 0.92; gc.use_physico_filter = True

# ════════ FoldPath-LLM ════════
print('[1/2] FoldPath-LLM...')
gen = ProteinGenerator(checkpoint_path='esmpro/foldpath_best_eopch10.pt', device=device)
seqs, _ = gen.generate(gc)
seqs = [s for s in seqs if len(s) >= 30]
seqs = pick_best(seqs, 13)
print(f'  Valid: {len(seqs)}, Selected best 13')
with open('experiment/foldpath_13.fasta', 'w') as f:
    for i, s in enumerate(seqs):
        f.write(f'>FP_{i+1}_len{len(s)}\n{s}\n')
        print(f'  [{i+1}] len={len(s)} C={s.count("C")}  {s[:50]}...')
print(f'  Saved: experiment/foldpath_13.fasta')

# ════════ NoStruct ════════
print('\n[2/2] NoStruct...')
ckpt = torch.load('esmpro/foldpath_best_eopch10.pt', map_location=device, weights_only=False)
enc = create_rita_encoder(model_name='RITA_m', device=device, freeze=True, local_dir='pretrained')
model = FoldPathLLM(ModelConfig(), esm_encoder=enc)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
for n, p in model.named_parameters():
    if 'structure_track' in n or 'chem_bias' in n:
        p.data.zero_()
model = model.to(device).eval()

gen2 = ProteinGenerator(model=model, device=device)
seqs2, _ = gen2.generate(gc)
seqs2 = [s for s in seqs2 if len(s) >= 30]
seqs2 = pick_best(seqs2, 13)
print(f'  Valid: {len(seqs2)}, Selected best 13')
with open('experiment/nostruct_13.fasta', 'w') as f:
    for i, s in enumerate(seqs2):
        f.write(f'>NS_{i+1}_len{len(s)}\n{s}\n')
        print(f'  [{i+1}] len={len(s)} C={s.count("C")}  {s[:50]}...')
print(f'  Saved: experiment/nostruct_13.fasta')

print('\nDone. Upload to ColabFold.')
