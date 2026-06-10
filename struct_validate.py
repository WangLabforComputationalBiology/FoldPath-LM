"""
结构轨道验证 v2 — 快速版
1. 用 ProteinGenerator 快速生成序列
2. 逐条过模型 forward 提取结构预测
用法: python struct_validate.py --checkpoint esmpro/foldpath_best_eopch10.pt --num-seqs 20
"""
import torch
import sys, os, json, argparse, numpy as np
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--num-seqs', type=int, default=20)
parser.add_argument('--output', type=str, default='struct_validate_result.json')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import IDX_TO_AA, AA_TO_IDX, PAD_IDX, BOS_IDX, EOS_IDX, TOTAL_VOCAB, ModelConfig
from model import FoldPathLLM
from rita_encoder import create_rita_encoder
from generate import ProteinGenerator
from config import GenerateConfig
from physicochemical import PHYSICO_MATRIX

# ════════ Step 1: Fast generate ════════
print(f'Loading model: {args.checkpoint}')
gen = ProteinGenerator(checkpoint_path=args.checkpoint, device=device)

gc = GenerateConfig()
gc.num_samples = args.num_seqs; gc.max_length = 256
gc.temperature = 0.24; gc.top_k = 50; gc.top_p = 0.92
gc.use_physico_filter = True

print(f'Generating {args.num_seqs} sequences...')
seqs, _ = gen.generate(gc)
seqs = [s for s in seqs if len(s) >= 20]
print(f'Valid: {len(seqs)}')

# ════════ Step 2: Re-load model for structure analysis ════════
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
use_rita = checkpoint.get('use_rita', True)
rita_name = checkpoint.get('rita_model_name', 'RITA_m')
esm_encoder = create_rita_encoder(model_name=rita_name, device=device, freeze=True, local_dir='pretrained') if use_rita else None

model_cfg = checkpoint.get('model_config', ModelConfig())
model = FoldPathLLM(model_cfg, esm_encoder=esm_encoder)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model = model.to(device).eval()

# ════════ Step 3: Extract structure predictions ════════
print(f'Extracting structure predictions for {len(seqs)} sequences...')
results = {'timestamp': datetime.now().isoformat(), 'sequences': []}

for i, seq in enumerate(seqs):
    L = len(seq)
    # Build input: BOS + sequence
    input_tokens = [BOS_IDX] + [AA_TO_IDX.get(aa, 0) for aa in seq]
    input_ids = torch.tensor([input_tokens], device=device)
    mask = torch.ones(1, L+1, dtype=torch.bool, device=device)
    targets = torch.tensor([[PAD_IDX] + [AA_TO_IDX.get(aa, 0) for aa in seq]], device=device)

    with torch.no_grad():
        _, _, _, aux = model.forward(input_ids, targets, mask, sequences=[seq], use_bias=True)

    exposure = aux['exposure'][0, 1:].cpu().numpy()[:L]  # skip BOS
    ss_logits = aux['ss_logits'][0, 1:].cpu().numpy()[:L]
    dist_matrix = aux.get('distance_matrix')
    if dist_matrix is not None:
        dm = dist_matrix[0, 1:, 1:].cpu().numpy()[:L, :L]

    # Analysis — focus on exposure and hydrophobicity (structure track's primary outputs)
    exp_arr = np.array(exposure)
    hydro = np.array([PHYSICO_MATRIX[AA_TO_IDX.get(aa, 0), 0] for aa in seq])

    # Helix periodicity (autocorr at 3-4 residue spacing)
    autocorrs = []
    for lag in [1, 2, 3, 4, 7]:
        if L > lag + 5:
            c = np.corrcoef(hydro[:-lag], hydro[lag:])[0, 1]
            autocorrs.append(0.0 if np.isnan(c) else c)
    helix_period = float(np.mean(autocorrs)) if autocorrs else 0.0

    # Transmembrane pattern: alternating low/high exposure in 20-aa windows
    window = 20
    tm_score = 0.0
    if L >= window * 2:
        exp_means = [float(np.mean(exp_arr[j:j+window])) for j in range(0, L - window, window)]
        if len(exp_means) >= 2:
            tm_score = float(np.std(exp_means))

    # Hydrophobicity distribution
    hydro_mean = float(np.mean(hydro))
    hydro_std = float(np.std(hydro))

    r = {
        'seq': seq[:60] + ('...' if L > 60 else ''),
        'length': L,
        'mean_exposure': round(float(np.mean(exp_arr)), 3),
        'burial_pct': round(float(np.mean(exp_arr < 0.3)) * 100, 1),
        'surface_pct': round(float(np.mean(exp_arr > 0.7)) * 100, 1),
        'mean_hydrophobicity': round(hydro_mean, 3),
        'hydro_std': round(hydro_std, 3),
        'tm_pattern': round(tm_score, 3),
        'helix_periodicity': round(helix_period, 3),
        'cytochrome_b_like': True if (tm_score > 0.08 and helix_period > 0.01) else False,
    }
    results['sequences'].append(r)
    if (i+1) % 10 == 0: print(f'  [{i+1}/{len(seqs)}]')

# ════════ Summary ════════
sl = results['sequences']; n = len(sl)
exp = [s['mean_exposure'] for s in sl]; bur = [s['burial_pct'] for s in sl]
surf = [s['surface_pct'] for s in sl]; hydro = [s['mean_hydrophobicity'] for s in sl]
tm = [s['tm_pattern'] for s in sl]; hp = [s['helix_periodicity'] for s in sl]
cb_like = sum(1 for x in sl if x.get('cytochrome_b_like', False))

print(f'\n{"="*55}')
print(f'  Structure Track Validation ({n} sequences)')
print(f'{"="*55}')
print(f'  Solvent Exposure:  mean={np.mean(exp):.3f}  buried={np.mean(bur):.1f}%  surface={np.mean(surf):.1f}%')
print(f'  Hydrophobicity:    mean={np.mean(hydro):.3f}')
print(f'  TM pattern score:  {np.mean(tm):.3f}  (alternating exposure = transmembrane helix)')
print(f'  Helix periodicity: {np.mean(hp):.3f}  (3.6-residue autocorrelation)')
print(f'  Cytochrome b-like structural profile: {cb_like}/{n}')
print(f'{"="*55}')
print(f'  Expected (cytochrome b): ~40% buried, alternating TM hydrophobicity, helix period ~3.6')

results['summary'] = {
    'num_sequences': n,
    'avg_exposure': round(np.mean(exp), 3),
    'avg_burial_pct': round(np.mean(bur), 1),
    'avg_surface_pct': round(np.mean(surf), 1),
    'avg_hydrophobicity': round(np.mean(hydro), 3),
    'avg_tm_pattern': round(np.mean(tm), 3),
    'avg_helix_periodicity': round(np.mean(hp), 3),
    'cytochrome_b_like': f'{cb_like}/{n}',
}

with open(args.output, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'\nSaved: {args.output}')
