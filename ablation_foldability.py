"""
消融实验折叠能力对比 — 统计证据
Step 1: 生成 FoldPath-LLM 和 NoStruct 各 50 条序列
Step 2: 上传 ColabFold，收集 pLDDT/pTM
Step 3: 统计分析 + 图表

用法: python ablation_foldability.py --foldpath-ckpt esmpro/foldpath_best_eopch10.pt --nostruct-ckpt esmpro/nostruct_eopch5.pt
"""
import torch, sys, os, json, argparse, time
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--foldpath-ckpt', type=str, required=True)
parser.add_argument('--nostruct-ckpt', type=str, default=None,
                    help='NoStruct checkpoint (可选，没有则用 FoldPathLLM 去掉结构轨)')
parser.add_argument('--num-seqs', type=int, default=50)
parser.add_argument('--output', type=str, default='colabfold_batch')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ModelConfig, IDX_TO_AA, BOS_IDX, EOS_IDX, PAD_IDX, TOTAL_VOCAB, AA_TO_IDX
from model import FoldPathLLM
from rita_encoder import create_rita_encoder
from physicochemical import PHYSICO_MATRIX

# ════════ Load RITA ════════
print('Loading RITA_m...')
rita_enc = create_rita_encoder(model_name='RITA_m', device=device, freeze=True, local_dir='pretrained')

# ════════ Generate function ════════
def generate_seqs(model, n=50, temp=0.24):
    """Autoregressive generation"""
    seqs_out = []
    with torch.no_grad():
        for i in range(n):
            batch = torch.ones(1, 1, dtype=torch.long, device=device) * BOS_IDX
            generated = []
            for step in range(256):
                prefix_ids = [idx for idx in batch[0].tolist() if idx not in (PAD_IDX, BOS_IDX)]
                prefix_seq = ''.join([IDX_TO_AA.get(idx, 'X') for idx in prefix_ids])
                seqs_arg = [prefix_seq] if prefix_seq else ['M']
                logits, _, _, _ = model.forward(batch, sequences=seqs_arg, use_bias=False)
                next_logits = logits[0, -1, :] / temp
                for idx in range(20, TOTAL_VOCAB):
                    if idx != EOS_IDX: next_logits[idx] = float('-inf')
                topk, _ = torch.topk(next_logits, min(50, next_logits.size(-1)))
                next_logits[next_logits < topk[-1]] = float('-inf')
                probs = torch.softmax(next_logits, dim=-1)
                if probs.sum() == 0 or torch.isnan(probs).any(): break
                next_aa = torch.multinomial(probs, 1)
                if next_aa.item() == EOS_IDX: break
                generated.append(next_aa.item())
                batch = torch.cat([batch, next_aa.unsqueeze(0)], dim=1)
            seq = ''.join([IDX_TO_AA.get(a, 'X') for a in generated])
            if len(seq) >= 30: seqs_out.append(seq)
            if (i+1) % 10 == 0: print(f'    [{i+1}/{n}]', end='\r')
    return seqs_out

# ════════ Score function ════════
def score_seq(seq):
    L = len(seq)
    hydro = np.array([PHYSICO_MATRIX[AA_TO_IDX.get(aa, 0), 0] for aa in seq])
    charge = np.array([PHYSICO_MATRIX[AA_TO_IDX.get(aa, 0), 2] for aa in seq])
    c_ratio = seq.count('C') / L
    c_score = max(0, 1 - c_ratio * 25)
    hydro_pct = np.mean(hydro > 0.5)
    h_score = max(0, 1 - abs(0.38 - hydro_pct) * 3)
    pos = np.mean(charge > 0.5); neg = np.mean(charge < -0.5)
    ch_score = max(0, 1 - abs(pos - neg) * 5)
    max_run = 1; cur_run = 1
    for i in range(1, L):
        if seq[i] == seq[i-1]: cur_run += 1; max_run = max(max_run, cur_run)
        else: cur_run = 1
    rep_score = max(0, 1 - (max_run - 2) * 0.3)
    len_score = min(1.0, L / 160) if L < 160 else max(0, 1 - (L - 200) / 200)
    return c_score*0.25 + h_score*0.20 + ch_score*0.15 + rep_score*0.15 + len_score*0.15

# ════════ Model 1: FoldPath-LLM ════════
print(f'\n{"="*50}')
print('Model 1: FoldPath-LLM (Full)')
print(f'{"="*50}')
ckpt1 = torch.load(args.foldpath_ckpt, map_location=device, weights_only=False)
model1 = FoldPathLLM(ModelConfig(), esm_encoder=rita_enc)
model1.load_state_dict(ckpt1['model_state_dict'], strict=False)
model1 = model1.to(device).eval()

seqs_full = generate_seqs(model1, n=args.num_seqs, temp=0.24)
print(f'\n  Generated: {len(seqs_full)} valid sequences')

# Score and pick best 25
scored = [(score_seq(s), s) for s in seqs_full]
scored.sort(key=lambda x: x[0], reverse=True)
best_full = [s for _, s in scored[:min(25, len(scored))]]

# ════════ Model 2: NoStruct ════════
if args.nostruct_ckpt and os.path.exists(args.nostruct_ckpt):
    print(f'\n{"="*50}')
    print('Model 2: No Structure Track')
    print(f'{"="*50}')
    ckpt2 = torch.load(args.nostruct_ckpt, map_location=device, weights_only=False)
    model2 = FoldPathLLM(ModelConfig(), esm_encoder=rita_enc)
    model2.load_state_dict(ckpt2['model_state_dict'], strict=False)
    model2 = model2.to(device).eval()
else:
    print(f'\n{"="*50}')
    print('Model 2: No Structure Track (from FoldPathLLM, stripping struct)')
    print(f'{"="*50}')
    # Create NoStruct by zeroing structure track parameters
    model2 = FoldPathLLM(ModelConfig(), esm_encoder=rita_enc)
    model2.load_state_dict(ckpt1['model_state_dict'], strict=False)
    # Zero out structure track
    for name, param in model2.named_parameters():
        if 'structure_track' in name or 'chem_bias' in name:
            param.data.zero_()
    model2 = model2.to(device).eval()

seqs_nostruct = generate_seqs(model2, n=args.num_seqs, temp=0.24)
print(f'\n  Generated: {len(seqs_nostruct)} valid sequences')
scored2 = [(score_seq(s), s) for s in seqs_nostruct]
scored2.sort(key=lambda x: x[0], reverse=True)
best_nostruct = [s for _, s in scored2[:min(25, len(scored2))]]

# ════════ Save FASTA ════════
os.makedirs(args.output, exist_ok=True)

# FoldPath-LLM batch
fasta_fp = os.path.join(args.output, 'foldpath_25.fasta')
with open(fasta_fp, 'w') as f:
    for i, s in enumerate(best_full):
        f.write(f'>FP_seq{i+1}_len{len(s)}\n{s}\n')
print(f'\nSaved: {fasta_fp} ({len(best_full)} sequences)')

# NoStruct batch
fasta_ns = os.path.join(args.output, 'nostruct_25.fasta')
with open(fasta_ns, 'w') as f:
    for i, s in enumerate(best_nostruct):
        f.write(f'>NS_seq{i+1}_len{len(s)}\n{s}\n')
print(f'Saved: {fasta_ns} ({len(best_nostruct)} sequences)')

# ════════ Instructions ════════
print(f'\n{"="*60}')
print(f'  NEXT STEPS')
print(f'{"="*60}')
print(f'''
  1. Upload {fasta_fp} to ColabFold, run all 25 sequences
  2. Upload {fasta_ns} to ColabFold, run all 25 sequences
  3. Download results, unzip to {args.output}/
  4. Run: python analyze_ablation_fold.py --colabfold-dir {args.output}
''')

# Save metadata
meta = {
    'timestamp': datetime.now().isoformat(),
    'foldpath_ckpt': args.foldpath_ckpt,
    'nostruct_ckpt': args.nostruct_ckpt or 'zeroed structure_track',
    'num_seqs': args.num_seqs,
    'foldpath_valid': len(seqs_full),
    'nostruct_valid': len(seqs_nostruct),
}
with open(os.path.join(args.output, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)
