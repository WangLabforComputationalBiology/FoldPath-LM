"""
ProGen2-small Baseline: Naturalness + Diversity + ESM score
用法: python progen2_baseline.py
"""
import torch, sys, os, json, numpy as np, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num-seqs', type=int, default=30)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--output', type=str, default='progen2_result.json')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import FoldPathBenchmark
from config import AA_TO_IDX

# ════════ Load ProGen2-small ════════
model_path = 'pretrained/progen2-small'
print(f'Loading ProGen2-small from {model_path}...')

from transformers import AutoModelForCausalLM, AutoTokenizer

# Try loading with trust_remote_code
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token or '<pad>'
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
print(f'Model loaded. Vocab size: {tokenizer.vocab_size}')

# ════════ Generate ════════
print(f'\nGenerating {args.num_seqs} sequences (T={args.temperature})...')
seqs = []

with torch.no_grad():
    for i in range(args.num_seqs):
        start_id = tokenizer.bos_token_id or 0
        input_ids = torch.tensor([[start_id]], device=device)
        generated = []

        for _ in range(256):
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :] / args.temperature

            # Mask special tokens
            for tid in [tokenizer.pad_token_id, tokenizer.bos_token_id,
                        tokenizer.eos_token_id, tokenizer.unk_token_id]:
                if tid is not None:
                    logits[tid] = float('-inf')

            topk, _ = torch.topk(logits, min(50, logits.size(-1)))
            logits[logits < topk[-1]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            if probs.sum() == 0 or torch.isnan(probs).any():
                break

            next_tok = torch.multinomial(probs, 1).item()
            if next_tok == tokenizer.eos_token_id:
                break
            generated.append(next_tok)
            input_ids = torch.cat([input_ids, torch.tensor([[next_tok]], device=device)], dim=1)

        seq = tokenizer.decode(generated, skip_special_tokens=True)
        seq = seq.replace(' ', '').upper()
        seq = ''.join(c for c in seq if c in 'ACDEFGHIKLMNPQRSTVWY')
        if len(seq) >= 30:
            seqs.append(seq)
        if (i+1) % 10 == 0:
            print(f'  [{i+1}/{args.num_seqs}] valid: {len(seqs)}')

print(f'Valid sequences: {len(seqs)}')

# ════════ Evaluate ════════
print('\nEvaluating...')
bench = FoldPathBenchmark(reference_fasta='data/train_sequences.fasta')
result = bench.evaluate(seqs[:min(len(seqs), 50)], verbose=True)

# ════════ ESM Score ════════
print('\nComputing ESM-2 score...')
from esm_encoder import create_esm_encoder
encoder = create_esm_encoder(model_name='esm2_t6_8M_UR50D', device=device, freeze=True, local_dir='pretrained')

@torch.no_grad()
def esm_score(seqs):
    vals = []
    for seq in seqs:
        clean = ''.join(c for c in seq if c in AA_TO_IDX)
        if len(clean) < 5: continue
        clean = clean[:200]
        emb, _ = encoder([clean])
        e = emb[0, :min(emb.size(1), len(clean)), :]
        normed = e / (e.norm(dim=1, keepdim=True) + 1e-8)
        sim = torch.mm(normed, normed.t())
        n = sim.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        vals.append(sim[mask].mean().item())
    return round(np.mean(vals), 4) if vals else 0

esm = esm_score(seqs)

# Also score RITA_m validation sequences for comparison
print('\nScoring validation set for RITA_m ESM baseline...')
from dataset import load_fasta as lf
val_all = lf('data/val_sequences.fasta')
rita_ref = [s for s in val_all[:50] if len(''.join(c for c in s if c in AA_TO_IDX)) >= 30]
esm_rita = esm_score(rita_ref[:20])

# ════════ Output ════════
output = {
    'model': 'ProGen2-small',
    'num_seqs': len(seqs),
    'temperature': args.temperature,
    'physico_mean': round(result['physico']['mean'], 4),
    'diversity': round(result['diversity']['total'], 4),
    'naturalness_mean': round(result['naturalness']['mean'], 4),
    'composite': round(result['composite'], 4),
    'grade': result['grade'],
    'esm_coherence': esm,
    'esm_rita': esm_rita,
}

print(f'\n{"="*60}')
print(f'  Comprehensive Baseline Comparison')
print(f'{"="*60}')
print(f'  {"Model":<20} {"Nat.":>7} {"Div.":>7} {"Phys.":>7} {"ESM":>7} {"Grade":>6}')
print(f'  {"-"*20} {"-"*7} {"-"*7} {"-"*7} {"-"*7} {"-"*6}')
print(f'  {"FoldPath-LLM":<20} {0.514:>7.3f} {0.819:>7.3f} {0.752:>7.3f} {0.698:>7.3f} {"B":>6}')
print(f'  {"RITA_m":<20} {0.483:>7.3f} {0.741:>7.3f} {0.786:>7.3f} {esm_rita:>7.3f} {"B":>6}')
print(f'  {"NoStruct":<20} {0.560:>7.3f} {0.809:>7.3f} {0.738:>7.3f} {0.712:>7.3f} {"B":>6}')
print(f'  {"ProGen2-small":<20} {output["naturalness_mean"]:>7.3f} {output["diversity"]:>7.3f} {output["physico_mean"]:>7.3f} {esm:>7.3f} {output["grade"]:>6}')
print(f'{"="*60}')

with open(args.output, 'w') as f:
    json.dump(output, f, indent=2)
print(f'\nSaved: {args.output}')
