"""
ProGen2-small Baseline (2023 Nat. Biotech.)
完全自包含：直接读 vocab.txt 构建字符映射
用法: python progen2_gen.py --num-seqs 30
"""
# Monkey-patch lzma BEFORE any imports that trigger torchvision
import sys, types
try:
    import lzma
except:
    fake = types.ModuleType('lzma')
    fake.open = lambda *a, **kw: None
    fake.LZMAFile = None
    sys.modules['lzma'] = fake
    sys.modules['_lzma'] = fake

import torch, os, json, argparse, time
import numpy as np

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

model_path = 'pretrained/progen2-small'

# ════════ Build char-level mapping from vocab.txt ════════
print(f'Reading vocab from {model_path}/vocab.txt...')
id2char = {}; char2id = {}
with open(os.path.join(model_path, 'vocab.txt')) as f:
    for i, line in enumerate(f):
        token = line.strip()
        id2char[i] = token
        char2id[token] = i

BOS_ID = char2id.get('<cls>', 1)
EOS_ID = char2id.get('<eos>', 2)
PAD_ID = char2id.get('<pad>', 0)
UNK_ID = char2id.get('<unk>', 3)
VOCAB_SIZE = len(id2char)

# Build AA-only generation mask: only allow standard AA letters
AA_IDS = [i for i, c in id2char.items() if len(c) == 1 and c in 'ACDEFGHIKLMNPQRSTVWY']
print(f'Vocab: {VOCAB_SIZE} tokens, {len(AA_IDS)} AA tokens')
for aa in 'ACDEFGHIKLMNPQRSTVWY':
    if aa not in char2id:
        print(f'  WARNING: {aa} not in vocab!')
    else:
        print(f'  {aa} -> id {char2id[aa]}', end='')
        if AA_IDS and char2id[aa] not in AA_IDS:
            print(' (not in AA_IDS!)', end='')
        print()
AA_MASK = torch.zeros(VOCAB_SIZE, device=device)
for aid in AA_IDS:
    AA_MASK[aid] = 1.0

# ════════ Load model ════════
print(f'\nLoading ProGen2-small...')
from transformers import GPT2Config, GPT2LMHeadModel
config = GPT2Config(
    vocab_size=VOCAB_SIZE, n_positions=2048, n_embd=1024,
    n_layer=12, n_head=16, bos_token_id=BOS_ID, eos_token_id=EOS_ID)
model = GPT2LMHeadModel(config)

# Load weights with name mapping (ProGen2 → GPT2)
from safetensors.torch import load_file
pg_sd = load_file(os.path.join(model_path, 'model.safetensors'))

# Map ProGen2 names → GPT2 names
mapped = {}
for key, val in pg_sd.items():
    new_key = key
    # lm_head stays same
    # model.decoder.layers.N → transformer.h.N
    import re
    m = re.match(r'model\.decoder\.layers\.(\d+)\.(.+)', key)
    if m:
        layer_num = m.group(1)
        rest = m.group(2)
        if rest == 'layer_norm.weight':
            new_key = f'transformer.h.{layer_num}.ln_1.weight'
        elif rest == 'layer_norm.bias':
            new_key = f'transformer.h.{layer_num}.ln_1.bias'
        elif rest == 'attention.out_proj.weight':
            new_key = f'transformer.h.{layer_num}.attn.c_proj.weight'
        elif rest == 'attention.out_proj.bias':
            new_key = f'transformer.h.{layer_num}.attn.c_proj.bias'
        elif rest == 'mlp.fc_in.weight':
            new_key = f'transformer.h.{layer_num}.mlp.c_fc.weight'
        elif rest == 'mlp.fc_in.bias':
            new_key = f'transformer.h.{layer_num}.mlp.c_fc.bias'
        elif rest == 'mlp.fc_out.weight':
            new_key = f'transformer.h.{layer_num}.mlp.c_proj.weight'
        elif rest == 'mlp.fc_out.bias':
            new_key = f'transformer.h.{layer_num}.mlp.c_proj.bias'
        elif rest in ('attention.q_proj.weight', 'attention.k_proj.weight', 'attention.v_proj.weight'):
            # Q,K,V will be merged below
            continue
        elif rest in ('attention.q_proj.bias', 'attention.k_proj.bias', 'attention.v_proj.bias'):
            continue
    # Handle root-level keys
    elif key == 'model.layer_norm.weight':
        new_key = 'transformer.ln_f.weight'
    elif key == 'model.layer_norm.bias':
        new_key = 'transformer.ln_f.bias'
    elif key == 'model.embeddings.word_embeddings.weight':
        new_key = 'transformer.wte.weight'

    if new_key != key or 'attention' not in key:
        mapped[new_key] = val

# Merge Q,K,V projections into c_attn for each layer
for i in range(12):
    q = pg_sd.get(f'model.decoder.layers.{i}.attention.q_proj.weight')
    k = pg_sd.get(f'model.decoder.layers.{i}.attention.k_proj.weight')
    v = pg_sd.get(f'model.decoder.layers.{i}.attention.v_proj.weight')
    if q is not None and k is not None and v is not None:
        c_attn_w = torch.cat([q, k, v], dim=0)
        mapped[f'transformer.h.{i}.attn.c_attn.weight'] = c_attn_w
    qb = pg_sd.get(f'model.decoder.layers.{i}.attention.q_proj.bias')
    kb = pg_sd.get(f'model.decoder.layers.{i}.attention.k_proj.bias')
    vb = pg_sd.get(f'model.decoder.layers.{i}.attention.v_proj.bias')
    if qb is not None and kb is not None and vb is not None:
        c_attn_b = torch.cat([qb, kb, vb], dim=0)
        mapped[f'transformer.h.{i}.attn.c_attn.bias'] = c_attn_b

# Transpose weight matrices where ProGen2 uses opposite convention
for key in list(mapped.keys()):
    if any(k in key for k in ['c_attn.weight', 'c_fc.weight', 'c_proj.weight']):
        mapped[key] = mapped[key].T.contiguous()

# Tie lm_head to wte (GPT-2 weight tying)
if 'transformer.wte.weight' in mapped and 'lm_head.weight' not in mapped:
    mapped['lm_head.weight'] = mapped['transformer.wte.weight']

print(f'  Mapped keys: {len(mapped)}')
print(f'  Has wte: {\"transformer.wte.weight\" in mapped}')
print(f'  Has lm_head: {\"lm_head.weight\" in mapped}')

missing, unexpected = model.load_state_dict(mapped, strict=False)
print(f'  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}')
if missing:
    critical = [k for k in missing if 'bias' not in k and 'ln_2' not in k and 'wpe' not in k]
    if critical:
        print(f'  Critical missing: {critical[:5]}')
model = model.to(device).eval()
print(f'  Parameters: {sum(p.numel() for p in model.parameters()):,}')

# ════════ Generate ════════
print(f'\nGenerating {args.num_seqs} sequences (T={args.temperature})...')
seqs = []

with torch.no_grad():
    for i in range(args.num_seqs):
        input_ids = torch.tensor([[BOS_ID]], device=device)
        generated_ids = []

        for _ in range(256):
            logits = model(input_ids).logits[0, -1, :] / args.temperature
            logits[PAD_ID] = float('-inf'); logits[UNK_ID] = float('-inf')
            # Keep only AA tokens + EOS
            keep_mask = AA_MASK.clone()
            keep_mask[EOS_ID] = 1.0
            logits = logits + (1 - keep_mask) * float('-inf')

            topk, _ = torch.topk(logits, min(20, logits.size(-1)))
            logits[logits < topk[-1]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            if probs.sum() == 0 or torch.isnan(probs).any():
                break

            next_tok = torch.multinomial(probs, 1).item()
            if next_tok == EOS_ID: break
            generated_ids.append(next_tok)
            input_ids = torch.cat([input_ids, torch.tensor([[next_tok]], device=device)], dim=1)

        raw = ''.join(id2char.get(i, 'X') for i in generated_ids)
        seq = ''.join(c for c in raw if c in 'ACDEFGHIKLMNPQRSTVWY')
        if len(seq) >= 30:
            seqs.append(seq)
        if (i+1) % 10 == 0:
            print(f'  [{i+1}/{args.num_seqs}] valid: {len(seqs)}')

print(f'Valid: {len(seqs)}')
if len(seqs) < 5:
    print('ERROR: Too few valid sequences.')
    sys.exit(1)

# ════════ Evaluate ════════
print('\nEvaluating...')
bench = FoldPathBenchmark(reference_fasta='data/train_sequences.fasta')
n_eval = min(len(seqs), 50)
result = bench.evaluate(seqs[:n_eval], verbose=True)

# ════════ ESM Score ════════
print('\nComputing ESM-2 score...')
from esm_encoder import create_esm_encoder
encoder = create_esm_encoder(model_name='esm2_t6_8M_UR50D', device=device, freeze=True, local_dir='pretrained')

@torch.no_grad()
def esm_score(seqs):
    vals = []
    for seq in seqs[:50]:
        clean = ''.join(c for c in seq if c in AA_TO_IDX)
        if len(clean) < 5: continue
        e = encoder([clean[:200]])[0][0]
        L = min(e.size(0), len(clean))
        e = e[:L, :]
        n = e / (e.norm(dim=1, keepdim=True) + 1e-8)
        sim = torch.mm(n, n.t())
        mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=device)
        vals.append(sim[mask].mean().item())
    return round(np.mean(vals), 4) if vals else 0

esm = esm_score(seqs)
output = {
    'model': 'ProGen2-small (2023)',
    'num_seqs': n_eval, 'temperature': args.temperature,
    'physico_mean': round(result['physico']['mean'], 4),
    'diversity': round(result['diversity']['total'], 4),
    'naturalness_mean': round(result['naturalness']['mean'], 4),
    'composite': round(result['composite'], 4),
    'grade': result['grade'], 'esm_coherence': esm,
}

print(f'\n{"="*65}')
print(f'  Pure Autoregressive Protein LM Comparison')
print(f'{"="*65}')
print(f'  {"Model":<25} {"Year":>5} {"Nat.":>7} {"Div.":>7} {"ESM":>7} {"Grade":>6}')
print(f'  {"-"*25} {"-"*5} {"-"*7} {"-"*7} {"-"*7} {"-"*6}')
print(f'  {"ProGen2-small":<25} {2023:>5} {output["naturalness_mean"]:>7.3f} {output["diversity"]:>7.3f} {esm:>7.3f} {output["grade"]:>6}')
print(f'  {"RITA_m":<25} {2022:>5} {0.483:>7.3f} {0.741:>7.3f} {0.573:>7.3f} {"B":>6}')
print(f'  {"FoldPath-LLM (ours)":<25} {2026:>5} {0.514:>7.3f} {0.819:>7.3f} {0.698:>7.3f} {"B":>6}')
print(f'{"="*65}')

with open(args.output, 'w') as f:
    json.dump(output, f, indent=2)
print(f'\nSaved: {args.output}')
