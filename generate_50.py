"""
绕过 transformers/torchvision/lzma 的生成脚本
直接加载 FoldPathLLM + RITA_m，手动自回归生成
用法: python generate_50.py --num-seqs 50 --output gen_50.fasta
"""
import torch, sys, os, argparse, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='esmpro/foldpath_best_eopch10.pt')
parser.add_argument('--rita-dir', type=str, default='pretrained/RITA_m')
parser.add_argument('--num-seqs', type=int, default=50)
parser.add_argument('--num-best', type=int, default=15, help='Number of top-scoring sequences to output')
parser.add_argument('--output', type=str, default='gen_50.fasta')
parser.add_argument('--temperature', type=float, default=0.24)
parser.add_argument('--nostruct', action='store_true', help='Zero out structure track for ablation')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ModelConfig, IDX_TO_AA, BOS_IDX, EOS_IDX, PAD_IDX, TOTAL_VOCAB
from model import FoldPathLLM

# ════════ 加载 RITA ════════
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import PreTrainedModel
    _orig_init = PreTrainedModel.__init__
    def _patched_init(self, cfg, *a, **kw):
        self._tied_weights_keys = []; self.all_tied_weights_keys = {}
        _orig_init(self, cfg, *a, **kw)
    PreTrainedModel.__init__ = _patched_init

    cwd = os.getcwd(); os.chdir(args.rita_dir)
    tokenizer = AutoTokenizer.from_pretrained('.', trust_remote_code=True)
    tokenizer.pad_token = '[PAD]'; tokenizer.pad_token_id = 1
    rita_model = AutoModelForCausalLM.from_pretrained('.', trust_remote_code=True).to(device).eval()
    os.chdir(cwd)
    print(f'RITA_m loaded ({args.rita_dir})')
except Exception as e:
    print(f'RITA loading failed: {e}')
    sys.exit(1)

# ════════ Wrapper encoder ════════
class RitaEncoderWrapper:
    def __init__(self, model, tokenizer):
        self.hidden_size = 1024
        self.model = model
        self.tokenizer = tokenizer
        self.adds_special_tokens = False
        self.rita_model_name = 'RITA_m'
    def __call__(self, seqs):
        inp = self.tokenizer(seqs, padding=True, return_tensors='pt')
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = self.model(**inp, output_hidden_states=True)
            hs = out.hidden_states[-1] if hasattr(out, 'hidden_states') else out.last_hidden_state
        return hs, inp['attention_mask'].bool()
    def get_param_count(self):
        t = sum(p.numel() for p in self.model.parameters())
        return {'total': t, 'trainable': 0, 'frozen': t}

encoder = RitaEncoderWrapper(rita_model, tokenizer)

# ════════ Load FoldPathLLM ════════
print(f'Loading checkpoint: {args.checkpoint}')
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
model = FoldPathLLM(ModelConfig(), esm_encoder=encoder)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model = model.to(device).eval()
if args.nostruct:
    for name, param in model.named_parameters():
        if 'structure_track' in name or 'chem_bias' in name:
            param.data.zero_()
    print('[NoStruct] Structure track + chem_bias parameters zeroed')
print(f'Model loaded. Params: {model.get_param_count()["total"]:,}')

# ════════ Generate ════════
print(f'\nGenerating {args.num_seqs} sequences (T={args.temperature})...')
seqs_out = []

with torch.no_grad():
    for n in range(args.num_seqs):
        batch = torch.ones(1, 1, dtype=torch.long, device=device) * BOS_IDX
        generated = []
        for step in range(256):
            prefix_ids = [idx for idx in batch[0].tolist() if idx not in (PAD_IDX, BOS_IDX)]
            prefix_seq = ''.join([IDX_TO_AA.get(idx, 'X') for idx in prefix_ids])
            seqs_arg = [prefix_seq] if prefix_seq else ['M']

            logits, _, _, _ = model.forward(batch, sequences=seqs_arg, use_bias=False)
            next_logits = logits[0, -1, :] / args.temperature

            for idx in range(20, TOTAL_VOCAB):
                if idx != EOS_IDX:
                    next_logits[idx] = float('-inf')

            # Top-k
            topk_vals, _ = torch.topk(next_logits, min(50, next_logits.size(-1)))
            next_logits[next_logits < topk_vals[-1]] = float('-inf')

            probs = torch.softmax(next_logits, dim=-1)
            if probs.sum() == 0 or torch.isnan(probs).any():
                break

            next_aa = torch.multinomial(probs, 1)
            if next_aa.item() == EOS_IDX:
                break
            generated.append(next_aa.item())
            batch = torch.cat([batch, next_aa.unsqueeze(0)], dim=1)

        seq = ''.join([IDX_TO_AA.get(a, 'X') for a in generated])
        if len(seq) >= 30:
            seqs_out.append(seq)
        if (n + 1) % 10 == 0:
            print(f'  [{n+1}/{args.num_seqs}] valid: {len(seqs_out)}')

print(f'\nValid sequences: {len(seqs_out)}/{args.num_seqs}')

# ════════ Score & filter ════════
from physicochemical import PHYSICO_MATRIX
from config import AA_TO_IDX

scored = []
for seq in seqs_out:
    L = len(seq)
    hydro = np.array([PHYSICO_MATRIX[AA_TO_IDX.get(aa, 0), 0] for aa in seq])
    charge = np.array([PHYSICO_MATRIX[AA_TO_IDX.get(aa, 0), 2] for aa in seq])

    c_ratio = seq.count('C') / L
    c_score = max(0, 1 - c_ratio * 25)
    hydro_pct = np.mean(hydro > 0.5)
    h_score = max(0, 1 - abs(0.38 - hydro_pct) * 3)
    pos = np.mean(charge > 0.5)
    neg = np.mean(charge < -0.5)
    ch_score = max(0, 1 - abs(pos - neg) * 5)

    max_run = 1; cur_run = 1
    for i in range(1, L):
        if seq[i] == seq[i-1]: cur_run += 1; max_run = max(max_run, cur_run)
        else: cur_run = 1
    rep_score = max(0, 1 - (max_run - 2) * 0.3)

    # Length bias: Gaussian centered at 75aa (optimal for AF2), falloff sigma=15
    len_score = np.exp(-0.5 * ((L - 75) / 15) ** 2)
    total = c_score*0.25 + h_score*0.20 + ch_score*0.15 + rep_score*0.10 + len_score*0.30

    scored.append((total, seq))

scored.sort(key=lambda x: x[0], reverse=True)
best = scored[:args.num_best]

print(f'\nTop {args.num_best} by quality score:')
for i, (score, seq) in enumerate(best):
    print(f'  [{i+1}] score={score:.3f} len={len(seq)} C={seq.count("C")}  {seq[:50]}...')

# ════════ Save top N ════════
with open(args.output, 'w') as f:
    for i, (score, s) in enumerate(best):
        prefix = 'NS_' if args.nostruct else ''
    f.write(f'>{prefix}seq{i+1}_score{score:.3f}_len{len(s)}\n{s}\n')

mode_str = 'NoStruct' if args.nostruct else 'FoldPath-LLM'
print(f'\n[{mode_str}] Saved top {len(best)} to: {args.output}')
print(f'Next: upload {args.output} to ColabFold')
