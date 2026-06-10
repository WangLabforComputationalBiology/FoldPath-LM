"""测试RITA生成100步的熵变化——看它到底能不能生成更长序列"""
import torch, os, sys
sys.path.insert(0, '.')
os.chdir('./pretrained/RITA_m')
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
_orig_init = PreTrainedModel.__init__
def _patched_init(self, config, *a, **kw):
    self._tied_weights_keys = []; self.all_tied_weights_keys = {}
    _orig_init(self, config, *a, **kw)
PreTrainedModel.__init__ = _patched_init
tok = AutoTokenizer.from_pretrained('.', trust_remote_code=True)
tok.pad_token = '[PAD]'; tok.pad_token_id = 1
m = AutoModelForCausalLM.from_pretrained('.', trust_remote_code=True).to('cuda').eval()
os.chdir('..')
from config import AA_TO_IDX, IDX_TO_AA

aa_to_r = {}
for aa in 'ACDEFGHIKLMNPQRSTVWY':
    ids = tok.encode(aa); aa_to_r[aa] = ids[0] if ids else None
r_to_aa = {v: AA_TO_IDX[k] for k, v in aa_to_r.items() if v is not None}
valid = torch.tensor(sorted(r_to_aa.keys()), device='cuda')

for run in [1, 2, 3, 4, 5]:
    input_ids = torch.tensor([[aa_to_r['M']]], device='cuda')
    entropies = []
    for step in range(100):
        out = m(input_ids)
        logits = out.logits[0, -1, :] / 1.0
        mask = torch.full_like(logits, float('-inf'))
        mask[valid] = logits[valid]
        probs = torch.softmax(mask, dim=-1)
        ent = -(probs * torch.log(probs + 1e-8)).sum().item()
        entropies.append(ent)
        next_tok = torch.multinomial(probs, 1).item()
        if next_tok in r_to_aa:
            input_ids = torch.cat([input_ids, torch.tensor([[next_tok]], device='cuda')], dim=1)
        else:
            print(f'  Run{run}: BREAK at step {step} (non-AA token {next_tok})')
            break

    seq = ''.join([IDX_TO_AA.get(r_to_aa.get(t.item(), 99), 'X') for t in input_ids[0]])
    avg_first10 = sum(entropies[:10]) / min(10, len(entropies))
    avg_last10 = sum(entropies[-10:]) / min(10, len(entropies))
    print(f'  Run{run}: len={len(seq)}, '
          f'entropy early={avg_first10:.2f} late={avg_last10:.2f}, '
          f'seq={seq[:40]}...')
