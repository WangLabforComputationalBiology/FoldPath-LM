"""调试 RITA 生成——看模型到底输出什么"""
import torch, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
device = torch.device('cuda')
m = AutoModelForCausalLM.from_pretrained('.', trust_remote_code=True).to(device).eval()

aa_to_r = {}
for aa in 'ACDEFGHIKLMNPQRSTVWY':
    ids = tok.encode(aa)
    aa_to_r[aa] = ids[0] if ids else None
r_to_aa = {v: k for k, v in aa_to_r.items() if v is not None}

print(f'Vocab size: {tok.vocab_size}')
print(f'AA tokens: {sorted(aa_to_r.values())}')

# 试 5 条短序列
for run in range(3):
    input_ids = torch.tensor([[aa_to_r['M']]], device=device)
    print(f'\n--- Run {run+1}: start=M ---')
    for step in range(20):
        out = m(input_ids)
        logits = out.logits[0, -1, :] / 1.0
        want = torch.tensor(sorted(r_to_aa.keys()), device=device)
        mask = torch.full_like(logits, float('-inf'))
        mask[want] = logits[want]
        probs = torch.softmax(mask, dim=-1)
        next_tok = torch.multinomial(probs, 1).item()
        if next_tok in r_to_aa:
            input_ids = torch.cat([input_ids, torch.tensor([[next_tok]], device=device)], dim=1)
        else:
            print(f'  BREAK step={step}: token={next_tok} (not AA)')
            break
    seq = ''.join([r_to_aa.get(t.item(), '?') for t in input_ids[0]])
    print(f'  Result ({len(seq)}): {seq[:80]}')
