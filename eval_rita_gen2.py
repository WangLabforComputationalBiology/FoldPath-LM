"""
RITA_m 原生生成 + 天然度评测 (50条) — 基于 debug 逻辑重写
用法: python eval_rita_gen2.py
"""
import torch, sys, os, json, time, numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import AA_TO_IDX, IDX_TO_AA
from evaluation import FoldPathBenchmark

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# ── 加载 RITA_m ──
PROJECT_ROOT = os.getcwd()
RITA_DIR = os.path.join(PROJECT_ROOT, 'pretrained', 'RITA_m')
os.chdir(RITA_DIR)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
_orig_init = PreTrainedModel.__init__
def _patched_init(self, config, *a, **kw):
    self._tied_weights_keys = []; self.all_tied_weights_keys = {}
    _orig_init(self, config, *a, **kw)
PreTrainedModel.__init__ = _patched_init

tok = AutoTokenizer.from_pretrained('.', trust_remote_code=True)
tok.pad_token = '[PAD]'; tok.pad_token_id = 1
m = AutoModelForCausalLM.from_pretrained('.', trust_remote_code=True).to(device).eval()
os.chdir(PROJECT_ROOT)  # 回到项目根目录

# AA 映射
aa_to_r = {}
for aa in 'ACDEFGHIKLMNPQRSTVWY':
    ids = tok.encode(aa)
    aa_to_r[aa] = ids[0] if ids else None
r_to_aa = {v: AA_TO_IDX[k] for k, v in aa_to_r.items() if v is not None}
valid = torch.tensor(sorted(r_to_aa.keys()), device=device)

print(f'AA 映射: {len(r_to_aa)}/20')

# ── 生成 (与 debug 完全一致) ──
NUM = 50
TEMP = 1.0
TOP_K = 50

print(f'生成 {NUM} 条 (temp={TEMP})...')
all_seqs = []
t0 = time.time()

MAX_STEPS = 115  # FoldPathLLM均值116 = 1(start M) + 115 steps

with torch.no_grad():
    for i in range(NUM):
        input_ids = torch.tensor([[aa_to_r['M']]], device=device)
        for step in range(MAX_STEPS):
            out = m(input_ids)
            logits = out.logits[0, -1, :] / TEMP

            # 屏蔽非 AA
            mask = torch.full_like(logits, float('-inf'))
            mask[valid] = logits[valid]

            # top-k
            k = min(TOP_K, (mask > float('-inf')).sum().item())
            if k > 0:
                tk_vals, _ = torch.topk(mask, k)
                mask[mask < tk_vals[-1]] = float('-inf')

            probs = torch.softmax(mask, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()

            if next_tok in r_to_aa:
                input_ids = torch.cat([input_ids, torch.tensor([[next_tok]], device=device)], dim=1)
            else:
                break

        seq = ''.join([IDX_TO_AA.get(r_to_aa.get(t.item(), 99), 'X') for t in input_ids[0]])
        if len(seq) >= 20:
            all_seqs.append(seq)

        print(f'  [{i+1}/{NUM}] 长度={len(seq)} 累计={len(all_seqs)}')

gen_time = time.time() - t0
print(f'有效: {len(all_seqs)}/{NUM}, 耗时 {gen_time:.1f}s')

if len(all_seqs) == 0:
    print('[ERROR] 无有效序列'); sys.exit(1)

# ── 评测 ──
ref_fasta = os.path.join(PROJECT_ROOT, 'data', 'train_sequences.fasta')
print(f'[DEBUG] CWD={os.getcwd()} ref_fasta={ref_fasta} exists={os.path.exists(ref_fasta)}')

bench = FoldPathBenchmark(reference_fasta=ref_fasta)
result = bench.evaluate(all_seqs, verbose=True)

# 单独打印每个序列的天然度，排查方差
nat_indiv = bench.naturalness.batch_score(all_seqs)
nat_scores = [s['total'] for s in nat_indiv['individual']]
print(f'\n个体天然度: mean={np.mean(nat_scores):.4f} std={np.std(nat_scores):.4f} '
      f'min={np.min(nat_scores):.4f} max={np.max(nat_scores):.4f}')
print(f'前10条: {[f"{x:.4f}" for x in nat_scores[:10]]}')
print(f'唯一值数: {len(set(round(x, 4) for x in nat_scores))}')

output = {
    'timestamp': datetime.now().isoformat(),
    'model': 'RITA_m (原生)',
    'num_valid': len(all_seqs),
    'temperature': TEMP,
    'generation_time_s': round(gen_time, 1),
    'physico_mean': round(result['physico']['mean'], 4),
    'physico_std': round(result['physico']['std'], 4),
    'diversity': round(result['diversity']['total'], 4),
    'naturalness_mean': round(float(np.mean(nat_scores)), 4),
    'naturalness_std': round(float(np.std(nat_scores)), 4),
    'naturalness_individual': [round(x, 4) for x in nat_scores],
    'composite': round(result['composite'], 4),
    'grade': result['grade'],
}
with open('rita_gen_result.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f'\n结果已保存: rita_gen_result.json')
