"""
一键折叠验证流水线: 过滤短序列 → ESMFold → 统计分析
用法: python run_fold_pipeline.py --input-dir colabfold_batch
"""
import torch, sys, os, json, argparse, time, re, subprocess
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, default='colabfold_batch')
parser.add_argument('--max-length', type=int, default=200)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ════════ Step 1: Filter short sequences ════════
print('\n[Step 1] Filtering sequences <= {args.max_length}aa...')
filtered = {}
for src_name in ['foldpath_25.fasta', 'nostruct_25.fasta']:
    src = os.path.join(args.input_dir, src_name)
    if not os.path.exists(src):
        print(f'  SKIP: {src} not found')
        continue
    dst = os.path.join(args.input_dir, src_name.replace('.fasta', '_short.fasta'))
    kept = 0
    with open(src) as fin, open(dst, 'w') as fout:
        header = ''; seq = ''
        for line in fin:
            if line.startswith('>'):
                if header and len(seq) <= args.max_length:
                    fout.write(f'{header}\n{seq}\n'); kept += 1
                header = line.strip(); seq = ''
            else: seq += line.strip()
        if header and len(seq) <= args.max_length:
            fout.write(f'{header}\n{seq}\n'); kept += 1
    filtered[src_name] = kept
    print(f'  {src_name}: {kept} sequences <= {args.max_length}aa')

# ════════ Step 2: ESMFold ════════
print('\n[Step 2] Running ESMFold...')
try:
    import esm
    model = esm.pretrained.esmfold_v1()
    model = model.eval().to(device)
    print('  ESMFold v1 loaded')
except Exception as e:
    print(f'  ESMFold failed: {e}')
    print('  Try: pip install omegaconf && pip install openfold')
    sys.exit(1)

aa3to1 = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
          'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
          'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}

all_results = {'FP': [], 'NS': []}

for prefix, src_name in [('FP', 'foldpath_25_short.fasta'), ('NS', 'nostruct_25_short.fasta')]:
    fasta_path = os.path.join(args.input_dir, src_name)
    if not os.path.exists(fasta_path):
        print(f'  SKIP: {fasta_path}')
        continue
    out_dir = os.path.join(args.input_dir, f'{prefix}_results')
    os.makedirs(out_dir, exist_ok=True)

    # Load sequences
    seqs = []
    with open(fasta_path) as f:
        cur_id, cur_seq = None, []
        for line in f:
            if line.startswith('>'):
                if cur_id: seqs.append({'id': cur_id, 'seq': ''.join(cur_seq)})
                cur_id = line.strip()[1:]; cur_seq = []
            else: cur_seq.append(line.strip())
        if cur_id: seqs.append({'id': cur_id, 'seq': ''.join(cur_seq)})

    print(f'\n  [{prefix}] {len(seqs)} sequences')

    for i, s in enumerate(seqs):
        seq = s['seq']; L = len(seq)
        t0 = time.time()
        pdb_path = os.path.join(out_dir, f'{s["id"]}.pdb')

        if os.path.exists(pdb_path):
            with open(pdb_path) as f: output = f.read()
            print(f'    [{i+1}/{len(seqs)}] {s["id"]} (cached)', end='\r')
        else:
            try:
                with torch.no_grad():
                    output = model.infer_pdb(seq)
                with open(pdb_path, 'w') as f: f.write(output)
                elapsed = time.time() - t0
                print(f'    [{i+1}/{len(seqs)}] {s["id"]} len={L} {elapsed:.1f}s')
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    torch.cuda.empty_cache()
                    print(f'    [{i+1}/{len(seqs)}] {s["id"]} OOM, skipping')
                    continue
                raise

        # Parse pLDDT
        plddt_vals = []
        for line in output.split('\n'):
            if line.startswith('ATOM') and line[13:15] == 'CA':
                try: plddt_vals.append(float(line[60:66]))
                except ValueError: pass

        if plddt_vals:
            plddt_arr = np.array(plddt_vals)
            r = {
                'id': s['id'], 'length': L,
                'mean_plddt': round(float(np.mean(plddt_arr)), 1),
                'plddt_std': round(float(np.std(plddt_arr)), 1),
                'high_conf_80': round(float(np.mean(plddt_arr > 80)) * 100, 1),
                'plddt_max': round(float(np.max(plddt_arr)), 1),
                'plddt_min': round(float(np.min(plddt_arr)), 1),
            }
            all_results[prefix].append(r)

    print(f'\n    [{prefix}] Done: {len(all_results[prefix])} valid structures')

# ════════ Step 3: Statistics ════════
print(f'\n[Step 3] Statistical analysis...')
fp = all_results['FP']; ns = all_results['NS']

if len(fp) < 3 or len(ns) < 3:
    print(f'  ERROR: Need >=3 sequences per model. FP={len(fp)}, NS={len(ns)}')
    sys.exit(1)

fp_plddt = [r['mean_plddt'] for r in fp]
ns_plddt = [r['mean_plddt'] for r in ns]
fp_conf  = [r['high_conf_80'] for r in fp]
ns_conf  = [r['high_conf_80'] for r in ns]
fp_max   = [r['plddt_max'] for r in fp]
ns_max   = [r['plddt_max'] for r in ns]

from scipy import stats
u1, p1 = stats.mannwhitneyu(fp_plddt, ns_plddt, alternative='greater')
u2, p2 = stats.mannwhitneyu(fp_conf, ns_conf, alternative='greater')
u3, p3 = stats.mannwhitneyu(fp_max, ns_max, alternative='greater')

print(f'\n{"="*60}')
print(f'  Ablation Foldability Results (ESMFold)')
print(f'{"="*60}')
print(f'  FoldPath-LLM (n={len(fp)}):')
print(f'    Mean pLDDT:     {np.mean(fp_plddt):.1f} +/- {np.std(fp_plddt):.1f}')
print(f'    High conf >80:  {np.mean(fp_conf):.1f}%')
print(f'    Best pLDDT:     {np.max(fp_plddt):.1f}')
print(f'  NoStruct (n={len(ns)}):')
print(f'    Mean pLDDT:     {np.mean(ns_plddt):.1f} +/- {np.std(ns_plddt):.1f}')
print(f'    High conf >80:  {np.mean(ns_conf):.1f}%')
print(f'    Best pLDDT:     {np.max(ns_plddt):.1f}')
print(f'  Statistics:')
print(f'    pLDDT:     U={u1:.0f} p={p1:.4f} {"***" if p1<0.001 else "**" if p1<0.01 else "*" if p1<0.05 else "ns"}')
print(f'    High conf: U={u2:.0f} p={p2:.4f} {"***" if p2<0.001 else "**" if p2<0.01 else "*" if p2<0.05 else "ns"}')
print(f'    Max pLDDT: U={u3:.0f} p={p3:.4f} {"***" if p3<0.001 else "**" if p3<0.01 else "*" if p3<0.05 else "ns"}')
print(f'{"="*60}')

# Save
output = {
    'foldpath': {'plddt': fp_plddt, 'high_conf': fp_conf, 'plddt_max': fp_max, 'n': len(fp)},
    'nostruct': {'plddt': ns_plddt, 'high_conf': ns_conf, 'plddt_max': ns_max, 'n': len(ns)},
    'statistics': {'plddt_u': round(u1,1), 'plddt_p': round(p1,4), 'high_conf_u': round(u2,1), 'high_conf_p': round(p2,4), 'max_plddt_u': round(u3,1), 'max_plddt_p': round(p3,4)}
}
with open(os.path.join(args.input_dir, 'foldability_stats.json'), 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f'\nSaved: {args.input_dir}/foldability_stats.json')
