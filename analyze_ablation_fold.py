"""
分析消融实验的 ColabFold 结果 — 统计证据
用法: python analyze_ablation_fold.py --colabfold-dir ./colabfold_batch
"""
import sys, os, json, argparse, re, numpy as np
from scipy import stats
parser = argparse.ArgumentParser()
parser.add_argument('--colabfold-dir', type=str, default='colabfold_batch')
parser.add_argument('--output', type=str, default='ablation_foldability.json')
args = parser.parse_args()

aa3to1 = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
          'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
          'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}


def parse_colabfold(dir_path, prefix):
    """Parse all ColabFold results for a given prefix (FP_ or NS_)"""
    results = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if f.endswith('.pdb') and f.startswith(prefix):
                pdb_path = os.path.join(root, f)
                try:
                    r = parse_pdb(pdb_path, f)
                    if r: results.append(r)
                except Exception as e:
                    print(f'  WARN: {f}: {e}')
    return results


def parse_pdb(pdb_path, name):
    with open(pdb_path, 'r') as f:
        pdb_text = f.read()

    plddt_vals = []
    seq = ''; prev_res = -999
    for line in pdb_text.split('\n'):
        if line.startswith('ATOM') and line[13:15] == 'CA':
            try:
                plddt_vals.append(float(line[60:66]))
                res_num = int(line[22:26])
                if res_num != prev_res:
                    aa3 = line[17:20].strip()
                    seq += aa3to1.get(aa3, 'X')
                    prev_res = res_num
            except ValueError: pass

    if not plddt_vals: return None
    plddt_arr = np.array(plddt_vals)

    # Extract pTM from filename or remarks
    ptm = 0.0
    for line in pdb_text.split('\n'):
        if 'pTM' in line or 'ptm' in line.lower():
            m = re.search(r'(\d+\.?\d*)', line)
            if m: ptm = float(m.group(1))

    # Extract from ColabFold ranking files
    rank_files = [f for f in os.listdir(os.path.dirname(pdb_path))
                  if 'rank' in f and f.endswith('.json')]
    for rf in rank_files:
        try:
            with open(os.path.join(os.path.dirname(pdb_path), rf)) as f:
                rank_data = json.load(f)
            if 'plddt' in str(rank_data).lower() or 'ptm' in str(rank_data).lower():
                pass
        except: pass

    return {
        'name': name, 'length': len(seq),
        'mean_plddt': round(float(np.mean(plddt_arr)), 1),
        'plddt_std': round(float(np.std(plddt_arr)), 1),
        'high_conf_pct': round(float(np.mean(plddt_arr > 80)) * 100, 1),
        'ptm': ptm if ptm > 0 else None,
        'plddt_max': round(float(np.max(plddt_arr)), 1),
        'plddt_min': round(float(np.min(plddt_arr)), 1),
    }


# ════════ Parse ════════
fp_results = parse_colabfold(args.colabfold_dir, 'FP_')
ns_results = parse_colabfold(args.colabfold_dir, 'NS_')

print(f'FoldPathLLM: {len(fp_results)} sequences')
print(f'NoStruct:    {len(ns_results)} sequences')

if len(fp_results) < 2 or len(ns_results) < 2:
    print('ERROR: Need at least 2 sequences from each model.')
    print('Please run ColabFold on both FASTA files first.')
    sys.exit(1)

# ════════ Statistics ════════
fp_plddt = [r['mean_plddt'] for r in fp_results]
ns_plddt = [r['mean_plddt'] for r in ns_results]
fp_ptm   = [r.get('ptm', 0) or 0 for r in fp_results]
ns_ptm   = [r.get('ptm', 0) or 0 for r in ns_results]
fp_conf  = [r['high_conf_pct'] for r in fp_results]
ns_conf  = [r['high_conf_pct'] for r in ns_results]

# Mann-Whitney U test (non-parametric, robust)
u_plddt, p_plddt = stats.mannwhitneyu(fp_plddt, ns_plddt, alternative='greater')
u_conf, p_conf = stats.mannwhitneyu(fp_conf, ns_conf, alternative='greater')

# pTM > 0.5 proportion
fp_foldable = sum(1 for p in fp_ptm if p > 0.5)
ns_foldable = sum(1 for p in ns_ptm if p > 0.5)

print(f'\n{"="*60}')
print(f'  Ablation Foldability Analysis')
print(f'{"="*60}')
print(f'\n  FoldPath-LLM (n={len(fp_results)}):')
print(f'    Mean pLDDT:     {np.mean(fp_plddt):.1f} +/- {np.std(fp_plddt):.1f}')
print(f'    Mean pTM:        {np.mean(fp_ptm):.3f} +/- {np.std(fp_ptm):.3f}')
print(f'    High conf (>80): {np.mean(fp_conf):.1f}%')
print(f'    pTM > 0.5:       {fp_foldable}/{len(fp_results)} ({fp_foldable/len(fp_results)*100:.0f}%)')

print(f'\n  No Structure Track (n={len(ns_results)}):')
print(f'    Mean pLDDT:     {np.mean(ns_plddt):.1f} +/- {np.std(ns_plddt):.1f}')
print(f'    Mean pTM:        {np.mean(ns_ptm):.3f} +/- {np.std(ns_ptm):.3f}')
print(f'    High conf (>80): {np.mean(ns_conf):.1f}%')
print(f'    pTM > 0.5:       {ns_foldable}/{len(ns_results)} ({ns_foldable/len(ns_results)*100:.0f}%)')

print(f'\n  Statistical Tests:')
print(f'    pLDDT:          U={u_plddt:.0f}, p={p_plddt:.4f} {"***" if p_plddt<0.001 else "**" if p_plddt<0.01 else "*" if p_plddt<0.05 else "ns"}')
print(f'    High conf (>80): U={u_conf:.0f}, p={p_conf:.4f} {"***" if p_conf<0.001 else "**" if p_conf<0.01 else "*" if p_conf<0.05 else "ns"}')
print(f'{"="*60}')

# ════════ Save ════════
output = {
    'foldpath': {'plddt': fp_plddt, 'ptm': fp_ptm, 'high_conf': fp_conf,
                 'n': len(fp_results), 'foldable': fp_foldable},
    'nostruct': {'plddt': ns_plddt, 'ptm': ns_ptm, 'high_conf': ns_conf,
                 'n': len(ns_results), 'foldable': ns_foldable},
    'statistics': {
        'plddt_mannwhitney_u': round(u_plddt, 1),
        'plddt_p_value': round(p_plddt, 4),
        'high_conf_mannwhitney_u': round(u_conf, 1),
        'high_conf_p_value': round(p_conf, 4),
        'foldpath_foldable_pct': round(fp_foldable / len(fp_results) * 100, 1),
        'nostruct_foldable_pct': round(ns_foldable / len(ns_results) * 100, 1),
    }
}
with open(args.output, 'w') as f:
    json.dump(output, f, indent=2)
print(f'\nSaved: {args.output}')

# ════════ LaTeX table ════════
print(f'\nLaTeX table for paper:')
print(r'\begin{table}[H]')
print(r'\centering')
print(r'\caption{Foldability comparison: FoldPath-LLM vs.\ No Structure Track.}')
print(r'\label{tab:fold_comp}')
print(r'\begin{tabular}{l c c c}')
print(r'\toprule')
print(r'Metric & FoldPath-LLM & No Struct Track & $p$-value \\')
print(r'\midrule')
print(f'  Mean pLDDT & {np.mean(fp_plddt):.1f}$\\pm${np.std(fp_plddt):.1f} & {np.mean(ns_plddt):.1f}$\\pm${np.std(ns_plddt):.1f} & {p_plddt:.4f} \\\\')
print(f'  High confidence (>80) \% & {np.mean(fp_conf):.1f}\\% & {np.mean(ns_conf):.1f}\\% & {p_conf:.4f} \\\\')
print(f'  pTM $>$ 0.5 count & {fp_foldable}/{len(fp_results)} ({fp_foldable/len(fp_results)*100:.0f}\\%) & {ns_foldable}/{len(ns_results)} ({ns_foldable/len(ns_results)*100:.0f}\\%) & -- \\\\')
print(r'\bottomrule')
print(r'\end{tabular}')
print(r'\end{table}')
