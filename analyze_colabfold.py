"""
分析 ColabFold 输出 — 提取 pLDDT、二级结构、生成论文图表
用法: python analyze_colabfold.py --colabfold-dir ./colabfold_results
"""
import sys, os, json, argparse, re
import numpy as np
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--colabfold-dir', type=str, default='.')
parser.add_argument('--output', type=str, default='colabfold_analysis.json')
args = parser.parse_args()

# ── 找到所有 .pdb 文件 ──
pdb_files = []
for root, dirs, files in os.walk(args.colabfold_dir):
    for f in files:
        if f.endswith('.pdb') and ('relaxed' in f or 'unrelaxed' in f):
            pdb_files.append(os.path.join(root, f))

if not pdb_files:
    pdb_files = []
    for root, dirs, files in os.walk(args.colabfold_dir):
        for f in files:
            if f.endswith('.pdb'):
                pdb_files.append(os.path.join(root, f))

print(f'Found {len(pdb_files)} PDB files')
if len(pdb_files) == 0:
    print('ERROR: No PDB files found.')
    print('Please download ColabFold results and extract to current directory.')
    sys.exit(1)

results = {'sequences': [], 'summary': {}}

for pdb_path in sorted(pdb_files):
    seq_id = os.path.basename(pdb_path).replace('.pdb', '').replace('_relaxed', '').replace('_unrelaxed', '')
    print(f'\nAnalyzing: {seq_id}')

    with open(pdb_path, 'r') as f:
        pdb_text = f.read()

    # ── 提取 pLDDT (存于 B-factor 列) ──
    plddt_values = []
    ca_residues = []
    for line in pdb_text.split('\n'):
        if line.startswith('ATOM') and line[13:15] == 'CA':
            try:
                bfactor = float(line[60:66])
                res_num = int(line[22:26])
                plddt_values.append(bfactor)
                ca_residues.append(res_num)
            except ValueError:
                pass

    if not plddt_values:
        print(f'  WARNING: No CA atoms with pLDDT found in {seq_id}')
        continue

    plddt_arr = np.array(plddt_values)
    mean_plddt = np.mean(plddt_arr)
    high_conf = np.mean(plddt_arr > 90) * 100
    med_conf = np.mean((plddt_arr >= 70) & (plddt_arr <= 90)) * 100
    low_conf = np.mean(plddt_arr < 70) * 100

    # ── 提取二级结构 (从 HELIX/SHEET 记录) ──
    helix_count = len([l for l in pdb_text.split('\n') if l.startswith('HELIX')])
    sheet_count = len([l for l in pdb_text.split('\n') if l.startswith('SHEET')])

    # ── 提取序列 ──
    seq = ''
    aa3to1 = {
        'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
        'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
        'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'
    }
    prev_res = -999
    for line in pdb_text.split('\n'):
        if line.startswith('ATOM') and line[13:15] == 'CA':
            res_num = int(line[22:26])
            if res_num != prev_res:
                aa3 = line[17:20].strip()
                seq += aa3to1.get(aa3, 'X')
                prev_res = res_num

    seq_len = len(seq)

    # ── TM-score 估算 (用 pLDDT 加权) ──
    # 粗略估计: >90 高置信 = 接近天然结构
    tm_proxy = np.mean(plddt_arr / 100.0)  # 0-1 scale proxy

    r = {
        'id': seq_id,
        'pdb_file': os.path.basename(pdb_path),
        'length': seq_len,
        'mean_plddt': round(mean_plddt, 1),
        'plddt_std': round(np.std(plddt_arr), 1),
        'high_conf_pct': round(high_conf, 1),
        'med_conf_pct': round(med_conf, 1),
        'low_conf_pct': round(low_conf, 1),
        'helix_count': helix_count,
        'sheet_count': sheet_count,
        'helix_pct': round(helix_count * 15 / max(seq_len, 1) * 100, 1) if seq_len > 0 else 0,
        'tm_score_proxy': round(tm_proxy, 3),
        'seq_preview': seq[:60] + ('...' if len(seq) > 60 else ''),
    }

    print(f'  Length: {seq_len}')
    print(f'  Mean pLDDT: {mean_plddt:.1f}')
    print(f'  High conf (>90): {high_conf:.1f}%')
    print(f'  Helices: {helix_count}, Sheets: {sheet_count}')
    print(f'  Sequence: {r["seq_preview"]}')

    results['sequences'].append(r)

# ── Summary ──
if not results['sequences']:
    print('\nNo valid structures found.')
    sys.exit(1)

sl = results['sequences']
plddts = [s['mean_plddt'] for s in sl]
lengths = [s['length'] for s in sl]
helices = [s['helix_count'] for s in sl]

print(f'\n{"="*60}')
print(f'  AlphaFold2 Structure Validation ({len(sl)} sequences)')
print(f'{"="*60}')
print(f'  Mean pLDDT:     {np.mean(plddts):.1f} (+/- {np.std(plddts):.1f})')
print(f'  Best pLDDT:     {np.max(plddts):.1f}')
print(f'  Worst pLDDT:    {np.min(plddts):.1f}')
print(f'  Mean helices:   {np.mean(helices):.1f}')
print(f'  Mean length:    {np.mean(lengths):.0f}')
print(f'  pLDDT > 90:     {sum(1 for p in plddts if p > 90)}/{len(sl)} sequences')
print(f'  pLDDT > 80:     {sum(1 for p in plddts if p > 80)}/{len(sl)} sequences')
print(f'  pLDDT > 70:     {sum(1 for p in plddts if p > 70)}/{len(sl)} sequences')
print(f'{"="*60}')

# Check for transmembrane pattern
tm_like = 0
for s in sl:
    if s['helix_count'] >= 5 and s['mean_plddt'] > 70:
        tm_like += 1
print(f'  TM-like (>=5 helices, pLDDT>70): {tm_like}/{len(sl)}')

results['summary'] = {
    'num_sequences': len(sl),
    'mean_plddt': round(np.mean(plddts), 1),
    'std_plddt': round(np.std(plddts), 1),
    'best_plddt': round(np.max(plddts), 1),
    'worst_plddt': round(np.min(plddts), 1),
    'plddt_gt_90': f'{sum(1 for p in plddts if p > 90)}/{len(sl)}',
    'plddt_gt_80': f'{sum(1 for p in plddts if p > 80)}/{len(sl)}',
    'plddt_gt_70': f'{sum(1 for p in plddts if p > 70)}/{len(sl)}',
    'mean_helices': round(np.mean(helices), 1),
    'tm_like_count': f'{tm_like}/{len(sl)}',
}

with open(args.output, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'\nSaved: {args.output}')

# ── 为论文生成表格 ──
print(f'\n论文表格 (for LaTeX):')
print(r'\begin{table}[H]')
print(r'\centering')
print(r'\caption{AlphaFold2 structure validation of FoldPath-LLM generated sequences.}')
print(r'\label{tab:af2}')
print(r'\begin{tabular}{c c c c c}')
print(r'\toprule')
print(r'Seq & Length & Mean pLDDT & High Conf. (\%) & Predicted Helices \\')
print(r'\midrule')
for i, s in enumerate(sl[:6]):  # Top 6 for the paper
    print(f'  #{i+1} & {s["length"]} & {s["mean_plddt"]:.1f} & {s["high_conf_pct"]:.0f}\\% & {s["helix_count"]} \\\\')
print(r'\bottomrule')
print(r'\end{tabular}')
print(r'\end{table}')
