"""
ESMFold 本地结构预测 — 最适合 de novo 序列
用法: python esmfold_local.py --fasta colabfold_input.fasta --num-seqs 10
"""
import torch, sys, os, json, argparse, time, re
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--fasta', type=str, default='colabfold_input.fasta')
parser.add_argument('--num-seqs', type=int, default=10)
parser.add_argument('--output', type=str, default='esmfold_results')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    free, total = torch.cuda.mem_get_info()
    print(f'VRAM: {free/1e9:.1f}/{total/1e9:.1f} GB free')

# Load sequences
seqs = []
with open(args.fasta, 'r') as f:
    cur_id, cur_seq = None, []
    for line in f:
        if line.startswith('>'):
            if cur_id:
                seqs.append({'id': cur_id, 'seq': ''.join(cur_seq)})
            cur_id = line.strip()[1:]
            cur_seq = []
        else:
            cur_seq.append(line.strip())
    if cur_id:
        seqs.append({'id': cur_id, 'seq': ''.join(cur_seq)})

seqs = seqs[:args.num_seqs]
print(f'Loaded {len(seqs)} sequences')
for s in seqs:
    print(f'  {s["id"]}: length={len(s["seq"])}')

# Load ESMFold
print('\nLoading ESMFold v1...')
import esm
model = esm.pretrained.esmfold_v1()
model = model.eval().to(device)
print('ESMFold ready')

# Predict structures
os.makedirs(args.output, exist_ok=True)
results = {'sequences': [], 'summary': {}}
aa3to1 = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
          'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
          'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}

for i, s in enumerate(seqs):
    seq = s['seq']
    L = len(seq)
    print(f'\n[{i+1}/{len(seqs)}] {s["id"]} (length={L})')

    t0 = time.time()
    with torch.no_grad():
        try:
            output = model.infer_pdb(seq)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                print(f'  OOM, retrying...')
                output = model.infer_pdb(seq)
            else:
                raise

    elapsed = time.time() - t0
    print(f'  Predicted in {elapsed:.1f}s')

    # Save PDB
    pdb_path = os.path.join(args.output, f'{s["id"]}.pdb')
    with open(pdb_path, 'w') as f:
        f.write(output)
    print(f'  Saved: {pdb_path}')

    # Parse pLDDT
    plddt_vals = []
    seq_from_pdb = ''
    prev_res = -999
    for line in output.split('\n'):
        if line.startswith('ATOM') and line[13:15] == 'CA':
            try:
                plddt_vals.append(float(line[60:66]))
                res_num = int(line[22:26])
                if res_num != prev_res:
                    aa3 = line[17:20].strip()
                    seq_from_pdb += aa3to1.get(aa3, 'X')
                    prev_res = res_num
            except ValueError:
                pass

    plddt_arr = np.array(plddt_vals)
    mean_plddt = np.mean(plddt_arr)
    high_conf = np.mean(plddt_arr > 80) * 100
    med_conf = np.mean((plddt_arr >= 60) & (plddt_arr <= 80)) * 100
    low_conf = np.mean(plddt_arr < 60) * 100

    # Secondary structure
    helix_count = len([l for l in output.split('\n') if l.startswith('HELIX')])
    sheet_count = len([l for l in output.split('\n') if l.startswith('SHEET')])

    # Transmembrane pattern from pLDDT
    window = 20
    plddt_means = []
    for j in range(0, L - window, window):
        plddt_means.append(np.mean(plddt_arr[j:j+window]))
    tm_likeness = float(np.std(plddt_means)) if len(plddt_means) >= 2 else 0

    r = {
        'id': s['id'], 'length': L,
        'mean_plddt': round(mean_plddt, 1),
        'plddt_std': round(float(np.std(plddt_arr)), 1),
        'high_conf_80_pct': round(high_conf, 1),
        'med_conf_pct': round(med_conf, 1),
        'low_conf_pct': round(low_conf, 1),
        'predicted_helices': helix_count,
        'predicted_sheets': sheet_count,
        'tm_likeness': round(tm_likeness, 3),
        'seq_preview': seq[:60] + ('...' if L > 60 else ''),
        'pdb_file': f'{s["id"]}.pdb',
    }

    print(f'  pLDDT: mean={mean_plddt:.1f} high(>80)={high_conf:.1f}%')
    print(f'  Helices: {helix_count}, Sheets: {sheet_count}')
    print(f'  TM likeness: {tm_likeness:.3f}')

    results['sequences'].append(r)

    # Free memory
    del output
    torch.cuda.empty_cache()

# Summary
sl = results['sequences']
plddts = [s['mean_plddt'] for s in sl]
helices = [s['predicted_helices'] for s in sl]
tm = [s['tm_likeness'] for s in sl]

print(f'\n{"="*60}')
print(f'  ESMFold Structure Validation ({len(sl)} sequences)')
print(f'{"="*60}')
print(f'  Mean pLDDT:     {np.mean(plddts):.1f} (+/- {np.std(plddts):.1f})')
print(f'  Best pLDDT:     {np.max(plddts):.1f} (id={sl[np.argmax(plddts)]["id"]})')
print(f'  Worst pLDDT:    {np.min(plddts):.1f}')
print(f'  pLDDT > 80:     {sum(1 for p in plddts if p > 80)}/{len(sl)}')
print(f'  pLDDT > 70:     {sum(1 for p in plddts if p > 70)}/{len(sl)}')
print(f'  Mean helices:   {np.mean(helices):.1f}')
print(f'  TM-like:        {sum(1 for t in tm if t > 5)}/{len(sl)}')
print(f'{"="*60}')

results['summary'] = {
    'num_sequences': len(sl),
    'mean_plddt': round(np.mean(plddts), 1),
    'best_plddt': round(np.max(plddts), 1),
    'worst_plddt': round(np.min(plddts), 1),
    'plddt_gt_80': f'{sum(1 for p in plddts if p > 80)}/{len(sl)}',
    'plddt_gt_70': f'{sum(1 for p in plddts if p > 70)}/{len(sl)}',
    'mean_helices': round(np.mean(helices), 1),
}

with open(f'{args.output}_summary.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f'\nSaved: {args.output}/ (PDB files) + {args.output}_summary.json')

# Print LaTeX table
print(f'\n论文表格:')
print(r'\begin{table}[H]')
print(r'\centering')
print(r'\caption{ESMFold structural validation of FoldPath-LLM generated sequences.}')
print(r'\label{tab:esmfold}')
print(r'\begin{tabular}{c c c c c c}')
print(r'\toprule')
print(r'Seq & Length & pLDDT & High Conf.\% & Helices & TM Pattern \\')
print(r'\midrule')
for s in sl[:6]:
    print(f'  {s["id"][:15]} & {s["length"]} & {s["mean_plddt"]:.1f} & {s["high_conf_80_pct"]:.0f}\\% & {s["predicted_helices"]} & {s["tm_likeness"]:.3f} \\\\')
print(r'\bottomrule')
print(r'\end{tabular}')
print(r'\end{table}')
