"""
消融折叠能力对比实验 — 完整流水线

Step 1 (服务器): python experiment_ablation_fold.py --mode generate
  → 生成 FoldPathLLM + NoStruct 各 30 条, 筛选最优 15 条
  → 输出: experiment/foldpath_15.fasta, experiment/nostruct_15.fasta

Step 2 (ColabFold 网页): 上传两个 FASTA → Run all → 下载结果 ZIP
  → 解压到 experiment/ 目录
  → https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb

Step 3 (服务器): python experiment_ablation_fold.py --mode analyze
  → 解析 PDB, 提取 pLDDT/pTM, Mann-Whitney U 检验
  → 输出: experiment/stats.json + LaTeX 表格
"""
import torch, sys, os, json, argparse, time, re, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['generate', 'analyze'])
parser.add_argument('--foldpath-ckpt', type=str, default='esmpro/foldpath_best_eopch10.pt')
parser.add_argument('--exp-dir', type=str, default='experiment')
parser.add_argument('--num-gen', type=int, default=30)
parser.add_argument('--num-best', type=int, default=15)
parser.add_argument('--temperature', type=float, default=0.6)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs(args.exp_dir, exist_ok=True)

# ════════════════════════════════════
# MODE: generate
# ════════════════════════════════════
if args.mode == 'generate':
    from config import ModelConfig, IDX_TO_AA, BOS_IDX, EOS_IDX, PAD_IDX, TOTAL_VOCAB, AA_TO_IDX
    from model import FoldPathLLM
    from rita_encoder import create_rita_encoder
    from physicochemical import PHYSICO_MATRIX

    print('Loading RITA_m...')
    rita_enc = create_rita_encoder(model_name='RITA_m', device=device, freeze=True, local_dir='pretrained')

    def generate_seqs(model, n, temp):
        seqs = []
        with torch.no_grad():
            for i in range(n):
                batch = torch.ones(1, 1, dtype=torch.long, device=device) * BOS_IDX
                generated = []
                for step in range(256):
                    prefix_ids = [idx for idx in batch[0].tolist() if idx not in (PAD_IDX, BOS_IDX)]
                    prefix_seq = ''.join([IDX_TO_AA.get(idx, 'X') for idx in prefix_ids])
                    seqs_arg = [prefix_seq] if prefix_seq else ['M']
                    logits, _, _, _ = model.forward(batch, sequences=seqs_arg, use_bias=False)
                    next_logits = logits[0, -1, :] / temp
                    for idx in range(20, TOTAL_VOCAB):
                        if idx != EOS_IDX: next_logits[idx] = float('-inf')
                    topk, _ = torch.topk(next_logits, min(50, next_logits.size(-1)))
                    next_logits[next_logits < topk[-1]] = float('-inf')
                    probs = torch.softmax(next_logits, dim=-1)
                    if probs.sum() == 0 or torch.isnan(probs).any(): break
                    next_aa = torch.multinomial(probs, 1)
                    if next_aa.item() == EOS_IDX: break
                    generated.append(next_aa.item())
                    batch = torch.cat([batch, next_aa.unsqueeze(0)], dim=1)
                seq = ''.join([IDX_TO_AA.get(a, 'X') for a in generated])
                if len(seq) >= 30: seqs.append(seq)
        return seqs

    def score_seq(seq):
        L = len(seq)
        hydro = np.array([PHYSICO_MATRIX[AA_TO_IDX.get(aa,0),0] for aa in seq])
        charge = np.array([PHYSICO_MATRIX[AA_TO_IDX.get(aa,0),2] for aa in seq])
        cr = seq.count('C') / L
        cs = max(0, 1 - cr * 25)
        hp = np.mean(hydro > 0.5)
        hs = max(0, 1 - abs(0.38 - hp) * 3)
        pos = np.mean(charge > 0.5); neg = np.mean(charge < -0.5)
        chs = max(0, 1 - abs(pos - neg) * 5)
        mr = 1; crun = 1
        for i in range(1, L):
            if seq[i] == seq[i-1]: crun += 1; mr = max(mr, crun)
            else: crun = 1
        rs = max(0, 1 - (mr - 2) * 0.3)
        ls = min(1.0, L / 160) if L < 160 else max(0, 1 - (L - 200) / 200)
        return cs*0.25 + hs*0.20 + chs*0.15 + rs*0.15 + ls*0.15

    # Model 1: FoldPath-LLM
    print(f'\n[1/2] FoldPath-LLM (T={args.temperature})')
    ckpt = torch.load(args.foldpath_ckpt, map_location=device, weights_only=False)
    model1 = FoldPathLLM(ModelConfig(), esm_encoder=rita_enc)
    model1.load_state_dict(ckpt['model_state_dict'], strict=False)
    model1 = model1.to(device).eval()

    seqs1 = generate_seqs(model1, n=args.num_gen, temp=args.temperature)
    scored1 = [(score_seq(s), s) for s in seqs1]
    scored1.sort(key=lambda x: x[0], reverse=True)
    best1 = [s for _, s in scored1[:args.num_best]]
    lens1 = [len(s) for s in best1]
    print(f'  Generated: {len(seqs1)}, Selected: {len(best1)}')
    print(f'  Lengths: {min(lens1)}-{max(lens1)} (mean {np.mean(lens1):.0f})')

    fasta1 = os.path.join(args.exp_dir, 'foldpath_15.fasta')
    with open(fasta1, 'w') as f:
        for i, s in enumerate(best1):
            f.write(f'>FP_{i+1}_len{len(s)}\n{s}\n')
    print(f'  Saved: {fasta1}')

    # Model 2: NoStruct (zero parameters)
    print(f'\n[2/2] NoStruct (T={args.temperature})')
    model2 = FoldPathLLM(ModelConfig(), esm_encoder=rita_enc)
    model2.load_state_dict(ckpt['model_state_dict'], strict=False)
    for name, param in model2.named_parameters():
        if 'structure_track' in name or 'chem_bias' in name:
            param.data.zero_()
    model2 = model2.to(device).eval()

    seqs2 = generate_seqs(model2, n=args.num_gen, temp=args.temperature)
    scored2 = [(score_seq(s), s) for s in seqs2]
    scored2.sort(key=lambda x: x[0], reverse=True)
    best2 = [s for _, s in scored2[:args.num_best]]
    lens2 = [len(s) for s in best2]
    print(f'  Generated: {len(seqs2)}, Selected: {len(best2)}')
    print(f'  Lengths: {min(lens2)}-{max(lens2)} (mean {np.mean(lens2):.0f})')

    fasta2 = os.path.join(args.exp_dir, 'nostruct_15.fasta')
    with open(fasta2, 'w') as f:
        for i, s in enumerate(best2):
            f.write(f'>NS_{i+1}_len{len(s)}\n{s}\n')
    print(f'  Saved: {fasta2}')

    print(f'\n{"="*60}')
    print(f'  NEXT: Upload to ColabFold')
    print(f'{"="*60}')
    print(f'''
  1. 浏览器: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb
  2. 左侧文件栏 → 上传 {fasta1} 和 {fasta2}
  3. 修改 query_sequence 参数指向你的 FASTA 文件名
  4. Runtime → Run all (~15-20 min for 30 sequences total)
  5. 下载结果 ZIP → 解压到 {args.exp_dir}/
  6. 运行: python experiment_ablation_fold.py --mode analyze
''')

# ════════════════════════════════════
# MODE: analyze
# ════════════════════════════════════
elif args.mode == 'analyze':
    aa3to1 = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
              'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
              'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}

    def parse_results(prefix):
        """Walk experiment dir, find all PDB files starting with prefix, extract pLDDT"""
        results = []
        for root, dirs, files in os.walk(args.exp_dir):
            for f in sorted(files):
                if not f.endswith('.pdb'): continue
                if not f.startswith(prefix): continue
                # Only use rank_001 (best model)
                if 'rank_001' not in f and 'relaxed' not in f: continue

                pdb_path = os.path.join(root, f)
                with open(pdb_path, 'r') as fh:
                    pdb_text = fh.read()

                plddt_vals = []
                for line in pdb_text.split('\n'):
                    if line.startswith('ATOM') and line[13:15] == 'CA':
                        try: plddt_vals.append(float(line[60:66]))
                        except ValueError: pass

                if not plddt_vals: continue
                plddt_arr = np.array(plddt_vals)

                # Get seq length from PDB
                seq = ''; prev = -999
                for line in pdb_text.split('\n'):
                    if line.startswith('ATOM') and line[13:15] == 'CA':
                        rn = int(line[22:26])
                        if rn != prev:
                            seq += aa3to1.get(line[17:20].strip(), 'X')
                            prev = rn

                results.append({
                    'file': f, 'length': len(seq),
                    'mean_plddt': round(float(np.mean(plddt_arr)), 1),
                    'plddt_max': round(float(np.max(plddt_arr)), 1),
                    'high_conf_pct': round(float(np.mean(plddt_arr > 80)) * 100, 1),
                })
        return results

    fp = parse_results('FP_')
    ns = parse_results('NS_')
    print(f'FoldPathLLM: {len(fp)} structures')
    print(f'NoStruct:    {len(ns)} structures')

    # If also have pTM from ColabFold's ranking JSON
    # Try to extract from rank JSON files
    ptm_fp = []; ptm_ns = []
    for root, dirs, files in os.walk(args.exp_dir):
        for f in files:
            if f.endswith('.json') and 'rank' in f:
                try:
                    with open(os.path.join(root, f)) as fh:
                        data = json.load(fh)
                    if 'pTM' in str(data) or 'ptm' in str(data).lower():
                        # Crude extraction
                        ptm_vals = re.findall(r'"pTM"?\s*:\s*([\d.]+)', str(data))
                        if ptm_vals:
                            prefix = 'FP' if f.startswith('FP') else 'NS'
                            for pv in ptm_vals[:1]:
                                v = float(pv)
                                if prefix == 'FP': ptm_fp.append(v)
                                else: ptm_ns.append(v)
                except: pass

    if len(fp) < 2 or len(ns) < 2:
        print('ERROR: Need >=2 structures per model. Run ColabFold first.')
        sys.exit(1)

    fp_plddt = [r['mean_plddt'] for r in fp]
    ns_plddt = [r['mean_plddt'] for r in ns]
    fp_conf = [r['high_conf_pct'] for r in fp]
    ns_conf = [r['high_conf_pct'] for r in ns]

    from scipy import stats
    u1, p1 = stats.mannwhitneyu(fp_plddt, ns_plddt, alternative='greater')
    u2, p2 = stats.mannwhitneyu(fp_conf, ns_conf, alternative='greater')

    print(f'\n{"="*60}')
    print(f'  Ablation Foldability Comparison')
    print(f'{"="*60}')
    print(f'  FoldPath-LLM: pLDDT={np.mean(fp_plddt):.1f}+/-{np.std(fp_plddt):.1f}')
    print(f'  NoStruct:     pLDDT={np.mean(ns_plddt):.1f}+/-{np.std(ns_plddt):.1f}')
    print(f'  pLDDT p-value: {p1:.4f} {"***" if p1<0.01 else "*" if p1<0.05 else "ns"}')
    print(f'  High conf p:   {p2:.4f} {"***" if p2<0.01 else "*" if p2<0.05 else "ns"}')

    # LaTeX table
    print(f'\n  LaTeX Table:')
    print(r'\begin{table}[H]')
    print(r'\centering')
    print(r'\caption{ESMFold foldability comparison: FoldPath-LLM vs.\ No Structure Track.}')
    print(r'\begin{tabular}{l c c c}')
    print(r'\toprule')
    print(r'Metric & FoldPath-LLM & NoStruct & $p$ \\')
    print(r'\midrule')
    print(f'  pLDDT & {np.mean(fp_plddt):.1f}$\\pm${np.std(fp_plddt):.1f} & {np.mean(ns_plddt):.1f}$\\pm${np.std(ns_plddt):.1f} & {p1:.3f} \\\\')
    print(f'  High conf (>80)\\% & {np.mean(fp_conf):.1f} & {np.mean(ns_conf):.1f} & {p2:.3f} \\\\')
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')

    # Save
    result = {
        'foldpath': {'plddt': fp_plddt, 'high_conf': fp_conf, 'ptm': ptm_fp, 'n': len(fp)},
        'nostruct': {'plddt': ns_plddt, 'high_conf': ns_conf, 'ptm': ptm_ns, 'n': len(ns)},
        'statistics': {'plddt_p': round(p1, 4), 'high_conf_p': round(p2, 4)},
    }
    with open(os.path.join(args.exp_dir, 'stats.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\nSaved: {args.exp_dir}/stats.json')
