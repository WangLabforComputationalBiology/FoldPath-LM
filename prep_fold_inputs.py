"""
准备 ColabFold 输入: 过滤短序列 → 输出就绪 FASTA
用法: python prep_fold_inputs.py --input-dir colabfold_batch --max-length 200
"""
import os, sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, default='colabfold_batch')
parser.add_argument('--max-length', type=int, default=200)
args = parser.parse_args()

for src_name in ['foldpath_25.fasta', 'nostruct_25.fasta']:
    src = os.path.join(args.input_dir, src_name)
    if not os.path.exists(src):
        print(f'SKIP: {src}')
        continue

    # Read all
    seqs = []
    with open(src) as f:
        header = ''; seq = ''
        for line in f:
            if line.startswith('>'):
                if header: seqs.append((header, seq))
                header = line.strip(); seq = ''
            else: seq += line.strip()
        if header: seqs.append((header, seq))

    # Filter
    short = [(h, s) for h, s in seqs if len(s) <= args.max_length]
    long  = [(h, s) for h, s in seqs if len(s) > args.max_length]
    print(f'{src_name}: {len(short)} <= {args.max_length}aa, {len(long)} > {args.max_length}aa')

    # Save filtered
    dst = os.path.join(args.input_dir, src_name.replace('.fasta', f'_{args.max_length}.fasta'))
    with open(dst, 'w') as f:
        for h, s in short:
            f.write(f'{h}\n{s}\n')
    print(f'  Saved: {dst} ({len(short)} seqs)')

    # If too few short seqs, include some of the longest short ones too
    if len(short) < 5:
        print(f'  WARNING: Only {len(short)} short seqs. Consider increasing --max-length.')
        # Fallback: use whatever we have
        all_sorted = sorted(seqs, key=lambda x: len(x[1]))
        fallback = all_sorted[:10]
        dst2 = os.path.join(args.input_dir, src_name.replace('.fasta', '_top10.fasta'))
        with open(dst2, 'w') as f:
            for h, s in fallback:
                f.write(f'{h}\n{s}\n')
        print(f'  Fallback: {dst2} (10 shortest)')

print('\nDone. Upload the *_200.fasta (or *_top10.fasta) files to ColabFold.')
print('https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb')
