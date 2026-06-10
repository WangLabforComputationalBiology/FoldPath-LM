"""
ESM-2 独立打分: 用预训练 ESM-2 (MLM) 对生成序列评估伪对数似然 (PLL) 和困惑度 (PPL)

方法: Pseudo-Log-Likelihood
  对每个位置 i, mask 掉 token_i, 用 ESM-2 预测该位置, 取真实 token 的 log 概率
  PLL = (1/L) * sum_i log P(token_i | masked_sequence)
  PPL = exp(-PLL)

用法:
  python esm2_score.py \
    --foldpath-seqs foldpath_50.txt \
    --rita-seqs rita_50.txt \
    --esm-model facebook/esm2_t6_8M_UR50D \
    --output esm2_scores.json

如果本地有模型:
  --esm-local pretrained/esm2_t6_8M_UR50D
"""

import torch
import numpy as np
import json
import argparse
import os
import sys
import time


def load_sequences(path):
    """从 FASTA 或纯文本文件加载序列"""
    seqs = []
    if path.endswith(('.fasta', '.fa')):
        with open(path) as f:
            seq = ''
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if seq:
                        seqs.append(seq)
                    seq = ''
                else:
                    seq += line
            if seq:
                seqs.append(seq)
    else:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('>'):
                    clean = ''.join(c for c in line if c in 'ACDEFGHIKLMNPQRSTVWY')
                    if clean:
                        seqs.append(clean)
    return seqs


def pll_score(model, tokenizer, sequences, device, batch_size=4, max_len=512):
    """
    计算 Pseudo-Log-Likelihood (PLL)
    对每个位置 mask 后预测, 取真实 token 的 log 概率
    返回每条序列的 PLL (per-residue avg) 和总 PLL
    """
    model.eval()
    results = []

    for start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[start:start + batch_size]
        batch_pll = []

        # 对每条序列单独处理 (长度可能不同)
        for seq in batch_seqs:
            if len(seq) < 5:
                batch_pll.append({'pll': 0.0, 'pll_per_residue': 0.0, 'length': len(seq)})
                continue

            # 截断过长序列
            trunc_seq = seq[:max_len]
            L = len(trunc_seq)

            # ESM-2 tokenizer: 空格分隔的氨基酸
            tokens = tokenizer(" ".join(list(trunc_seq)), return_tensors="pt",
                               truncation=True, max_length=max_len + 2)
            input_ids = tokens["input_ids"].to(device)

            # input_ids: [CLS, A1, A2, ..., AL, EOS], 长度 L+2
            # 氨基酸位置: 1..L
            total_log_prob = 0.0
            valid_positions = 0

            with torch.no_grad():
                for i in range(1, L + 1):
                    # 保存原始 token
                    original_token = input_ids[0, i].item()

                    # Mask 该位置
                    masked_ids = input_ids.clone()
                    masked_ids[0, i] = tokenizer.mask_token_id

                    # 前向
                    outputs = model(masked_ids)
                    logits = outputs.logits  # [1, L+2, vocab]

                    # 取位置 i 的预测分布
                    pos_logits = logits[0, i, :]

                    # Log-softmax
                    log_probs = torch.log_softmax(pos_logits, dim=-1)

                    # 取真实 token 的 log 概率
                    log_prob = log_probs[original_token].item()
                    total_log_prob += log_prob
                    valid_positions += 1

            pll = total_log_prob
            pll_per_residue = pll / max(valid_positions, 1)
            batch_pll.append({
                'pll': pll,
                'pll_per_residue': pll_per_residue,
                'length': L,
                'valid_positions': valid_positions
            })

            # 进度
            idx = start + batch_seqs.index(seq)
            if (idx + 1) % 5 == 0 or idx == len(sequences) - 1:
                print(f"  [{idx+1}/{len(sequences)}] len={L}  PLL/res={pll_per_residue:.4f}")

        results.extend(batch_pll)

    return results


def main():
    parser = argparse.ArgumentParser(description='ESM-2 Independent Scoring')
    parser.add_argument('--foldpath-seqs', type=str, required=True,
                        help='FoldPath-LLM 生成序列文件 (FASTA 或 txt)')
    parser.add_argument('--rita-seqs', type=str, required=True,
                        help='RITA_m 生成序列文件 (FASTA 或 txt)')
    parser.add_argument('--nostruct-seqs', type=str, default=None,
                        help='NoStruct 消融序列文件 (可选)')
    parser.add_argument('--esm-model', type=str, default='facebook/esm2_t6_8M_UR50D',
                        help='ESM-2 模型名')
    parser.add_argument('--esm-local', type=str, default=None,
                        help='ESM-2 本地模型目录 (优先于 esm-model)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='批大小 (逐序列处理, 此参数仅用于分组)')
    parser.add_argument('--max-len', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--output', type=str, default='esm2_scores.json',
                        help='输出 JSON 文件')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info()
        print(f'VRAM: {free/1e9:.1f}/{total/1e9:.1f} GB free')

    # ── 加载 ESM-2 ──
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    if args.esm_local and os.path.exists(args.esm_local):
        print(f'Loading ESM-2 from local: {args.esm_local}')
        tokenizer = AutoTokenizer.from_pretrained(args.esm_local)
        model = AutoModelForMaskedLM.from_pretrained(args.esm_local)
    else:
        print(f'Loading ESM-2 from HuggingFace: {args.esm_model}')
        tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
        model = AutoModelForMaskedLM.from_pretrained(args.esm_model)

    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'ESM-2 params: {n_params/1e6:.1f}M')

    # ── 加载序列 ──
    all_groups = {}
    for name, path in [('FoldPath-LLM', args.foldpath_seqs),
                       ('RITA_m', args.rita_seqs)]:
        seqs = load_sequences(path)
        seqs = [s for s in seqs if len(s) >= 20]
        print(f'{name}: {len(seqs)} valid sequences (len >= 20)')
        all_groups[name] = seqs

    if args.nostruct_seqs:
        seqs = load_sequences(args.nostruct_seqs)
        seqs = [s for s in seqs if len(s) >= 20]
        print(f'NoStruct: {len(seqs)} valid sequences')
        all_groups['NoStruct'] = seqs

    # ── 打分 ──
    all_results = {}

    for group_name, seqs in all_groups.items():
        print(f'\n{"="*50}')
        print(f'Scoring: {group_name} ({len(seqs)} sequences)')
        print(f'{"="*50}')

        t0 = time.time()
        scores = pll_score(model, tokenizer, seqs, device,
                           batch_size=args.batch_size, max_len=args.max_len)
        elapsed = time.time() - t0

        pll_values = [s['pll_per_residue'] for s in scores]
        avg_pll = np.mean(pll_values)
        std_pll = np.std(pll_values)
        ppl = np.exp(-avg_pll)

        lengths = [s['length'] for s in scores]

        all_results[group_name] = {
            'n_sequences': len(seqs),
            'avg_pll': round(float(avg_pll), 4),
            'std_pll': round(float(std_pll), 4),
            'ppl': round(float(ppl), 4),
            'avg_length': round(float(np.mean(lengths)), 1),
            'total_pll': round(float(np.sum([s['pll'] for s in scores])), 2),
            'scoring_time_s': round(elapsed, 1),
            'esm_model': args.esm_local or args.esm_model,
            'per_sequence': scores,
        }

        print(f'\n  {group_name} Results:')
        print(f'    ESM-2 Avg PLL (per residue): {avg_pll:.4f} ± {std_pll:.4f}')
        print(f'    ESM-2 PPL:                    {ppl:.4f}')
        print(f'    Avg Length:                   {np.mean(lengths):.0f}')

    # ── 汇总表格 ──
    print(f'\n{"="*65}')
    print(f'  ESM-2 Independent Evaluation Summary')
    print(f'{"="*65}')
    print(f'  {"Model":<20} {"ESM-2 Avg LL ↑":>15} {"ESM-2 PPL ↓":>14} {"N":>5} {"AvgLen":>7}')
    print(f'  {"-"*20} {"-"*15} {"-"*14} {"-"*5} {"-"*7}')
    for name, res in all_results.items():
        print(f'  {name:<20} {res["avg_pll"]:>15.4f} {res["ppl"]:>14.4f} {res["n_sequences"]:>5} {res["avg_length"]:>7.0f}')

    # ── 保存 ──
    # 保存时去掉 per_sequence 的冗余详细信息以节省空间
    save_results = {}
    for name, res in all_results.items():
        save_results[name] = {k: v for k, v in res.items() if k != 'per_sequence'}

    with open(args.output, 'w') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f'\nSaved: {args.output}')

    # ── LaTeX 表格片段 ──
    print(f'\n% LaTeX table snippet:')
    print(r'\begin{table}[h]')
    print(r'\centering')
    print(r'\caption{ESM-2 independent evaluation of generated sequences.}')
    print(r'\label{tab:esm2}')
    print(r'\begin{tabular}{l c c c}')
    print(r'\toprule')
    print(r'Model & ESM-2 Avg LL $\uparrow$ & ESM-2 PPL $\downarrow$ & Avg Length \\')
    print(r'\midrule')
    for name, res in all_results.items():
        print(f'  {name} & {res["avg_pll"]:.4f} $\\pm$ {res["std_pll"]:.4f} & {res["ppl"]:.2f} & {res["avg_length"]:.0f} \\\\')
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')


if __name__ == '__main__':
    main()
