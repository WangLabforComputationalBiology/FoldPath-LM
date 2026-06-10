"""
SOTA Baseline Comparison (ProGen2-small / ProtGPT2)
用法: python baseline_protgpt2.py --num-seqs 50 --model progen2 --output result.json
"""
import torch, sys, os, json, argparse, time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--num-seqs', type=int, default=50)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--model', type=str, default='progen2',
                    choices=['progen2', 'protgpt2'],
                    help='progen2 (local) or protgpt2 (HuggingFace)')
parser.add_argument('--output', type=str, default='baseline_result.json')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import FoldPathBenchmark

# ════════ Load Model ════════
if args.model == 'progen2':
    model_path = 'pretrained/progen2-small'
    print(f'Loading ProGen2-small from {model_path}...')
else:
    model_path = 'nferruz/ProtGPT2'
    print(f'Loading ProtGPT2 from HuggingFace...')

from transformers import AutoModelForCausalLM, BertTokenizer

tokenizer = BertTokenizer(
    vocab_file=f'{model_path}/vocab.txt',
    do_lower_case=False,
    bos_token='<cls>', eos_token='<eos>',
    unk_token='<unk>', pad_token='<pad>',
)
print('  Tokenizer: BertTokenizer (vocab-only)')
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
print(f'Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}')

# ════════ Generate ════════
print(f'\nGenerating {args.num_seqs} sequences (T={args.temperature})...')
seqs = []
t0 = time.time()

with torch.no_grad():
    for i in range(args.num_seqs):
        start_id = tokenizer.bos_token_id
        input_ids = torch.tensor([[start_id]], device=device)
        generated = []

        for _ in range(256):
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :] / args.temperature

            # Mask special tokens
            for tid in [tokenizer.pad_token_id, tokenizer.bos_token_id,
                        tokenizer.eos_token_id, tokenizer.unk_token_id]:
                if tid is not None:
                    logits[tid] = float('-inf')

            topk, _ = torch.topk(logits, min(50, logits.size(-1)))
            logits[logits < topk[-1]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            if probs.sum() == 0 or torch.isnan(probs).any():
                break

            next_tok = torch.multinomial(probs, 1).item()
            if next_tok == tokenizer.eos_token_id:
                break
            generated.append(next_tok)
            input_ids = torch.cat([input_ids, torch.tensor([[next_tok]], device=device)], dim=1)

        seq = tokenizer.decode(generated, skip_special_tokens=True)
        seq = seq.replace(' ', '').upper()
        seq = ''.join(c for c in seq if c in 'ACDEFGHIKLMNPQRSTVWY')
        if len(seq) >= 30:
            seqs.append(seq)
        if (i+1) % 10 == 0:
            print(f'  [{i+1}/{args.num_seqs}] valid: {len(seqs)}')

t1 = time.time()
print(f'Generated {len(seqs)} valid sequences in {t1-t0:.0f}s')

# ════════ Evaluate ════════
print('\nEvaluating...')
bench = FoldPathBenchmark(reference_fasta='data/train_sequences.fasta')
result = bench.evaluate(seqs[:args.num_seqs], verbose=True)

output = {
    'model': args.model,
    'num_seq': len(seqs[:args.num_seqs]),
    'temperature': args.temperature,
    'physico_mean': round(result['physico']['mean'], 4),
    'physico_std': round(result['physico']['std'], 4),
    'diversity': round(result['diversity']['total'], 4),
    'naturalness_mean': round(result['naturalness']['mean'], 4),
    'naturalness_std': round(result['naturalness']['std'], 4),
    'composite': round(result['composite'], 4),
    'grade': result['grade'],
}

print(f'\n{"="*55}')
print(f'  {args.model} Baseline (T={args.temperature})')
print(f'  Physicochemical: {output["physico_mean"]:.4f}')
print(f'  Diversity:       {output["diversity"]:.4f}')
print(f'  Naturalness:     {output["naturalness_mean"]:.4f}')
print(f'  Composite:       {output["composite"]:.4f} ({output["grade"]})')
print(f'{"="*55}')

# Compare with FoldPath-LLM
print(f'\n  Comparison:')
label = 'ProGen2' if args.model == 'progen2' else 'ProtGPT2'
print(f'  {"Metric":<20} {label:>10} {"FoldPath-LLM":>14} {"RITA_m":>10}')
print(f'  {"-"*20} {"-"*10} {"-"*14} {"-"*10}')
print(f'  {"Naturalness":<20} {output["naturalness_mean"]:>10.3f} {0.514:>14.3f} {0.483:>10.3f}')
print(f'  {"Physicochemical":<20} {output["physico_mean"]:>10.3f} {0.752:>14.3f} {0.786:>10.3f}')
print(f'  {"Diversity":<20} {output["diversity"]:>10.3f} {0.819:>14.3f} {0.741:>10.3f}')
print(f'  {"Composite":<20} {output["composite"]:>10.3f} {0.67:>14.3f} {0.65:>10.3f}')

with open(args.output, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f'\nSaved: {args.output}')
