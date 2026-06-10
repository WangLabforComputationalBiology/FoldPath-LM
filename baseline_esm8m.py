"""
ESM-2 8M Baseline — Teacher-Forcing P/R on validation set
用法: python baseline_esm8m.py
"""
import torch, sys, os, json, argparse
import torch.nn as nn, torch.nn.functional as F
from torch.amp import autocast

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--lr', type=float, default=5e-4)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esm_encoder import create_esm_encoder
from dataset import create_dataloaders
from config import TOTAL_VOCAB, PAD_IDX

# ════════ Load ESM-2 8M ════════
print('Loading ESM-2 8M...')
encoder = create_esm_encoder(model_name='esm2_t6_8M_UR50D', device=device, freeze=True, local_dir='pretrained')
esm_dim = encoder.hidden_size

# ════════ Simple prediction head ════════
head = nn.Sequential(
    nn.Linear(esm_dim, 512), nn.GELU(),
    nn.Linear(512, TOTAL_VOCAB)
).to(device)

optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

train_loader, val_loader = create_dataloaders(use_synthetic=False, batch_size=args.batch_size)
print(f'Train: {len(train_loader)} batches, Val: {len(val_loader)} batches')

# ════════ Train ════════
print(f'\nFine-tuning head for {args.epochs} epochs...')
for epoch in range(args.epochs):
    head.train()
    total_loss = 0; nb = 0
    for batch in train_loader:
        seqs = batch.get('sequence')
        if not seqs: continue
        clean = [''.join(c for c in s if c in 'ACDEFGHIKLMNPQRSTVWY') for s in seqs]
        clean = [s if s else 'M' for s in clean]
        targets = batch['target_ids'].to(device)  # [B, L]
        mask = batch['mask'].to(device)

        emb, _ = encoder(clean)  # [B, L_esm, D]
        # Align: pad/crop ESM output to target length
        B, Le, D = emb.shape; _, Lt = targets.shape
        if Le < Lt:
            pad = torch.zeros(B, Lt - Le, D, device=device)
            emb = torch.cat([emb, pad], dim=1)
        elif Le > Lt:
            emb = emb[:, :Lt, :]

        logits = head(emb)  # [B, Lt, V]

        if scaler:
            with autocast(device_type='cuda'):
                loss = F.cross_entropy(logits.reshape(-1, TOTAL_VOCAB), targets.reshape(-1), ignore_index=PAD_IDX)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            loss = F.cross_entropy(logits.reshape(-1, TOTAL_VOCAB), targets.reshape(-1), ignore_index=PAD_IDX)
            loss.backward(); optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item(); nb += 1

    print(f'  Epoch {epoch+1}: Loss={total_loss/max(nb,1):.4f}')

# ════════ Eval ════════
print('\nEvaluating teacher-forcing P/R...')
head.eval()
correct = 0; total = 0
with torch.no_grad():
    for batch in val_loader:
        seqs = batch.get('sequence')
        if not seqs: continue
        clean = [''.join(c for c in s if c in 'ACDEFGHIKLMNPQRSTVWY') for s in seqs]
        clean = [s if s else 'M' for s in clean]
        targets = batch['target_ids'].to(device)
        mask = batch['mask'].to(device)

        emb, _ = encoder(clean)
        B, Le, D = emb.shape; _, Lt = targets.shape
        if Le < Lt:
            pad = torch.zeros(B, Lt - Le, D, device=device)
            emb = torch.cat([emb, pad], dim=1)
        elif Le > Lt:
            emb = emb[:, :Lt, :]

        logits = head(emb)
        pred = logits.argmax(dim=-1)
        correct += ((pred == targets) & mask.bool()).sum().item()
        total += mask.sum().item()

p = correct / max(total, 1)
print(f'\n{"="*50}')
print(f'  ESM-2 8M Teacher-Forcing Baseline')
print(f'  Precision/Recall: {p:.4f}')
print(f'  Tokens evaluated: {total}')
print(f'{"="*50}')
print(f'\n  Comparison:')
print(f'  ESM-2 8M (bidirectional, teacher-forcing): P/R = {p:.4f}')
print(f'  RITA_m (causal, teacher-forcing):          P/R = 0.717')
print(f'  FoldPath-LLM (causal, teacher-forcing):    P/R ~ 0.613')
print(f'\n  Note: ESM-2 P/R is inflated due to bidirectional attention (sees future tokens).')
print(f'  This baseline demonstrates the ceiling for bidirectional encoding on this task.')

with open('esm8m_baseline.json', 'w') as f:
    json.dump({'model': 'ESM-2 8M + head', 'P/R': round(p, 4), 'tokens': total}, f)
