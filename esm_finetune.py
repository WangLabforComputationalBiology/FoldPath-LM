"""
ESM-2 35M 快速微调 — 基线对比
用法: python esm_finetune.py --epochs 5 --batch-size 8
"""
import torch, torch.nn as nn, torch.nn.functional as F
import sys, os, argparse, time
from torch.amp import autocast

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--output', type=str, default='esmpro/esm2_ft.pt')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esm_encoder import create_esm_encoder
from dataset import create_dataloaders
from config import TOTAL_VOCAB, IDX_TO_AA, AA_TO_IDX, BOS_IDX, EOS_IDX

# Load ESM-2, unfreeze last 2 layers
print('Loading ESM-2 35M...')
encoder = create_esm_encoder(model_name='esm2_t6_8M_UR50D', device=device, freeze=False, local_dir='pretrained')

# Fine-tune head: ESM embeddings -> AA logits
class ESMGenerativeHead(nn.Module):
    def __init__(self, esm_dim=320, hidden=512, vocab=TOTAL_VOCAB):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(esm_dim, hidden), nn.GELU(),
            nn.Linear(hidden, vocab)
        )
    def forward(self, x):
        return self.net(x)

head = ESMGenerativeHead(esm_dim=encoder.hidden_size).to(device)
optimizer = torch.optim.AdamW(
    list(encoder.model.parameters()) + list(head.parameters()),
    lr=args.lr, weight_decay=0.01
)

train_loader, val_loader = create_dataloaders(use_synthetic=False, batch_size=args.batch_size)
scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

print(f'\nFine-tuning ESM-2 for {args.epochs} epochs (batch={args.batch_size})...')
encoder.model.train()

for epoch in range(args.epochs):
    t0 = time.time(); running_loss = 0.0; nb = 0
    for batch in train_loader:
        sequences = batch.get('sequence', None)
        if not sequences: continue
        clean_seqs = [''.join(c for c in s if c in AA_TO_IDX) for s in sequences]
        clean_seqs = [s if s else 'M' for s in clean_seqs]

        target_ids = batch['target_ids'].to(device)
        mask = batch['mask'].to(device)

        esm_emb, esm_mask = encoder(clean_seqs)
        # Align ESM embeddings to target length
        B, L_esm, D = esm_emb.shape
        _, L_tgt = target_ids.shape
        # Simple approach: pad/crop to match
        if L_esm < L_tgt:
            pad = torch.zeros(B, L_tgt - L_esm, D, device=device)
            esm_emb = torch.cat([esm_emb, pad], dim=1)
        elif L_esm > L_tgt:
            esm_emb = esm_emb[:, :L_tgt, :]

        logits = head(esm_emb)

        loss = F.cross_entropy(
            logits.reshape(-1, TOTAL_VOCAB), target_ids.reshape(-1),
            ignore_index=1  # PAD
        )

        if scaler:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item(); nb += 1

    epoch_loss = running_loss / max(nb, 1)
    # Quick val
    encoder.model.eval(); vloss = 0; vnb = 0
    with torch.no_grad():
        for batch in val_loader:
            sequences = batch.get('sequence', None)
            if not sequences: continue
            clean_seqs = [''.join(c for c in s if c in AA_TO_IDX) for s in sequences]
            clean_seqs = [s if s else 'M' for s in clean_seqs]
            target_ids = batch['target_ids'].to(device)
            esm_emb, _ = encoder(clean_seqs)
            B, L_esm, D = esm_emb.shape
            _, L_tgt = target_ids.shape
            if L_esm < L_tgt:
                pad = torch.zeros(B, L_tgt - L_esm, D, device=device)
                esm_emb = torch.cat([esm_emb, pad], dim=1)
            elif L_esm > L_tgt:
                esm_emb = esm_emb[:, :L_tgt, :]
            logits = head(esm_emb)
            vloss += F.cross_entropy(logits.reshape(-1, TOTAL_VOCAB), target_ids.reshape(-1), ignore_index=1).item()
            vnb += 1
    encoder.model.train()
    print(f'  Epoch {epoch+1}: Train Loss={epoch_loss:.4f}  Val Loss={vloss/max(vnb,1):.4f}  Time={time.time()-t0:.0f}s')

# Save
os.makedirs('esmpro', exist_ok=True)
torch.save({
    'head_state_dict': head.state_dict(),
    'esm_state_dict': encoder.model.state_dict(),
    'epoch': args.epochs,
}, args.output)
print(f'\nSaved: {args.output}')
