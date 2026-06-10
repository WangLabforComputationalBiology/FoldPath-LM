"""Check ProGen2 weight keys — find embedding and all root keys"""
from safetensors.torch import load_file
sd = load_file('pretrained/progen2-small/model.safetensors')

# Find embedding-related keys
print('=== Embedding/LM head keys ===')
for k in sorted(sd.keys()):
    if any(x in k.lower() for x in ['embed', 'wte', 'lm_head', 'token', 'norm']):
        print(f'  {k}: {list(sd[k].shape)}')

# Find root-level keys (not decoder.layers.N)
print('\n=== Root-level keys (no layer number) ===')
for k in sorted(sd.keys()):
    if 'decoder.layers.' not in k:
        print(f'  {k}: {list(sd[k].shape)}')

# List ALL key prefixes
print('\n=== All unique key prefixes ===')
prefixes = set()
for k in sd.keys():
    parts = k.split('.')
    for p in range(1, len(parts)):
        prefixes.add('.'.join(parts[:p]))
for p in sorted(prefixes):
    print(f'  {p}')

