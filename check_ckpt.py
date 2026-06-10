import torch, sys
path = sys.argv[1] if len(sys.argv) > 1 else 'esmpro/foldpath_best.pt'
ckpt = torch.load(path, map_location='cpu')
print('use_esm:', ckpt.get('use_esm'))
print('use_rita:', ckpt.get('use_rita'))
print('esm_model:', ckpt.get('esm_model_name'))
print('rita_model:', ckpt.get('rita_model_name'))
