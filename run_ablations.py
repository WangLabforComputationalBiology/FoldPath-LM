"""
消融实验启动器: 并行启动6个消融变体 + 1个完整模型对照
每个变体独立进程, 可分布在多GPU或多台机器上运行

用法:
  # 终端1:   python run_ablations.py --ablation no_struct
  # 终端2:   python run_ablations.py --ablation no_physico
  # 终端3:   python run_ablations.py --ablation no_chembias
  # 终端4:   python run_ablations.py --ablation no_divreg
  # 终端5:   python run_ablations.py --ablation no_natreg
  # 终端6:   python run_ablations.py --ablation no_mlm
  # 终端7:   python run_ablations.py --ablation full

所有变体: no_struct | no_physico | no_chembias | no_divreg | no_natreg | no_mlm | full

消融说明:
  no_struct   : 移除结构轨 → 验证结构偏置对天然度/长度的贡献
  no_physico  : 移除理化编码器 → 验证理化约束对理化合理性的贡献
  no_chembias : 移除化学交互偏置 → 验证化学配对对二肽相关性的贡献
  no_divreg   : 移除所有多样性正则化 (熵+边际+重复+均匀) → 验证防崩塌体系的贡献
  no_natreg   : 移除二肽KL+k-mer惩罚 → 验证天然度正则化的贡献
  no_mlm      : 移除MLM辅助任务 → 验证双向结构轨预测的贡献
  full        : 完整模型 (对照)
"""
import torch, sys, os, time, json, argparse
from config import DEVICE, ModelConfig, TrainConfig, LogConfig, PAD_IDX
from model import FoldPathLLM
from dataset import create_dataloaders
from train import Trainer

ABLATION_DESC = {
    'no_struct':   '结构轨 (StructureTrack) — 移除溶剂暴露/二级结构/距离预测及结构偏置',
    'no_physico':  '理化编码器 (PhysicoEncoder) — 移除12维AA理化属性嵌入',
    'no_chembias': '化学交互偏置 (ChemBias) — 移除理化×结构双线性交互偏置',
    'no_divreg':   '多样性正则化 — 移除熵/边际/重复/均匀性四项损失',
    'no_natreg':   '天然度正则化 — 移除二肽KL散度 + k-mer存在性惩罚',
    'no_mlm':      'MLM辅助任务 — 移除10% token掩码预测',
    'full':        '完整模型 (所有组件启用)',
}

parser = argparse.ArgumentParser()
parser.add_argument('--ablation', type=str, required=True,
                    choices=['no_struct', 'no_physico', 'no_chembias', 'no_divreg',
                             'no_natreg', 'no_mlm', 'full'],
                    help='消融变体名称')
parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
args = parser.parse_args()

# ── 配置 ──
model_cfg = ModelConfig()
model_cfg.ablation = None if args.ablation == 'full' else args.ablation

train_cfg = TrainConfig()
log_cfg = LogConfig()

# 输出路径区隔
tag = args.ablation
log_cfg.log_dir = f'./logs_ablation/{tag}'
log_cfg.checkpoint_dir = f'./checkpoints_ablation/{tag}'
os.makedirs(log_cfg.log_dir, exist_ok=True)
os.makedirs(log_cfg.checkpoint_dir, exist_ok=True)

print(f'\n{"="*60}')
print(f'  消融实验: {tag}')
print(f'{"="*60}')
print(f'  移除组件: {ABLATION_DESC.get(tag, "完整模型")}')
print(f'  日志目录: {log_cfg.log_dir}')
print(f'  Checkpoint: {log_cfg.checkpoint_dir}')
print(f'  训练轮数: {args.epochs}')
print(f'{"="*60}\n')

# ── 训练 ──
trainer = Trainer(
    config=model_cfg, train_config=train_cfg, log_config=log_cfg,
    use_esm=False, use_rita=True, rita_model_name='RITA_m', rita_local_dir='pretrained'
)
trainer.train(num_epochs=args.epochs)

# ── 评测 ──
print(f'\n评测最佳checkpoint...')
from generate import ProteinGenerator
from evaluation import FoldPathBenchmark
from config import GenerateConfig

gen = ProteinGenerator(checkpoint_path=os.path.join(log_cfg.checkpoint_dir, 'best_model.pt'), device=DEVICE)
gc = GenerateConfig()
gc.num_samples = 50; gc.max_length = 256; gc.temperature = 0.24
gc.top_k = 50; gc.top_p = 0.92; gc.use_physico_filter = True

seqs, _ = gen.generate(gc)
seqs = [s for s in seqs if len(s) >= 20]

bench = FoldPathBenchmark(reference_fasta='data/train_sequences.fasta')
result = bench.evaluate(seqs, verbose=True)

# 保存评测结果
eval_path = os.path.join(log_cfg.log_dir, 'eval_result.json')
with open(eval_path, 'w', encoding='utf-8') as f:
    json.dump({
        'ablation': tag,
        'physico_mean': result['physico']['mean'],
        'diversity': result['diversity']['total'],
        'naturalness': result['naturalness']['mean'],
        'composite': result['composite'],
        'grade': result['grade'],
    }, f, indent=2, ensure_ascii=False)
print(f'\n评测保存: {eval_path}')
