"""
FoldPath-LLM: Protein Generation
折叠路径引导的蛋白质设计大语言模型 - 蛋白质生成模块
支持 ESM-2 基座模式
"""

import torch
import os
from config import DEVICE, IDX_TO_AA, GenerateConfig, ModelConfig
from model import FoldPathLLM
from physicochemical import PhysicochemicalEvaluator
from esm_encoder import create_esm_encoder
from rita_encoder import create_rita_encoder
import numpy as np


def _load_encoder(use_esm, use_rita, esm_model_name, rita_model_name,
                  esm_local_dir, rita_local_dir, device, config=None):
    """统一加载基座编码器 (ESM 或 RITA)"""
    if use_rita:
        name = rita_model_name or (config.rita_model_name if config else "RITA_m")
        print(f"[INFO] 加载 RITA 基座: {name}")
        return create_rita_encoder(model_name=name, device=device, freeze=True,
                                   local_dir=rita_local_dir), True
    elif use_esm:
        name = esm_model_name or (config.esm_model_name if config else "esm2_t12_35M_UR50D")
        print(f"[INFO] 加载 ESM-2 基座: {name}")
        return create_esm_encoder(model_name=name, device=device, freeze=True,
                                  local_dir=esm_local_dir), False
    return None, False


class ProteinGenerator:
    """蛋白质生成器 (支持 ESM-2 / RITA 基座)"""

    def __init__(self, model=None, checkpoint_path=None, device=None,
                 use_esm=True, esm_model_name=None, esm_local_dir="pretrained",
                 use_rita=False, rita_model_name=None, rita_local_dir="pretrained"):
        self.device = device or DEVICE
        self.evaluator = PhysicochemicalEvaluator(device='cpu')
        self.use_esm = use_esm and not use_rita
        self.use_rita = use_rita
        self.esm_encoder = None

        if model is not None:
            self.model = model
            self.use_esm = model.use_esm
        elif checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            config = checkpoint.get('model_config', ModelConfig())
            use_esm_ckpt = checkpoint.get('use_esm', use_esm)
            use_rita_ckpt = checkpoint.get('use_rita', False)
            esm_name_ckpt = checkpoint.get('esm_model_name', esm_model_name)

            if use_esm_ckpt or use_rita_ckpt:
                self.use_esm = True if use_esm_ckpt and not use_rita_ckpt else False
                self.use_rita = use_rita_ckpt
                self.esm_encoder, _ = _load_encoder(
                    use_esm=use_esm_ckpt and not use_rita_ckpt,
                    use_rita=use_rita_ckpt,
                    esm_model_name=esm_name_ckpt or esm_model_name,
                    rita_model_name=checkpoint.get('rita_model_name', rita_model_name),
                    esm_local_dir=esm_local_dir,
                    rita_local_dir=rita_local_dir,
                    device=self.device, config=config
                )
            else:
                self.use_esm = False
                self.use_rita = False

            self.model = FoldPathLLM(config, esm_encoder=self.esm_encoder)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if 'output_scale' not in checkpoint.get('model_state_dict', {}):
                self.model.output_scale.data.fill_(1.0)
            print(f"[INFO] 已加载模型: {checkpoint_path}")
        else:
            self.esm_encoder, self.use_rita = _load_encoder(
                use_esm=self.use_esm, use_rita=self.use_rita,
                esm_model_name=esm_model_name, rita_model_name=rita_model_name,
                esm_local_dir=esm_local_dir, rita_local_dir=rita_local_dir,
                device=self.device
            )
            if self.esm_encoder:
                self.use_esm = True
            self.model = FoldPathLLM(ModelConfig(), esm_encoder=self.esm_encoder)
            print("[INFO] 使用随机初始化模型 (用于演示)")

        self.model = self.model.to(self.device)
        self.model.eval()

        if self.use_rita:
            mode_tag = "[RITA]"
        elif self.model.use_esm:
            mode_tag = "[ESM-2]"
        else:
            mode_tag = "[Scratch]"
        params = self.model.get_param_count()
        print(f"  {mode_tag} 总参数: {params['total']:,} | "
              f"可训练: {params['trainable']:,}")

    def generate(self, config=None):
        if config is None:
            config = GenerateConfig()
        sequences = []
        all_physico_scores = []
        for i in range(config.num_samples):
            generated, physico_scores = self.model.generate(
                max_length=config.max_length,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                physico_threshold=config.physico_threshold if config.use_physico_filter else None,
                use_physico_filter=config.use_physico_filter,
                device=self.device
            )
            seq = ''.join([IDX_TO_AA.get(idx, 'X') for idx in generated])
            sequences.append(seq)
            all_physico_scores.append(physico_scores)
            print(f"  样本 {i+1}: {seq[:50]}{'...' if len(seq) > 50 else ''} (长度: {len(seq)})")
        return sequences, all_physico_scores

    def evaluate_sequence(self, sequence):
        # 注意: evaluate_all 需要 exposure, distance, ss_types
        # 在真实场景中，这些应从模型预测或实验数据获取
        L = len(sequence)
        exposure = np.random.beta(2, 5, size=L)
        distance = np.random.exponential(10, size=(L, L))
        distance = (distance + distance.T) / 2
        np.fill_diagonal(distance, 0)
        ss_types = np.random.choice([0, 1, 2], size=L, p=[0.4, 0.2, 0.4])
        results = self.evaluator.evaluate_all(sequence, exposure, distance, ss_types)
        return results

    def generate_with_evaluation(self, config=None):
        print("\n🧬 开始生成蛋白质...")
        sequences, physico_scores = self.generate(config)
        print("\n📊 开始理化评估...")
        all_results = []
        for i, seq in enumerate(sequences):
            print(f"\n--- 样本 {i+1} ---")
            results = self.evaluate_sequence(seq)
            all_results.append(results)
            report = self.evaluator.format_report(results)
            print(report)
        return sequences, all_results


def main():
    """生成入口"""
    import argparse
    parser = argparse.ArgumentParser(description='FoldPath-LLM Protein Generation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型 checkpoint 路径')
    parser.add_argument('--no-esm', action='store_true',
                        help='禁用基座编码器 (纯 From-Scratch)')
    parser.add_argument('--esm-model', type=str, default=None,
                        help='ESM 模型名')
    parser.add_argument('--esm-local-dir', type=str, default='pretrained',
                        help='ESM 本地模型目录')
    parser.add_argument('--rita-model', type=str, default=None,
                        help='使用 RITA 因果模型 (RITA_s / RITA_m / RITA_l / RITA_xl)')
    parser.add_argument('--rita-local-dir', type=str, default='pretrained',
                        help='RITA 本地模型目录')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='生成样本数')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='采样温度')
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--no-eval', action='store_true',
                        help='跳过理化评估')
    args = parser.parse_args()

    use_esm = not args.no_esm and not args.rita_model
    use_rita = args.rita_model is not None

    generator = ProteinGenerator(
        checkpoint_path=args.checkpoint,
        use_esm=use_esm,
        esm_model_name=args.esm_model,
        esm_local_dir=args.esm_local_dir,
        use_rita=use_rita,
        rita_model_name=args.rita_model,
        rita_local_dir=args.rita_local_dir
    )

    gen_config = GenerateConfig()
    gen_config.num_samples = args.num_samples
    gen_config.temperature = args.temperature
    gen_config.top_k = args.top_k
    gen_config.top_p = args.top_p

    if args.no_eval:
        sequences, physico_scores = generator.generate(gen_config)
        print(f"\n✅ 共生成 {len(sequences)} 条蛋白质序列")
        for i, seq in enumerate(sequences):
            print(f"  >Protein_{i+1}: {seq}")
    else:
        sequences, results = generator.generate_with_evaluation(gen_config)
        print(f"\n✅ 共生成 {len(sequences)} 条蛋白质序列")


if __name__ == "__main__":
    main()