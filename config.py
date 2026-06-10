"""
FoldPath-LLM: Configuration
折叠路径引导的蛋白质设计大语言模型 - 配置文件
"""

import torch

# ============================================================
# 设备配置 - 自动检测CUDA
# ============================================================
def _detect_device():
    """检测并返回最佳计算设备，提供详细诊断信息"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[CUDA] 检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"[CUDA] 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"[CUDA] CUDA版本: {torch.version.cuda}")
        return device
    else:
        print("[WARN] CUDA不可用，使用CPU训练")
        print(f"[INFO] PyTorch版本: {torch.__version__}")
        if '+cpu' in torch.__version__:
            print("[HINT] 当前安装的是CPU版PyTorch，请运行 setup_cuda.bat 安装CUDA版本")
        return torch.device("cpu")

DEVICE = _detect_device()
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

# 氨基酸理化属性名称 (12维)
PHYSICO_PROPERTY_NAMES = [
    '疏水指数', '侧链体积', '电荷(pH7)', '柔性',
    '氢键供体', '氢键受体', '螺旋偏好', '折叠偏好',
    '转角偏好', '芳香性', '二硫键能力', 'pKa'
]

# 氨基酸颜色映射 (基于理化分类)
AA_COLORS = {
    'A': '#C8C8C8', 'V': '#C8C8C8', 'I': '#C8C8C8', 'L': '#C8C8C8',  # 非极性
    'M': '#C8C8C8', 'F': '#C8C8C8', 'W': '#C8C8C8',
    'G': '#85D68A', 'P': '#85D68A',                                  # 特殊
    'S': '#6FA8DC', 'T': '#6FA8DC', 'C': '#6FA8DC', 'N': '#6FA8DC', 'Q': '#6FA8DC',  # 极性
    'Y': '#E69138',                                                  # 芳香
    'D': '#E06666', 'E': '#E06666',                                  # 负电荷
    'K': '#8E7CC3', 'R': '#8E7CC3', 'H': '#8E7CC3',                  # 正电荷
}

# 氨基酸分类标签
AA_CATEGORIES = {
    'A': '非极性', 'V': '非极性', 'I': '非极性', 'L': '非极性',
    'M': '非极性', 'F': '芳香族', 'W': '芳香族',
    'G': '特殊',   'P': '特殊',
    'S': '极性',   'T': '极性',   'C': '极性',   'N': '极性', 'Q': '极性',
    'Y': '芳香族',
    'D': '负电荷', 'E': '负电荷',
    'K': '正电荷', 'R': '正电荷', 'H': '正电荷',
}

# ============================================================
# 氨基酸词表
# ============================================================
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}
VOCAB_SIZE = len(AMINO_ACIDS)

# 特殊token
PAD_IDX = 20
BOS_IDX = 21
EOS_IDX = 22
MASK_IDX = 23
TOTAL_VOCAB = 24  # 20 AA + PAD + BOS + EOS + MASK

# ============================================================
# 模型超参数
# ============================================================
class ModelConfig:
    # 主轨 (序列轨) Transformer
    d_model = 1024
    n_heads = 16
    n_layers_main = 6
    d_ff = 4096
    dropout = 0.2

    # 副轨 (结构轨) Transformer
    n_layers_struct = 3
    struct_latent_dim = 256

    # 理化性质编码
    physico_dim = 64
    physico_raw_dim = 12

    # 理化交互偏置
    chem_bias_dim = 32

    # 损失权重
    lambda_seq = 1.0
    lambda_struct = 0.5
    lambda_physico = 0.3

    # 标签平滑 (防止模型过度自信 → 生成崩塌)
    label_smoothing = 0.1

    # 熵正则化权重 (鼓励输出分布多样性)
    lambda_entropy = 0.3

    # 边际多样性权重 (防止全局氨基酸聚类偏好)
    lambda_marginal = 0.15

    # 连续重复惩罚权重 (防止模型对重复残基过度自信)
    lambda_repeat = 0.2

    # 对比学习均匀性权重 (防止表示空间模式崩塌)
    lambda_uniform = 0.1

    # MLM辅助任务权重 (基于结构轨双向预测，降低权重避免喧宾夺主)
    lambda_mlm = 0.1

    # 天然度正则化权重 (二肽频率 + k-mer)
    lambda_dipep = 0.15
    lambda_kmer = 0.05

    # ── 消融实验开关 ──
    ablation = None  # None=完整模型, 可选值见 ABLATION_VARIANTS

    # 最大序列长度
    max_seq_len = 256

    # ── 动态 Lambda 调度 ──
    # (epoch_start, epoch_end, lambda_struct, lambda_physico)
    # 训练初期强化物理引导 → 中后期逐步衰减 → 主轨主导
    LAMBDA_SCHEDULE = [
        (0,  2,  0.8, 0.5),   # Epoch 0-2:   强化结构+理化引导
        (3,  6,  0.6, 0.4),   # Epoch 3-6:   维持较高权重
        (7,  12, 0.4, 0.25),  # Epoch 7-12:  逐步衰减
        (13, 20, 0.2, 0.12),  # Epoch 13-20: 主轨主导
        (21, 99, 0.1, 0.06),  # Epoch 21+:    最小维持
    ]

    # ── ESM-2 基座配置 ──
    use_esm_base = True                       # 是否使用 ESM-2 作为基座编码器
    esm_model_name = "esm2_t12_35M_UR50D"     # 可切换: 35M / 150M / 650M
    esm_embed_dim = 480                        # ESM 输出维度 (35M→480, 150M→640, 650M→1280)
    freeze_esm = True                          # 冻结 ESM 参数
    esm_local_dir = "pretrained"               # 本地模型根目录

    # ── RITA 基座配置 ──
    use_rita_base = False                     # 是否使用 RITA 作为基座编码器 (因果, 无信息泄漏)
    rita_model_name = "RITA_m"                # 可切换: RITA_s / RITA_m / RITA_l / RITA_xl
    rita_embed_dim = 1024                      # RITA 输出维度 (S→768, M→1024, L→1536, XL→2048)
    freeze_rita = True                         # 冻结 RITA 参数
    rita_local_dir = "pretrained"              # 本地模型根目录

# ============================================================
# 训练超参数
# ============================================================
class TrainConfig:
    batch_size = 16
    learning_rate = 1e-4
    resume_learning_rate = 2e-5      # 续训时默认 LR (避免从 1e-4 恢复导致 loss 震荡)
    weight_decay = 0.05
    num_epochs = 50
    warmup_steps = 1000
    max_grad_norm = 1.0

    # 损失权重
    lambda_seq = 1.0
    lambda_struct = 0.5
    lambda_physico = 0.3

    # 学习率调度
    scheduler = "plateau"

    # 混合精度训练
    use_amp = True

    # 梯度累积
    gradient_accumulation_steps = 1

# ============================================================
# 生成配置
# ============================================================
class GenerateConfig:
    max_length = 256
    temperature = 1.2     # 从0.95提高 → 更强随机性, 防止崩塌
    top_k = 40            # 从50降低 → 略收窄但仍然合理
    top_p = 0.92          # 从0.95略降 → 减少尾部噪声
    num_samples = 10

    # 理化约束
    physico_threshold = -0.5
    use_physico_filter = True
    use_rejection_sampling = True

# ============================================================
# 数据配置
# ============================================================
class DataConfig:
    data_dir = "./data"
    train_file = "train_sequences.fasta"
    val_file = "val_sequences.fasta"
    test_file = "test_sequences.fasta"
    struct_dir = "structures"
    num_workers = 4
    pin_memory = True

# ============================================================
# 日志与保存
# ============================================================
class LogConfig:
    log_dir = "./logs"
    checkpoint_dir = "./checkpoints"
    save_every = 5
    log_every = 50
    eval_every = 1