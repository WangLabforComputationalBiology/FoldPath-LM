"""
FoldPath-LLM: ESM-2 Encoder Wrapper
ESM-2 预训练蛋白质语言模型编码器封装
支持多尺寸模型切换 (35M / 150M / 650M)
"""

import torch
import torch.nn as nn
import os
from transformers import AutoTokenizer, AutoModel


class ESMEncoder(nn.Module):
    """
    ESM-2 编码器封装
    - 冻结所有预训练参数
    - 支持本地离线加载 & HuggingFace 在线下载
    - 输出维度: 35M→480维, 150M→640维, 650M→1280维
    """

    MODEL_CONFIGS = {
        "esm2_t12_35M_UR50D":  {"hidden_size": 480,  "layers": 12,  "params": "35M"},
        "esm2_t30_150M_UR50D": {"hidden_size": 640,  "layers": 30,  "params": "150M"},
        "esm2_t33_650M_UR50D": {"hidden_size": 1280, "layers": 33,  "params": "650M"},
    }

    def __init__(self, model_name="esm2_t12_35M_UR50D", device=None, freeze=True,
                 local_path=None):
        """
        Args:
            model_name: HuggingFace 模型名称
            device: torch device
            freeze: 是否冻结参数
            local_path: 本地模型路径 (优先使用)
        """
        super().__init__()
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.freeze = freeze

        # 加载模型
        if local_path and os.path.exists(local_path):
            print(f"[ESM] 从本地加载: {local_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(local_path)
            self.model = AutoModel.from_pretrained(local_path)
        else:
            print(f"[ESM] 从 HuggingFace 加载: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
            self.model = AutoModel.from_pretrained(f"facebook/{model_name}")

        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        self.model.to(self.device)
        self.model.eval()

        # 冻结所有参数
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"[ESM] 已冻结 {sum(p.numel() for p in self.model.parameters()):,} 参数")

        # 验证 hidden_size
        expected = self.MODEL_CONFIGS.get(model_name, {}).get("hidden_size", self.hidden_size)
        if self.hidden_size != expected:
            print(f"[ESM] 警告: hidden_size {self.hidden_size} != 预期 {expected}")

        print(f"[ESM] 模型: {model_name} | 隐藏层维度: {self.hidden_size} | "
              f"层数: {self.num_layers} | 设备: {self.device}")

    def tokenize(self, sequences):
        """将氨基酸序列转换为 ESM token IDs"""
        # ESM-2 使用空格分隔的氨基酸序列
        spaced_seqs = [" ".join(list(seq)) for seq in sequences]
        encoded = self.tokenizer(
            spaced_seqs,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def forward(self, sequences):
        """
        编码蛋白质序列
        Args:
            sequences: list of str, 氨基酸序列, e.g. ["MKALIVL", "GLVL..."]
        Returns:
            embeddings: [B, L, hidden_size] 每个残基的嵌入向量
            attention_mask: [B, L] 有效位置掩码
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        inputs = self.tokenize(sequences)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # last_hidden_state: [B, L, hidden_size]
        embeddings = outputs.last_hidden_state

        # ESM tokenizer 在序列首尾加了 <cls> 和 <eos>，需要去掉
        # 实际上对于氨基酸序列，ESM tokenizer 不会加特殊 token
        # 但我们仍需要处理 padding

        return embeddings, inputs["attention_mask"].bool()

    def get_param_count(self):
        """返回参数量统计"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def get_model_size_mb(self):
        """估算模型显存占用 (MB)"""
        total_params = sum(p.numel() for p in self.model.parameters())
        # FP32: 4 bytes per param
        size_mb = total_params * 4 / (1024 * 1024)
        return round(size_mb, 1)


class ESMProjection(nn.Module):
    """
    ESM 嵌入 → FoldPath-LLM 维度的投影层
    将 ESM-2 的高维嵌入 (480/640/1280) 投影到模型的 d_model (256)
    """

    def __init__(self, esm_dim, d_model=256, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(esm_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, esm_embeddings):
        """
        Args:
            esm_embeddings: [B, L, esm_dim]
        Returns:
            projected: [B, L, d_model]
        """
        return self.projection(esm_embeddings)


# ── 便捷工厂函数 ────────────────────────────────────────────

def create_esm_encoder(model_name="esm2_t12_35M_UR50D", device=None, freeze=True,
                       local_dir="pretrained"):
    """
    创建 ESM 编码器，优先从本地加载
    Args:
        model_name: 模型名 (不含 facebook/ 前缀)
        device: torch device
        freeze: 是否冻结
        local_dir: 本地模型根目录 (如果是相对路径，相对于 esm_encoder.py 所在目录)
    """
    # 将 local_dir 解析为绝对路径
    if not os.path.isabs(local_dir):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_dir = os.path.join(base_dir, local_dir)
    local_path = os.path.join(local_dir, model_name)
    if os.path.exists(local_path):
        print(f"[ESM] 发现本地模型: {local_path}")
        return ESMEncoder(model_name=model_name, device=device, freeze=freeze,
                         local_path=local_path)
    else:
        print(f"[ESM] 本地未找到 {local_path}，从 HuggingFace 下载...")
        return ESMEncoder(model_name=model_name, device=device, freeze=freeze)
