"""
FoldPath-LLM: Jupyter Interactive Interface
============================================
基于 ipywidgets + matplotlib 的 Jupyter 可视化界面
适用于无法访问浏览器的局域网 Jupyter 环境
保留全部功能: 训练 / 生成 / 分析

使用方法:
    from jupyter_app import JupyterApp
    app = JupyterApp()
    app.display()
"""

import sys
import os
import time
import json
import threading
import copy
from io import BytesIO

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
from torch.amp import autocast

# ── 项目模块 ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEVICE, IDX_TO_AA, AMINO_ACIDS, AA_TO_IDX,
    ModelConfig, TrainingConfig, GenerateConfig
)
from model import FoldPathLLM
from dataset import ProteinDataset, create_dataloader
from physicochemical import PhysicochemicalEvaluator, PHYSICOCHEMICAL_TABLE, PHYSICO_MATRIX
from esm_encoder import create_esm_encoder
from generate import ProteinGenerator

# ── ipywidgets ────────────────────────────────────────
import ipywidgets as widgets
from IPython.display import display, clear_output, Image as IPImage, HTML

# ========================================================
# 全局样式 & 颜色
# ========================================================

AA_COLORS = {
    'A': '#8C8C8C', 'V': '#8C8C8C', 'I': '#8C8C8C', 'L': '#8C8C8C', 'M': '#8C8C8C',
    'F': '#6B6BCD', 'W': '#6B6BCD',
    'G': '#5CB85C', 'P': '#5CB85C',
    'S': '#5CA8D6', 'T': '#5CA8D6', 'C': '#5CA8D6', 'N': '#5CA8D6', 'Q': '#5CA8D6',
    'Y': '#E69138',
    'D': '#E06666', 'E': '#E06666',
    'K': '#8E7CC3', 'R': '#8E7CC3', 'H': '#8E7CC3',
    'X': '#555555',
}

AA_GROUPS = {
    '非极性': ['A', 'V', 'I', 'L', 'M'],
    '芳香族': ['F', 'W', 'Y'],
    '特殊': ['G', 'P'],
    '极性': ['S', 'T', 'C', 'N', 'Q'],
    '负电荷': ['D', 'E'],
    '正电荷': ['K', 'R', 'H'],
}
AA_COLORS_GROUP = {
    '#8C8C8C': '非极性 (A V I L M)',
    '#6B6BCD': '芳香族 (F W Y)',
    '#5CB85C': '特殊 (G P)',
    '#5CA8D6': '极性 (S T C N Q)',
    '#E06666': '负电荷 (D E)',
    '#8E7CC3': '正电荷 (K R H)',
}

CAT_COLORS = {
    '内部稳定性': '#5b9cff',
    '静电与电荷': '#ffd54f',
    '骨架与构象': '#4dd0e1',
    '特异性相互作用': '#b388ff',
    '全局性质': '#f06292',
}

INDICATOR_NAMES = {
    'hydrophobic_core': '疏水核心', 'packing_density': '堆积密度', 'steric_clash': '空间位阻',
    'hydrogen_bonds': '氢键', 'salt_bridges': '盐桥', 'surface_electrostatics': '表面静电',
    'ph_dependence': 'pH依赖性', 'rama_compliance': '拉氏构象', 'special_residue': 'Pro/Gly位置',
    'sidechain_conformation': '侧链构象', 'aromatic_stacking': '芳香堆叠', 'disulfide_bonds': '二硫键',
    'cation_pi': '阳离子-π', 'hydrophobic_moment': '疏水矩', 'aggregation': '聚集倾向',
}

PHYSICO_PROP_NAMES = [
    '疏水指数', '侧链体积', '电荷(pH7)', '柔性',
    'H键供体', 'H键受体', '螺旋偏好', '折叠偏好',
    '转角偏好', '芳香性', '二硫键', 'pKa',
]

# matplotlib 中文 & 样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#0d1117'
plt.rcParams['axes.edgecolor'] = '#30363d'
plt.rcParams['axes.labelcolor'] = '#8b949e'
plt.rcParams['text.color'] = '#c9d1d9'
plt.rcParams['xtick.color'] = '#484f58'
plt.rcParams['ytick.color'] = '#484f58'
plt.rcParams['grid.color'] = '#21262d'
plt.rcParams['grid.alpha'] = 0.6


def fig_to_image(fig, width=None):
    """将 matplotlib figure 转为 PNG bytes，用于 ipywidgets Image"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def get_gpu_info():
    info = {
        'available': torch.cuda.is_available(),
        'devices': [],
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'pytorch_version': torch.__version__,
    }
    if info['available']:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info['devices'].append({
                'name': props.name,
                'memory_gb': round(props.total_memory / 1e9, 1),
                'compute_capability': f"{props.major}.{props.minor}",
            })
    return info


def safe_eval(seq, evaluator):
    """安全调用 evaluator.evaluate_all，提供默认 exposure/distance"""
    L = len(seq)
    exposure = np.ones(L) * 0.5
    # 生成假的距离矩阵（对角线为0）
    distance = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        for j in range(i + 1, L):
            d = abs(i - j) * 3.8  # 近似Cα-Cα距离
            distance[i, j] = d
            distance[j, i] = d
    ss_types = ['C'] * L
    return evaluator.evaluate_all(seq, exposure, distance, ss_types)


# ========================================================
# JupyterApp 主类
# ========================================================

class JupyterApp:
    """FoldPath-LLM Jupyter 交互界面"""

    def __init__(self):
        self.device = DEVICE
        self.gpu_info = get_gpu_info()
        self.evaluator = PhysicochemicalEvaluator(device='cpu')

        # 模型 & 生成器
        self.model = None
        self.generator = None
        self.use_esm = True
        self.esm_model_name = "esm2_t12_35M_UR50D"
        self.esm_encoder = None
        self._model_loaded = False

        # 训练状态
        self._training_thread = None
        self._training_state = {
            'running': False,
            'current_epoch': 0,
            'current_batch': 0,
            'total_batches': 0,
            'total_epochs': 0,
            'batch_metrics': {},
            'history': {
                'train_loss': [], 'val_loss': [], 'train_seq_loss': [],
                'train_struct_loss': [], 'train_physico_loss': [],
                'learning_rate': [], 'epoch_time': [], 'precision': [], 'recall': [],
            },
            'log_lines': [],
        }
        self._state_lock = threading.Lock()

        # 生成结果缓存
        self.generated_sequences = []
        self.current_analysis = None
        self.current_analysis_seq = ""

        # ── 构建 UI ──
        self._build_ui()

    # ──────── 模型加载 ──────────────────────────────────

    def _load_or_create_model(self, use_esm=None, esm_model_name=None):
        """加载或创建模型"""
        if use_esm is None:
            use_esm = self.use_esm
        if esm_model_name is None:
            esm_model_name = self.esm_model_name

        # 检查是否需要重建
        if self._model_loaded and self.use_esm == use_esm and self.esm_model_name == esm_model_name:
            return

        self.use_esm = use_esm
        self.esm_model_name = esm_model_name

        if use_esm:
            self.esm_encoder = create_esm_encoder(
                model_name=esm_model_name,
                device=self.device,
                freeze=True,
                local_dir="pretrained"
            )
        else:
            self.esm_encoder = None

        self.model = FoldPathLLM(ModelConfig(), esm_encoder=self.esm_encoder).to(self.device)
        self.model.eval()
        self.generator = ProteinGenerator(model=self.model, device=self.device)

        params = self.model.get_param_count()
        self._model_loaded = True
        return params

    # ──────── UI 构建 ───────────────────────────────────

    def _build_ui(self):
        """构建完整的 ipywidgets 界面"""

        # ━━━━ 头部：标题 + GPU 信息 ━━━━━━━━━━━━━━
        self._header_title = widgets.HTML(
            value='<h2 style="margin:0;color:#58a6ff;">FoldPath-LLM <span style="font-size:14px;color:#8b949e;">蛋白质设计</span></h2>'
        )
        gpu_text = "GPU 检测中..."
        if self.gpu_info['available'] and self.gpu_info['devices']:
            d = self.gpu_info['devices'][0]
            gpu_text = f"🟢 {d['name']} ({d['memory_gb']} GB)"
        else:
            gpu_text = "⚫ 无GPU — 仅CPU"
        self._header_gpu = widgets.HTML(
            value=f'<div style="color:#8b949e;font-size:12px;">{gpu_text}</div>'
        )

        # ━━━━ Tab 切换 ━━━━━━━━━━━━━━━━━━━━━━━━━━
        self._tab = widgets.Tab()
        self._tab_train_content = widgets.VBox()
        self._tab_gen_content = widgets.VBox()
        self._tab_analysis_content = widgets.VBox()
        self._tab.children = [
            self._tab_train_content,
            self._tab_gen_content,
            self._tab_analysis_content,
        ]
        self._tab.set_title(0, '训练')
        self._tab.set_title(1, '生成')
        self._tab.set_title(2, '分析')

        # 构建各 Tab 内容
        self._build_train_tab()
        self._build_generate_tab()
        self._build_analysis_tab()

        # Tab 切换回调：刷新分析页（如果刚切换过来）
        self._tab.observe(self._on_tab_change, names='selected_index')

        # ━━━━ 组装主布局 ━━━━━━━━━━━━━━━━━━━━━━
        self.main_layout = widgets.VBox([
            widgets.HBox([self._header_title, self._header_gpu],
                         layout=widgets.Layout(justify_content='space-between', align_items='center',
                                               padding='8px 12px')),
            self._tab,
        ], layout=widgets.Layout(width='100%', padding='0'))

    # ──────── 训练 Tab ──────────────────────────────────

    def _build_train_tab(self):
        """训练页：左侧配置 + 右侧指标/图表/日志"""

        # ── 左侧 Sidebar ──
        self._cfg_epochs = widgets.IntText(value=10, min=1, max=200, description='训练轮次:', layout=widgets.Layout(width='200px'))
        self._cfg_batch = widgets.Dropdown(options=['4', '8', '16', '32', '64', '128', '256'], value='8', description='批次大小:', layout=widgets.Layout(width='200px'))
        self._cfg_lr = widgets.Text(value='1e-4', description='学习率:', layout=widgets.Layout(width='200px'))
        self._cfg_amp = widgets.Checkbox(value=True, description='混合精度')
        self._cfg_synth = widgets.Checkbox(value=False, description='合成数据')
        self._cfg_esm = widgets.Checkbox(value=True, description='ESM-2 基座')
        self._cfg_esm_model = widgets.Dropdown(
            options=[('ESM-2 8M (最小)', 'esm2_t6_8M_UR50D'),
                     ('ESM-2 35M (推荐)', 'esm2_t12_35M_UR50D'),
                     ('ESM-2 150M', 'esm2_t30_150M_UR50D'),
                     ('ESM-2 650M', 'esm2_t33_650M_UR50D')],
            value='esm2_t12_35M_UR50D',
            description='ESM 模型:',
            layout=widgets.Layout(width='250px')
        )
        self._cfg_esm.observe(lambda c: self._update_esm_visibility(), names='value')

        self._btn_start = widgets.Button(description='开始训练', button_style='primary',
                                         layout=widgets.Layout(width='120px'))
        self._btn_stop = widgets.Button(description='停止', button_style='danger',
                                        layout=widgets.Layout(width='120px'))
        self._btn_start.on_click(self._on_start_train)
        self._btn_stop.on_click(self._on_stop_train)

        self._status_device = widgets.HTML(value='<b>设备:</b> --')
        self._status_epoch = widgets.HTML(value='<b>轮次:</b> 0 / 0')
        self._status_batch = widgets.HTML(value='<b>批次:</b> 0 / 0')
        self._status_state = widgets.HTML(value='⚫ 空闲')
        self._status_progress = widgets.FloatProgress(value=0, min=0, max=100, description='进度:',
                                                      layout=widgets.Layout(width='100%'))

        self._model_info_html = widgets.HTML(value='<i>点击 ESM 基座选框加载模型信息...</i>')

        sidebar = widgets.VBox([
            widgets.HTML(value='<h4 style="color:#58a6ff;">训练配置</h4>'),
            self._cfg_epochs, self._cfg_batch, self._cfg_lr,
            self._cfg_amp, self._cfg_synth, self._cfg_esm, self._cfg_esm_model,
            widgets.HBox([self._btn_start, self._btn_stop]),
            widgets.HTML(value='<h4 style="color:#58a6ff;margin-top:8px;">运行状态</h4>'),
            self._status_device, self._status_epoch, self._status_batch, self._status_state,
            self._status_progress,
            widgets.HTML(value='<h4 style="color:#58a6ff;margin-top:8px;">模型信息</h4>'),
            self._model_info_html,
        ], layout=widgets.Layout(width='280px', padding='8px'))

        # ── 右侧 Main ──
        # 指标卡片行
        self._metric_loss = widgets.HTML(value='<div style="text-align:center"><div style="color:#8b949e;font-size:10px">损失</div><div style="font-size:20px;font-weight:bold;color:#ef5350">--</div></div>')
        self._metric_val = widgets.HTML(value='<div style="text-align:center"><div style="color:#8b949e;font-size:10px">验证损失</div><div style="font-size:20px;font-weight:bold;color:#ff9800">--</div></div>')
        self._metric_seq = widgets.HTML(value='<div style="text-align:center"><div style="color:#8b949e;font-size:10px">序列损失</div><div style="font-size:20px;font-weight:bold;color:#4dd0e1">--</div></div>')
        self._metric_struct = widgets.HTML(value='<div style="text-align:center"><div style="color:#8b949e;font-size:10px">结构损失</div><div style="font-size:20px;font-weight:bold;color:#b388ff">--</div></div>')
        self._metric_phys = widgets.HTML(value='<div style="text-align:center"><div style="color:#8b949e;font-size:10px">理化损失</div><div style="font-size:20px;font-weight:bold;color:#4caf50">--</div></div>')
        self._metric_prec = widgets.HTML(value='<div style="text-align:center"><div style="color:#8b949e;font-size:10px">精确率</div><div style="font-size:20px;font-weight:bold;color:#ff9800">--</div></div>')
        self._metric_rec = widgets.HTML(value='<div style="text-align:center"><div style="color:#8b949e;font-size:10px">召回率</div><div style="font-size:20px;font-weight:bold;color:#f06292">--</div></div>')
        metrics_row = widgets.HBox([
            self._metric_loss, self._metric_val, self._metric_seq,
            self._metric_struct, self._metric_phys, self._metric_prec, self._metric_rec,
        ], layout=widgets.Layout(justify_content='space-around', padding='4px'))

        # 图表区
        self._img_loss = widgets.Image(format='png', layout=widgets.Layout(width='49%'))
        self._img_pr = widgets.Image(format='png', layout=widgets.Layout(width='49%'))
        charts_row = widgets.HBox([self._img_loss, self._img_pr],
                                  layout=widgets.Layout(justify_content='space-between'))

        # 日志区
        self._log_output = widgets.Output(layout=widgets.Layout(height='200px', overflow_y='auto',
                                                                 border='1px solid #30363d'))

        main = widgets.VBox([
            metrics_row,
            charts_row,
            widgets.HTML(value='<h5 style="color:#8b949e;">训练日志</h5>'),
            self._log_output,
        ], layout=widgets.Layout(padding='8px', flex='1'))

        self._tab_train_content.children = [widgets.HBox([sidebar, main])]

    def _update_esm_visibility(self):
        self._cfg_esm_model.layout.display = '' if self._cfg_esm.value else 'none'
        self._refresh_model_info()

    def _refresh_model_info(self):
        try:
            params = self._load_or_create_model(
                use_esm=self._cfg_esm.value,
                esm_model_name=self._cfg_esm_model.value,
            )
            esm_name = self.esm_model_name if self._cfg_esm.value else '无 (From Scratch)'
            esm_total = sum(p.numel() for p in self.esm_encoder.parameters()) if self.esm_encoder else 0
            self._model_info_html.value = (
                f'<div style="font-size:11px;line-height:1.8;">'
                f'<b>总参数:</b> {params["total"]:,}<br>'
                f'<b>可训练:</b> {params["trainable"]:,}<br>'
                f'<b>基座:</b> {esm_name}<br>'
                f'<b>ESM参数:</b> {esm_total:,}'
                f'</div>'
            )
        except Exception as e:
            self._model_info_html.value = f'<i style="color:#ef5350;">加载失败: {e}</i>'

    # ──────── 训练逻辑 ──────────────────────────────────

    def _on_start_train(self, btn):
        if self._training_state['running']:
            return
        self._load_or_create_model()
        self._training_state['running'] = True
        self._training_state['history'] = {
            'train_loss': [], 'val_loss': [], 'train_seq_loss': [],
            'train_struct_loss': [], 'train_physico_loss': [],
            'learning_rate': [], 'epoch_time': [], 'precision': [], 'recall': [],
        }
        self._training_state['log_lines'] = []
        self._training_state['total_epochs'] = self._cfg_epochs.value
        self._btn_start.disabled = True
        self._status_state.value = '🟢 训练中'
        device_label = 'GPU' if self.gpu_info['available'] else 'CPU'
        self._status_device.value = f'<b>设备:</b> {device_label}'

        self._training_thread = threading.Thread(target=self._train_worker, daemon=True)
        self._training_thread.start()

    def _on_stop_train(self, btn):
        with self._state_lock:
            self._training_state['running'] = False
        self._status_state.value = '⏳ 停止中...'

    def _train_worker(self):
        """训练线程（不依赖 Flask SSE）"""
        try:
            device = self.device
            model = self.model
            model.train()

            epochs = self._cfg_epochs.value
            batch_size = int(self._cfg_batch.value)
            lr = float(self._cfg_lr.value)
            use_amp = self._cfg_amp.value
            use_synthetic = self._cfg_synth.value

            # 数据集
            dataset = ProteinDataset(split='train', synthetic=use_synthetic)
            train_loader = create_dataloader(dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=0, pin_memory=(device.type == 'cuda'))
            val_dataset = ProteinDataset(split='val', synthetic=use_synthetic)
            if len(val_dataset) == 0:
                val_dataset = ProteinDataset(split='train', synthetic=use_synthetic)
            val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False,
                                           num_workers=0, pin_memory=(device.type == 'cuda'))

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            total_steps = epochs * len(train_loader)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)
            scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == 'cuda') else None

            self._log_print(f"[启动] FoldPath-LLM 训练开始 ({epochs}轮, batch={batch_size})")
            best_val_loss = float('inf')

            for epoch in range(epochs):
                with self._state_lock:
                    if not self._training_state['running']:
                        break

                epoch_start = time.time()
                total_loss = 0; total_seq = 0; total_struct = 0; total_physico = 0
                total_prec = 0; total_rec = 0; nb = 0

                for batch_idx, batch in enumerate(train_loader):
                    with self._state_lock:
                        if not self._training_state['running']:
                            break

                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    target_ids = batch['target_ids'].to(device, non_blocking=True)
                    mask = batch['mask'].to(device, non_blocking=True)
                    sequences = batch.get('sequence', None)

                    if scaler is not None:
                        with autocast(device_type=device.type):
                            _, loss_dict, logits, _ = model(input_ids, target_ids, mask, sequences=sequences)
                            loss_val = loss_dict.get('loss_total', 0)
                        scaler.scale(loss_val).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        _, loss_dict, logits, _ = model(input_ids, target_ids, mask, sequences=sequences)
                        loss_val = loss_dict.get('loss_total', 0)
                        loss_val.backward()
                        if (batch_idx + 1) % 4 == 0:
                            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    # Token metrics
                    pred_ids = logits.argmax(dim=-1)
                    prec = ((pred_ids == target_ids) & mask.bool()).float().sum() / (mask.sum() + 1e-8)
                    rec = prec  # simplified
                    prec = prec.item(); rec = rec.item()

                    total_loss += loss_dict.get('loss_total', 0)
                    total_seq += loss_dict.get('loss_seq', 0)
                    total_struct += loss_dict.get('loss_struct', 0)
                    total_physico += loss_dict.get('loss_physico', 0)
                    total_prec += prec; total_rec += rec; nb += 1

                    with self._state_lock:
                        self._training_state['current_batch'] = batch_idx + 1
                        self._training_state['total_batches'] = len(train_loader)
                        self._training_state['batch_metrics'] = {
                            'loss': round(loss_dict.get('loss_total', 0), 4),
                            'seq_loss': round(loss_dict.get('loss_seq', 0), 4),
                            'struct_loss': round(loss_dict.get('loss_struct', 0), 4),
                            'physico_loss': round(loss_dict.get('loss_physico', 0), 4),
                            'precision': round(prec, 4),
                            'recall': round(rec, 4),
                        }

                    log_line = (f"E{epoch+1:02d} [{batch_idx+1}/{len(train_loader)}] "
                                f"Loss:{loss_dict.get('loss_total',0):.4f} "
                                f"Seq:{loss_dict.get('loss_seq',0):.4f} "
                                f"Struct:{loss_dict.get('loss_struct',0):.4f} "
                                f"Phys:{loss_dict.get('loss_physico',0):.4f} "
                                f"P:{prec:.3f} R:{rec:.3f}")
                    with self._state_lock:
                        self._training_state['log_lines'].append(log_line)

                    # 每 5 个 batch 刷新一次 UI
                    if batch_idx % 5 == 0:
                        self._refresh_training_ui(epoch, epoch_start)

                with self._state_lock:
                    if not self._training_state['running']:
                        break

                n = max(nb, 1)
                epoch_loss = total_loss / n
                epoch_seq = total_seq / n
                epoch_struct = total_struct / n
                epoch_physico = total_physico / n
                epoch_prec = total_prec / n
                epoch_rec = total_rec / n

                # 验证
                model.eval()
                vloss = 0; vnb = 0
                with torch.no_grad():
                    for vbatch in val_loader:
                        vid = vbatch['input_ids'].to(device, non_blocking=True)
                        vtid = vbatch['target_ids'].to(device, non_blocking=True)
                        vmask = vbatch['mask'].to(device, non_blocking=True)
                        vseq = vbatch.get('sequence', None)
                        if scaler is not None:
                            with autocast(device_type=device.type):
                                _, vld, _, _ = model(vid, vtid, vmask, sequences=vseq)
                        else:
                            _, vld, _, _ = model(vid, vtid, vmask, sequences=vseq)
                        vloss += vld.get('loss_total', 0); vnb += 1
                val_loss = vloss / max(vnb, 1)
                model.train()

                epoch_time_elapsed = time.time() - epoch_start
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss

                with self._state_lock:
                    h = self._training_state['history']
                    h['train_loss'].append(epoch_loss)
                    h['val_loss'].append(val_loss)
                    h['train_seq_loss'].append(epoch_seq)
                    h['train_struct_loss'].append(epoch_struct)
                    h['train_physico_loss'].append(epoch_physico)
                    h['learning_rate'].append(optimizer.param_groups[0]['lr'])
                    h['epoch_time'].append(epoch_time_elapsed)
                    h['precision'].append(epoch_prec)
                    h['recall'].append(epoch_rec)
                    self._training_state['current_epoch'] = epoch + 1

                self._log_print(
                    f"--- Epoch {epoch+1} 完成 --- "
                    f"Loss:{epoch_loss:.4f} Val:{val_loss:.4f} "
                    f"Seq:{epoch_seq:.4f} Struct:{epoch_struct:.4f} Phys:{epoch_physico:.4f} "
                    f"P:{epoch_prec:.3f} R:{epoch_rec:.3f} "
                    f"Time:{epoch_time_elapsed:.1f}s {'[BEST]' if is_best else ''}"
                )
                self._refresh_training_ui(epoch, epoch_start)

            # 训练结束
            model.eval()
            self._log_print("[完成] 训练结束！")
            with self._state_lock:
                self._training_state['running'] = False

        except Exception as e:
            self._log_print(f"[错误] {e}")
            import traceback
            traceback.print_exc()
            with self._state_lock:
                self._training_state['running'] = False
        finally:
            self._btn_start.disabled = False
            self._status_state.value = '⚫ 空闲'

    def _log_print(self, line):
        """线程安全地写入日志"""
        with self._log_output:
            print(line)

    def _refresh_training_ui(self, epoch, epoch_start):
        """刷新训练指标、图表"""
        with self._state_lock:
            state = copy.deepcopy(self._training_state)
            current_epoch = state.get('current_epoch', epoch + 1)
            total_epochs = state.get('total_epochs', self._cfg_epochs.value)
            batch_metrics = state.get('batch_metrics', {})
            history = state['history']

        self._status_epoch.value = f'<b>轮次:</b> {current_epoch} / {total_epochs}'
        pct = (current_epoch / total_epochs * 100) if total_epochs > 0 else 0
        self._status_progress.value = pct

        bm = batch_metrics
        self._metric_loss.value = self._make_metric_html('损失', bm.get('loss', '--'), '#ef5350')
        self._metric_seq.value = self._make_metric_html('序列损失', bm.get('seq_loss', '--'), '#4dd0e1')
        self._metric_struct.value = self._make_metric_html('结构损失', bm.get('struct_loss', '--'), '#b388ff')
        self._metric_phys.value = self._make_metric_html('理化损失', bm.get('physico_loss', '--'), '#4caf50')
        self._metric_prec.value = self._make_metric_html('精确率', bm.get('precision', '--'), '#ff9800')
        self._metric_rec.value = self._make_metric_html('召回率', bm.get('recall', '--'), '#f06292')

        # 验证损失（在epoch结束时更新）
        if history['val_loss']:
            self._metric_val.value = self._make_metric_html('验证损失', round(history['val_loss'][-1], 4), '#ff9800')

        # 更新图表
        if history['train_loss']:
            self._update_loss_chart(history)
            self._update_pr_chart(history)

    def _make_metric_html(self, label, value, color):
        v = f'{value:.4f}' if isinstance(value, (int, float)) else str(value)
        return f'<div style="text-align:center"><div style="color:#8b949e;font-size:10px">{label}</div><div style="font-size:20px;font-weight:bold;color:{color}">{v}</div></div>'

    def _update_loss_chart(self, history):
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        epochs = list(range(1, len(history['train_loss']) + 1))
        ax.plot(epochs, history['train_loss'], '#5b9cff', linewidth=2, label='Train Loss', marker='o', markersize=3)
        if history['val_loss']:
            ax.plot(epochs, history['val_loss'], '#ef5350', linewidth=2, label='Val Loss', marker='s', markersize=3)
        ax.plot(epochs, history['train_seq_loss'], '#4dd0e1', linewidth=1, linestyle='--', alpha=0.7, label='Seq')
        ax.plot(epochs, history['train_struct_loss'], '#b388ff', linewidth=1, linestyle='--', alpha=0.7, label='Struct')
        ax.plot(epochs, history['train_physico_loss'], '#4caf50', linewidth=1, linestyle='--', alpha=0.7, label='Phys')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('损失曲线', fontsize=11, color='#c9d1d9')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._img_loss.value = fig_to_image(fig)

    def _update_pr_chart(self, history):
        fig, ax1 = plt.subplots(figsize=(5.5, 3.5))
        epochs = list(range(1, len(history['train_loss']) + 1))
        ax1.plot(epochs, history['precision'], '#ff9800', linewidth=2, label='Precision', marker='o', markersize=3)
        ax1.plot(epochs, history['recall'], '#f06292', linewidth=2, label='Recall', marker='s', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Precision / Recall', color='#ff9800')
        ax1.tick_params(axis='y', labelcolor='#ff9800')
        ax1.set_ylim(0, 1.05)

        if history['learning_rate']:
            ax2 = ax1.twinx()
            ax2.plot(epochs, [lr * 1e4 for lr in history['learning_rate']],
                     '#ffd54f', linewidth=1.5, label='LR (×1e4)')
            ax2.set_ylabel('LR (×1e4)', color='#ffd54f')
            ax2.tick_params(axis='y', labelcolor='#ffd54f')

        ax1.set_title('精确率 / 召回率 / 学习率', fontsize=11, color='#c9d1d9')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center right')
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        self._img_pr.value = fig_to_image(fig)

    # ──────── 生成 Tab ──────────────────────────────────

    def _build_generate_tab(self):
        # 左侧配置
        self._gen_len = widgets.IntText(value=64, min=16, max=256, description='最大长度:', layout=widgets.Layout(width='200px'))
        self._gen_temp = widgets.FloatSlider(value=0.8, min=0.1, max=2.0, step=0.1, description='温度:', layout=widgets.Layout(width='250px'))
        self._gen_topk = widgets.IntSlider(value=50, min=1, max=200, step=5, description='Top-K:', layout=widgets.Layout(width='250px'))
        self._gen_topp = widgets.FloatSlider(value=0.95, min=0.5, max=1.0, step=0.05, description='Top-P:', layout=widgets.Layout(width='250px'))
        self._gen_num = widgets.IntText(value=3, min=1, max=20, description='生成数量:', layout=widgets.Layout(width='200px'))
        self._gen_filter = widgets.Checkbox(value=True, description='理化过滤')
        self._btn_gen = widgets.Button(description='生成序列', button_style='primary', layout=widgets.Layout(width='120px'))
        self._btn_gen.on_click(self._on_generate)

        self._custom_seq = widgets.Textarea(placeholder='例如: MKALIVLGLVLLSVTVQGK', layout=widgets.Layout(width='100%', height='60px'))
        self._btn_analyze_custom = widgets.Button(description='分析自定义序列', button_style='info', layout=widgets.Layout(width='140px'))
        self._btn_analyze_custom.on_click(self._on_analyze_custom)

        sidebar = widgets.VBox([
            widgets.HTML(value='<h4 style="color:#58a6ff;">生成配置</h4>'),
            self._gen_len, self._gen_temp, self._gen_topk, self._gen_topp, self._gen_num,
            self._gen_filter, self._btn_gen,
            widgets.HTML(value='<h4 style="color:#58a6ff;margin-top:8px;">自定义序列</h4>'),
            self._custom_seq, self._btn_analyze_custom,
        ], layout=widgets.Layout(width='280px', padding='8px'))

        # 右侧结果区
        self._gen_output = widgets.Output(layout=widgets.Layout(flex='1', padding='8px', overflow_y='auto'))
        with self._gen_output:
            display(HTML('<div style="text-align:center;color:#8b949e;padding:60px"><div style="font-size:40px">🧬</div><div>生成蛋白质序列</div><div style="font-size:12px">在左侧配置参数，然后点击生成</div></div>'))

        self._tab_gen_content.children = [widgets.HBox([sidebar, self._gen_output])]

    def _on_generate(self, btn):
        self._load_or_create_model()
        self._gen_output.clear_output()
        with self._gen_output:
            print("⏳ 正在生成...")

        try:
            config = GenerateConfig()
            config.max_length = self._gen_len.value
            config.temperature = self._gen_temp.value
            config.top_k = self._gen_topk.value
            config.top_p = self._gen_topp.value
            config.num_samples = self._gen_num.value
            config.use_physico_filter = self._gen_filter.value

            sequences, physico_scores = self.generator.generate(config)
            self.generated_sequences = sequences
            # 把每个序列转为字典格式（兼容分析）
            self.generated_sequences = [{'sequence': s, 'physico_scores': ps} for s, ps in zip(sequences, physico_scores)]

            self._gen_output.clear_output()
            with self._gen_output:
                # 显示结果
                html_parts = [f'<div style="color:#8b949e;margin-bottom:8px;">已在 <b style="color:#58a6ff;">{self.device.type.upper()}</b> 上生成 {len(sequences)} 条序列</div>']
                for i, (seq, ps) in enumerate(zip(sequences, physico_scores)):
                    chain_html = self._render_chain_html(seq)
                    html_parts.append(
                        f'<div style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px;margin-bottom:10px;">'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
                        f'<span style="font-weight:bold;color:#58a6ff;">序列 {i+1}</span>'
                        f'<span style="color:#8b949e;font-size:12px;">长度: {len(seq)}</span>'
                        f'</div>'
                        f'<div style="font-family:Consolas,monospace;font-size:11px;line-height:2;letter-spacing:1px;word-break:break-all;">{chain_html}</div>'
                        f'<div style="margin-top:6px;font-size:11px;color:#8b949e;">{seq}</div>'
                        f'</div>'
                    )
                display(HTML(''.join(html_parts)))

                # 分析按钮
                for i in range(len(sequences)):
                    btn_analyze = widgets.Button(description=f'分析序列 {i+1}', button_style='info',
                                                  layout=widgets.Layout(width='120px', margin='2px'))
                    btn_analyze.on_click(lambda b, idx=i: self._run_analysis_from_gen(idx))
                    display(btn_analyze)

        except Exception as e:
            self._gen_output.clear_output()
            with self._gen_output:
                print(f'❌ 错误: {e}')
            import traceback; traceback.print_exc()

    def _render_chain_html(self, seq):
        html = ''
        for aa in seq:
            c = AA_COLORS.get(aa, '#555')
            html += f'<span style="display:inline-block;background:{c};color:#fff;padding:1px 4px;border-radius:2px;font-weight:bold;min-width:14px;text-align:center;margin:1px;">{aa}</span>'
        return html

    def _render_chain_image(self, seq):
        """matplotlib 绘制蛋白质链可视化"""
        aa_width, aa_height, gap = 22, 28, 2
        max_cols = 40
        L = len(seq)
        if L <= max_cols:
            cols = L
            rows = 1
        else:
            cols = max_cols
            rows = (L + max_cols - 1) // max_cols

        fig_width = cols * (aa_width + gap) / 100 + 0.5
        fig_height = rows * (aa_height + gap) / 100 + 0.8
        fig, ax = plt.subplots(figsize=(max(fig_width, 6), max(fig_height, 2)))
        ax.set_xlim(0, cols * (aa_width + gap))
        ax.set_ylim(0, rows * (aa_height + gap))
        ax.set_aspect('equal')
        ax.axis('off')

        for i, aa in enumerate(seq):
            col = i % cols
            row = i // cols
            x = col * (aa_width + gap) + gap / 2
            y = (rows - 1 - row) * (aa_height + gap) + gap / 2  # 从上到下
            color = AA_COLORS.get(aa, '#555')
            rect = FancyBboxPatch((x, y), aa_width, aa_height,
                                  boxstyle='round,pad=1', facecolor=color,
                                  edgecolor='none', linewidth=0)
            ax.add_patch(rect)
            ax.text(x + aa_width / 2, y + aa_height / 2, aa,
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white', fontfamily='monospace')
            if (i + 1) % 10 == 0:
                ax.text(x + aa_width / 2, y - 4, str(i + 1),
                        ha='center', va='top', fontsize=7, color='#8b949e')

        fig.tight_layout(pad=0.3)
        return fig_to_image(fig)

    def _on_analyze_custom(self, btn):
        seq = self._custom_seq.value.strip()
        if not seq:
            with self._gen_output:
                print('请输入序列')
            return
        self._run_analysis(seq)

    def _run_analysis_from_gen(self, idx):
        if idx < len(self.generated_sequences):
            seq = self.generated_sequences[idx]['sequence']
            self._tab.selected_index = 2  # 切换到分析页
            self._run_analysis(seq)

    # ──────── 分析 Tab ──────────────────────────────────

    def _build_analysis_tab(self):
        self._an_seq_display = widgets.HTML(value='未加载序列', layout=widgets.Layout(word_break='break-all',
                                            font_family='Consolas,monospace', font_size='11px',
                                            max_height='80px', overflow_y='auto'))
        self._an_len = widgets.HTML(value='--')
        self._an_score = widgets.HTML(value='--')

        sidebar = widgets.VBox([
            widgets.HTML(value='<h4 style="color:#58a6ff;">当前序列</h4>'),
            self._an_seq_display,
            widgets.HTML(value='<h4 style="color:#58a6ff;">信息</h4>'),
            self._an_len, self._an_score,
        ], layout=widgets.Layout(width='280px', padding='8px'))

        # 主内容区：总分卡片 + 5类别 + 图表（用 VBox 装 Image widgets）
        self._an_output = widgets.VBox(layout=widgets.Layout(flex='1', padding='8px', overflow_y='auto'))
        self._tab_analysis_content.children = [widgets.HBox([sidebar, self._an_output])]

    def _run_analysis(self, sequence):
        self.current_analysis_seq = sequence
        self._an_seq_display.value = sequence
        self._an_len.value = f'<b>长度:</b> {len(sequence)}'

        # 加载模型用于评估
        try:
            self._load_or_create_model()
        except Exception:
            pass  # 即使模型没加载也能做理化评估

        try:
            result = safe_eval(sequence, self.evaluator)
            self.current_analysis = result
        except Exception as e:
            self.current_analysis = None
            self._an_output.children = [widgets.HTML(value=f'<div style="color:#ef5350;">分析错误: {e}</div>')]
            return

        r = result['results']
        cats = r['category_scores']
        total_pct = r['total_score'] * 100
        self._an_score.value = f'<b>总分:</b> {total_pct:.1f}%'

        # 构建分析内容
        children = []

        # 总分卡片
        children.append(widgets.HTML(
            f'<div style="background:#161b22;border:1px solid #58a6ff;border-radius:8px;padding:14px;'
            f'text-align:center;margin-bottom:10px;">'
            f'<div style="color:#8b949e;font-size:13px;">理化性质总分</div>'
            f'<div style="font-size:28px;font-weight:bold;color:#58a6ff;">{total_pct:.1f}%</div>'
            f'</div>'
        ))

        # 5 类别分数
        cat_html = '<div style="display:flex;gap:8px;margin-bottom:10px;">'
        for name, val in cats.items():
            c = CAT_COLORS.get(name, '#58a6ff')
            v = val * 100
            cat_html += f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;text-align:center;flex:1;">'
            cat_html += f'<div style="color:#8b949e;font-size:10px;">{name}</div>'
            cat_html += f'<div style="font-size:20px;font-weight:bold;color:{c};">{v:.0f}%</div>'
            cat_html += f'</div>'
        cat_html += '</div>'
        children.append(widgets.HTML(cat_html))

        # 链可视化
        chain_img = widgets.Image(value=self._render_chain_image(sequence), format='png',
                                  layout=widgets.Layout(width='100%'))
        children.append(widgets.HTML('<div style="color:#8b949e;font-size:12px;margin-bottom:4px;">蛋白质链可视化</div>'))
        children.append(chain_img)

        # 图例
        legend_html = '<div style="display:flex;flex-wrap:wrap;gap:10px;margin:8px 0;">'
        for color, label in AA_COLORS_GROUP.items():
            legend_html += f'<div style="display:flex;align-items:center;gap:4px;font-size:10px;color:#8b949e;">'
            legend_html += f'<div style="width:10px;height:10px;background:{color};border-radius:2px;"></div>{label}</div>'
        legend_html += '</div>'
        children.append(widgets.HTML(legend_html))

        # 图表行1: 雷达图 + 热力图
        fig_radar, fig_heatmap = self._plot_radar_and_heatmap(cats)
        children.append(widgets.HBox([
            widgets.Image(value=fig_to_image(fig_radar), format='png', layout=widgets.Layout(width='49%')),
            widgets.Image(value=fig_to_image(fig_heatmap), format='png', layout=widgets.Layout(width='49%')),
        ]))

        # 图表行2: 15指标 + 距离矩阵
        fig_indicators, fig_dist = self._plot_indicators_and_dist(r, result.get('distance_matrix'), sequence)
        children.append(widgets.HBox([
            widgets.Image(value=fig_to_image(fig_indicators), format='png', layout=widgets.Layout(width='49%')),
            widgets.Image(value=fig_to_image(fig_dist), format='png', layout=widgets.Layout(width='49%')),
        ]))

        # 图表行3: 暴露曲线 + AA组成
        fig_exp, fig_comp = self._plot_exposure_and_comp(result.get('exposure'), sequence)
        children.append(widgets.HBox([
            widgets.Image(value=fig_to_image(fig_exp), format='png', layout=widgets.Layout(width='49%')),
            widgets.Image(value=fig_to_image(fig_comp), format='png', layout=widgets.Layout(width='49%')),
        ]))

        # 15指标详细列表
        ind_html = '<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;margin-top:8px;">'
        ind_html += '<div style="color:#8b949e;font-size:12px;margin-bottom:6px;">详细指标评分</div>'
        ind_html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">'
        for key, name in INDICATOR_NAMES.items():
            val = r.get(key, 0)
            color = '#4caf50' if val >= 0.7 else '#ff9800' if val >= 0.4 else '#ef5350'
            ind_html += f'<div style="display:flex;align-items:center;gap:6px;font-size:10px;">'
            ind_html += f'<span style="color:#8b949e;min-width:80px;">{name}</span>'
            ind_html += f'<div style="flex:1;height:5px;background:#21262d;border-radius:3px;overflow:hidden;">'
            ind_html += f'<div style="width:{val*100}%;height:100%;background:{color};border-radius:3px;"></div></div>'
            ind_html += f'<span style="color:{color};font-weight:bold;min-width:32px;text-align:right;">{val*100:.0f}%</span>'
            ind_html += f'</div>'
        ind_html += '</div></div>'
        children.append(widgets.HTML(ind_html))

        self._an_output.children = children

    def _plot_radar_and_heatmap(self, cats):
        """雷达图 + 理化性质热力图"""
        # 雷达图
        labels = list(cats.keys())
        values = [cats[k] for k in labels]
        N = len(labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig_radar, ax = plt.subplots(figsize=(4, 3.5), subplot_kw=dict(polar=True))
        ax.set_facecolor('#0d1117')
        ax.fill(angles, values, alpha=0.15, color='#5b9cff')
        ax.plot(angles, values, color='#5b9cff', linewidth=2)
        ax.scatter(angles[:-1], values[:-1], color='#5b9cff', s=30, zorder=5)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8, color='#8b949e')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7, color='#555')
        ax.set_title('类别雷达图', fontsize=10, color='#c9d1d9', pad=12)
        ax.grid(color='#21262d', alpha=0.6)
        fig_radar.tight_layout()

        # 热力图
        n_aa = len(AMINO_ACIDS)
        n_prop = PHYSICO_MATRIX.shape[1]
        fig_heatmap, ax_hm = plt.subplots(figsize=(6, 4))
        # 每列归一化
        mat = PHYSICO_MATRIX.copy()
        col_min = mat.min(axis=0, keepdims=True)
        col_max = mat.max(axis=0, keepdims=True)
        mat_norm = (mat - col_min) / (col_max - col_min + 1e-8)
        im = ax_hm.imshow(mat_norm, aspect='auto', cmap='RdYlBu_r')
        ax_hm.set_xticks(range(n_prop))
        ax_hm.set_xticklabels(PHYSICO_PROP_NAMES, rotation=45, ha='right', fontsize=7, color='#8b949e')
        ax_hm.set_yticks(range(n_aa))
        ax_hm.set_yticklabels(list(AMINO_ACIDS), fontsize=9, fontweight='bold')
        # 颜色标记氨基酸
        for i, aa in enumerate(AMINO_ACIDS):
            ax_hm.get_yticklabels()[i].set_color(AA_COLORS.get(aa, '#888'))
        # 数值标注（小网格）
        for i in range(n_aa):
            for j in range(n_prop):
                ax_hm.text(j, i, f'{mat[i,j]:.1f}', ha='center', va='center', fontsize=5.5, color='#111')
        ax_hm.set_title('理化性质矩阵 (20种AA × 12项性质)', fontsize=10, color='#c9d1d9')
        fig_heatmap.tight_layout()

        return fig_radar, fig_heatmap

    def _plot_indicators_and_dist(self, r, distance_matrix, seq):
        """15指标横向条形图 + 距离矩阵热力图"""
        # 指标图
        keys = list(INDICATOR_NAMES.keys())
        names = [INDICATOR_NAMES[k] for k in keys]
        values = [r.get(k, 0) for k in keys]
        colors = ['#4caf50' if v >= 0.7 else '#ff9800' if v >= 0.4 else '#ef5350' for v in values]

        fig_ind, ax = plt.subplots(figsize=(5, 4))
        bars = ax.barh(names, values, color=[c + '88' for c in colors], edgecolor=colors, linewidth=1, height=0.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel('评分', fontsize=9)
        ax.set_title('15项理化指标', fontsize=10, color='#c9d1d9')
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(axis='x', alpha=0.3)
        for bar, val, c in zip(bars, values, colors):
            ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{val*100:.0f}%', va='center', fontsize=7, color=c)
        fig_ind.tight_layout()

        # 距离矩阵
        if distance_matrix is not None and len(distance_matrix) > 0:
            L = min(len(distance_matrix), 80)
            sub_dist = np.array(distance_matrix)[:L, :L]
            max_d = sub_dist.max() or 1
            sub_dist_norm = sub_dist / max_d
        else:
            L = min(len(seq), 30)
            sub_dist_norm = np.zeros((L, L))

        fig_dist, ax_d = plt.subplots(figsize=(4, 3.5))
        im = ax_d.imshow(sub_dist_norm, cmap='hot_r', aspect='equal')
        ax_d.set_title('距离矩阵热力图', fontsize=10, color='#c9d1d9')
        ax_d.set_xlabel('残基位置', fontsize=8)
        ax_d.set_ylabel('残基位置', fontsize=8)
        fig_dist.tight_layout()

        return fig_ind, fig_dist

    def _plot_exposure_and_comp(self, exposure, seq):
        """暴露曲线 + AA组成"""
        L = len(seq)
        if exposure is None:
            exposure = np.ones(L) * 0.5

        step = max(1, L // 100)
        xs = list(range(0, L, step))
        ys = [exposure[i] if i < len(exposure) else 0.5 for i in xs]

        fig_exp, ax = plt.subplots(figsize=(5, 3))
        ax.fill_between(xs, ys, alpha=0.15, color='#4dd0e1')
        ax.plot(xs, ys, color='#4dd0e1', linewidth=1.5)
        ax.set_xlabel('位置', fontsize=9)
        ax.set_ylabel('暴露度', fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title('溶剂暴露曲线', fontsize=10, color='#c9d1d9')
        ax.grid(alpha=0.3)
        fig_exp.tight_layout()

        # AA组成
        AAs = list(AMINO_ACIDS)
        cnt = {aa: seq.count(aa) for aa in AAs}
        total = max(len(seq), 1)
        freqs = [cnt[aa] / total for aa in AAs]
        colors = [AA_COLORS.get(aa, '#888') + 'CC' for aa in AAs]
        colors_edge = [AA_COLORS.get(aa, '#888') for aa in AAs]

        fig_comp, ax_c = plt.subplots(figsize=(5, 3))
        ax_c.bar(AAs, freqs, color=colors, edgecolor=colors_edge, linewidth=0.5)
        ax_c.set_xlabel('氨基酸', fontsize=9)
        ax_c.set_ylabel('频率', fontsize=9)
        ax_c.set_title('氨基酸组成分布', fontsize=10, color='#c9d1d9')
        ax_c.grid(axis='y', alpha=0.3)
        # 颜色标记
        for i, aa in enumerate(AAs):
            ax_c.get_xticklabels()[i].set_color(AA_COLORS.get(aa, '#888'))
        fig_comp.tight_layout()

        return fig_exp, fig_comp

    # ──────── Tab 切换回调 ──────────────────────────────

    def _on_tab_change(self, change):
        idx = change['new']
        if idx == 0 and not self._model_loaded:
            # 自动刷新模型信息
            try:
                self._refresh_model_info()
            except Exception:
                pass

    # ──────── 公共接口 ──────────────────────────────────

    def display(self):
        """在 Jupyter Cell 中渲染整个界面"""
        display(self.main_layout)

        # 初始加载模型信息
        try:
            self._refresh_model_info()
        except Exception:
            pass


# ========================================================
# 快捷启动函数
# ========================================================

def launch():
    """
    在 Jupyter Notebook/Cell 中启动可视化界面。
    使用方法:
        from jupyter_app import launch
        launch()
    """
    app = JupyterApp()
    app.display()
    return app


if __name__ == '__main__':
    # 命令行模式：仍可使用
    print("FoldPath-LLM Jupyter Interface")
    print("在 Jupyter Notebook 中运行: from jupyter_app import launch; launch()")
    launch()