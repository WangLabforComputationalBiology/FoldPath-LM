"""
FoldPath-LLM: Training Curve Plotting (YOLO-Style)
训练完成后生成 6 张独立曲线图 + 1 张汇总图
"""

import json
import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── 中文字体支持 ──
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_all_curves(history, save_dir="logs", model_info=None):
    """
    生成全部训练曲线图和汇总图。
    
    Args:
        history: dict, 包含 train_loss, val_loss, train_seq_loss, 
                 train_struct_loss, train_physico_loss, learning_rate, 
                 epoch_time 等列表
        save_dir: 保存目录
        model_info: dict, 可选，模型信息如 {'use_esm': True, 'esm_model': 'esm2_t12_35M_UR50D', 
                     'total_params': 35000000, 'trainable_params': 2400000}
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = list(range(1, len(history.get('train_loss', [])) + 1))
    if not epochs:
        print("[plot_results] 无训练数据，跳过绘图")
        return

    model_info = model_info or {}
    best_idx = np.argmin(history.get('val_loss', history.get('train_loss', [0])))
    best_val = history.get('val_loss', history.get('train_loss', [0]))[best_idx] if history.get('val_loss') else None

    # ── 1. Train + Val Loss ──
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'train_loss' in history and history['train_loss']:
        ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    if 'val_loss' in history and history['val_loss']:
        ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
        if best_idx < len(epochs):
            ax.axvline(x=epochs[best_idx], color='green', linestyle='--', alpha=0.5, label=f'Best Epoch ({epochs[best_idx]})')
            ax.scatter([epochs[best_idx]], [best_val], color='green', s=80, zorder=5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('FoldPath-LLM: Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, '01_loss_train_val.png'), dpi=150)
    plt.close(fig)

    # ── 2. Sub-component Losses (Seq / Struct / Physico) ──
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'train_seq_loss': ('#4dd0e1', 'Sequence Loss'),
              'train_struct_loss': ('#b388ff', 'Structure Loss'),
              'train_physico_loss': ('#4caf50', 'Physicochemical Loss')}
    for key, (color, label) in colors.items():
        if key in history and history[key]:
            ax.plot(epochs, history[key], color=color, linewidth=2, label=label, marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('FoldPath-LLM: Sub-Component Losses (Seq / Struct / Physico)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, '02_loss_subcomponents.png'), dpi=150)
    plt.close(fig)

    # ── 3. Precision + Recall ──
    fig, ax = plt.subplots(figsize=(10, 6))
    if history.get('precision'):
        ax.plot(epochs, history['precision'], '#ff9800', linewidth=2, label='Precision', marker='o', markersize=4)
    if history.get('recall'):
        ax.plot(epochs, history['recall'], '#f06292', linewidth=2, label='Recall', marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('FoldPath-LLM: Token Precision & Recall', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, '03_precision_recall.png'), dpi=150)
    plt.close(fig)

    # ── 4. Learning Rate ──
    fig, ax = plt.subplots(figsize=(10, 6))
    if history.get('learning_rate'):
        ax.plot(epochs, history['learning_rate'], '#ffd54f', linewidth=2, marker='o', markersize=4)
        ax.fill_between(epochs, 0, history['learning_rate'], alpha=0.15, color='#ffd54f')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('FoldPath-LLM: Cosine Annealing Learning Rate', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, '04_learning_rate.png'), dpi=150)
    plt.close(fig)

    # ── 5. Epoch Time ──
    fig, ax = plt.subplots(figsize=(10, 6))
    if history.get('epoch_time'):
        colors_bar = plt.cm.Blues(np.linspace(0.4, 0.9, len(epochs)))
        ax.bar(epochs, history['epoch_time'], color=colors_bar, edgecolor='#1565c0', linewidth=0.5)
        avg_time = np.mean(history['epoch_time'])
        ax.axhline(y=avg_time, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Average: {avg_time:.1f}s')
        ax.legend(fontsize=10)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('FoldPath-LLM: Training Time per Epoch', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, '05_epoch_time.png'), dpi=150)
    plt.close(fig)

    # ── 6. Final Metrics Table ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    metrics = []
    if history.get('train_loss'):
        metrics.append(('Final Train Loss', f"{history['train_loss'][-1]:.4f}"))
    if history.get('val_loss'):
        metrics.append(('Final Val Loss', f"{history['val_loss'][-1]:.4f}"))
    if best_val is not None:
        metrics.append(('Best Val Loss', f"{best_val:.4f} (Epoch {epochs[best_idx]})"))
    if history.get('train_seq_loss'):
        metrics.append(('Final Seq Loss', f"{history['train_seq_loss'][-1]:.4f}"))
    if history.get('train_struct_loss'):
        metrics.append(('Final Struct Loss', f"{history['train_struct_loss'][-1]:.4f}"))
    if history.get('train_physico_loss'):
        metrics.append(('Final Physico Loss', f"{history['train_physico_loss'][-1]:.4f}"))
    if history.get('precision'):
        metrics.append(('Final Precision', f"{history['precision'][-1]:.4f}"))
    if history.get('recall'):
        metrics.append(('Final Recall', f"{history['recall'][-1]:.4f}"))
    if history.get('epoch_time'):
        metrics.append(('Total Time', f"{sum(history['epoch_time']):.1f}s ({sum(history['epoch_time'])/60:.1f} min)"))
    if model_info:
        if model_info.get('use_esm'):
            metrics.append(('Base Model', f"ESM-2 ({model_info.get('esm_model', 'Unknown')})"))
        else:
            metrics.append(('Base Model', 'From Scratch'))
        metrics.append(('Trainable Params', f"{model_info.get('trainable_params', 'N/A'):,}"))
        metrics.append(('Total Params', f"{model_info.get('total_params', 'N/A'):,}"))

    if metrics:
        col_labels = ['Metric', 'Value']
        table_data = [[m[0], m[1]] for m in metrics]
        table = ax.table(cellText=table_data, colLabels=col_labels,
                         cellLoc='left', loc='center',
                         colWidths=[0.45, 0.45])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        for key, cell in table.get_celld().items():
            cell.set_edgecolor('#aaaaaa')
            if key[1] == 0:
                cell.set_facecolor('#1a1a2e')
                cell.set_text_props(weight='bold', color='white')
            elif key[0] == 0:
                cell.set_facecolor('#f5f5f5')
            elif key[0] == 1:
                cell.set_facecolor('#fff3e0')
            elif key[0] == 2:
                cell.set_facecolor('#e8f5e9')

    ax.set_title('FoldPath-LLM: Final Training Metrics', fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, '06_metric_table.png'), dpi=150)
    plt.close(fig)

    # ── 7. Summary Figure (YOLO-style 2×3 grid) ──
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('FoldPath-LLM Training Results', fontsize=18, fontweight='bold', y=0.98)
    
    # Subtitle
    subtitle_parts = []
    if model_info.get('use_esm'):
        subtitle_parts.append(f"Base: ESM-2 {model_info.get('esm_model', '')}")
    else:
        subtitle_parts.append("Base: From Scratch")
    subtitle_parts.append(f"Epochs: {len(epochs)}")
    if best_val is not None:
        subtitle_parts.append(f"Best Val Loss: {best_val:.4f}")
    fig.text(0.5, 0.94, " | ".join(subtitle_parts), ha='center', fontsize=11, color='gray')

    # Panel 1: Train + Val Loss
    ax1 = fig.add_subplot(2, 3, 1)
    if history.get('train_loss'):
        ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train')
    if history.get('val_loss'):
        ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val')
    if best_idx < len(epochs) and history.get('val_loss'):
        ax1.axvline(x=epochs[best_idx], color='green', linestyle='--', alpha=0.5)
    ax1.set_title('A. Train & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Sub-components
    ax2 = fig.add_subplot(2, 3, 2)
    for key, (color, label) in [('train_seq_loss', ('#4dd0e1', 'Seq')),
                                 ('train_struct_loss', ('#b388ff', 'Struct')),
                                 ('train_physico_loss', ('#4caf50', 'Physico'))]:
        if key in history and history[key]:
            ax2.plot(epochs, history[key], color=color, linewidth=1.8, label=label)
    ax2.set_title('B. Sub-Component Losses', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Precision + Recall
    ax3 = fig.add_subplot(2, 3, 3)
    if history.get('precision'):
        ax3.plot(epochs, history['precision'], '#ff9800', linewidth=2, label='Precision')
    if history.get('recall'):
        ax3.plot(epochs, history['recall'], '#f06292', linewidth=2, label='Recall')
    ax3.set_title('C. Token Precision & Recall', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)

    # Panel 4: Learning Rate
    ax4 = fig.add_subplot(2, 3, 4)
    if history.get('learning_rate'):
        ax4.plot(epochs, history['learning_rate'], '#ffd54f', linewidth=2)
        ax4.fill_between(epochs, 0, history['learning_rate'], alpha=0.15, color='#ffd54f')
    ax4.set_title('D. Learning Rate (Cosine Annealing)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Panel 5: Epoch Time
    ax5 = fig.add_subplot(2, 3, 5)
    if history.get('epoch_time'):
        ax5.bar(epochs, history['epoch_time'], color=plt.cm.Blues(np.linspace(0.4, 0.9, len(epochs))))
        avg_t = np.mean(history['epoch_time'])
        ax5.axhline(y=avg_t, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Avg: {avg_t:.1f}s')
        ax5.legend(fontsize=9)
    ax5.set_title('E. Epoch Time', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # Panel 6: Summary Table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    summary_lines = []
    summary_lines.append(f"Best Val Loss:    {best_val:.4f}" if best_val is not None else "Best Val Loss: N/A")
    if history.get('train_loss'):
        summary_lines.append(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    if history.get('train_seq_loss'):
        summary_lines.append(f"Final Seq Loss:   {history['train_seq_loss'][-1]:.4f}")
    if history.get('train_struct_loss'):
        summary_lines.append(f"Final Struct Loss:{history['train_struct_loss'][-1]:.4f}")
    if history.get('train_physico_loss'):
        summary_lines.append(f"Final Physico Loss:{history['train_physico_loss'][-1]:.4f}")
    if history.get('precision') and history.get('recall'):
        summary_lines.append(f"P={history['precision'][-1]:.3f} / R={history['recall'][-1]:.3f}")
    if history.get('epoch_time'):
        summary_lines.append(f"Total Time: {sum(history['epoch_time']):.1f}s")
    if model_info:
        if model_info.get('use_esm'):
            summary_lines.append(f"Base: ESM-2 {model_info.get('esm_model', '')}")
        else:
            summary_lines.append("Base: From Scratch")
        summary_lines.append(f"Trainable: {model_info.get('trainable_params', 0):,} / {model_info.get('total_params', 0):,}")
    for i, line in enumerate(summary_lines):
        ax6.text(0.1, 0.9 - i * 0.09, line, transform=ax6.transAxes, fontsize=11, fontfamily='monospace')
    ax6.set_title('F. Final Metrics', fontsize=12, fontweight='bold')

    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(os.path.join(save_dir, 'training_results_summary.png'), dpi=150)
    plt.close(fig)

    print(f"[plot_results] 已生成 7 张训练曲线图到 {save_dir}/")


def main():
    """从 history JSON 文件生成曲线图（用于事后补生成）"""
    import argparse
    parser = argparse.ArgumentParser(description='FoldPath-LLM Plot Curves')
    parser.add_argument('--history', type=str, default='logs/training_history.json',
                        help='训练历史 JSON 文件路径')
    parser.add_argument('--save-dir', type=str, default='logs',
                        help='图片保存目录')
    args = parser.parse_args()

    if not os.path.exists(args.history):
        print(f"[ERROR] 找不到历史文件: {args.history}")
        return

    with open(args.history, 'r') as f:
        history = json.load(f)

    plot_all_curves(history, save_dir=args.save_dir)


if __name__ == '__main__':
    main()