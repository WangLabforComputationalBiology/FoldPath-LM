"""
APBC Figures: Ablation, Novelty, Foldability Boxplot
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BG='white'; NAVY='#1A2A3C'; SLATE='#5A7184'; AMBER='#C4953A'; GRAY='#999999'

# ═══════ Fig4: Ablation ═══════
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), facecolor=BG)
metrics = ['Naturalness', 'pTM (best)', 'P/R (Epoch 5)']
full_vals = [0.514, 0.619, 0.416]
nostruct_vals = [0.560, 0.459, 0.268]

for i, (ax, metric, fv, nv) in enumerate(zip(axes, metrics, full_vals, nostruct_vals)):
    x = [0, 1]
    bars = ax.bar(x, [fv, nv], color=[NAVY, GRAY], width=0.4, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(['Full', 'No Struct'], fontsize=10)
    ax.set_title(metric, fontsize=12, fontweight='bold', color=NAVY)
    for bar, val in zip(bars, [fv, nv]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(fv,nv)*1.25)
    ax.grid(axis='y', alpha=0.3)
    if metric == 'Naturalness':
        ax.text(0.5, max(fv,nv)*1.18, 'NoStruct higher', ha='center', fontsize=8, color=GRAY)
    else:
        ax.text(0.0, max(fv,nv)*1.18, 'FoldPath higher', ha='center', fontsize=8, color=NAVY)

fig.suptitle('Ablation: Full Model vs. No Structure Track', fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('fig_ablation.png', dpi=150, facecolor=BG, edgecolor='none', bbox_inches='tight')
plt.close(fig)
print('Saved: fig_ablation.png')

# ═══════ Fig5: Novelty ═══════
fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=BG)
bins = ['0-20%', '20-40%', '40-60%', '60-100%']
counts = [48, 1, 1, 0]
colors = [NAVY, SLATE, AMBER, '#DDD']
bars = ax.bar(bins, counts, color=colors, width=0.5, edgecolor='white')
for bar, count in zip(bars, counts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{count}', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Sequences', fontsize=11)
ax.set_title('Sequence Novelty Distribution (n=50)', fontsize=12, fontweight='bold', color=NAVY)
ax.set_ylim(0, 55)
ax.text(0.5, 50, 'Mean: 14.6% | Median: 12.8% | Max: 57.8%\n96% below strict 20% novelty threshold',
        ha='center', fontsize=9, color='#666',
        bbox=dict(boxstyle='round', facecolor='#fafafa', edgecolor='#ccc', alpha=0.8))
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('fig_novelty.png', dpi=150, facecolor=BG, edgecolor='none', bbox_inches='tight')
plt.close(fig)
print('Saved: fig_novelty.png')

# ═══════ Fig6: AF2 Boxplot ═══════
fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
fp_plddt = [51.7, 52.3, 51.0, 63.2, 55.8, 70.9, 75.1, 55.1, 47.2, 35.2, 61.9, 82.4, 60.6, 50.1, 64.1, 46.0, 82.1, 34.9, 71.7, 36.6]
ns_plddt = [45.2, 45.3, 44.8, 39.6, 73.0, 52.5, 50.0, 41.8, 32.8, 49.0, 36.5, 32.9, 32.9, 32.9, 30.7, 61.2, 44.4, 43.2, 43.5, 49.2]

bp = ax.boxplot([fp_plddt, ns_plddt], labels=['FoldPath-LLM', 'NoStruct'],
                patch_artist=True, widths=0.4,
                medianprops=dict(color='white', linewidth=2))
bp['boxes'][0].set_facecolor(NAVY); bp['boxes'][0].set_alpha(0.8)
bp['boxes'][1].set_facecolor(GRAY); bp['boxes'][1].set_alpha(0.7)

np.random.seed(42)
for i, (data, color) in enumerate(zip([fp_plddt, ns_plddt], [NAVY, GRAY])):
    x = np.random.normal(i+1, 0.06, len(data))
    ax.scatter(x, data, c=color, alpha=0.7, s=40, edgecolors='white', linewidth=0.5, zorder=3)

ax.axhline(y=50, color='#ccc', linestyle='--', linewidth=1, alpha=0.5)
ax.text(2.3, 51, 'pLDDT=50', fontsize=8, color='#999')
ax.set_ylabel('Mean pLDDT', fontsize=11)
ax.set_title('AlphaFold2 Foldability Comparison (n=20 each)', fontsize=12, fontweight='bold', color=NAVY)

from scipy import stats
u, p = stats.mannwhitneyu(fp_plddt, ns_plddt, alternative='greater')
ax.text(0.5, 0.95, f'FoldPath mean: {np.mean(fp_plddt):.1f}\nNoStruct mean: {np.mean(ns_plddt):.1f}\nMann-Whitney p<0.001',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='#fafafa', edgecolor='#ccc', alpha=0.8))
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('fig_foldability_boxplot.png', dpi=150, facecolor=BG, edgecolor='none', bbox_inches='tight')
plt.close(fig)
print('Saved: fig_foldability_boxplot.png')
