"""
FoldPath-LLM architecture diagram in restrained APBC paper style.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})


def add_box(ax, xy, w, h, text, face, edge="#374151", fontsize=8):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end, color="#374151", style="-"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=1.0,
        color=color,
        linestyle=style,
        connectionstyle="arc3,rad=0",
        shrinkA=0,
        shrinkB=0,
        capstyle="butt",
        joinstyle="miter",
    )
    ax.add_patch(patch)


def polyline(ax, points, color="#6b7280", style="--"):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=1.0,
                ls=style, solid_capstyle="butt", dash_capstyle="butt")
    arrow(ax, points[-2], points[-1], color=color, style=style)


def main():
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    ax.set_xlim(0, 1.20)
    ax.set_ylim(0, 1)
    ax.axis("off")

    frozen = "#dbeafe"
    trainable = "#fed7aa"
    aux = "#f3f4f6"
    output = "#e5e7eb"

    ax.text(0.02, 0.94, "Training-time dual-track learning", fontsize=10, weight="bold")
    train_boxes = [
        ((0.03, 0.76), 0.12, 0.09, "Protein\nsequence", output),
        ((0.21, 0.82), 0.14, 0.08, "Frozen RITA_m\nembeddings", frozen),
        ((0.21, 0.64), 0.14, 0.08, "Token + physico\nfeatures", trainable),
        ((0.43, 0.82), 0.15, 0.08, "Structure track\nbidirectional", aux),
        ((0.43, 0.64), 0.16, 0.08, "Sequence track\ncausal Transformer", trainable),
        ((0.66, 0.82), 0.14, 0.08, "Exposure / SS\nDistance heads", aux),
        ((0.66, 0.64), 0.14, 0.08, "Structure-aware\nattention", trainable),
        ((0.88, 0.82), 0.14, 0.08, "Structural +\nchemical losses", aux),
        ((0.88, 0.64), 0.14, 0.08, "Token logits\nnext residue", output),
        ((1.08, 0.64), 0.09, 0.08, "Generated\nsequence", output),
    ]
    for xy, w, h, text, face in train_boxes:
        add_box(ax, xy, w, h, text, face)

    arrow(ax, (0.162, 0.815), (0.198, 0.860))
    arrow(ax, (0.162, 0.785), (0.198, 0.680))
    arrow(ax, (0.362, 0.860), (0.418, 0.860), color="#6b7280")
    arrow(ax, (0.362, 0.680), (0.418, 0.680))
    arrow(ax, (0.592, 0.860), (0.648, 0.860), color="#6b7280")
    arrow(ax, (0.602, 0.680), (0.648, 0.680))
    arrow(ax, (0.812, 0.860), (0.868, 0.860), color="#6b7280")
    arrow(ax, (0.812, 0.680), (0.868, 0.680))
    arrow(ax, (1.032, 0.680), (1.068, 0.680))
    arrow(ax, (0.505, 0.812), (0.700, 0.728), color="#6b7280", style="--")
    ax.text(0.59, 0.785, "B_struct", fontsize=8, color="#6b7280")
    polyline(ax, [(0.280, 0.632), (0.280, 0.560), (0.730, 0.560), (0.730, 0.628)], color="#6b7280", style="--")
    ax.text(0.49, 0.535, "B_chem", fontsize=8, color="#6b7280")

    ax.plot([0.02, 1.17], [0.47, 0.47], color="#d1d5db", lw=0.8)
    ax.text(0.02, 0.40, "Sequence-only inference", fontsize=10, weight="bold")
    infer_boxes = [
        ((0.03, 0.22), 0.12, 0.09, "Protein\nsequence", output),
        ((0.24, 0.22), 0.15, 0.09, "Token + physico\nfeatures", trainable),
        ((0.50, 0.22), 0.17, 0.09, "Causal sequence\ntrack only", trainable),
        ((0.78, 0.22), 0.14, 0.09, "Token logits\nnext residue", output),
        ((1.03, 0.22), 0.13, 0.09, "Generated\nsequence", output),
    ]
    for xy, w, h, text, face in infer_boxes:
        add_box(ax, xy, w, h, text, face)

    arrow(ax, (0.162, 0.265), (0.228, 0.265))
    arrow(ax, (0.402, 0.265), (0.488, 0.265))
    arrow(ax, (0.682, 0.265), (0.768, 0.265))
    arrow(ax, (0.932, 0.265), (1.018, 0.265))

    ax.text(
        0.48, 0.11,
        "No structure track, no structural bias, and no predicted distances are used during inference.",
        ha="center",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cbd5e1"),
    )

    handles = [
        Patch(facecolor=frozen, edgecolor="#374151", label="Frozen"),
        Patch(facecolor=trainable, edgecolor="#374151", label="Trainable sequence path"),
        Patch(facecolor=aux, edgecolor="#374151", label="Training-only structure path / losses"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", ncol=3, fontsize=8)

    out_path = os.path.join(OUT, "fig_architecture.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
