# GPT Image 2 Prompts — FoldPath-LLM 论文配图

## 风格统一要求（每条 prompt 末尾附加）
```
Academic journal figure style. Muted, sophisticated color palette: warm ivory background (#FAFAF5), 
deep navy (#1A2A3A), slate blue (#4A6A8A), warm gray (#8A8A8A), muted amber (#C4A35A), 
soft teal (#5A8A8A). No neon, no pure black, no pure white, no gradients, no drop shadows, 
no 3D effects, no glowing elements. Clean thin lines. Sans-serif labels (Helvetica or Inter style). 
2D flat vector graphic. Nature Methods / Cell Systems aesthetic. 
White space is intentional — do not fill every corner.
```

---

## Figure 1: Dual-Track Transformer Architecture (Fig 1, Architecture Diagram)

```
A professional scientific architecture diagram for a computational biology paper, 
showing the FoldPath-LLM dual-track Transformer model. 
This will be the anchor figure of the paper — it must be polished, publication-ready, 
and communicate complex information with visual clarity.

LAYOUT: Top-to-bottom information flow, organized in three horizontal bands.

═══════════════════════════════════════
BAND 1 — INPUT (top 20% of canvas)
═══════════════════════════════════════

Center-left: A horizontal strip of amino acid letters "M-K-A-L-I-V-L-G-F-..." 
in muted category colors (hydrophobic=navy, polar=slate, charged=amber, aromatic=gray). 
Label above: "Input Protein Sequence."

Two arrows diverge from this strip — one going left-down, one going right-down.

LEFT PATH (sequence track): Arrow labeled "Causal (left-to-right)" 
→ Box "Token Embedding" (icon of embedding matrix, subtitle "V=24, d=1024")
→ Box "Physicochemical Encoder" (icon of a small 20×12 grid, subtitle "12 properties → 64 dims")
→ The two outputs merge at a ⊕ symbol with label "Concat + MLP Fusion"
→ Faint arrow continuing downward.

RIGHT PATH (structure track): Arrow labeled "Bidirectional (full sequence)" 
→ Box "RITA_m Encoder" (larger box, subtitle "300M params, frozen, 24 layers")
→ Subtle horizontal line separating frozen from trainable region
→ Box "Projection MLP" (subtitle "1024→2048→1024, LayerNorm")
→ Same "Physicochemical Encoder" as left (shared icon, dashed connector)
→ Same ⊕ "Concat + MLP Fusion"
→ Faint arrow continuing downward.

A subtle dashed vertical line between the two paths, 
with annotation: "Causal track: no future-token leakage. Structure track: full bidirectional context."

═══════════════════════════════════════
BAND 2 — PROCESSING (middle 50% of canvas)
═══════════════════════════════════════

Two parallel vertical columns, side by side.

LEFT COLUMN (navy-accented): "Sequence Track (Causal)" header.
A stack of 6 identical rounded rectangles, each labeled "Transformer Block 1" … "6."
Inside each block, a simplified diagram showing:
  Pre-LN → [Attention: QK^T/√d + B_struct + B_chem] → Pre-LN → FFN (GELU)
The B_struct and B_chem have small dashed arrows feeding in from the RIGHT column.
Annotation: "Structure-Aware Attention — structural/chemical biases injected per layer."

RIGHT COLUMN (teal-accented): "Structure Track (Bidirectional)" header.
A stack of 3 similar blocks labeled "Encoder Layer 1" … "3."
Inside: Multi-Head Self-Attention → FFN.
From the bottom of this column, FOUR thin arrows branch out to four small prediction heads:
  • "Exposure" (0-1, sigmoid)
  • "SS (3-class)" (helix/sheet/coil)
  • "Distance (pairwise)" (Softplus)
  • "Structure Bias (pairwise)" — this one has a CURVED DASHED ARROW 
    looping back to the LEFT column's attention blocks.
Label along the curved arrow: "B_struct + B_chem (scaled by 1/√L, clamped ±2.0)"
Label: "Biases disabled during inference (use_bias=False)."

═══════════════════════════════════════
BAND 3 — OUTPUT & LOSS (bottom 30% of canvas)
═══════════════════════════════════════

From the left column's bottom: "LayerNorm" → "Output Head (1024→1024→24)" 
→ "Token Logits → Generated Sequence" (shown as another amino acid strip "M-K-A-L...")

A gray horizontal divider.

Below it, a compact grid of 9 loss terms in 3 rows of 3:
  L1: Cross-Entropy (CE, λ=1.0)
  L2: Structure Self-Sup (λ=0.4→0.1)
  L3: Physicochemical (λ=0.25→0.06)
  L4: Entropy Reg (λ=0.3)
  L5: Marginal Diversity (λ=0.15)
  L6: Repeat Penalty (λ=0.2)
  L7: Contrastive Uniformity (λ=0.1)
  L8: Dipeptide KL (λ=0.15)
  L9: K-mer Penalty (λ=0.05)

All 9 loss terms feed into a Σ summation symbol → "Total Loss → Backward()"

At the very bottom: a small box "Lambda Schedule: struct/physico weights start high 
(epochs 0-2: 0.8/0.5), decay to low (epochs 21+: 0.1/0.06)."

═══════════════════════════════════════
STYLE (critical — read carefully):
═══════════════════════════════════════

This is an academic computer science / computational biology figure.
DO NOT make it look like a marketing infographic.
DO NOT use bright colors, neon, or saturated hues.
DO NOT add decorative elements, icons, or emoji.
DO NOT use painterly shading or 3D-like boxes.

USE: clean geometric shapes, thin precise lines (1-2px), subtle rounded corners (4-6px), 
generous whitespace, consistent typography, restrained color application.
The diagram should feel calm, precise, and information-dense without clutter.

Colors:
- Background: warm ivory (#FAFAF5)
- Sequence track elements: deep navy (#1A2A3A) fills with white text
- Structure track elements: soft teal (#5A8A8A) fills with white text
- Arrow lines: warm gray (#8A8A8A), 1.5px
- Text labels: warm dark gray (#3A3A3A), 9-11pt
- Borders: #CCCCCC or navy/teal as appropriate

Academic journal figure style. Muted, sophisticated color palette: warm ivory background (#FAFAF5), 
deep navy (#1A2A3A), slate blue (#4A6A8A), warm gray (#8A8A8A), muted amber (#C4A35A), 
soft teal (#5A8A8A). No neon, no pure black, no pure white, no gradients, no drop shadows, 
no 3D effects, no glowing elements. Clean thin lines. Sans-serif labels (Helvetica or Inter style). 
2D flat vector graphic. Nature Methods / Cell Systems aesthetic. 
White space is intentional — do not fill every corner.
```
 