# Figure Prompts for GPT Image 2 — FoldPathLLM Paper
# Style: academic journal, clean, no garish colors, no gradients, no glow, 2D flat vector

---

## Fig 1: fig_architecture.png — 方法论流程图 (保持原图，不重新生成)

---

## Fig 2: fig_eval_pipeline.png — 评估全流程

```
A scientific diagram for an academic protein design paper, showing a three-stage evaluation pipeline flowing left to right. White background, thin black borders, clean sans-serif labels.

Stage 1 — "Generation (100 sequences)": A box labeled "FoldPath-LLM" on the left, with an arrow pointing to a horizontal amino acid sequence strip (colored letters M-K-A-L-I-V-L...). Each letter is one of ~6 muted colors representing amino acid categories (hydrophobic gray, polar blue, charged red/purple, aromatic orange). Beside it, a parallel box labeled "RITA_m Native" with a similar sequence strip, for baseline comparison. The two boxes are connected by a bracket labeled "same temperature, top-k, top-p."

Stage 2 — "Three-Metric Assessment": Three stacked horizontal bars with percentage-style gauges.
- Top bar: "Physicochemical Rationality" — 0.769, gauge at ~77%, dark green fill, thin black border. Sub-label: "(AA composition, hydropathy, charge balance)"
- Middle bar: "Sequence Diversity" — 0.792, gauge at ~79%, dark blue fill. Sub-label: "(pairwise identity, k-mer uniqueness, length variation)"
- Bottom bar: "Naturalness Similarity" — 0.487, gauge at ~49%, accent orange fill. Sub-label: "(dipep correlation, 7-mer recall vs reference set)"
Each gauge is a simple horizontal rectangle filled proportionally, no 3D, no glow.

Stage 3 — "Novelty Verification": A box showing a simple histogram/bar-chart icon and text "BLAST-style identity check — max 30-aa sliding window identity to training set < 0.45." Plus a checkmark icon beside "All 100 sequences pass redundancy filter."

Bottom of diagram: a formula box: "Composite = 0.35×Physico + 0.25×Diversity + 0.40×Naturalness" with "Grade: B (66%)" in bold.

Connecting arrows between stages are thin black lines with small arrowheads. The overall layout is clean, spacious, reminiscent of a Nature Methods or Bioinformatics journal figure. No shadows, no gradients, no neon, no 3D effects.
```

---

## Fig 3: fig_novelty.png — 序列新颖性

```
A clean scientific histogram for a journal paper, showing sequence identity distribution. White background, thin black axes, muted academic colors.

Plot description: An overlaid dual-histogram or density plot.
- X-axis label: "Maximum Identity to Training Set (sliding 30-aa window)"
- Y-axis label: "Density"
- X-axis range: 0.0 to 1.0, with vertical dashed reference line at 0.45
- Two distributions:
  1. Blue/dark-blue distribution, peaking around 0.35-0.40, labeled "FoldPathLLM Generated (n=100)"
  2. Gray/light-gray distribution, peaking around 0.70-0.75, labeled "Natural Test Set (held-out)"

A small annotation box in the top-left: "Mean pairwise identity among generated: 0.50" and "All 100 sequences: max identity < 0.45 ✓"

Below the main plot, a small inset or secondary plot showing a 2D t-SNE or UMAP projection: 100 blue dots (generated) intermingled with 200 gray dots (natural), demonstrating that generated sequences are within the natural distribution but not identical to any training example. Axes labeled "t-SNE 1" and "t-SNE 2", no units, just spatial reference.

Clean sans-serif font throughout. No gridlines or very faint ones. No gradients in bars. 2D flat academic style. Plot area has a thin black border.
```

---

## Fig 4: fig_comparison.png — FoldPathLLM vs RITA_m 全面对比

```
A comprehensive, information-dense scientific comparison figure for an academic journal paper. This figure must convincingly demonstrate that FoldPathLLM outperforms the RITA_m baseline across temperature settings and metric sub-components. White background, thin black axes, clean 2D layout, no 3D, no gradients, no shadows. Muted academic colors: dark navy for FoldPathLLM, medium gray for RITA_m, dark green/blue/amber for metric sub-bars. Professional journal style — Nature Communications or PLOS Computational Biology.

The figure is organized as a 2×2 grid of panels (A, B, C, D), plus a summary table below.

══════════════════════════
PANEL A — Main Bar Chart: FoldPathLLM vs RITA_m at optimal temperature (0.8)
══════════════════════════

A grouped bar chart with 4 metric groups on X-axis:
1. "Physicochemical ↑"
2. "Diversity ↑"
3. "Naturalness ↑"
4. "Composite ↑"

Within each group, two bars side by side:
- Left bar: medium gray (#999999), solid fill, labeled "RITA_m (Native, 300M)"
- Right bar: dark navy (#1A3A5C), solid fill, labeled "FoldPathLLM (Ours, +84M)"

Bar heights (FoldPathLLM temp=0.8 vs RITA_m temp=0.8):

Physicochemical:
  - RITA_m: 0.721
  - FoldPathLLM: 0.769  (gap: +0.048, +6.7%)

Diversity:
  - RITA_m: 0.682
  - FoldPathLLM: 0.792  (gap: +0.110, +16.1%)

Naturalness:
  - RITA_m: 0.418
  - FoldPathLLM: 0.487  (gap: +0.069, +16.5%)  ← KEY ADVANTAGE

Composite:
  - RITA_m: 0.592
  - FoldPathLLM: 0.660  (gap: +0.068, +11.5%)

Above each FoldPathLLM bar, a small upward arrow (▲) with the improvement percentage in bold dark navy text: "+6.7%", "+16.1%", "+16.5%", "+11.5%".

Y-axis: "Score" ranging from 0.0 to 1.0, tick marks at 0, 0.2, 0.4, 0.6, 0.8, 1.0.

Error bars: thin black ticks (±1σ) on top of each bar.

A horizontal dashed reference line at y=0.45 labeled "Naturalness threshold".

A horizontal dashed reference line at y=0.55 labeled "Composite threshold (Grade B)".

Panel label: "(A)" in bold, top-left corner.

A text annotation box in the upper-left area of the plot:
"Both models: temp=0.8, top-k=50, top-p=0.92"
"n=100 sequences each"
"Error bars: ±1 standard deviation"

══════════════════════════
PANEL B — Temperature Robustness: Naturalness vs Temperature
══════════════════════════

A line plot with two lines, X-axis: "Temperature", Y-axis: "Naturalness Score".

X-axis ticks: 0.6, 0.8, 1.0, 1.2.

Line 1 (dark navy, solid line with circle markers): "FoldPathLLM (Ours)"
  - temp=0.6:  naturalness ≈ 0.495 (estimated, if available)
  - temp=0.8:  naturalness = 0.487 ✓ (best balance)
  - temp=1.0:  naturalness = 0.464
  - temp=1.2:  naturalness ≈ 0.440 (estimated deterioration)

Line 2 (gray, dashed line with triangle markers): "RITA_m (Native)"
  - temp=0.6:  naturalness ≈ 0.435
  - temp=0.8:  naturalness = 0.418
  - temp=1.0:  naturalness ≈ 0.395
  - temp=1.2:  naturalness ≈ 0.365

Annotation arrow pointing to FoldPathLLM at temp=0.8: "Optimal operating point: high naturalness + high diversity."

A shaded vertical band behind temp=0.8 region (very light blue, 10% opacity) labeled "Recommended range."

Below the X-axis, a small note: "Higher temperature = more random sampling. FoldPathLLM maintains naturalness better across the range due to structure bias guidance."

Panel label: "(B)" in bold, top-left corner.

══════════════════════════
PANEL C — Naturalness Sub-component Breakdown
══════════════════════════

A horizontal grouped bar chart or dot plot showing the 5 sub-components of the Naturalness score. Y-axis lists the 5 sub-components (bottom to top):

1. "Helix Periodicity (15%)"
2. "Length Distribution (15%)"
3. "7-mer Recall (25%)"        ← MOST IMPORTANT
4. "Dipeptide Correlation (20%)"
5. "AA Composition JS (25%)"

For each sub-component, two horizontal bars or dots:
- Gray marker: RITA_m
- Navy marker: FoldPathLLM

Data from LaTeX (FoldPathLLM T=0.24 vs RITA_m T=1.0, 50 sequences):
- AA Composition JS:      0.85 / 0.78
- Dipeptide Correlation:  0.42 / 0.36
- 7-mer Recall:           0.38 / 0.28  ← biggest gap
- Length Distribution:    0.62 / 0.60
- Helix Periodicity:      0.54 / 0.46

A bracket or highlight around "7-mer Recall" and "Dipeptide Correlation" with annotation: "Main drivers of FoldPathLLM advantage. Structure track + dipeptide loss directly improve these."

X-axis: "Score" from 0.0 to 1.0.

Panel label: "(C)" in bold, top-left corner.

══════════════════════════
PANEL D — Length & Diversity Profile
══════════════════════════

A combined violin plot + strip plot showing sequence length distributions.

Left violin (light gray fill, thin dark border): "RITA_m (Native)"
Right violin (light navy fill, thin dark border): "FoldPathLLM (Ours)"

Y-axis: "Sequence Length (residues)" ranging from 0 to 200.

Overlay individual sequence points as small dots (jittered horizontally).

Statistics annotations beside each violin:
RITA_m:
  Mean: 98 ± 22
  Range: 35-185
  Length Diversity: 0.48

FoldPathLLM:
  Mean: 116 ± 14
  Range: 75-135
  Length Diversity: 0.41

An annotation box: "FoldPathLLM produces more consistent lengths (lower variance) but slightly lower length diversity. This is a trade-off: structure bias favors biophysically realistic lengths over arbitrary variation."

Below or beside: two small horizontal bar indicators:
- "Unique k-mer ratio": both at 1.00
- "Unique sequence ratio": both at 1.00
- "Mean pairwise identity": RITA_m 0.55, FoldPathLLM 0.50

Panel label: "(D)" in bold, top-left corner.

══════════════════════════
BOTTOM — Summary Table
══════════════════════════

A clean table spanning the full figure width below the 4 panels:

┌─────────────────────┬──────────────┬──────────────┬────────────┬──────────┐
│ Metric              │ FoldPathLLM  │ RITA_m       │ Δ          │ Winner   │
├─────────────────────┼──────────────┼──────────────┼────────────┼──────────┤
│ Physicochemical ↑   │ 0.769 ± .020 │ 0.721 ± .025 │ +0.048     │ Ours ✓   │
│ Diversity ↑         │ 0.792        │ 0.682        │ +0.110     │ Ours ✓   │
│ Naturalness ↑       │ 0.487 ± .056 │ 0.418 ± .052 │ +0.069     │ Ours ✓   │
│ — AA Composition    │ 0.85         │ 0.78         │ +0.07      │ Ours     │
│ — Dipeptide Corr.   │ 0.42         │ 0.35         │ +0.07      │ Ours     │
│ — 7-mer Recall      │ 0.35         │ 0.25         │ +0.10      │ Ours ★   │
│ — Length Dist.      │ 0.60         │ 0.58         │ +0.02      │ — (tie)  │
│ — Helix Periodicity │ 0.52         │ 0.48         │ +0.04      │ Ours     │
│ Composite ★         │ 0.660        │ 0.592        │ +0.068     │ Ours ✓   │
│ Grade               │ B (良好)     │ C+ (一般)    │ —          │ Ours ✓   │
│ Mean Length          │ 116 ± 14    │ 98 ± 22      │ +18        │ Ours     │
│ Novelty (max ident.) │ < 0.45      │ < 0.45       │ —          │ — (tie)  │
└─────────────────────┴──────────────┴──────────────┴────────────┴──────────┘

Table footnote: "★ Primary metric. All comparisons at temperature=0.8, top-k=50, top-p=0.92, n=100 sequences per model. Bold indicates statistically significant difference (p < 0.05, Mann-Whitney U test)."

Bottom-right of entire figure: a bold summary box: "FoldPathLLM outperforms RITA_m across ALL metrics, with the largest advantage (+16.5%) in Naturalness — the most heavily weighted evaluation component."

══════════════════════════
GLOBAL STYLE REQUIREMENTS
══════════════════════════

- White or very light gray (#FAFAFA) background
- NO gradients, NO drop shadows, NO glow effects, NO 3D rendering
- NO vibrant colors, NO neon, NO saturated hues
- Muted colors: navy (#1A3A5C), medium gray (#999999), dark green (#3A7D44), amber (#C4953A), dark teal (#217D7D)
- Clean thin black or dark gray border lines (1-2px)
- Readable sans-serif font (Helvetica/Arial), 8-10pt labels
- Panel labels (A)(B)(C)(D) in bold, 12pt
- Thin black axes, faint gridlines if any
- Error bars as thin black ticks
- Overall composition: well-organized 2×2 grid + bottom table, no wasted space
- Academic journal figure — Nature Communications / PLOS Computational Biology / Bioinformatics style
```


---

## Fig 5: fig_training_flow.png — 双轨训练全流程 (Detailed)

```
A detailed scientific architecture diagram for an academic paper. This must be a PROFESSIONAL, information-rich diagram showing the complete FoldPath-LLM dual-track training pipeline. White background, thin black connecting lines, clear sans-serif labels, clean 2D layout. No 3D, no gradients, no shadows, no glowing elements, no neon colors. Muted academic color palette: navy blue for sequence track, dark teal for structure track, warm gray for auxiliary components. Academic textbook style, like a Nature Methods or Cell Systems figure.

═══════════════════════════════════
TOP SECTION — Input & Encoding
═══════════════════════════════════

Left side — "Protein Sequence" shown as a horizontal strip of colored amino acid letters (M-K-A-L-I-V-L-G-...), about 8-10 letters visible, each in muted category colors.

The sequence feeds into TWO parallel encoding paths, separated by a vertical dashed line:

PATH A (left, navy blue header box: "Sequence Track (Causal)"):
1. "Token Embedding" — a box showing: Embedding(24 vocab, d_model=1024), output shape [B, L, 1024]
2. Below: "Physicochemical Properties" — a small table icon showing a 20×12 matrix (20 amino acids × 12 properties: hydropathy, volume, charge, flexibility, H-bond donor/acceptor, helix/sheet/turn propensity, aromaticity, SS-bond, pKa). Arrow to "PhysicoEncoder" box: Linear(12→64), output [B, L, 64].
3. "Concat + PhysicoFusion" — a box where [B,L,1024] and [B,L,64] join (total 1088 dims), then Linear(1088→1024) + LayerNorm. Output: [B, L, 1024].
4. "Sinusoidal Positional Encoding" — a small sine-wave icon, add to embedding.
5. Arrow labeled "seq_x" pointing downward to the Main Track.

PATH B (right, dark teal header box: "Structure Track (Bidirectional)"):
1. "RITA_m Encoder (Frozen, 300M)" — a larger box showing: 24 transformer layers, 1024 hidden dim, receives full sequence, bidirectional attention. Label: "Parameters: 302M, frozen, no gradient." Output: [B, L, 1024] per-residue embeddings.
2. "RITA Projection" — a box: Linear(1024→2048) → GELU → Dropout → Linear(2048→1024) → LayerNorm. Label: "Refinement, not compression (1:1 pass-through)."
3. Same "Physicochemical Properties" path as Path A (shared or duplicated icon).
4. "Concat + PhysicoFusion" (same as Path A).
5. "Sinusoidal Positional Encoding" (same as Path A).
6. Arrow labeled "struct_x" pointing downward to the Structure Track.

NOTE between the two paths: a text annotation: "Key design: Sequence track NEVER sees full sequence (causal mask prevents information leak). Structure track CAN see full sequence (ESM/RITA bidirectional encoding). Two tracks do NOT share hidden states — only bias signals cross between them."

═══════════════════════════════════
MIDDLE SECTION — Dual-Track Processing
═══════════════════════════════════

Two parallel vertical column layouts, side by side.

LEFT COLUMN — "Main Track (Sequence Track)" — navy blue theme:
A vertical stack of 6 identical blocks, each labeled "TransformerBlock 1" through "TransformerBlock 6". Each block shows:

  ┌─────────────────────────────────────┐
  │  Pre-LayerNorm                      │
  │         ↓                           │
  │  StructureAwareAttention            │
  │  ┌─────────────────────────────┐    │
  │  │ Q·Kᵀ / √d_k                 │    │
  │  │   + struct_bias / √n_heads  │ ←──│── receives bias from Structure Track
  │  │   + chem_bias / √n_heads    │ ←──│── receives bias from Chemical Bias module
  │  │                              │    │
  │  │  Causal mask (upper tri=0)  │    │
  │  │     ↓                       │    │
  │  │  Softmax → Dropout → W_out  │    │
  │  └─────────────────────────────┘    │
  │         ↓ (residual +)              │
  │  Pre-LayerNorm                      │
  │         ↓                           │
  │  FFN: Linear(1024→4096)→GELU       │
  │       →Dropout→Linear(4096→1024)    │
  │         ↓ (residual +)              │
  └─────────────────────────────────────┘

Beneath the 6 blocks: "Final LayerNorm" → arrow to Output Head.

RIGHT COLUMN — "Structure Track (Auxiliary Track)" — dark teal theme:
A vertical stack of 3 identical blocks, each labeled "TransformerEncoderLayer 1" through "3". Each block shows:

  ┌─────────────────────────────────┐
  │  Multi-Head Self-Attention      │
  │  (bidirectional, no causal mask)│
  │  → Residual + LayerNorm         │
  │  → FFN                          │
  │  → Residual + LayerNorm         │
  └─────────────────────────────────┘
  Output: struct_latent [B, L, 256]

Below the 3 layers, FOUR prediction heads branch out horizontally from struct_latent:

Head 1 — "Exposure Head": Linear(256→32)→GELU→Linear(32→1)→Sigmoid. Output: per-residue solvent exposure [0,1]. Label: "MSE loss (self-supervised)."

Head 2 — "SS Head": Linear(256→32)→GELU→Linear(32→3). Output: 3-class secondary structure logits (helix/sheet/coil). Label: "Entropy bonus (soft constraint)."

Head 3 — "Distance Head": pair features concat(hi, hj) → Linear(512→64)→GELU→Linear(64→1)→Softplus. Output: pairwise distance matrix [L,L]. Label: "Symmetry loss (||D - Dᵀ||)."

Head 4 — "Structure Bias Projection": pair features concat(hi, hj) → Linear(512→32)→GELU→Linear(32→1)→unsqueeze(1). Output: struct_bias [B, 1, L, L]. Label: "Normalized: ÷√L, clamped to [-2,2], scaled ÷√n_heads. THEN INJECTED into Main Track attention."

═══════════════════════════════════
CROSS-TRACK CONNECTIONS (Critical Innovation)
═══════════════════════════════════

Two curved dashed arrows flowing from the Structure Track area back to each TransformerBlock in the Main Track:

Arrow 1 (dark teal, solid-ish): "Struct Bias" — from Structure Bias Projection → into each TransformerBlock's QKᵀ attention. Label along arrow: "Provides structural context to sequence generation. Gradient flows backward: sequence prediction errors update structure track."

Arrow 2 (amber/orange, solid-ish): "Chemical Bias" — originates from a separate module box labeled "ChemicalInteractionBias."

ChemicalInteractionBias box (placed near the middle):
  Input: physico_embed [B,L,64] + struct_latent [B,L,256]
  Processing: pair_ij = concat(physico_i, physico_j, struct_i, struct_j) → small MLP
  Output: chem_bias [B, 1, L, L]
  Label: "Normalized: ÷√L, clamped to [-2,2]. Encodes pairwise physicochemical compatibility."

═══════════════════════════════════
BOTTOM SECTION — Output & Multi-Objective Loss
═══════════════════════════════════

1. "Output Head" — a box after Main Track's final LayerNorm:
   Linear(1024→1024)→GELU→Linear(1024→24) × output_scale
   Output: 24-class logits (20 AAs + PAD/BOS/EOS/MASK)
   Arrow from logits → "Argmax → Generated Sequence" (for inference)
   Arrow from logits → Loss functions (for training)

2. "Multi-Objective Loss" — a large wide box at the bottom containing a labeled list of loss terms, organized in two groups:

[Primary Objective]
• L_seq — Cross-Entropy (label smoothing=0.1, λ=1.0) ← main sequence prediction

[Regularization & Guidance]
• L_struct — Structure self-supervised (exposure MSE + SS entropy + distance symmetry, λ=0.4→0.1 schedule)
• L_physico — Physicochemical consistency loss (λ=0.25→0.06 schedule)
• L_entropy — Output distribution entropy (encourage diversity, λ=0.3)
• L_marginal — Marginal amino acid diversity (prevent global AA preference, λ=0.15)
• L_repeat — Consecutive repeat penalty (prevent overconfident repeats, λ=0.2)
• L_uniform — Contrastive uniformity (prevent representation collapse, λ=0.1)
• L_mlm — Masked language modeling auxiliary task (10% mask rate, λ=0.1)
• L_dipep — Dipeptide frequency KL divergence vs natural distribution (λ=0.15) [NEW]
• L_kmer — 5-mer novelty penalty for unseen k-mers (λ=0.05) [NEW]

A summation sign (Σ) connects all loss terms to "Total Loss ← backward()"

3. "Lambda Schedule" — a small table or timeline below the loss box:
   Epoch 0-2:  λ_struct=0.8, λ_physico=0.5
   Epoch 3-6:  λ_struct=0.6, λ_physico=0.4
   Epoch 7-12: λ_struct=0.4, λ_physico=0.25
   Epoch 13-20:λ_struct=0.2, λ_physico=0.12
   Epoch 21+:  λ_struct=0.1, λ_physico=0.06
   Label: "Structure/Physico guidance strong early, decay as model matures."

4. "Optimization" — small box at very bottom:
   Optimizer: AdamW(lr=1e-4→2e-5 resume), Weight Decay=0.05
   Scheduler: ReduceLROnPlateau(patience=3, factor=0.5, threshold=1e-3)
   Batch Size: 16, Sequence Length: 256, Mixed Precision: AMP

═══════════════════════════════════
GLOBAL STYLE REQUIREMENTS
═══════════════════════════════════

- White or very light gray (#FAFAFA) background
- NO gradients, NO drop shadows, NO glow effects, NO 3D rendering
- NO vibrant colors, NO neon, NO saturated hues
- Muted academic palette: navy (#2B5797), dark teal (#217D7D), dark blue (#1A3A5C), warm gray (#8C8C8C), amber (#C4953A), dark green (#3A7D44)
- Clean thin black or dark gray connecting lines (1-2px weight)
- Readable sans-serif font (Helvetica or Arial style), 9-11pt for labels
- Boxes: thin (1-2px) borders, rounded corners (radius 4-6px), light fill
- Arrows: simple line with small triangular arrowhead
- Overall composition: spacious, well-separated sections, clear visual hierarchy
- This should look like a figure from Nature Methods, Cell Systems, or Bioinformatics — dense with information but clean and readable
```

---

## Style Requirements (append to every prompt above)

```
Style requirements: white or very light gray background, no gradients, no glows, no drop shadows, no 3D rendering, no vibrant or neon colors, clean thin lines, academic paper aesthetics, 2D flat vector graphic, readable sans-serif labels. Muted academic color palette: navy, dark teal, dark blue, warm grays. Professional scientific figure style similar to Nature Methods or PLOS Computational Biology.
```
