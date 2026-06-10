# FoldPath-LLM: Training-Time Structural Supervision for Sequence-Only Autoregressive Protein Generation

**APBC 2026**

---

## Overview

Autoregressive protein language models can reproduce family-level sequence statistics, but **sequence naturalness does not guarantee foldability**. A sequence can match family-level amino-acid and k-mer statistics while failing to encode a stable 3D fold.

FoldPath-LLM investigates **training-time structural supervision as a methodological design axis** for autoregressive protein generation. Structural signals guide learning during training, but inference remains fully sequence-only — the structure track is entirely disabled at sampling time.

## Key Results

- **Naturalness ≠ foldability**: A no-structure-track ablation achieves higher sequence naturalness (0.560 vs. 0.514) yet produces zero sequences above the pTM 0.5 threshold. A representative no-structure sequence scores naturalness 0.601 but pTM 0.233.
- **Foldability improvement**: FoldPath-LLM achieves mean pTM 0.365 vs. 0.251 (no-structure), with 4/20 sequences exceeding pTM 0.5 and best pTM 0.619.
- **Learning efficiency**: The structure track accelerates learning (+35% precision/recall at epoch 5).
- **Novelty**: 96% of generated sequences have <20% identity to training data (mean 14.6%).

## Paper

| File | Description |
|------|-------------|
| [`FoldPathLLM_v4.tex`](FoldPathLLM_v4.tex) | LaTeX source |
| [`FoldPathLLM_v4.pdf`](FoldPathLLM_v4.pdf) | Compiled PDF |

## Architecture

```
Training (dual-track)                    Inference (sequence-only)
───────────────────────                  ─────────────────────────
Sequence tokens                          Sequence tokens
     │                                         │
RITA-m encoder (frozen)                  RITA-m encoder (frozen)
     │                                         │
     ├── Sequence Track (causal)               ├── Sequence Track (causal)
     │    └── Structure-Aware Attention        │    └── Standard attention
     │         + B_struct + B_chem                  (biases disabled)
     │                                              │
     └── Structure Track (bidirectional)             ▼
          └── Exposure / SS / Distance          Token logits
     (training only)
```

## Repository Structure

```
proteinllm/
├── FoldPathLLM_v4.tex            # Manuscript source
├── FoldPathLLM_v4.pdf            # Compiled PDF
├── model.py                      # Dual-track Transformer architecture
├── train.py                      # Multi-objective training (9 loss terms)
├── evaluation.py                 # Naturalness / diversity / physicochemical scoring
├── generate.py                   # Autoregressive generation
├── alphafold_validate.py         # AlphaFold2 foldability validation
├── esm2_score.py                 # ESM-2 embedding coherence evaluation
├── check_novelty.py              # Novelty verification (30-aa sliding window)
├── sweep_temperature.py          # Temperature sweep optimization
├── naturalness_loss.py           # Naturalness loss functions
├── physicochemical.py            # Amino acid physicochemical encoder
├── rita_encoder.py               # RITA-m base encoder
├── esm_encoder.py                # ESM representation encoder
├── dataset.py                    # Dataset and data loading
├── data/                         # Cytochrome b sequence data
├── pretrained/                   # Pretrained model weights
├── checkpoints_esm/              # Training checkpoints
├── fig_*.png                     # Paper figures
├── config.py                     # Model and training configuration
└── README.md
```

## Quick Start

```bash
# Train FoldPath-LLM
python train.py

# Generate 50 sequences
python generate_50.py

# Evaluate generation quality
python evaluation.py

# Run AlphaFold2 foldability validation
python alphafold_validate.py

# Run no-structure-track ablation
python gen_ablation_13.py

# ESM-2 embedding evaluation
python esm2_score.py

# Novelty check
python check_novelty.py

# Temperature sweep
python sweep_temperature.py
```

## Citation

```bibtex
@misc{yang2025foldpath,
  title   = {Training-Time Structural Supervision as an Inductive Bias for
             Sequence-Only Autoregressive Protein Generation},
  author  = {Yang, Ruofan and Liang, Lixin and Huang, Bingding and Wang, Xin},
  year    = {2025},
  note    = {APBC 2026 submission}
}
```

## Authors

Ruofan Yang, Lixin Liang, Bingding Huang*, Xin Wang*  
Shenzhen Technology University, Shenzhen, China  
\* Corresponding authors: huangbingding@sztu.edu.cn, wangxin@sztu.edu.cn
