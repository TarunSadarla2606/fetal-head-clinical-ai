<div align="center">

# Fetal Head Circumference Clinical AI

**Automated HC measurement · Gestational age estimation · Temporal uncertainty quantification · Model compression**

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-FFD21E)](https://huggingface.co/spaces/TarunSadarla2606/fetal-head-clinical-ai)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-FF4B4B?logo=streamlit)](https://huggingface.co/spaces/TarunSadarla2606/fetal-head-clinical-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ISUOG](https://img.shields.io/badge/ISUOG_≤3mm-all_models_pass-brightgreen)]()

</div>

---

> **Independent research continuation** of a course project ([CSCE 6260, UNT Fall 2025](https://github.com/TarunSadarla2606/fetal-head-cine-segmentation)) — developed after the semester to push from 86% Dice / 17.25mm MAE to a deployable clinical-grade pipeline. All work here is sole-authored.

---

## Live Demo

**[→ Try the deployed app on HuggingFace Spaces](https://huggingface.co/spaces/TarunSadarla2606/fetal-head-clinical-ai)**

Upload any 2D fetal head ultrasound (or pick a demo subject). Choose single-frame or cine-clip mode. Get HC in mm, gestational age, GradCAM++ explainability, and a downloadable PDF clinical report — in seconds, on CPU.

---

## What This Is

Fetal head circumference (HC) is one of three standard biometric measurements in every routine antenatal scan — currently performed manually by sonographers using calipers on frozen ultrasound frames. It is the primary indicator for gestational age estimation and intrauterine growth restriction screening.

This system automates the full pipeline:
1. Segments the fetal skull boundary with a trained Residual U-Net
2. Fits a Ramanujan ellipse to the predicted mask
3. Computes HC in mm and converts to gestational age (Hadlock 1984 cubic polynomial)
4. Optionally processes 16-frame cine-loops for temporal consensus + reliability scoring
5. Generates GradCAM++ explainability maps showing which regions drove the boundary decision
6. Produces downloadable PDF clinical reports (LLM-generated narrative via Claude Haiku, or rule-based fallback)

All four model variants pass the **ISUOG ≤3mm acceptable error threshold** for second-trimester biometry.

---

## Four Model Variants

| Phase | Type | Architecture | Dice (%) | MAE (mm) | Params | vs Baseline |
|-------|------|-------------|----------|----------|--------|-------------|
| **Phase 0** | Static | Residual U-Net + deep supervision | **97.75** | **1.65** | 8.11M | — |
| **Phase 4a** | Static | Phase 0 — hybrid crossover pruned | 97.64 | 1.76 | 4.57M | **−43.7% params** |
| **Phase 2** | Temporal (16 frames) | 2D U-Net + temporal self-attention | 95.95 | 2.10 | 8.90M | — |
| **Phase 4b** | Temporal (16 frames) | Phase 2 — backbone pruned, TAM intact | 96.00 | 2.06 | 5.20M | **−41.6% params** |

All results on **HC18 held-out test set** (335 images, 551 pregnancies, Radboud UMC Netherlands).  
✅ All models pass ISUOG ≤3mm threshold for second-trimester screening biometry.

---

## Development Arc

This project evolved through four phases, each building on the last:

```
Phase 0 — Static Baseline
  Residual U-Net with deep supervision auxiliary heads at dec3, dec2
  Boundary-weighted BCE + Dice loss
  Augmentation: elastic deformation, Rician noise injection, coarse dropout
  Training: HC18 static frames (800 train / 5% val / test)
  Result: Dice 97.75% · MAE 1.65 mm · 8.11M params

       ↓

Phase 1 — Pseudo-LDDM v2: Synthetic Cine Generation
  Physics-inspired conversion of static HC18 frames → realistic 16-frame cine-loops
  Key improvements over prior work (v1, course project):
    - Ornstein-Uhlenbeck probe motion: non-periodic, mean-reverting, stochastic
    - Cross-sectional mask variation: per-frame ellipse axis perturbation
      (the critical fix — makes temporal task non-trivial)
    - Rician speckle noise (physically correct ultrasound noise model)
    - Depth-dependent intensity attenuation
    - Acoustic shadowing behind the skull boundary
    - TGC (time-gain compensation) drift between frames
  Output: 806 high-fidelity cine clips · Mean temporal HC std: 10.33 px
  (vs ~0.0 px in v1 — degenerate sequences provided no temporal learning signal)

       ↓

Phase 2 — Temporal Attention System
  Shared 2D Residual U-Net encoder (Phase 0 pretrained)
  Lightweight temporal self-attention module at bottleneck (~200K additional params)
  Three-stage training: frozen backbone → partial unfreeze → full fine-tune
  Processes 16-frame clips · Per-frame uncertainty · Reliability scoring
  Result: Dice 95.95% · MAE 2.10 mm · 8.90M params

       ↓

Phase 4 — Structured Pruning (Hybrid Crossover Filter Synthesis)
  Method from CNN structured pruning research (CSCE 5934, UNT — see related work)
  Phase 0 → Phase 4a: −43.7% params · Dice 97.64% · MAE 1.76 mm (−0.11pp Dice)
  Phase 2 → Phase 4b: −41.6% params · Dice 96.00% · MAE 2.06 mm (+0.05pp Dice)
  TAM preserved intact in Phase 4b — bottleneck pruning accounts for attention channels
```

---

## Ablation Study: Why Temporal Attention Is Essential

| Config | Architecture | Dice (%) | MAE (mm) | Note |
|--------|-------------|----------|----------|------|
| **A** | Phase 0 — static single frame | 97.36 | 1.75 | Baseline |
| **B** | 16-frame cine, identity attention | 81.48 | 19.37 | No TAM |
| **C** | 16-frame cine + temporal attention | **95.71** | **2.10** | Full system |

The **B→C gap (+14.23pp Dice, −17.27mm MAE)** shows that temporal attention is load-bearing, not incidental. Without it, inconsistent per-frame predictions collapse into poor consensus masks. The ~200K TAM parameter addition recovers near-baseline performance on temporal data and closes 93% of the B→A Dice gap.

---

## System Architecture

### Phase 0 — Residual U-Net with Deep Supervision

```
Input: [B, 1, 256, 384] — grayscale, normalized (mean=0.2, std=0.15)
  │
  ├─ Enc1: ResBlock(1→32)    → MaxPool(2)   [B, 32, 128, 192]
  ├─ Enc2: ResBlock(32→64)   → MaxPool(2)   [B, 64,  64,  96]
  ├─ Enc3: ResBlock(64→128)  → MaxPool(2)   [B, 128, 32,  48]
  ├─ Enc4: ResBlock(128→256) → MaxPool(2)   [B, 256, 16,  24]
  │
  Bottleneck: ResBlock(256→512)             [B, 512, 16,  24]
  │
  ├─ Dec4: ConvTranspose + Skip(Enc4) → ResBlock → aux_d3 head  ←── deep supervision
  ├─ Dec3: ConvTranspose + Skip(Enc3) → ResBlock → aux_d2 head  ←── deep supervision
  ├─ Dec2: ConvTranspose + Skip(Enc2) → ResBlock
  └─ Dec1: ConvTranspose + Skip(Enc1) → ResBlock → sigmoid output

ResBlock: BN → ReLU → Conv(in→out, k=3) → BN → ReLU → Conv(out→out, k=3) + skip
Loss: L_main + 0.3×L_aux3 + 0.1×L_aux2   (boundary-weighted BCE + Dice)
```

---

### Phase 1 — Pseudo-LDDM v2 Cine Synthesis

Converts a single static HC18 frame into a 16-frame cine sequence with clinically realistic motion and noise. The core challenge: generating sequences where per-frame HC values actually vary (non-trivial temporal signal) while staying anatomically consistent.

```python
def generate_cine(img_gray, n_frames=16):
    # 1. Ornstein-Uhlenbeck probe motion (mean-reverting, stochastic)
    tx  = ornstein_uhlenbeck(n_frames, theta=0.15, sigma=2.0)   # translation x
    ty  = ornstein_uhlenbeck(n_frames, theta=0.15, sigma=1.5)   # translation y
    rot = ornstein_uhlenbeck(n_frames, theta=0.20, sigma=0.40)  # rotation

    for i, frame:
        # 2. Per-frame ellipse axis perturbation (cross-sectional variation)
        #    → makes HC vary across frames, forcing attention to be useful
        # 3. Warp affine transform with OU motion
        # 4. Rician speckle: sqrt(real² + imag²) noise model
        # 5. Depth attenuation: multiply by exp(-k × depth_fraction)
        # 6. Acoustic shadowing behind skull boundary
        # 7. TGC drift between frames

    return frames  # uint8, shape (16, H, W)
```

**v1 vs v2 comparison:**

| Component | v1 (Course Project) | v2 (This Work) |
|-----------|--------------------|-----------------|
| Motion model | Sinusoidal (periodic) | Ornstein-Uhlenbeck (stochastic) |
| Cross-sectional variation | None | Per-frame ellipse axis perturbation |
| Noise model | Gaussian | Rician (physically correct) |
| Intensity variation | None | Depth-dependent attenuation |
| Acoustic effects | None | Shadowing + TGC drift |
| Clips generated | — | 806 (Stage 4 full fidelity) |
| Temporal HC std | ~0.0 px (degenerate) | **10.33 px** |

---

### Phase 2 — Temporal Attention Architecture

```
Input clip: [B, 16, 1, 256, 384]
  │
  Reshape to [B×16, 1, 256, 384]
  │
  2D Residual U-Net encoder (shared Phase 0 weights)
  │
  Bottleneck: [B×16, 512, h, w]
  │
  Reshape to [B, 16, 512, h, w]
  │
  ┌─────────────────────────────────────────────────┐
  │   Temporal Attention Module (TAM) — ~200K params │
  │                                                   │
  │   Spatial avg pool  → [B, 16, 512]               │
  │   Linear(512→256)   → [B, 16, 256]               │
  │   + positional encoding                           │
  │   Multi-head attention (8 heads, dim=256)         │
  │   FFN: 256 → 1024 → 256                          │
  │   Linear(256→512) → sigmoid gate                 │
  │   Element-wise multiply with bottleneck           │
  │   Output: [B, 16, 512, h, w] + attn [16, 16]     │
  └─────────────────────────────────────────────────┘
  │
  Reshape to [B×16, 512, h, w]
  │
  2D Decoder (shared weights)
  │
  Output: [B, 16, 1, 256, 384]  per-frame logits
  │
  Consensus mask: mean(sigmoid(logits)) > threshold
  Uncertainty map: std(sigmoid(logits) > threshold, across T)
  Reliability: max(0, 1 − std(HC_per_frame) / mean(HC_per_frame))
```

**Three-stage training:**
1. Frozen encoder — train TAM + decoder only (10 epochs, lr=1e-3)
2. Partial unfreeze — enc3, enc4, bottleneck + TAM + decoder (10 epochs, lr=1e-4)
3. Full fine-tune — all layers (20 epochs, lr=1e-5, ReduceLROnPlateau)

---

## Deployed Application — Feature Overview

Live Streamlit app with four tabs, all models selectable at runtime:

### Tab 1 — Static Analysis (Phase 0 / Phase 4a)
- Upload image or pick demo subject (automatic grayscale + validation)
- Segmentation overlay (red skull boundary on original image)
- GradCAM++ heatmap (which regions drove boundary prediction)
- HC (mm) · GA (weeks + days) · Trimester · Inference latency
- HIGH / MODERATE / LOW confidence badge
- Optional: upload ground-truth annotation → live Dice + MAE vs annotation
- Three-colour overlay (TP=yellow, FP=red, FN=green)
- Downloadable PDF clinical report (Claude Haiku narrative or rule-based template)

### Tab 2 — Cine Analysis (Phase 2 / Phase 4b)
- Static frame → synthetic 16-frame cine (Pseudo-LDDM v2) → animated GIF
- Temporal consensus segmentation (mean probability across 16 frames)
- Per-frame uncertainty heatmap (inter-frame disagreement visualization)
- T×T temporal attention weight matrix
- Per-frame HC stability chart (value ± 1 std band)
- Reliability score → confidence badge
- Downloadable cine clinical report PDF

### Tab 3 — Head-to-Head Comparison
- All four models run simultaneously on the same input
- 4-column segmentation overlays side by side
- Comparison table: HC · GA · Dice · MAE · params · FLOPs · compression ratio
- Combined PDF report (single document for all four variants)

### Tab 4 — Model Card
- Responsible AI documentation (NIST AI RMF framework)
- Performance stratified by GA trimester (Early / Mid / Late)
- Bias analysis: known degradation at <20w (small skull, low SNR)
- Regulatory pathway information (FDA 510(k), EU IVDR)

---

## Key Implementation Details

**HC Computation**  
Ramanujan's ellipse approximation on the largest connected component of the predicted mask:
```
h  = ((a − b) / (a + b))²
HC = π(a + b) × [1 + (3h) / (10 + √(4 − 3h))]
```
Scaled to mm: `HC_px × pixel_spacing_mm × (ORIG_W / INPUT_W)` where `ORIG_W=800` corrects for the 800→384 resize.

**Gestational Age**  
Hadlock (1984) cubic polynomial: `GA = 8.96 + 0.540(HC/10) − 0.0040(HC/10)² + 0.000399(HC/10)³`, clipped to [10, 42] weeks.

**Reliability Score** (cine mode)  
`max(0, 1 − std(HC_per_frame) / mean(HC_per_frame))`. Maps to: ≥0.97 → HIGH · ≥0.92 → MODERATE · <0.92 → LOW.

**Input Validation**  
7 clinical sanity checks (shape, resolution ≥64×64, blank detection, saturation, dynamic range, aspect ratio, Laplacian texture). Non-blocking warnings shown in UI; only hard failures block inference.

**LLM Report Generation**  
Claude Haiku (Anthropic API) generates structured clinical narrative from HC, GA, confidence, and trimester fields. Falls back to deterministic rule-based template if API key not set.

**Fill Hollow Mask**  
HC18 annotations are ellipse outlines (rings), not solid regions. `fill_hollow_mask()` flood-fills from the image border and inverts to convert ring → solid disk before training and metric computation.

---

## Results

### Model Comparison

| Phase | Dice (%) | IoU (%) | MAE (mm) | RMSE (mm) | Params | Size (MB) |
|-------|----------|---------|----------|-----------|--------|----------|
| Phase 0 | 97.75 | — | 1.65 | — | 8.11M | 97.5 |
| Phase 4a | 97.64 | — | 1.76 | — | 4.57M | 18.3 |
| Phase 2 | 95.95 | — | 2.10 | — | 8.90M | 35.7 |
| Phase 4b | 96.00 | — | 2.06 | — | 5.20M | 20.9 |

*Full per-trimester breakdown and figure outputs will be added when training notebooks are uploaded.*

### Compression Summary

| Pruning | Original | Pruned | Δ Params | Δ Dice | Δ MAE |
|---------|----------|--------|----------|--------|-------|
| Phase 0 → 4a | 8.11M | 4.57M | **−43.7%** | −0.11pp | +0.11mm |
| Phase 2 → 4b | 8.90M | 5.20M | **−41.6%** | +0.05pp | −0.04mm |

Phase 4b's accuracy **improves** slightly after pruning — the compression regularizes the backbone without touching the temporal attention module, acting as implicit dropout.

---

## Repository Structure

```
fetal-head-clinical-ai/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── app/                                  ← Deployed Streamlit app (mirrors HF Space)
│   ├── README.md
│   ├── app.py                            ← 4-tab Streamlit UI
│   ├── inference.py                      ← Model architectures + inference pipelines
│   ├── report.py                         ← PDF clinical report (LLM + rule-based)
│   ├── xai.py                            ← GradCAM++, temporal attention, uncertainty
│   └── model_card.py                     ← Responsible AI / NIST AI RMF docs
│
├── notebooks/                            ← Kaggle training notebooks [to be uploaded]
│   ├── README.md
│   ├── phase0_residual_unet.ipynb        ← Phase 0: ResUNet + deep supervision
│   ├── phase1_pseudo_lddm_v2.ipynb       ← Cine synthesis: OU motion + Rician
│   ├── phase2_temporal_attention.ipynb   ← Phase 2: 3-stage TAM training
│   ├── phase4_pruning.ipynb              ← Phase 4a/4b: crossover pruning + fine-tune
│   └── ablation_study.ipynb             ← Config A/B/C ablation
│
├── src/                                  ← Modular Python equivalents
│   ├── models/
│   │   ├── residual_unet.py              ← ResidualUNetDS (Phase 0/2 backbone)
│   │   ├── temporal_net.py              ← TemporalFetaSegNet + TAM (Phase 2/4b)
│   │   └── pruned_unet.py               ← PrunedResidualUNetDS (Phase 4a/4b)
│   ├── data/
│   │   ├── dataset.py                   ← HC18Dataset + augmentation pipeline
│   │   └── pseudo_lddm_v2.py            ← Cine generation engine
│   └── evaluate.py                      ← Dice, MAE, RMSE, R², ablation reporting
│
├── results/
│   ├── README.md                         ← Full results tables
│   └── figures/                          ← Segmentation outputs, attention maps
│
├── data/
│   └── README.md                         ← HC18 download + preprocessing guide
│
└── models/
    └── README.md                         ← Weights on HF Space (too large for GitHub)
```

---

## Quickstart

**Option 1 — Live app (no setup)**
```
https://huggingface.co/spaces/TarunSadarla2606/fetal-head-clinical-ai
```

**Option 2 — Run locally**
```bash
git clone https://github.com/TarunSadarla2606/fetal-head-clinical-ai.git
cd fetal-head-clinical-ai
pip install -r requirements.txt

# Download weights from HuggingFace Space (see models/README.md)
# Place phase0_model.pth, phase2_model.pth, 4a_best_pruned_ft_v10.pth,
# 4b_best_pruned_ft_v10.pth in the project root

streamlit run app/app.py
```

**Option 3 — Inference only**
```python
from app.inference import load_phase0, predict_single_frame
import cv2

model = load_phase0("phase0_model.pth")
img   = cv2.imread("ultrasound.png", cv2.IMREAD_GRAYSCALE)

result = predict_single_frame(model, img, pixel_spacing_mm=0.070)

print(f"HC:         {result['hc_mm']:.1f} mm")
print(f"GA:         {result['ga_str']}")
print(f"Confidence: {result['confidence_label']}")
print(f"Latency:    {result['elapsed_ms']:.1f} ms")
```

---

## Dataset

**[HC18 Grand Challenge](https://hc18.grand-challenge.org/)** — van den Heuvel et al., *PLOS ONE* 2018  
999 training + 335 test fetal head ultrasound images from 551 pregnancies, Radboud University Medical Center, Netherlands.

- Image size: 800×540 px at 0.070 mm/pixel (default)
- Annotation format: semi-major axis, semi-minor axis, angle (ellipse outline, not solid)
- GA range: 14–40 weeks (predominantly 2nd trimester)
- Available on Kaggle: [hc18 dataset](https://www.kaggle.com/datasets/sahliz/hc18)

See `data/README.md` for download and preprocessing instructions.

---

## Model Weights

Weights are too large for GitHub. All four checkpoints are hosted on the HuggingFace Space:

| File | Size | Phase |
|------|------|-------|
| `phase0_model.pth` | 97.5 MB | Static Residual U-Net |
| `phase2_model.pth` | 35.7 MB | Temporal attention system |
| `4a_best_pruned_ft_v10.pth` | 18.3 MB | Pruned static (−43.7%) |
| `4b_best_pruned_ft_v10.pth` | 20.9 MB | Pruned temporal (−41.6%) |

Download: `https://huggingface.co/spaces/TarunSadarla2606/fetal-head-clinical-ai/tree/main`  
See `models/README.md` for programmatic download.

---

## Related Work

**[CNN Structured Pruning — VGG16 (CSCE 5934, UNT)](https://github.com/TarunSadarla2606/cnn-structured-pruning-vgg16)**  
The hybrid crossover filter synthesis method used in Phase 4 was first developed as a directed research project. This clinical pipeline is its first application to a medical image segmentation model.

**[Fetal Head Cine Segmentation — Course Project](https://github.com/TarunSadarla2606/fetal-head-cine-segmentation)** (CSCE 6260, UNT Fall 2025)  
The predecessor: 3D U-Net + Pseudo-LDDM v1 (sinusoidal motion), Dice 86.17%, MAE 17.25 mm, joint work with Ramyasri Murugesan. This repo is the independent solo continuation.

---

## Regulatory Notice

⚠️ **Research prototype. Not FDA-cleared. Not CE-marked. Not for clinical use.**

As a biometric measurement Software as a Medical Device (SaMD), clinical deployment would require:
- **US:** FDA 510(k) clearance (Class II, 21 CFR Part 892.2050)
- **EU:** IVDR Class B certification

All HC measurements and GA estimates require verification by a certified sonographer before any clinical decision.

---

## Citation

```bibtex
@misc{sadarla2026fetalhead,
  author    = {Sadarla, Tarun},
  title     = {Fetal Head Circumference Estimation: Static and Temporal Clinical AI
               with Synthetic Cine Generation and Structured Pruning},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/TarunSadarla2606/fetal-head-clinical-ai}
}
```

HC18 Dataset:
```bibtex
@article{vandenheuvel2018,
  author  = {van den Heuvel, Thomas L A and de Bruijn, Dagmar and de Korte, Chris L and van Ginneken, Bram},
  title   = {Automated measurement of fetal head circumference using 2D ultrasound images},
  journal = {PLOS ONE},
  year    = {2018},
  volume  = {13},
  number  = {8},
  doi     = {10.1371/journal.pone.0200412}
}
```

---

<div align="center">
  <i>Independent research · MS Artificial Intelligence (Biomedical Concentration) · University of North Texas · 2026</i><br>
  <i>Tarun Sadarla &nbsp;·&nbsp; <a href="mailto:tarunsadarla26@gmail.com">tarunsadarla26@gmail.com</a> &nbsp;·&nbsp; <a href="https://linkedin.com/in/tarun-sadarla-715026231">LinkedIn</a></i>
</div>
