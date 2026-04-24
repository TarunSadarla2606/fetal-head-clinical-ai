# Fetal Head Circumference Clinical AI

**Automated HC measurement · Gestational age estimation · Structural pruning compression · Temporal uncertainty quantification · Clinical report generation**

[![HuggingFace Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/TarunSadarla2606/fetal-head-clinical-ai)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ISUOG](https://img.shields.io/badge/ISUOG-±3mm%20PASS-brightgreen)](https://www.isuog.org)

Independent research continuation of a course project (CSCE 6260, UNT Fall 2025) — developed post-semester to advance from 86% Dice / 17.25mm MAE to a deployable clinical-grade pipeline. All work sole-authored.  
**Directed study:** Prof. Russel Pears · **Fetal head project:** Prof. Xiaohui Yuan · University of North Texas, 2026.

---

## Live Demo
**→ [Try the deployed app on HuggingFace Spaces](https://huggingface.co/spaces/TarunSadarla2606/fetal-head-clinical-ai)**

Upload any 2D fetal head ultrasound (or pick a demo subject). Choose single-frame or cine-clip mode. Get HC in mm, gestational age, GradCAM++ explainability, and a downloadable PDF clinical report — in seconds, on CPU.

---

## What This Is

Fetal head circumference (HC) is one of three standard biometric measurements in every routine antenatal scan — currently performed manually by sonographers using calipers on frozen ultrasound frames. It is the primary indicator for gestational age estimation and intrauterine growth restriction screening.

This system automates the full pipeline:
- Segments the fetal skull boundary with a trained Residual U-Net (pre-activation, deep supervision)
- Fits a Ramanujan ellipse to the predicted mask and computes HC in mm
- Converts HC to gestational age using the Hadlock (1984) cubic polynomial
- Optionally processes 16-frame synthetic cine-loops for temporal consensus + reliability scoring
- Generates GradCAM++ explainability maps and downloadable PDF clinical reports
- Audits model fairness across GA trimesters, image quality, and HC size range (Phase 3)

All four model variants pass the ISUOG ±3mm acceptable error threshold for second-trimester biometry.

---

## Four Model Variants

| Phase | Type | Architecture | Dice (%) | MAE (mm) | Params | vs Baseline |
|-------|------|--------------|----------|----------|--------|-------------|
| Phase 0 | Static | Residual U-Net + deep supervision | **97.75** | **1.65** | 8.11M | — |
| Phase 4a | Static | Phase 0 — Hybrid Crossover pruned | **97.64** | **1.76** | 4.57M | −43.7% params |
| Phase 2 | Temporal (16 frames) | 2D U-Net + temporal self-attention | **95.95** | **2.10** | 8.90M | — |
| Phase 4b | Temporal (16 frames) | Phase 2 — backbone pruned, TAM intact | **96.00** | **2.06** | 5.20M | −41.6% params |

All results on HC18 test set (Radboud UMC, Netherlands). ✅ All models pass ISUOG ≤3mm threshold.  
**MAE context:** 1.65mm vs published SOTA 5.95mm on HC18 — **3.6× lower error**.

---

## Development Arc

### Phase 0 — Static Residual U-Net
- Pre-activation residual blocks (BN → ReLU → Conv) with deep supervision
- Auxiliary segmentation heads at dec3 and dec2 (loss: main + 0.5×aux2 + 0.3×aux3)
- Boundary-weighted BCE+Dice loss with Sobel-derived edge weight maps
- Augmentation: HorizontalFlip, Rotate(±15°), ElasticTransform, ShiftScaleRotate,
                GaussNoise, RandomBrightnessContrast, CoarseDropout (albumentations)
- Training: 80 epochs, AdamW lr=3e-4, CosineAnnealingLR(eta_min=lr/100)
- Dataset: 999 HC18 images → 70% train / 15% val / 15% test
- Environment: Google Colab, NVIDIA T4 GPU
- **Result: Dice 97.75% · MAE 1.65mm · 8.11M params**

↓

### Phase 1 — Pseudo-LDDM v2: Synthetic Cine Generation
- Physics-inspired conversion of static HC18 frames → 16-frame cine-loops
- Key fix over v1 (course project): cross-sectional mask variation (per-frame
  ellipse axis perturbation) makes temporal HC std non-trivial (10.33 px vs ~0.0)
- Ornstein-Uhlenbeck probe motion (mean-reverting, non-periodic)
- Rician speckle noise (physically correct ultrasound model)
- Depth-dependent attenuation, acoustic shadowing, TGC drift (Stage 4)
- Environment: Google Colab, NVIDIA T4 GPU
- **Output: 806 high-fidelity cine clips**

↓

### Phase 2 — Temporal Attention U-Net
- Shared 2D encoder (Phase 0 pretrained, base_ch=32)
- Lightweight TAM at bottleneck: pool → Linear(512→256) → MHA(8 heads) → FFN → gate
- Three-stage training: Stage 1 (TAM only, lr=3e-4) → Stage 2 (decoder+TAM, lr=1e-4) → Stage 3 (full fine-tune, lr=3e-5)
- Dataset: 806 clips → 564 train / 121 val / 121 test
- Environment: Google Colab, NVIDIA T4 GPU
- **Result: Dice 95.95% · MAE 2.10mm · 8.90M params**

↓

### Phase 3 — XAI, Bias Audit & Governance
Not a training phase — a structured clinical validation and accountability layer:
- GradCAM++ (custom implementation, no external packages) on Phase 0 final decoder layer
- Temporal attention T×T matrix and per-frame attention weights for Phase 2
- GA-trimester bias audit: Early (<20w) / Mid (20–30w) / Late (>30w)
  — Mid trimester achieves best performance; Late shows elevated MAE (acoustic shadowing)
  — No systematic over/under-measurement bias by HC size range
- Business case: 2 min manual → 10 sec AI at 20 scans/day = **153 sonographer hrs/year saved (~$5,347 USD/year)**
  — Inter-observer CV reduction from 3.2% → 1.1% (Papageorghiou et al. 2014)
- Model Card (Mitchell et al. 2019 framework): intended use, fairness analysis, known limitations
- Regulatory classification: **SaMD Class II** (FDA 21 CFR Part 892 · 510(k) pathway) · **IVDR Class B** (EU)

↓

### Phase 4 — Structured Pruning (Hybrid Crossover Filter Synthesis)
- ILR importance scoring: 0.6×RMS activation + 0.4×filter L1 norm + 0.2×Frobenius
- Hybrid Crossover merging: dropped channel features synthesised into kept channel
  via 50-step Adam regression (information preservation, not discard)
- Burst-sequential pruning with adaptive burst size + rollback guard rails
- 3 prune-FT cycles with KD recovery (teacher = frozen baseline, α=0.5, T=4.0)
- Hard floors: enc3≥64, enc4≥128, bottleneck≥256, dec4≥128, dec3≥64
- Guard rails: Dice drop ≤4pp, MAE increase ≤1.5mm

**Phase 4a (static):** 8.11M → 4.57M (−43.7%) · Dice 97.64% · MAE 1.76mm · Wilcoxon p=0.0049  
Final channels: enc3: 128→71 · enc4: 256→129 · bottleneck: 512→257 · dec4: 256→129 · dec3: 128→65

**Phase 4b (temporal):** 8.90M → 5.20M (−41.6%) · Dice 96.00% · MAE 2.06mm · Wilcoxon p=0.1013 (NS)  
Final channels: enc3: 128→65 · enc4: 256→129 · bottleneck: 512→257 · dec4: 256→129 · dec3: 128→65  
FT recovery rate 103.8% — pruned+KD model marginally exceeded its unpruned teacher  
Phase 4b critical fix: TAM proj_in/proj_out resized with bottleneck pruning via concat-index weight slicing

---

## Ablation Study: Why Temporal Attention Is Essential

| Config | Architecture | Dice (%) | MAE (mm) | Note |
|--------|-------------|----------|----------|------|
| A | Phase 0 — static single frame | 97.75 | 1.65 | Baseline |
| B | 16-frame cine, identity attention | 81.48 | 19.37 | No TAM |
| C | 16-frame cine + temporal attention | 95.95 | 2.10 | Full system |

The B→C gap (+13.5pp Dice, −17.3mm MAE) confirms the TAM is load-bearing: without it, inconsistent per-frame predictions collapse consensus masks.

---

## System Architecture

### Phase 0 — Pre-Activation Residual U-Net (base_ch=32)

```
Input: [B, 1, 256, 384]  grayscale, per-image z-score normalised
  │
  ├─ enc1: ResidualBlock(1→32)    → [B,  32, 256, 384]
  │  MaxPool(2)                   → [B,  32, 128, 192]
  ├─ enc2: ResidualBlock(32→64)   → [B,  64, 128, 192]
  │  MaxPool(2)                   → [B,  64,  64,  96]
  ├─ enc3: ResidualBlock(64→128)  → [B, 128,  64,  96]
  │  MaxPool(2)                   → [B, 128,  32,  48]
  ├─ enc4: ResidualBlock(128→256) → [B, 256,  32,  48]
  │  MaxPool(2)                   → [B, 256,  16,  24]
  │
  bottleneck: ResidualBlock(256→512)              [B, 512, 16, 24]
  │
  ├─ up4: ConvTranspose2d(512→256, stride=2)
  │  dec4: ResidualBlock(512→256)  [cat up4 + enc4]
  ├─ up3: ConvTranspose2d(256→128, stride=2)
  │  dec3: ResidualBlock(256→128)  [cat up3 + enc3] → aux_d3: Conv2d(128→1) ← deep supervision
  ├─ up2: ConvTranspose2d(128→64,  stride=2)
  │  dec2: ResidualBlock(128→64)   [cat up2 + enc2] → aux_d2: Conv2d(64→1)  ← deep supervision
  ├─ up1: ConvTranspose2d(64→32,   stride=2)
  │  dec1: ResidualBlock(64→32)    [cat up1 + enc1]
  └─ final: Conv2d(32→1)

Loss: main_loss + 0.5×aux_d2_loss + 0.3×aux_d3_loss
      each = BoundaryWeightedBCE+DiceLoss (Sobel boundary weight map)
Training: 80 epochs · AdamW lr=3e-4 · CosineAnnealingLR(eta_min=3e-6)
```

### Phase 1 — Pseudo-LDDM v2 Cine Synthesis

```python
# Ornstein-Uhlenbeck probe motion (non-periodic, mean-reverting)
tx  = ornstein_uhlenbeck(n_frames, theta=0.15, sigma=2.0)  # translation x (pixels)
ty  = ornstein_uhlenbeck(n_frames, theta=0.15, sigma=1.5)  # translation y
rot = ornstein_uhlenbeck(n_frames, theta=0.20, sigma=0.40) # rotation (degrees)

# Per frame:
# 1. Perturb ellipse semi-axes (cross-sectional variation → non-trivial HC changes)
# 2. Warp affine with (tx, ty, rot)
# 3. Rician speckle: sqrt(real² + imag²)
# 4. Depth attenuation: multiply by exp(−k × depth_fraction)
# 5. Acoustic shadowing behind skull boundary  (Stage 4)
# 6. TGC drift between frames                  (Stage 4)
```

| Component | Course project v1 | This work v2 |
|-----------|------------------|--------------|
| Motion model | Sinusoidal (periodic) | Ornstein-Uhlenbeck (stochastic) |
| Cross-sectional variation | None | Per-frame ellipse axis perturbation |
| Noise model | Gaussian | Rician (physically correct) |
| Intensity variation | None | Depth-dependent attenuation |
| Acoustic effects | None | Shadowing + TGC drift |
| Clips generated | — | 806 |
| Temporal HC std | ~0.0 px (degenerate) | 10.33 px |

### Phase 2 — Temporal Attention Architecture

```
Input clip: [B, 16, 1, H, W]
  │
  Shared 2D Encoder (Phase 0 weights, base_ch=32)
  │
  Bottleneck: [B×16, 512, h, w]  →  reshape  →  [B, 16, 512, h, w]
  │
  ┌─────────────────────────────────────────────────┐
  │  Temporal Attention Module (TAM)  ~200K params   │
  │  Spatial avg-pool → [B, 16, 512]                 │
  │  Linear(512→256) + positional encoding           │
  │  LayerNorm → MHA(8 heads, dim=256) → residual    │
  │  LayerNorm → FFN(256→512→256, GELU) → residual   │
  │  Linear(256→512) → Sigmoid → gate                │
  │  Element-wise multiply with bottleneck sequence   │
  └─────────────────────────────────────────────────┘
  │
  Shared 2D Decoder  →  consensus + uncertainty
```

### Phase 4 — Hybrid Crossover Structural Pruning

```python
# ILR Importance Scoring
ILR(ch) = [0.6 × RMS_activation + 0.4 × filter_L1_norm + 0.2 × Frobenius_norm] / 1.2

# Phase 4b critical fix — TAM projection resizing with bottleneck pruning:
new_pi.weight = nn.Parameter(old_pi.weight.data[:, keep_indices])   # proj_in
new_po.weight = nn.Parameter(old_po.weight.data[keep_indices])       # proj_out
```

---

## Phase 3 — Bias Audit Results

| Subgroup | Finding |
|----------|---------|
| Early GA (<20w) | Modestly higher MAE — smaller skull, lower SNR |
| Mid GA (20–30w) | **Best performance** — optimal 2nd-trimester imaging window |
| Late GA (>30w) | Slight degradation — increasing acoustic shadowing |
| Low image quality (low Laplacian variance) | Higher MAE, consistent with clinical expectation |
| HC size range | No systematic over- or under-measurement bias |

**Business case:** 153 sonographer hrs/year saved per unit (~$5,347 USD/year at $35/hr, 20 scans/day).  
Inter-observer CV: 3.2% → 1.1% (Papageorghiou et al. 2014).

**Limitation:** HC18 contains no patient demographic metadata (ethnicity, BMI, scanner model). True demographic bias analysis requires multi-centre data.

---

## Deployed Application — Feature Overview

| Tab | What It Does |
|-----|-------------|
| Static Analysis (Phase 0 / 4a) | Single-frame segmentation · GradCAM++ · HC + GA · PDF report |
| Cine Analysis (Phase 2 / 4b) | Synthetic cine generation · temporal consensus · uncertainty · T×T attention · per-frame HC chart · reliability score · PDF report |
| Head-to-Head Comparison | All 4 models simultaneously · comparison table (HC, Dice, MAE, params, FLOPs, compression) |
| Model Card | Responsible AI documentation · bias analysis · regulatory pathway |

LLM integration: Claude Haiku generates clinical-language PDF reports. Rule-based fallback always available.  
Input validation: 7 clinical sanity checks block inference on corrupted input.  
Ground-truth upload: Optional live Dice + MAE vs annotation.

---

## Repository Structure

```
fetal-head-clinical-ai/
├── app/
│   ├── app.py              ← 4-tab Streamlit UI
│   ├── inference.py        ← Model architectures + inference pipelines
│   ├── report.py           ← PDF clinical report (LLM + rule-based)
│   ├── xai.py              ← GradCAM++, temporal attention, uncertainty
│   └── model_card.py       ← Responsible AI documentation
├── notebooks/
│   ├── fetal_head_phase0_baseline.ipynb
│   ├── fetal_head_phase1_lddm_v2.ipynb
│   ├── fetal_head_phase2_temporal_attention.ipynb
│   ├── fetal_head_phase3_v2.ipynb
│   ├── fetal_head_phase4a.ipynb
│   └── fetal_head_phase4b.ipynb
├── src/
│   ├── models/
│   │   ├── residual_unet.py
│   │   ├── temporal_net.py
│   │   └── pruned_unet.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── pseudo_lddm_v2.py
│   └── evaluate.py
└── results/
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
streamlit run app/app.py
```

**Option 3 — Inference module directly**
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

**HC18 Grand Challenge** — van den Heuvel et al., PLOS ONE 2018  
999 training + 335 test fetal head ultrasound images from 551 pregnancies, Radboud University Medical Center, Netherlands.

| Split | Static (Phase 0) | Temporal cine (Phase 2) |
|-------|-----------------|------------------------|
| Train | ~699 | 564 clips |
| Val | ~150 | 121 clips |
| Test | ~150 | 121 clips |
| Total | 999 | 806 clips |

---

## Model Weights

| File | Size | Phase |
|------|------|-------|
| phase0_model.pth | 97.5 MB | Static Residual U-Net |
| phase2_model.pth | 35.7 MB | Temporal attention system |
| 4a_best_pruned_ft_v10.pth | 18.3 MB | Pruned static (−43.7%) |
| 4b_best_pruned_ft_v10.pth | 20.9 MB | Pruned temporal (−41.6%) |

Download: https://huggingface.co/spaces/TarunSadarla2606/fetal-head-clinical-ai/tree/main

---

## Regulatory Notice

⚠️ **Research prototype. Not FDA-cleared. Not CE-marked. Not for clinical use.**

Classified as SaMD Class II under FDA 21 CFR Part 892. Requires FDA 510(k) clearance (US) and IVDR Class B certification (EU) before clinical deployment. All measurements require verification by a certified sonographer.

---

## Citation

```bibtex
@misc{sadarla2026fetalhead,
  author      = {Sadarla, Tarun},
  title       = {Structural Pruning of Residual U-Net and Temporal Segmentation Models
                 for Efficient Fetal Head Circumference Estimation in Ultrasound Cine Clips},
  year        = {2026},
  institution = {University of North Texas},
  note        = {MS Artificial Intelligence (Biomedical Concentration).
                 Directed study: Prof. Russel Pears · Fetal head project: Prof. Xiaohui Yuan},
  url         = {https://huggingface.co/spaces/TarunSadarla2606/fetal-head-clinical-ai}
}
```

---

*Independent research · MS Artificial Intelligence (Biomedical Concentration) · University of North Texas · 2026*  
*Tarun Sadarla · tarunsadarla26@gmail.com · [LinkedIn]((https://www.linkedin.com/in/tarun-sadarla-715026231/)) · [Portfolio](https://tarunsadarla2606.github.io)*
