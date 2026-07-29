---
title: Fetal Head Clinical AI
emoji: 🫇
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Fetal Head Circumference Clinical AI

**Automated HC measurement · Gestational age estimation · Structural pruning compression · Temporal uncertainty quantification · Clinical report generation**

[![Tests](https://github.com/TarunSadarla2606/fetal-head-clinical-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/TarunSadarla2606/fetal-head-clinical-ai/actions/workflows/ci.yml)
[![HuggingFace Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/TarunSadarla2606/fetal-head-clinical-ai)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ISUOG](https://img.shields.io/badge/ISUOG-±3mm%20PASS-brightgreen)](https://www.isuog.org)

Independent research continuation of a course project (CSCE 6260, UNT Fall 2025) — developed post-semester to advance from 86% Dice / 17.25mm MAE to a deployable clinical-grade pipeline. All work sole-authored.  
**Directed study:** Prof. Russel Pears · **Fetal head project:** Prof. Xiaohui Yuan · University of North Texas, 2026.

---

## API Endpoints

This Space exposes a FastAPI inference server:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Status, loaded models, device |
| GET | `/demo/list` | List demo subject filenames |
| GET | `/demo/{filename}` | Serve demo subject image |
| POST | `/infer` | Fetal head HC measurement |
| GET | `/api/docs` | Interactive Swagger UI |

### Cine mode returns an animated overlay

Selecting a temporal variant (`phase2` / `phase4b`) puts `/infer` in cine mode.
The single uploaded frame is expanded into a 16-frame cine-loop with
Pseudo-LDDM v2 (Ornstein-Uhlenbeck probe motion + Rician speckle + depth
attenuation), the temporal model runs over the whole sequence, and the
response carries an extra field:

```jsonc
{
  "mode": "cine_clip",
  "cine_overlay_gif": "data:image/gif;base64,R0lGODlh…"  // animated overlay
}
```

`cine_overlay_gif` is an animated GIF showing the predicted skull contour
drawn on **every** frame of the loop, so the boundary can be watched tracking
the head through probe motion rather than judged from a single consensus
still. It is a `data:` URI — nothing is written to disk, and the response
stays self-contained.

The animation is budgeted to stay under ~300 KB: frames are subsampled and the
image downscaled along a quality ladder until it fits. If it cannot be built
the field is `null`, and clients fall back to the static `overlay_b64`.

`mode: "single_frame"` responses (`phase0` / `phase4a`) do not carry the field
and are unchanged.

> **Note:** cine mode previously tiled 16 identical copies of the uploaded
> frame, which made `reliability` a constant 1.0. It now measures real
> inter-frame agreement, so temporal-variant reliability and `hc_std_mm`
> values differ from earlier releases.

---

## Four Model Variants

| Phase | Type | Architecture | Dice (%) | MAE (mm) | Params | vs Baseline |
|-------|------|--------------|----------|----------|--------|-------------|
| Phase 0 | Static | Residual U-Net + deep supervision | **97.75** | **1.65** | 8.11M | — |
| Phase 4a | Static | Phase 0 — Hybrid Crossover pruned | **97.64** | **1.76** | 4.57M | −43.7% params |
| Phase 2 | Temporal (16 frames) | 2D U-Net + temporal self-attention | **95.95** | **2.10** | 8.90M | — |
| Phase 4b | Temporal (16 frames) | Phase 2 — backbone pruned, TAM intact | **96.00** | **2.06** | 5.20M | −41.6% params |

All results on HC18 test set (Radboud UMC, Netherlands). ✅ All models pass ISUOG ≤3mm threshold.

---

## Model Weights

| File | Size | Phase |
|------|------|-------|
| phase0_model.pth | 97.5 MB | Static Residual U-Net |
| phase2_model.pth | 35.7 MB | Temporal attention system |
| 4a_best_pruned_ft_v10.pth | 18.3 MB | Pruned static (−43.7%) |
| 4b_best_pruned_ft_v10.pth | 20.9 MB | Pruned temporal (−41.6%) |

---

## Regulatory Notice

⚠️ **Research prototype. Not FDA-cleared. Not CE-marked. Not for clinical use.**

---

*Independent research · MS Artificial Intelligence (Biomedical Concentration) · University of North Texas · 2026*  
*Tarun Sadarla · tarunsadarla26@gmail.com*
