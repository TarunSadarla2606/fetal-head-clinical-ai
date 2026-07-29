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
  "cine_overlay_gif": "data:image/gif;base64,R0lGODlh…", // contour on every frame
  "cine_loop_gif":    "data:image/gif;base64,R0lGODlh…", // raw loop, no prediction
  "cine_per_frame_hc": [92.1, 92.3, null, 92.0, …],      // per-frame HC, frame-aligned
  "cine_frame_count": 16,
  "cine_key_frame_index": 8
}
```

- **`cine_overlay_gif`** — the predicted skull contour drawn on **every** frame,
  so the boundary can be watched tracking the head through probe motion rather
  than judged from a single consensus still. Each frame is labelled with its
  position in the loop (`9/16`) and its own HC, and the frame that
  `overlay_b64` is rendered from is outlined in amber and tagged `KEY FRAME`,
  so the animation and the still view can be reconciled.
- **`cine_loop_gif`** — the same loop with nothing drawn on it: what the
  temporal model was actually fed. Lets a reader judge the prediction against
  the raw frames.
- **`cine_per_frame_hc`** — one HC per frame, aligned with frame index (`null`
  where HC was unmeasurable). This is what `reliability` and `hc_std_mm` are
  computed from, so exposing it makes the stability claim checkable.

Both GIFs are `data:` URIs — nothing is written to disk, and the response stays
self-contained. Each is budgeted (300 KB overlay / 200 KB loop) by subsampling
frames and downscaling along a quality ladder until it fits; a typical response
is ~450 KB of GIF. If an animation cannot be built its field is `null` and
clients fall back to the static `overlay_b64`.

`mode: "single_frame"` responses (`phase0` / `phase4a`) carry none of these
fields and are unchanged.

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

## Deployment

`main` is mirrored to both Hugging Face Spaces by
`.github/workflows/deploy-hf.yml` on every push. A Space is a **separate git
remote**, not a GitHub mirror, so without this workflow merging to `main` does
not update the live app.

Requires one repository secret:

| Secret | Value |
|---|---|
| `HF_TOKEN` | A Hugging Face access token with **write** permission — https://huggingface.co/settings/tokens |

Optionally override the Space names with repository *variables* `HF_API_SPACE`
and `HF_STREAMLIT_SPACE`. The workflow can also be run manually
(**Actions → Deploy to Hugging Face Spaces → Run workflow**) to push just one
Space. It checks out Git LFS objects so the Spaces receive real model weights
rather than pointer files.

### How it handles large files

Hugging Face enforces **two** separate rules, and this repo tripped both:

| Rule | Caught |
|---|---|
| No non-LFS file over **10 MiB** | `4a`/`4b` weights (18 MB / 21 MB), committed as raw blobs before `.gitattributes` tracked `*.pth` |
| No **binary** file outside LFS/Xet (~100 KB) | 9 of the 12 `demo_subjects/*.png` — the 3 smallest slipped under |

So the deploy declares LFS for every binary *extension* rather than chasing
size thresholds, and a guard enforces the invariant directly: no non-empty
binary blob may ride as a raw git object.

Rather than rewrite GitHub history, the workflow builds a **single squashed
orphan commit** and converts on the way out:

```bash
git checkout --orphan hf-deploy
git rm -r --cached .   # empties the index ...
git add -A             # ... so every path is re-run through its clean filter
```

Emptying the index is the load-bearing part. A plain `git add -A` sees
byte-identical content, skips the clean filter, and leaves the raw blobs in
place — the push then fails. Re-adding from an empty index forces the `*.pth`
LFS filter to run, so the snapshot carries pointers even though the source
commit does not, and the objects upload to HF's LFS store on push.

Squashing also sidesteps the history problem: the hook scans the whole push, so
the old raw-blob commits would keep failing it however clean the tip is. A Space
is a deployment target, not a history archive.

The extension list is appended to `.gitattributes` **in CI only**, after the
index is emptied — editing it before `git rm --cached` makes that command refuse
the file, and committing the rules upstream would require pushing PNG LFS
objects to GitHub too.

The guard detects binary content by looking for a **NUL byte in the first 8 KB**,
which is git's own heuristic. It deliberately does *not* use `grep -I`: in the C
locale that reports valid UTF-8 (em dashes, arrows, emoji) as binary, which
false-positives on `app.py`, `inference.py`, and every notebook.

**Adding a new large file:** make sure `.gitattributes` covers its extension. If
it is already committed as a raw blob the deploy still handles it, but the GitHub
repo carries the full blob forever — to fix that properly, re-add it:

```bash
git rm --cached path/to/file && git add path/to/file
git show :path/to/file | head -c 40   # -> "version https://git-lfs..."
```

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
