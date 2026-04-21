# notebooks/

Kaggle training notebooks for all four development phases. Run on Kaggle (dual NVIDIA T4, 16GB GDDR6 each, mixed-precision).

Notebooks will be uploaded here after review. The table below describes what each contains.

---

## Notebook Index

| Notebook | Phase | What It Does | Key Output |
|----------|-------|-------------|------------|
| `phase0_residual_unet.ipynb` | Phase 0 | Full training pipeline: HC18 loading, preprocessing, augmentation (elastic deformation + Rician noise + coarse dropout), Residual U-Net with deep supervision, boundary-weighted BCE+Dice loss, evaluation | `phase0_model.pth` · Dice 97.75% · MAE 1.65mm |
| `phase1_pseudo_lddm_v2.ipynb` | Phase 1 | Pseudo-LDDM v2 cine synthesis: OU motion, cross-sectional mask variation, Rician speckle, depth attenuation, acoustic shadowing, TGC drift. Generates 806 cine clips. | 806 synthetic cine-loop clips |
| `phase2_temporal_attention.ipynb` | Phase 2 | Three-stage TAM training (frozen → partial → full fine-tune) on synthetic cines, ablation study (Config A/B/C), per-frame uncertainty computation | `phase2_model.pth` · Dice 95.95% · MAE 2.10mm |
| `phase4_pruning.ipynb` | Phase 4 | Hybrid crossover filter synthesis pruning of Phase 0 → 4a and Phase 2 → 4b, sensitivity analysis, fine-tuning, compression vs accuracy trade-off | `4a_best_pruned_ft_v10.pth` · `4b_best_pruned_ft_v10.pth` |
| `ablation_study.ipynb` | — | Side-by-side comparison of Config A (static), Config B (cine, no attention), Config C (full system). Produces ablation table and per-case analysis. | Ablation table: A=97.36%, B=81.48%, C=95.71% |

---

## Environment

- Platform: Kaggle (dual T4, 32GB total VRAM)
- Mixed-precision: `torch.cuda.amp` (Phase 0 TF32, Phase 2+ AMP)
- Framework: PyTorch 2.x
- Data path: `/kaggle/input/hc18/`
