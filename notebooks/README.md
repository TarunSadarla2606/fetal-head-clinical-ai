# notebooks/

Six Google Colab training notebooks covering the full development pipeline.  
Environment: **Google Colab, NVIDIA T4 GPU** (not Kaggle).

---

## Notebook Index

| Notebook | Phase | Type | Description | Key Output |
|----------|-------|------|-------------|------------|
| `fetal_head_phase0_baseline.ipynb` | Phase 0 | Training | Residual U-Net with pre-activation blocks and deep supervision. BoundaryWeightedDiceLoss (Sobel edge weights). 100 epochs, Adam lr=1e-3, ReduceLROnPlateau. 80/10/10 split. | `phase0_model.pth` · Dice 97.75% · MAE 1.65 mm |
| `fetal_head_phase1_lddm_v2.ipynb` | Phase 1 | Data generation | Pseudo-LDDM v2 synthetic cine synthesis. Ornstein-Uhlenbeck motion, cross-sectional mask variation (critical fix), Rician speckle, depth attenuation, acoustic shadowing, TGC drift. 4 fidelity stages. | 806 cine clips (.npz) · Temporal HC std 10.33 px |
| `fetal_head_phase2_temporal_attention.ipynb` | Phase 2 | Training | TemporalFetaSegNet: shared 2D U-Net encoder + TAM at bottleneck. Three-stage training (frozen→partial→full). 564/121/121 clip split. | `phase2_model.pth` · Val Dice 96.29% · Test Dice 95.95% · MAE 2.10 mm |
| `fetal_head_phase3_v2.ipynb` | Phase 3 | Audit | XAI, bias audit, and governance. GradCAM++ on Phase 0. T×T temporal attention heatmaps for Phase 2. Subgroup Dice/MAE by GA trimester, image quality, HC range. No model is trained here. | Bias report · fairness findings by trimester |
| `fetal_head_phase4a.ipynb` | Phase 4a | Pruning | Hybrid Crossover structural pruning of Phase 0 static model. ILR importance scoring, burst-sequential pruning, 3-cycle KD recovery. Guard rails: Dice drop ≤4pp, MAE increase ≤1.5mm. | `4a_best_pruned_ft_v10.pth` · 4.57M params (−43.7%) · Dice 97.64% · MAE 1.76 mm |
| `fetal_head_phase4b.ipynb` | Phase 4b | Pruning | Same Hybrid Crossover framework applied to Phase 2 temporal model. Additional TAM proj_in/proj_out resizing when bottleneck is pruned. Surgical concat-index decoder fix. | `4b_best_pruned_ft_v10.pth` · 5.20M params (−41.6%) · Dice 96.00% · MAE 2.06 mm |

---

## Key Shared Constants

```python
INPUT_H   = 256     # model input height
INPUT_W   = 384     # model input width
N_FRAMES  = 16      # frames per cine clip
BASE_CH   = 32      # U-Net base channel multiplier
SEED      = 42
DEVICE    = 'cuda'  # NVIDIA T4 on Colab
```

## Dataset

All notebooks use the **HC18 Grand Challenge** dataset.  
Annotations are ellipse rings — `fill_hollow_mask()` converts to solid disks before training.

```
Static (Phase 0):    799 train / 100 val / 100 test  (from 999 total)
Temporal (Phase 2):  564 train / 121 val / 121 test  (from 806 clips)
```
