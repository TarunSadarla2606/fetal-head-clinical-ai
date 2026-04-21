# models/

Model weights are too large to host on GitHub. All four checkpoints are hosted on the HuggingFace Space.

---

## Download

**Manual (browser)**  
https://huggingface.co/spaces/TarunSadarla2606/fetal-head-clinical-ai/tree/main

**Programmatic (huggingface_hub)**
```python
from huggingface_hub import hf_hub_download

files = [
    "phase0_model.pth",
    "phase2_model.pth",
    "4a_best_pruned_ft_v10.pth",
    "4b_best_pruned_ft_v10.pth",
]

for fname in files:
    hf_hub_download(
        repo_id="TarunSadarla2606/fetal-head-clinical-ai",
        filename=fname,
        repo_type="space",
        local_dir=".",  # saves to project root
    )
```

**CLI (wget)**
```bash
base="https://huggingface.co/spaces/TarunSadarla2606/fetal-head-clinical-ai/resolve/main"
wget $base/phase0_model.pth
wget $base/phase2_model.pth
wget "$base/4a_best_pruned_ft_v10.pth"
wget "$base/4b_best_pruned_ft_v10.pth"
```

---

## Checkpoint Summary

| File | Size | Phase | Architecture | Dice | MAE |
|------|------|-------|-------------|------|-----|
| `phase0_model.pth` | 97.5 MB | 0 | ResidualUNetDS | 97.75% | 1.65 mm |
| `phase2_model.pth` | 35.7 MB | 2 | TemporalFetaSegNet | 95.95% | 2.10 mm |
| `4a_best_pruned_ft_v10.pth` | 18.3 MB | 4a | PrunedResidualUNetDS | 97.64% | 1.76 mm |
| `4b_best_pruned_ft_v10.pth` | 20.9 MB | 4b | TemporalFetaSegNet (pruned) | 96.00% | 2.06 mm |

---

## Loading

```python
from app.inference import load_phase0, load_phase2, load_phase4a, load_phase4b

# Each loader auto-detects CUDA, extracts channel_counts from checkpoint,
# and returns model in .eval() mode on the target device
phase0  = load_phase0("phase0_model.pth")
phase4a = load_phase4a("4a_best_pruned_ft_v10.pth")
phase2  = load_phase2("phase2_model.pth")
phase4b = load_phase4b("4b_best_pruned_ft_v10.pth")
```
