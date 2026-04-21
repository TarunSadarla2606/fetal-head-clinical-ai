# data/

Raw data is **not committed** to this repository. Download and prepare as follows.

---

## HC18 Grand Challenge Dataset

**Source:** van den Heuvel TLA et al. *PLOS ONE* 2018.  
**DOI:** [10.1371/journal.pone.0200412](https://doi.org/10.1371/journal.pone.0200412)

### Download options

**Option 1 — Kaggle (recommended, fast)**
```bash
kaggle datasets download -d sahliz/hc18
unzip hc18.zip -d data/hc18/
```

**Option 2 — Grand Challenge (official)**
```
https://hc18.grand-challenge.org/
```
Requires free account registration.

### Dataset structure after download

```
data/hc18/
├── training_set/
│   ├── 000_HC.png           ← Ultrasound image
│   ├── 000_HC_Annotation.png  ← Ground truth mask (ellipse outline)
│   ├── 001_HC.png
│   ├── 001_HC_Annotation.png
│   └── ... (999 pairs total)
├── test_set/
│   ├── 000_HC.png
│   └── ... (335 images, no annotations)
└── training_set_pixel_size_and_HC.csv   ← pixel_size_mm, HC_mm per image
```

### Key dataset facts

| Property | Value |
|----------|-------|
| Training images | 999 |
| Test images | 335 |
| Image size | 800×540 px |
| Default pixel spacing | 0.070 mm/pixel |
| Gestational age range | ~14–40 weeks |
| Source hospital | Radboud University Medical Center, Netherlands |
| Pregnancies | 551 |

### Important: annotation format

HC18 ground truth masks are **ellipse outlines (1px rings)**, not solid regions.  
The `fill_hollow_mask()` function in `src/data/dataset.py` converts them to solid disks before training and evaluation. Do not skip this step — Dice loss on ring annotations produces unstable gradients.

### Splits used in this project

| Split | Proportion | Images |
|-------|-----------|--------|
| Train | 75% | ~749 |
| Val | 20% | ~200 |
| Test | 5% | ~50 |

Stratified by HC range to ensure coverage across all gestational ages.
