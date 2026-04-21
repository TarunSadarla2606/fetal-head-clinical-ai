# app/

This directory mirrors the deployed HuggingFace Space:
https://huggingface.co/spaces/TarunSadarla2606/fetal-head-clinical-ai

## Files

| File | Size | Purpose |
|------|------|---------|
| `app.py` | ~38 KB | 4-tab Streamlit UI — Static, Cine, Comparison, Model Card |
| `inference.py` | ~34 KB | Model architectures (all 4 phases) + inference pipelines |
| `report.py` | ~49 KB | PDF clinical report generation (LLM + rule-based) |
| `xai.py` | ~13 KB | GradCAM++, temporal attention heatmap, uncertainty visualization |
| `model_card.py` | ~12 KB | Responsible AI documentation (NIST AI RMF) |

## Running locally

```bash
# From repo root
pip install -r requirements.txt

# Download weights (see models/README.md) to repo root, then:
streamlit run app/app.py
```

## Architecture modules (defined in inference.py)

| Class | Phase | Type | Params |
|-------|-------|------|--------|
| `ResidualUNetDS` | 0 | Static Residual U-Net + deep supervision | 8.11M |
| `PrunedResidualUNetDS` | 4a | Hybrid crossover pruned static | 4.57M |
| `TemporalFetaSegNet` | 2 | 2D U-Net + temporal self-attention | 8.90M |
| `TemporalFetaSegNet` (pruned backbone) | 4b | Pruned temporal | 5.20M |
| `TemporalAttentionModule` | — | Bottleneck TAM (~200K) | ~200K |
| `PrunedResidualBlock` | 4a/4b | Post-pruning residual block | — |
