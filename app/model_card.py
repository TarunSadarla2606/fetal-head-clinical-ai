"""
model_card.py — Model Card for Fetal Head Circumference Clinical AI

Implements the Model Card tab in the Streamlit application.
Based on the Mitchell et al. (2019) model card framework and
directly references Phase 3 (XAI/bias audit) findings from this project.

Rendered as a static structured page — no model inference occurs here.
"""

import streamlit as st


def render_model_card():
    st.subheader("Model Card — Fetal Head Circumference Clinical AI")
    st.caption(
        "Following Mitchell et al. (2019) model card framework. "
        "This card documents intended use, training data, evaluation results, "
        "fairness considerations, and known limitations."
    )

    # ── Model details ─────────────────────────────────────────────────────────
    with st.expander("📌 Model details", expanded=True):
        st.markdown("""
**Model family:** Four variants across two architecture classes.

| Variant | Architecture | Parameters | Role |
|---------|-------------|------------|------|
| Phase 0 | ResidualUNetDS (base_ch=32) | 8.11M | Static baseline |
| Phase 4a | ResidualUNetDS (pruned) | 4.57M | Compressed static |
| Phase 2 | TemporalFetaSegNet | 8.90M | Temporal baseline |
| Phase 4b | TemporalFetaSegNet (backbone pruned) | 5.20M | Compressed temporal |

**Model type:** Supervised segmentation CNN (Phase 0/4a); CNN-Transformer hybrid with temporal self-attention (Phase 2/4b).

**Input:** Single grayscale ultrasound image (Phase 0/4a) or 16-frame cine clip (Phase 2/4b). Images resized to 256×384 internally.

**Output:** Binary segmentation mask of the fetal skull boundary, from which head circumference (HC) and gestational age (GA) are derived using the Ramanujan ellipse approximation and Hadlock (1984) polynomial respectively.

**Training framework:** PyTorch 2.x. Training hardware: NVIDIA GPU (Colab/Kaggle environment).

**Version:** Phase 4a v10 / Phase 4b v10 (April 2026). Pruning: Hybrid Crossover channel merging with iterative prune-fine-tune cycles and Knowledge Distillation.

**License:** Research prototype. Not for clinical deployment without regulatory review.

**Contact:** Tarun Sadarla · University of North Texas · MS Artificial Intelligence 2026
        """)

    # ── Intended use ──────────────────────────────────────────────────────────
    with st.expander("🎯 Intended use"):
        st.markdown("""
**Primary intended use:**
Research demonstration of automated fetal head circumference estimation from 2D obstetric ultrasound, incorporating structural model compression for resource-constrained deployment scenarios.

**Primary intended users:**
- Researchers in medical imaging AI and neural network compression
- Sonographers and radiologists evaluating AI-assisted measurement tools (research context only)
- ML engineers assessing deployment-ready medical segmentation pipelines

**Out-of-scope uses:**
- Clinical diagnosis or treatment decision support without regulatory clearance
- Replacement of trained sonographer measurement and interpretation
- Deployment on non-HC18-domain ultrasound equipment without revalidation
- Use outside the 14–42 week gestational age range where Hadlock (1984) is validated
- Measurement of fetal structures other than the fetal head/skull boundary
        """)

    # ── Training data ─────────────────────────────────────────────────────────
    with st.expander("📊 Training data"):
        st.markdown("""
**Dataset:** HC18 Grand Challenge — Automated Measurement of Fetal Head Circumference
- **Source:** Radboud University Medical Center, Nijmegen, Netherlands
- **Images:** 999 fetal head ultrasound images from 551 patients
- **Annotations:** Ellipse-fit skull boundary contours, converted to binary masks
- **Split:** 70% train (699) / 10% validation (100) / 20% test (199)
- **Resolution:** 800×540 pixels (variable), resized to 384×256 for training

**Temporal training data (Phase 2/4b):**
- 806 synthetic 16-frame cine-loops generated from HC18 images via Pseudo-LDDM v2
- Pseudo-LDDM v2 applies Ornstein-Uhlenbeck probe motion + Rician speckle simulation + depth attenuation to synthesise realistic ultrasound motion artefacts
- No real video ultrasound data was used; the temporal pipeline is validated on synthetic sequences only

**Data limitations:**
- Single institution (Radboud UMC) — no multi-centre data
- No demographic metadata available (scanner model, gestational age at scan, maternal BMI)
- No ethnicity or body habitus stratification possible with this dataset
        """)

    # ── Evaluation results ────────────────────────────────────────────────────
    with st.expander("📈 Evaluation results"):
        st.markdown("""
All metrics reported on the HC18 held-out test set (n=199 images / n=121 cine clips for temporal models).

**Static models (Phase 0 / Phase 4a)**

| Metric | Phase 0 (baseline) | Phase 4a (compressed) | Delta |
|--------|-------------------|-----------------------|-------|
| Dice coefficient | 97.75% | 97.64% | −0.12pp |
| MAE (mm) | 1.65 mm | 1.76 mm | +0.11 mm |
| R² | 0.9985 | 0.9983 | — |
| ISUOG ≤3mm | PASS | PASS | — |
| Parameters | 8.11M | 4.57M | −43.7% |
| FLOPs | 21.58 GMACs | 16.56 GMACs | −23.3% |
| Inference latency | 11.6 ms | 9.9 ms | 1.17× faster |
| Wilcoxon p (Dice) | — | 0.0049 | Significant |

**Temporal models (Phase 2 / Phase 4b)**

| Metric | Phase 2 (baseline) | Phase 4b (compressed) | Delta |
|--------|-------------------|-----------------------|-------|
| Dice coefficient | 95.95% | 96.00% | +0.05pp |
| MAE (mm) | 2.10 mm | 2.06 mm | −0.04 mm |
| ISUOG ≤3mm | PASS | PASS | — |
| Parameters | 8.90M | 5.20M | −41.6% |
| FLOPs/frame | 21.58 GMACs | 16.44 GMACs | −23.8% |
| Clip latency | 182.7 ms | 171.5 ms | 1.07× faster |
| Wilcoxon p (Dice) | — | 0.1013 (NS) | Statistically indistinguishable |

**Ablation study (Phase 2 — temporal attention removed):**
Dice 81.48% · MAE 19.37mm — confirming that the temporal attention module is essential, not decorative.
        """)

    # ── Fairness and bias ─────────────────────────────────────────────────────
    with st.expander("⚖️ Fairness and bias analysis (Phase 3)"):
        st.markdown("""
A structured bias audit was conducted in Phase 3 of this project as part of the clinical AI pipeline validation.

**Subgroup analysis performed:**
- **By gestational age trimester:** First (<14w), Second (14–28w), Third (≥28w)
- **By image quality proxy:** Laplacian variance tertiles (low / mid / high sharpness)
- **By HC size:** Small (<250mm), Medium (250–310mm), Large (>310mm)

**Key findings:**
- Performance is consistent across gestational age groups; no significant trimester bias detected
- Low-quality images (low Laplacian variance) show modestly higher MAE — expected and consistent with clinical practice
- No systematic over- or under-measurement bias by HC size range

**Limitations of bias analysis:**
- HC18 contains no patient demographic metadata (ethnicity, BMI, scanner model)
- True subgroup fairness analysis across demographic groups is not possible with this dataset
- Multi-centre validation on diverse ultrasound equipment is required before clinical deployment

**GradCAM++ XAI findings (Phase 3):**
- The model correctly attends to the skull boundary (calvarium)
- High-activation regions correspond anatomically to the expected measurement plane
- No systematic attention to irrelevant background structures was observed
- Uncertainty maps (temporal models) correctly flag frames with high probe motion

**Recommendation:** Before clinical deployment, validate on multi-centre data with documented demographic and equipment metadata. Commission a prospective subgroup fairness study stratified by gestational age, maternal characteristics, and scanner model.
        """)

    # ── Ethical considerations ────────────────────────────────────────────────
    with st.expander("🔒 Ethical considerations"):
        st.markdown("""
**Regulatory classification:**
This system performs an automated biometric measurement that influences clinical management decisions (gestational age assessment, growth monitoring). It would be classified as **SaMD (Software as a Medical Device) Class II** under FDA 21 CFR Part 880 and EU MDR Annex VIII. It is **not FDA-cleared** and **not CE-marked**. It must not be used for clinical decisions without regulatory approval.

**Human oversight requirement:**
All measurements must be verified by a trained sonographer before clinical use. The system is designed to assist, not replace, clinical expertise.

**Transparency:**
- Model weights, architecture, and training code are publicly available
- This model card documents known limitations and failure modes
- Uncertainty quantification (temporal models) flags low-confidence predictions

**Data privacy:**
- This application does not store, log, or transmit uploaded images
- No patient identifiers are collected
- Uploaded images are processed in-memory and discarded after the session

**Known failure modes:**
- Poor image quality (heavy speckle, shadowing, partial skull views) reduces segmentation accuracy
- Incorrect pixel spacing entry directly degrades HC/GA accuracy — always verify DICOM metadata
- Out-of-distribution images (non-HC18 domain ultrasound machines) may produce unreliable results
- The Hadlock (1984) formula is validated for 14–42 weeks; extrapolation outside this range is unreliable
- The temporal model is validated on synthetic cine-loops only; real video ultrasound has not been tested
        """)

    # ── Caveats and recommendations ───────────────────────────────────────────
    with st.expander("📝 Caveats and recommendations for future work"):
        st.markdown("""
**Current limitations:**
1. Single-institution dataset (HC18, Radboud UMC) — multi-centre validation required
2. Temporal model trained on synthetic cine-loops — real video validation needed
3. TemporalAttentionModule not pruned — 794K parameters remain (TAM head pruning or low-rank decomposition identified as next step)
4. No INT8/FP16 quantisation applied — complementary 2–4× gains available
5. No ONNX/TensorRT export validated — edge deployment (Jetson, mobile) requires additional work

**Recommended next steps:**
- Per-block sequential fine-tuning (freeze all except one block) for improved pruning recovery
- Multi-objective Pareto optimisation for Dice vs compression trade-off exploration
- Prospective multi-centre validation study
- Post-pruning INT8 quantisation
- ONNX export and TensorRT optimisation for point-of-care deployment

**References:**
- HC18 dataset: van den Heuvel et al., Data in Brief, 2018
- Hadlock formula: Hadlock FP et al., AJR, 1984;143:97-100
- ISUOG guidelines: Ultrasound Obstet Gynecol, 2010;35(3):348-361
- Model card framework: Mitchell et al., FAccT, 2019
- Hybrid Crossover pruning: adapted from filter merging literature (Li et al., ICLR 2017)
- SIRFP (segmentation pruning): Wu et al., AAAI 2025
        """)

    st.markdown("---")
    st.caption(
        "Model card last updated: April 2026 · "
        "Tarun Sadarla · University of North Texas · MS Artificial Intelligence · "
        "Directed study under Prof. Russel Pears"
    )
