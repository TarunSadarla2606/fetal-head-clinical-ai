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

## RAG Q&A — grounded answers with citations

`POST /findings/{id}/ask` answers a free-text question about a specific
measurement using **only** the reference material in `knowledge/` plus that
measurement's own numbers. The point is auditability: an unsourced LLM
paragraph cannot be checked, so every answer ships with the evidence it used.

```jsonc
{
  "answer": "Reliability is inter-frame agreement … [project_metrics.md § Temporal reliability score — definition]",
  "citations": ["project_metrics.md § Temporal reliability score — definition"],
  "chunks": [ /* every excerpt supplied to the model, verbatim, with scores */ ],
  "grounded": true,        // false ⇒ nothing retrieved, model never called
  "used_llm": true,        // false ⇒ fallback path, excerpts returned raw
  "any_provisional": true, // a cited section is not yet verbatim-sourced
  "disclaimer": "…not cleared for clinical diagnosis…"
}
```

**Guarantees, each covered by a test:**

- **No retrieval ⇒ no LLM call.** An off-topic question returns a refusal;
  the model is never given the chance to answer from memory.
- **Evidence is returned, not just cited.** `chunks` carries the full text the
  model saw, so an answer can be checked against it.
- **Citations reflect the answer**, not everything retrieved — only labels the
  answer actually referenced are listed.
- **Diagnosis is refused** in the system prompt, and the standard disclaimer
  rides on every response.
- **Degrades rather than fails.** No `ANTHROPIC_API_KEY`, or a failed call,
  returns the retrieved excerpts with `used_llm: false`.
- **Degrading says why.** A failed call returns the reason in `llm_error`
  rather than swallowing it. Graceful degradation that hides the cause is
  indistinguishable from a feature that quietly never worked — which is exactly
  what happened here: every question fell back, and the generic message gave
  nobody anything to act on.

`GET /knowledge/status` reports the index: chunk count, files, and which
sections are still provisional.

### Diagnosing a silent LLM failure

`GET /llm/status` makes one minimal live call and reports what happened:

```jsonc
{
  "key_present": true,
  "key_prefix": "sk-ant-…",   // never the full key
  "sdk_installed": true,
  "model": "claude-haiku-4-5-20251001",
  "call_ok": false,
  "error": "AuthenticationError: invalid x-api-key",
  "remedy": "The ANTHROPIC_API_KEY on the server is not valid…"
}
```

When the failure looks like connectivity, the response also carries a
`network` block that tests each layer separately:

```jsonc
{
  "network": {
    "host": "api.anthropic.com",
    "dns_ok": true,
    "resolved_ips": ["160.79.104.10"],
    "tcp_ok": false,
    "tls_ok": null,          // not attempted — the layer below it failed
    "proxy_env": {"HTTPS_PROXY": "http://***@proxy:8080"},
    "failed_layer": "tcp"
  }
}
```

This exists because `APIConnectionError` collapses DNS failure, a refused
socket, TLS interception and a misconfigured proxy into the single string
`"Connection error."` — which names none of them. `failed_layer` is the deepest
layer that actually works, and the `https` step deliberately bypasses the
Anthropic SDK: if raw HTTPS succeeds while the SDK fails, the fault is the
client or its proxy handling, not the network.

Proxy environment variables are reported because httpx honours them, so a stale
`HTTPS_PROXY` breaks every call while the network itself is fine. Values are
redacted to strip any embedded `user:password@`.

`_sanitize_error` walks the `__cause__`/`__context__` chain: the wrapper
exception is often contentless and the real reason sits one frame down.

### Redaction runs in three layers

A single pattern-matching layer is not enough, and this repo has the scar to
prove it: `sk-[A-Za-z0-9_-]{8,}` stops at a newline, so a key pasted wrapped
across two lines was redacted up to the break and printed from there on — a
live credential, through a public endpoint. So:

1. **Rejected header values are dropped wholesale.** The bytes of a refused
   header *are* the credential; no pattern has to be trusted with them.
2. **Known secret values are replaced by value**, read from the environment —
   the whole string and each whitespace-separated fragment of it. This does not
   depend on the credential having any particular shape.
3. **Pattern matching last**, now spanning internal whitespace, for anything the
   first two layers did not know about.

Messages are also length-capped. Any new secret-bearing environment variable
belongs in `_SECRET_ENV_VARS`.

### Malformed keys are named, not misattributed

`get_api_key()` strips surrounding whitespace — env vars routinely collect a
trailing newline and dropping it is always safe. Whitespace *inside* the value
is reported rather than repaired: httpx refuses to send a header containing a
newline, which raises `APIConnectionError` and reads as a network outage when
the real fault is a credential pasted across two lines. Silently joining the
fragments would mean guessing at a credential, so `/llm/status` names it
instead and never calls the API.

> `app/report.py` swallows LLM errors the same way this code used to, so
> LLM-mode report generation can also fall back to rule-based prose without
> saying why. `/llm/status` diagnoses the shared key and client, but that
> function has not been converted yet.

### The knowledge base

Markdown in `knowledge/`, one retrievable chunk per `##` heading. See
`knowledge/README.md` for the chunking rules.

⚠️ Sections marked `TODO(verbatim)` are **paraphrase, not sourced guideline
text** — deliberately avoiding specific numeric thresholds I could not verify
against the primary document. Replace them with licensed text before relying on
them. Those chunks are flagged `provisional` through the API and badged in the
UI. `project_metrics.md` is derived from this repository's own source and needs
no such caveat.

### Retrieval

TF-IDF over word n-grams plus a prefix-stem branch, cosine similarity,
scikit-learn only — no new dependency, no embedding model fetched at container
start, and a deterministic index tests can assert on. For a few dozen curated
chunks whose key terms are rare and exact, this outperforms what a dense store
would add. `Retriever` is a single class; swap it for an embedding backend if
the corpus grows.

The stem branch is load-bearing: without it "how **reliable** is this
measurement" scores the **reliability** sections at zero and lands on unrelated
ISUOG text.

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

The extension list is appended to `.gitattributes` **in CI only** — committing
the rules upstream would require pushing PNG LFS objects to GitHub too.

The index is emptied with `git read-tree --empty`, not `git rm -r --cached .`.
`rm` refuses to act on a path whose staged content differs from both the file
and HEAD, which is precisely the state the `4a`/`4b` weights are in (HEAD holds
a raw blob; the `*.pth` filter cleans the working file to a pointer). That
aborted deploy run #3. `read-tree` empties the index unconditionally and leaves
the working tree alone.

That failure only reproduces once git's stat cache is invalidated: a plain fresh
checkout hides the discrepancy because git trusts stat info and skips the clean
filter, while `git lfs checkout` in CI touches the files and forces the
comparison. When testing this workflow locally, `touch *.pth` after checkout to
recreate the CI state — otherwise the bug is invisible.

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
