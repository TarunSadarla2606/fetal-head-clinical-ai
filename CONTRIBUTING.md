# Contributing

Thank you for your interest in this project. Contributions are welcome.

## Getting started

```bash
git clone https://github.com/TarunSadarla2606/fetal-head-clinical-ai.git
cd fetal-head-clinical-ai
pip install -r requirements.txt
pip install pytest
```

## Running tests

```bash
pytest tests/ -v
```

All 27 tests must pass before submitting a pull request.

## What belongs where

| Directory | Purpose |
|-----------|---------|
| `src/` | Reusable Python package — models, data pipeline, evaluation |
| `app/` | Streamlit application — UI, inference wrappers, report generation |
| `notebooks/` | Training notebooks (one per phase) — read-only reference |
| `tests/` | Automated tests — add a test for any new function in `src/` |

## Pull request guidelines

- **One concern per PR.** Bug fixes and new features in separate PRs.
- **Tests required.** Any new function in `src/` needs at least one test.
- **No model weights.** Never commit `.pth` files — they go on HuggingFace Hub.
- **Commit message format:**

```
<type>: <short description>           ← 50 chars max

<longer explanation if needed>        ← wrap at 72 chars
```

Types: `feat` · `fix` · `docs` · `test` · `refactor` · `ci` · `chore`

Examples:
```
feat: add per-image pixel spacing support to evaluate_predictions
fix: correct BoundaryWeightedDiceLoss unsqueeze crash on 4D target
docs: add ISUOG threshold context to README metrics table
test: add HC scaling test to verify pixel spacing multiplication
```

## Reporting issues

Use the issue templates — they ask for the minimum information needed to reproduce a bug or evaluate a feature request.

## Code style

- Python 3.11+, type hints on all public functions
- Line length ≤ 100 characters
- No comments that explain *what* the code does — only *why* when non-obvious
