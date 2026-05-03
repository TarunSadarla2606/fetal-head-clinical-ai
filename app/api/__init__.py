"""FetalScan AI — FastAPI inference API (Batch 1)."""
import sys
from pathlib import Path

# Make app/ importable so inference.py, report.py etc. resolve without
# depending on the caller's working directory.
_app_dir = Path(__file__).parent.parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))
