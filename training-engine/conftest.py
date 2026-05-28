import sys
from pathlib import Path

# inference-engine tasks (annotate_clips, auto_label, etc.) are needed by some
# training-engine unit tests. Add it before training-engine so imports resolve
# to the right package — no name conflicts between the two tasks/ directories.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_INFERENCE_ENGINE = str(_REPO_ROOT / "inference-engine")
if _INFERENCE_ENGINE not in sys.path:
    sys.path.insert(0, _INFERENCE_ENGINE)
