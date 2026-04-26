import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.db.models import (  # noqa: F401
    Base,
    Clip, ClipSource, ClipStatus,
)