# requires: pip install pytest httpx
import sys
from pathlib import Path

# Add inference-engine root so `tasks`, `db`, `core`, `config` are importable
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: pure function tests, no external deps")
    config.addinivalue_line("markers", "integration: requires postgres DB")
    config.addinivalue_line("markers", "network: requires internet access")
    config.addinivalue_line("markers", "smoke: quick health checks")
    config.addinivalue_line("markers", "gpu: requires GPU and model weights")
    config.addinivalue_line("markers", "slow: takes more than 30s")
