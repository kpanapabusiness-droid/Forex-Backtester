import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402


def make_run_id(config: dict) -> str:
    core = json.dumps(config, sort_keys=True, default=str).encode()
    h = hashlib.sha1(core).hexdigest()[:8]
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{h}"
