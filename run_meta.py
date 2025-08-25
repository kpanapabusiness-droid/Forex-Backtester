import json
import hashlib
import time

def make_run_id(config: dict) -> str:
    core = json.dumps(config, sort_keys=True, default=str).encode()
    h = hashlib.sha1(core).hexdigest()[:8]
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{h}"
