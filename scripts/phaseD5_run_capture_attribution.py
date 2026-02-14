"""
Phase D-5: Wrapper script for Opportunity Capture Attribution.

Delegates to analytics.phaseD5_opportunity_capture.main (same CLI, --find-latest-trades supported).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.phaseD5_opportunity_capture import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
