"""Streamlit Community Cloud entry point.

The cloud tier allows about 2.7 GB of RAM, so this entry runs the health
assistant in lite mode: the identical six-stage pipeline, rule layer and
guardrails, with small models swapped in (see config.LITE_MODE). The full
model set runs locally as described in cloud-native-web/health_assist/README.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("HEALTH_ASSIST_LITE", "1")

APP = Path(__file__).parent / "cloud-native-web" / "health_assist" / "src" / "app.py"
sys.path.insert(0, str(APP.parent))

exec(
    compile(APP.read_text(encoding="utf-8"), str(APP), "exec"),
    {"__name__": "__main__", "__file__": str(APP)},
)
