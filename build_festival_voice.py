"""Compatibility launcher for the source-tree voice builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import runpy
import sys


SOURCE_ROOT = Path(__file__).resolve().parent / "src" / "festvox_tts"
IMPLEMENTATION = SOURCE_ROOT / "build_festival_voice.py"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


if __name__ == "__main__":
    runpy.run_path(str(IMPLEMENTATION), run_name="__main__")
else:
    _spec = importlib.util.spec_from_file_location(__name__, IMPLEMENTATION)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Cannot load FestVox builder from {IMPLEMENTATION}")
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
