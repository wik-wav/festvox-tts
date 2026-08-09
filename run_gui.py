"""Launch the FestVox desktop GUI from the organized source tree."""
from __future__ import annotations

from pathlib import Path
import runpy
import sys


SOURCE_ROOT = Path(__file__).resolve().parent / "src" / "festvox_tts"
GUI_ENTRY = SOURCE_ROOT / "festvox_gui" / "festvox_gui.py"

for path in (SOURCE_ROOT, GUI_ENTRY.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

runpy.run_path(str(GUI_ENTRY), run_name="__main__")
