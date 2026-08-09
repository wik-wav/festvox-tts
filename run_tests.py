"""Run the complete repository test suite with the source paths configured."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "src" / "festvox_tts"
GUI_ROOT = SOURCE_ROOT / "festvox_gui"

for path in (SOURCE_ROOT, GUI_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
