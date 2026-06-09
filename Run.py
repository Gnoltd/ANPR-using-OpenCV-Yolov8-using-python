import os
import sys
import types

# ── Fix: register this directory as "ANPR_Yolo" regardless of folder name ──
_here   = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)

if _parent not in sys.path:
    sys.path.insert(0, _parent)

if "ANPR_Yolo" not in sys.modules:
    _pkg = types.ModuleType("ANPR_Yolo")
    _pkg.__path__    = [_here]
    _pkg.__file__    = os.path.join(_here, "__init__.py")
    _pkg.__package__ = "ANPR_Yolo"
    sys.modules["ANPR_Yolo"] = _pkg
# ───────────────────────────────────────────────────────────────────────────

from ANPR_Yolo.gui_tk import main  # noqa: E402

if __name__ == "__main__":
    main()
