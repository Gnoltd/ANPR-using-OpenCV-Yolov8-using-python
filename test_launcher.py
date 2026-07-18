import os
import sys
import types

# Register current directory as "ANPR_Yolo" in sys.modules for testing
def _register_anpr_yolo_package():
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    if "ANPR_Yolo" not in sys.modules:
        pkg = types.ModuleType("ANPR_Yolo")
        pkg.__path__ = [here]
        pkg.__file__ = os.path.join(here, "__init__.py")
        pkg.__package__ = "ANPR_Yolo"
        sys.modules["ANPR_Yolo"] = pkg

_register_anpr_yolo_package()

import unittest
from unittest.mock import patch, MagicMock

class TestLauncher(unittest.TestCase):
    @patch("gui.app.App")
    def test_launcher_main(self, mock_app):
        import gui_tk
        gui_tk.main()
        self.assertTrue(mock_app.called)
