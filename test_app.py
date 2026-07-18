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

class TestApp(unittest.TestCase):
    def test_app_initialization(self):
        try:
            from gui.app import App
            app = App()
            self.assertIsNotNone(app.settings_tab)
            self.assertIsNotNone(app.dashboard_tab)
            self.assertIsNotNone(app.registry_tab)
            app.destroy()
        except ImportError as e:
            self.fail(f"Failed to import main App: {e}")
