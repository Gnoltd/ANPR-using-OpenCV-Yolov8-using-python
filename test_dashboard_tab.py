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
import customtkinter as ctk

class TestDashboardTab(unittest.TestCase):
    def test_instantiate_dashboard(self):
        root = ctk.CTk()
        try:
            from gui.dashboard_tab import DashboardTab
            tab = DashboardTab(root, on_image=lambda: None, on_video=lambda: None, on_webcam=lambda: None, on_stop=lambda: None)
            self.assertIsNotNone(tab.canvas)
            self.assertIsNotNone(tab.btn_image)
        finally:
            root.destroy()
