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
import pandas as pd
import tempfile
from unittest.mock import patch, MagicMock

class TestRegistryTab(unittest.TestCase):
    def setUp(self):
        self.root = ctk.CTk()
        self.temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        self.temp_csv.close()
        
        # Write minimal dummy registry
        df = pd.DataFrame(columns=["plate", "owner_name", "phone", "notes", "photo", "plate_norm"])
        df.to_csv(self.temp_csv.name, index=False)

    def tearDown(self):
        self.root.destroy()
        os.unlink(self.temp_csv.name)

    def test_registry_crud(self):
        # We need mock to return a dummy df
        dummy_df = pd.DataFrame([
            {"plate": "29D1-999.99", "owner_name": "Nguyen Test", "phone": "090000000", "notes": "Test note", "photo": "", "plate_norm": "29D199999"}
        ])
        
        # Wait, since from gui.registry_tab import RegistryTab will run, it imports load_registry
        # Let's mock it
        with patch("DetectNP.load_registry", return_value=dummy_df):
            with patch("DetectNP.save_registry") as mock_save:
                try:
                    from gui.registry_tab import RegistryTab
                    tab = RegistryTab(self.root)
                    tab.plate_var.set("29D1-999.99")
                    tab.owner_name_var.set("Nguyen Test Updated")
                    tab.phone_var.set("090000000")
                    tab.notes_var.set("Test note updated")
                    
                    # Simulate saving
                    tab.on_save_record()
                    self.assertTrue(mock_save.called)
                except ImportError as e:
                    self.fail(f"Failing to import RegistryTab: {e}")
