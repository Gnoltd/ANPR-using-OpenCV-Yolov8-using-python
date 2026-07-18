import unittest
import customtkinter as ctk

class TestSettingsTab(unittest.TestCase):
    def test_instantiate_settings_tab(self):
        root = ctk.CTk()
        try:
            from gui.settings_tab import SettingsTab
            tab = SettingsTab(root)
            config = tab.get_config()
            self.assertEqual(config["detect_every_n"], 8)
            self.assertEqual(config["ocr_every_n"], 24)
            self.assertTrue(config["save_csv_log"])
        finally:
            root.destroy()
