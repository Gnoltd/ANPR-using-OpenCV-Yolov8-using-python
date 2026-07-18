"""
ANPR – CustomTkinter GUI Redesign
Launches the new modular CustomTkinter application.
"""
import sys

def main():
    try:
        from gui.app import App
        app = App()
        app.mainloop()
    except Exception as e:
        print(f"Error launching ANPR GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
