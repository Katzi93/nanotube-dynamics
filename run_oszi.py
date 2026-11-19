"""Launch the oscilloscope recorder with VS Code's Run Python File button."""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    app_path = Path(__file__).resolve().parent / "osci_interface" / "app.py"
    sys.path.insert(0, str(app_path.parent))
    runpy.run_path(str(app_path), run_name="__main__")
