"""Launch the tube tracking interface with VS Code's Run Python File button."""

import os
from pathlib import Path
import runpy


if __name__ == "__main__":
    project_directory = Path(__file__).resolve().parent
    os.chdir(project_directory)
    runpy.run_path(str(project_directory / "tests" / "test_interface.py"), run_name="__main__")
