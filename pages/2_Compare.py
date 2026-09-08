"""Streamlit page wrapper for the radial signature comparison app."""

import runpy
from pathlib import Path


runpy.run_path(str(Path(__file__).parents[1] / "compare_app.py"), run_name="__main__")
