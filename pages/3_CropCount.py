"""Streamlit page wrapper for the general CropCount model app."""

import runpy
from pathlib import Path


runpy.run_path(str(Path(__file__).parents[1] / "cropcount_app.py"), run_name="__main__")
