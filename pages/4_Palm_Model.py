"""Streamlit page wrapper for the palm-specific model app."""

import runpy
from pathlib import Path


runpy.run_path(str(Path(__file__).parents[1] / "palm_model_app.py"), run_name="__main__")
