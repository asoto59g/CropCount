"""Streamlit page for model generation and signature comparison."""

import runpy
from pathlib import Path


runpy.run_path(str(Path(__file__).parents[1] / "compare_app.py"), run_name="__main__")
