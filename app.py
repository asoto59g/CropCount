from pathlib import Path

import streamlit as st


st.set_page_config(page_title="CropCount", page_icon="🌱", layout="wide")

root = Path(__file__).parent
pages = [
    st.Page(str(root / "radial_signature_page.py"), title="Radial Signature", icon="🌴"),
    st.Page(str(root / "pages" / "2_Model_Generation.py"), title="Model Generation", icon="🧬"),
    st.Page(str(root / "pages" / "3_CropCount.py"), title="CropCount", icon="🌱"),
]
st.navigation(pages).run()
