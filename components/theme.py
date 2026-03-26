from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parents[1]


def load_css() -> None:
    css = (ROOT / "assets" / "css" / "main.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def load_js() -> None:
    js = (ROOT / "assets" / "js" / "app.js").read_text()
    components.html(f"<script>{js}</script>", height=0)
