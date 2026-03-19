from __future__ import annotations

import streamlit as st

from finance_tracker.ui import render_app


st.set_page_config(page_title="Tracker", page_icon=":chart_with_upwards_trend:", layout="wide")
render_app()

