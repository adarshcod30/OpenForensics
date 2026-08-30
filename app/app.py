"""OpenForensics dashboard.

Multi-page: a detector plus the evidence behind it. Only the Detect page
imports TensorFlow, so the rest of the site loads in a fraction of the
memory — which matters on a host with roughly a gigabyte to spend.
"""
import sys
from pathlib import Path

import streamlit as st

# Make `shared` and `views` importable however the app is laid out, and put
# the openforensics package on the path for the pages that need it.
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
for _cand in (_here.parent / "src", _here / "src", _here.parent):
    if (_cand / "openforensics").is_dir():
        sys.path.insert(0, str(_cand))
        break

st.set_page_config(
    page_title="OpenForensics — Deepfake Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from views import (  # noqa: E402
    about, data, detect, evaluation, model, overview, robustness, training,
)

# url_path is explicit because every page's entrypoint is named `render`;
# Streamlit would otherwise infer the same pathname for all of them and
# refuse to build the navigation.
PAGES = [
    st.Page(overview.render, title="Overview", icon=":material/home:",
            url_path="overview", default=True),
    st.Page(detect.render, title="Detect", icon=":material/search:",
            url_path="detect"),
    st.Page(model.render, title="Model", icon=":material/account_tree:",
            url_path="model"),
    st.Page(training.render, title="Training", icon=":material/trending_up:",
            url_path="training"),
    st.Page(evaluation.render, title="Evaluation", icon=":material/query_stats:",
            url_path="evaluation"),
    st.Page(robustness.render, title="Robustness", icon=":material/shield:",
            url_path="robustness"),
    st.Page(data.render, title="Data", icon=":material/database:",
            url_path="data"),
    st.Page(about.render, title="About", icon=":material/info:",
            url_path="about"),
]

nav = st.navigation(PAGES, position="sidebar")
nav.run()
