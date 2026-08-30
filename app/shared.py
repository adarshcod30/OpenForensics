"""Data access and shared helpers for the dashboard.

The guiding constraint: **the model is never loaded unless a page needs it.**
Every metrics page reads a few kilobytes of JSON from the Hub, so the site is
browsable without importing TensorFlow at all. Only the detector page pays
the ~900 MB cost, and only once a visitor asks for a prediction.

Nothing here reads the repository or a run directory -- neither exists on a
hosting platform. Everything ships alongside the weights on the Hub.
"""
from __future__ import annotations

import csv
import io
import json
import os
from functools import lru_cache

import streamlit as st

MODEL_REPO = os.environ.get("OF_MODEL_REPO", "adarshcod30/openforensics-ensemble")
MODEL_DIR = os.environ.get("OF_MODEL_DIR")
IMG_SIZE = (224, 224)

REPO_URL = "https://github.com/adarshcod30/OpenForensics"
HUB_URL = f"https://huggingface.co/{MODEL_REPO}"

# Consistent meaning across every chart on the site.
C_REAL = "#2E9EB8"     # genuine / cool
C_FAKE = "#D9752F"     # manipulated / warm
C_GOOD = "#2BA574"
C_BAD = "#D95742"
C_MUTED = "#7E8F9A"


# ---------------------------------------------------------------- artifacts
@st.cache_data(show_spinner=False)
def fetch(name: str) -> bytes | None:
    """One small artifact from the Hub (or OF_MODEL_DIR). None if absent."""
    if MODEL_DIR:
        p = os.path.join(MODEL_DIR, name)
        return open(p, "rb").read() if os.path.exists(p) else None
    try:
        from huggingface_hub import hf_hub_download
        return open(hf_hub_download(MODEL_REPO, name, repo_type="model"), "rb").read()
    except Exception:
        return None


def load_json(name: str) -> dict | None:
    raw = fetch(name)
    return json.loads(raw) if raw else None


def load_csv(name: str) -> list[dict] | None:
    raw = fetch(name)
    if not raw:
        return None
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    out = []
    for r in rows:
        rec = {}
        for k, v in r.items():
            try:
                rec[k] = float(v)
            except (TypeError, ValueError):
                rec[k] = v
        out.append(rec)
    return out


@st.cache_data(show_spinner=False)
def bundle() -> dict:
    """Everything the metrics pages need. A few tens of KB in total."""
    return {
        "card": load_json("serving.json") or {},
        "report": load_json("evaluation_report.json") or {},
        "config": load_json("config.json") or {},
        "leakage": load_json("leakage_report.json") or {},
        "stage1": load_json("stage1_history.json") or {},
        "stage2": load_json("stage2_history.json") or {},
    }


# ------------------------------------------------------------------- model
@st.cache_resource(show_spinner=False)
def load_model():
    """Import TensorFlow and load the weights. Deliberately deferred."""
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    for cand in (here.parent / "src", here / "src", here.parent):
        if (cand / "openforensics").is_dir():
            sys.path.insert(0, str(cand))
            break

    import tensorflow as tf
    from openforensics.models.layers import PreprocessLayer

    if MODEL_DIR:
        weights = os.path.join(MODEL_DIR, "model.keras")
    else:
        from huggingface_hub import hf_hub_download
        weights = hf_hub_download(MODEL_REPO, "model.keras", repo_type="model")
    return tf.keras.models.load_model(
        weights, compile=False, custom_objects={"PreprocessLayer": PreprocessLayer}
    )


# ------------------------------------------------------------------- pieces
def metric_row(items: list[tuple[str, str, str | None]]):
    """(label, value, help) tuples laid out in equal columns."""
    for col, (label, value, hint) in zip(st.columns(len(items)), items):
        col.metric(label, value, help=hint)


def section(title: str, caption: str | None = None):
    st.subheader(title)
    if caption:
        st.caption(caption)


@lru_cache(maxsize=1)
def corruption_names() -> tuple[str, ...]:
    return (
        "desaturate", "colour_cast", "gaussian_noise", "speckle", "blur",
        "jpeg_artifact", "pixelate", "brightness_shift", "occlusion",
    )


CORRUPTION_BLURB = {
    "desaturate": "Colour pulled toward grey.",
    "colour_cast": "Per-channel gain and offset — false colour.",
    "gaussian_noise": "Additive sensor-style noise.",
    "speckle": "Sparse bright and dark pixels.",
    "blur": "Defocus or resize blur.",
    "jpeg_artifact": "Low-quality recompression; blocking.",
    "pixelate": "Downscale then upscale.",
    "brightness_shift": "Heavy under- or over-exposure.",
    "occlusion": "A region erased and filled.",
}
