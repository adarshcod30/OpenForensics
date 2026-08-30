"""OpenForensics — deepfake face detector.

Written for Hugging Face Spaces. Weights are fetched from the Hub at first
run rather than committed: a 45M-parameter ensemble does not belong in git,
and the previous deployment shipped with no weights at all, so the upload
panel could never do anything.

Set OF_MODEL_REPO to the Hub model repo, or OF_MODEL_DIR to a local folder.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Layout-agnostic: the repo keeps the package at ../src/openforensics, while
# a Hugging Face Space puts app.py and src/ side by side at the root.
_here = Path(__file__).resolve().parent
for _candidate in (_here.parent / "src", _here / "src", _here.parent):
    if (_candidate / "openforensics").is_dir():
        sys.path.insert(0, str(_candidate))
        break

st.set_page_config(page_title="OpenForensics — Deepfake Detector",
                   page_icon="🔬", layout="wide")

MODEL_REPO = os.environ.get("OF_MODEL_REPO", "adarshcod30/openforensics-ensemble")
MODEL_DIR = os.environ.get("OF_MODEL_DIR")
IMG_SIZE = (224, 224)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_bundle():
    """Return (model, serving_card). Raises with an actionable message."""
    import tensorflow as tf
    from openforensics.models.layers import PreprocessLayer

    if MODEL_DIR:
        folder = Path(MODEL_DIR)
        if not folder.exists():
            raise FileNotFoundError(f"OF_MODEL_DIR={folder} does not exist")
    else:
        from huggingface_hub import snapshot_download
        folder = Path(snapshot_download(repo_id=MODEL_REPO, repo_type="model"))

    weights = folder / "model.keras"
    if not weights.exists():
        found = sorted(p.name for p in folder.glob("*"))
        raise FileNotFoundError(
            f"model.keras not found in {folder}. Contents: {found or 'empty'}"
        )

    card_path = folder / "serving.json"
    card = json.loads(card_path.read_text()) if card_path.exists() else {}
    model = tf.keras.models.load_model(
        str(weights), compile=False,
        custom_objects={"PreprocessLayer": PreprocessLayer},
    )
    return model, card


def preprocess(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB").resize(IMG_SIZE), dtype=np.float32) / 255.0
    return arr[None, ...]


def predict(model, x, card, use_tta=True) -> float:
    import tensorflow as tf
    p = float(model.predict(x, verbose=0).ravel()[0])
    if use_tta:
        flipped = tf.image.flip_left_right(tf.convert_to_tensor(x)).numpy()
        p = (p + float(model.predict(flipped, verbose=0).ravel()[0])) / 2.0
    t = float(card.get("decision", {}).get("temperature", 1.0))
    if t != 1.0:
        eps = 1e-7
        pc = np.clip(p, eps, 1 - eps)
        p = float(1.0 / (1.0 + np.exp(-(np.log(pc / (1 - pc)) / t))))
    return p


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("OpenForensics — Deepfake Detector")
st.caption("A three-backbone ensemble (ResNet50 · VGG16 · EfficientNetV2-B0) "
           "trained on face crops from the OpenForensics dataset.")

try:
    with st.spinner("Loading model (first run downloads weights)…"):
        model, card = load_bundle()
    loaded = True
except Exception as exc:
    loaded = False
    st.error("**The model could not be loaded, so no prediction is possible.**")
    st.code(f"{type(exc).__name__}: {exc}")
    st.info(
        f"Set `OF_MODEL_REPO` (currently `{MODEL_REPO}`) to a Hub repo holding "
        "`model.keras`, or `OF_MODEL_DIR` to a local folder. Package one with:\n\n"
        "`python -m openforensics.export.package --run_dir runs/v2 "
        "--push_to <user>/<repo>`"
    )

if loaded:
    decision = card.get("decision", {})
    threshold = float(decision.get("threshold", 0.5))
    calibrated = bool(decision.get("calibrated", False))

    with st.sidebar:
        st.subheader("Decision settings")
        threshold = st.slider("Threshold on P(Real)", 0.0, 1.0, threshold, 0.01)
        use_tta = st.checkbox("Test-time augmentation", value=True,
                              help="Average the prediction over the image and its mirror.")
        show_cam = st.checkbox("Grad-CAM overlay", value=True)
        st.divider()
        if calibrated:
            st.success(f"Calibrated · T={decision.get('temperature', 1):.3f}\n\n"
                       f"Operating point chosen on validation "
                       f"({decision.get('criterion', 'n/a')}).")
        else:
            st.warning("Uncalibrated — threshold is a default, not a fitted "
                       "operating point.")
        if "test_metrics" in card:
            m = card["test_metrics"]
            st.metric("Test accuracy", f"{m['accuracy']:.4f}")
            st.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
            st.metric("Real images called fake",
                      f"{m['false_accusation_rate']*100:.1f}%")

    uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded is None:
        st.info("Upload an image to analyse.")
    else:
        img = Image.open(uploaded)
        x = preprocess(img)
        with st.spinner("Analysing…"):
            p_real = predict(model, x, card, use_tta)

        verdict = "Real" if p_real >= threshold else "Fake"
        left, right = st.columns([1, 1])

        with left:
            st.image(img, caption="Input", use_container_width=True)

        with right:
            if verdict == "Real":
                st.success(f"### Likely **Real**")
            else:
                st.error(f"### Likely **Fake**")
            st.metric("P(Real)", f"{p_real:.4f}")
            st.progress(float(np.clip(p_real, 0, 1)))
            st.caption(
                f"Threshold {threshold:.2f}. "
                + ("Confidence is calibrated on held-out data."
                   if calibrated else
                   "Confidence is uncalibrated and likely overstated.")
            )
            margin = abs(p_real - threshold)
            if margin < 0.10:
                st.warning("**Borderline.** The score sits close to the "
                           "threshold; treat this as inconclusive rather than "
                           "as evidence either way.")

        if show_cam:
            st.subheader("Where each backbone looked")
            st.caption("Gradient of the *fake* score. Bright regions pushed the "
                       "model toward calling this image forged. The branches are "
                       "fused by concatenation, so no single weighted ensemble "
                       "map exists — the mean is a summary, not an attribution.")
            try:
                from openforensics.evaluation import explain
                cams = explain.all_branches(model, x, size=IMG_SIZE)
                if not cams:
                    raise RuntimeError(
                        "no addressable backbone found — this checkpoint has its "
                        "backbones inlined into the parent graph, so per-branch "
                        "attribution is not available"
                    )
                base = x[0]
                cols = st.columns(len(cams))
                for col, (name, hm) in zip(cols, cams.items()):
                    with col:
                        st.image(explain.overlay(base, hm),
                                 caption=name, use_container_width=True)
            except Exception as exc:
                st.warning(f"Grad-CAM unavailable: {type(exc).__name__}: {exc}")

st.divider()
st.caption(
    "Built on the OpenForensics dataset (Le et al., ICCV 2021). "
    "Research and educational use — not a forensic authority. "
    "A prediction is evidence to weigh, not proof."
)
