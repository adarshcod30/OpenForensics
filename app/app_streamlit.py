import os
import io
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

import streamlit as st

st.set_page_config(page_title="OpenForensics — Deepfake Detector", layout="wide")
st.title("OpenForensics — Deepfake Detector")

import tensorflow as tf

# Add project root to path so we can import src
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import custom layer early so model deserialization succeeds
from src.model import model_def

# -----------------------
# Paths / config
# -----------------------
BASE_DIR = Path('.')
RUN_INITIAL = BASE_DIR / 'runs' / 'exp1'
RUN_FINETUNE = BASE_DIR / 'runs' / 'exp1_finetune'

MODEL_INIT = RUN_INITIAL / 'best_model.keras'
MODEL_INIT_FINAL = RUN_INITIAL / 'final_model.keras'
HISTORY_INIT = RUN_INITIAL / 'history.json'
EVAL_INIT_DIR = RUN_INITIAL / 'eval'

MODEL_FT = RUN_FINETUNE / 'best_model_finetuned.keras'
MODEL_FT_FINAL = RUN_FINETUNE / 'final_finetuned.keras'
HISTORY_FT = RUN_FINETUNE / 'history_finetune.json'
EVAL_FT_DIR = RUN_FINETUNE / 'eval'

IMG_SIZE = (224, 224)

# -----------------------
# Utilities
# -----------------------
@st.cache_resource
def load_model_safe(path: str):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        # fallback to custom_objects pointing to our PreprocessLayer
        return tf.keras.models.load_model(path, compile=False,
                                          custom_objects={'PreprocessLayer': model_def.PreprocessLayer})

def read_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def model_exists(path):
    return path and os.path.exists(path)

def model_summary_str(model):
    buf = io.StringIO()
    model.summary(print_fn=lambda s: buf.write(s + '\n'))
    return buf.getvalue()

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def display_eval_metrics_from_report(eval_report):
    """
    Return (roc_auc, pr_auc, note). Use robust parsing and return note if missing.
    """
    if not eval_report:
        return None, None, "No eval_report.json found."
    # prefer keys 'roc_auc' and 'pr_auc' top-level
    roc = safe_float(eval_report.get('roc_auc')) if isinstance(eval_report, dict) else None
    pr = safe_float(eval_report.get('pr_auc')) if isinstance(eval_report, dict) else None
    note = None
    if roc is None or pr is None:
        # try to find nested values or older field names
        # no more sources present -> add note
        note = "roc_auc/pr_auc missing or invalid in eval_report.json. Re-run evaluate.py to regenerate accurate metrics."
    return roc, pr, note

# Grad-CAM helpers (same as previous implementation)
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sub in reversed(layer.layers):
                if hasattr(sub, 'output_shape') and isinstance(sub.output_shape, tuple) and len(sub.output_shape) == 4:
                    return f"{layer.name}/{sub.name}" if '/' not in sub.name else sub.name
        else:
            if hasattr(layer, 'output_shape') and isinstance(layer.output_shape, tuple) and len(layer.output_shape) == 4:
                return layer.name
    for layer in reversed(model.layers):
        if layer.__class__.__name__.lower().startswith('conv'):
            return layer.name
    return None

def get_layer_by_name_recursive(model, layer_name):
    if '/' in layer_name:
        model_name, sub_name = layer_name.split('/', 1)
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and layer.name == model_name:
                for sub in layer.layers:
                    if sub.name == sub_name:
                        return sub
    else:
        for layer in model.layers:
            if layer.name == layer_name:
                return layer
            if isinstance(layer, tf.keras.Model):
                for sub in layer.layers:
                    if sub.name == layer_name:
                        return sub
    return None

def make_gradcam(model, img_array, class_idx=None, layer_name=None):
    if layer_name is None:
        layer_name = find_last_conv_layer(model)
        if layer_name is None:
            raise RuntimeError("No conv layer found for Grad-CAM")
    target_layer = get_layer_by_name_recursive(model, layer_name)
    if target_layer is None:
        raise RuntimeError(f"Could not locate target layer '{layer_name}' in model")
    conv_output = target_layer.output
    grad_model = tf.keras.models.Model([model.inputs], [conv_output, model.output])
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        tape.watch(inputs)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, 0] if class_idx is None else predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients are None; cannot compute Grad-CAM")
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    cam = tf.zeros(conv_outputs.shape[0:2], dtype=tf.float32)
    for i in range(pooled_grads.shape[-1]):
        cam += pooled_grads[i] * conv_outputs[:, :, i]
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., tf.newaxis], (IMG_SIZE[0], IMG_SIZE[1]))
    heatmap = tf.squeeze(cam).numpy()
    img = (img_array[0] * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(img)
    cmap = plt.get_cmap('jet')
    heatmap_rgb = cmap(heatmap)[:, :, :3]
    heatmap_img = Image.fromarray((heatmap_rgb * 255).astype(np.uint8)).resize(pil_img.size, Image.BILINEAR)
    overlay = Image.blend(pil_img.convert('RGBA'), heatmap_img.convert('RGBA'), alpha=0.45)
    return overlay, heatmap

# -----------------------
# Streamlit UI
# -----------------------
st.markdown("Dashboard: initial vs fine-tuned models, evaluation artifacts, single-image prediction with Grad-CAM overlay.")

# Sidebar
st.sidebar.header("Controls")
model_choice = st.sidebar.radio("Compare / select model", ("Comparison", "Initial", "Fine-tuned"))
if st.sidebar.button("Reload all"):
    st.experimental_rerun()

# load models if present (cache)
model_init = None
model_ft = None

with st.spinner("Loading deep learning models into memory... This may take up to a minute on first run."):
    if model_exists(MODEL_INIT):
        try:
            model_init = load_model_safe(str(MODEL_INIT))
        except Exception as e:
            st.sidebar.error(f"Failed to load initial model: {e}")
    if model_exists(MODEL_FT):
        try:
            model_ft = load_model_safe(str(MODEL_FT))
        except Exception as e:
            st.sidebar.error(f"Failed to load fine-tuned model: {e}")

# Comparison panel
if model_choice == "Comparison":
    st.header("Initial vs Fine-tuned — Comparison")
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Initial (exp1)")
        st.code(str(MODEL_INIT) if MODEL_INIT.exists() else "Not found")
        eval_init = read_json(str(Path(EVAL_INIT_DIR) / 'eval_report.json'))
        roc_init, pr_init, note_init = display_eval_metrics_from_report(eval_init)
        if roc_init:
            st.metric("ROC AUC", f"{roc_init:.4f}")
        else:
            st.metric("ROC AUC", "N/A")
            if note_init:
                st.warning(note_init)
        if pr_init:
            st.metric("PR AUC", f"{pr_init:.4f}")
        else:
            st.metric("PR AUC", "N/A")
        if eval_init and 'classification_report' in eval_init:
            cr = pd.DataFrame(eval_init['classification_report']).transpose()
            st.write("Classification report (excerpt)")
            st.dataframe(cr.style.format("{:.4f}"), use_container_width=True)

        rpath = Path(EVAL_INIT_DIR) / 'roc_curve.png'
        ppath = Path(EVAL_INIT_DIR) / 'pr_curve.png'
        if rpath.exists():
            st.image(str(rpath), caption="ROC (initial)")
        if ppath.exists():
            st.image(str(ppath), caption="PR (initial)")

    with colB:
        st.subheader("Fine-tuned (exp1_finetune)")
        st.code(str(MODEL_FT) if MODEL_FT.exists() else "Not found")
        eval_ft = read_json(str(Path(EVAL_FT_DIR) / 'eval_report.json'))
        roc_ft, pr_ft, note_ft = display_eval_metrics_from_report(eval_ft)
        if roc_ft:
            st.metric("ROC AUC", f"{roc_ft:.4f}")
        else:
            st.metric("ROC AUC", "N/A")
            if note_ft:
                st.warning(note_ft)
        if pr_ft:
            st.metric("PR AUC", f"{pr_ft:.4f}")
        else:
            st.metric("PR AUC", "N/A")
        if eval_ft and 'classification_report' in eval_ft:
            cr2 = pd.DataFrame(eval_ft['classification_report']).transpose()
            st.write("Classification report (excerpt)")
            st.dataframe(cr2.style.format("{:.4f}"), use_container_width=True)

        rpath2 = Path(EVAL_FT_DIR) / 'roc_curve.png'
        ppath2 = Path(EVAL_FT_DIR) / 'pr_curve.png'
        if rpath2.exists():
            st.image(str(rpath2), caption="ROC (finetuned)", use_container_width=True)
        if ppath2.exists():
            st.image(str(ppath2), caption="PR (finetuned)", use_container_width=True)

    st.markdown("---")
    st.subheader("Confusion matrices")
    c1, c2 = st.columns(2)
    if eval_init and 'confusion_matrix' in eval_init:
        with c1:
            st.markdown("Initial")
            cm = eval_init['confusion_matrix']
            cm_df = pd.DataFrame(cm, index=['Fake','Real'], columns=['Pred_Fake','Pred_Real'])
            st.dataframe(cm_df, use_container_width=True)
    else:
        with c1:
            st.info("No initial confusion matrix")
    if eval_ft and 'confusion_matrix' in eval_ft:
        with c2:
            st.markdown("Fine-tuned")
            cm2 = eval_ft['confusion_matrix']
            cm2_df = pd.DataFrame(cm2, index=['Fake','Real'], columns=['Pred_Fake','Pred_Real'])
            st.dataframe(cm2_df, use_container_width=True)
    else:
        with c2:
            st.info("No finetuned confusion matrix")

# Single-model view
selected_model = None
selected_model_path = None
selected_eval_dir = None
selected_history = None
if model_choice == "Initial":
    selected_model = model_init
    selected_model_path = MODEL_INIT if MODEL_INIT.exists() else MODEL_INIT_FINAL if MODEL_INIT_FINAL.exists() else None
    selected_eval_dir = EVAL_INIT_DIR
    selected_history = HISTORY_INIT if HISTORY_INIT.exists() else None
elif model_choice == "Fine-tuned":
    selected_model = model_ft
    selected_model_path = MODEL_FT if MODEL_FT.exists() else MODEL_FT_FINAL if MODEL_FT_FINAL.exists() else None
    selected_eval_dir = EVAL_FT_DIR
    selected_history = HISTORY_FT if HISTORY_FT.exists() else None

if model_choice in ("Initial","Fine-tuned"):
    st.header(f"{model_choice} model — Details & Predict")
    st.write("Model file:")
    st.code(str(selected_model_path) if selected_model_path else "Not found")
    # Model summary in an expander
    if selected_model is not None:
        try:
            s = model_summary_str(selected_model)
            with st.expander("Show model summary"):
                st.code(s)
        except Exception as e:
            st.error("Failed to produce model summary")
            st.exception(e)
    else:
        st.info("Model not loaded")

    # Eval metrics
    eval_json = Path(selected_eval_dir) / 'eval_report.json'
    eval_report = read_json(str(eval_json)) if eval_json.exists() else None
    roc, pr, note = display_eval_metrics_from_report(eval_report)
    st.metric("ROC AUC", f"{roc:.4f}" if roc else "N/A")
    st.metric("PR AUC", f"{pr:.4f}" if pr else "N/A")
    if note:
        st.warning(note + " If you want I can show the command to re-run evaluation.")

    if eval_report and 'classification_report' in eval_report:
        st.write("Classification report (excerpt)")
        st.dataframe(pd.DataFrame(eval_report['classification_report']).transpose().style.format("{:.4f}"), use_container_width=True)

    # history plots (if exists)
    hist = read_json(str(selected_history)) if selected_history else None
    if hist:
        df = pd.DataFrame(hist)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("Loss (train/val)")
            fig, ax = plt.subplots(figsize=(6,3))
            if 'loss' in df and 'val_loss' in df:
                ax.plot(df['loss'], label='train_loss'); ax.plot(df['val_loss'], label='val_loss')
                ax.legend(); st.pyplot(fig, use_container_width=True)
            else:
                st.write("No loss data")
        with c2:
            st.markdown("AUC (train/val)")
            fig2, ax2 = plt.subplots(figsize=(6,3))
            if 'auc' in df and 'val_auc' in df:
                ax2.plot(df['auc'], label='train_auc'); ax2.plot(df['val_auc'], label='val_auc')
                ax2.legend(); st.pyplot(fig2, use_container_width=True)
            else:
                st.write("No auc data")
    else:
        st.info("No history JSON found for this run")

    st.markdown("---")
    st.subheader("Upload image and run prediction + Grad-CAM")
    col_up, col_opts = st.columns([2,1])
    with col_up:
        uploaded = st.file_uploader("Upload image (jpg/png)", type=['jpg','jpeg','png'])
        if uploaded:
            img = Image.open(uploaded).convert('RGB')
            st.image(img, caption="Uploaded image", use_container_width=True)
    with col_opts:
        threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5)
        show_heatmap = st.checkbox("Show Grad-CAM overlay", value=True)
        layer_override = st.text_input("Grad-CAM target layer (optional)", value="")

    if uploaded:
        if selected_model is None:
            st.error("Selected model not loaded; cannot predict.")
        else:
            x = np.array(img.resize(IMG_SIZE), dtype=np.float32) / 255.0
            x = np.expand_dims(x, 0)
            try:
                prob = float(selected_model.predict(x).ravel()[0])
                label = "Real" if prob >= threshold else "Fake"
                st.metric("Probability Real", f"{prob:.4f}")
                st.markdown(f"Predicted: **{label}** (threshold {threshold:.2f})")
                st.bar_chart(pd.DataFrame({'score':[prob, 1-prob]}, index=['Real','Fake']))
            except Exception as e:
                st.error("Prediction failed")
                st.exception(e)
                prob = None

            if show_heatmap and prob is not None:
                try:
                    target_layer_name = layer_override.strip() or None
                    overlay, heatmap = make_gradcam(selected_model, x, class_idx=None, layer_name=target_layer_name)
                    st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
                except Exception as e:
                    st.error("Grad-CAM failed")
                    st.exception(e)

st.markdown("---")
st.caption("Notes: If ROC/PR display as N/A or 0.00, re-run evaluate.py to regenerate eval_report.json with the desired --test_per_class. Use the sidebar to switch views.")
