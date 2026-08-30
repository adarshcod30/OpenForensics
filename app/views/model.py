"""Architecture, and how it is trained."""
import streamlit as st

from shared import HUB_URL, bundle, section

DIAGRAM = """
input (224, 224, 3)  float32 in [0, 1]
│
├─ preprocess_resnet50 ──────→ ResNet50           (7, 7, 2048) ─┐
├─ preprocess_vgg16 ─────────→ VGG16              (7, 7, 512)  ─┤  per branch:
└─ preprocess_efficientnetv2 → EfficientNetV2-B0  (7, 7, 1280) ─┘  GAP → Dropout(0.4)
                                                                   → Dense(256, relu) → BN
                                    ↓
                  Concatenate (768) ┘
                                    ↓
              Dropout(0.4) → Dense(256, relu) → BN → Dropout(0.3)
                                    ↓
                          Dense(1, sigmoid) → P(genuine)
"""

BACKBONES = [
    ("ResNet50", "23,587,712", "Global structure and macro facial geometry.",
     "caffe-style: [0,255], channel means subtracted"),
    ("VGG16", "14,714,688", "Fine texture and blending seams.",
     "caffe-style: [0,255], different channel means"),
    ("EfficientNetV2-B0", "5,919,312", "Strongest features per parameter; added in v2.",
     "raw [0,255] — it rescales internally, so its preprocess_input is a no-op"),
]


def render():
    st.title("Model")
    b = bundle()
    cfg = b["config"].get("model", {})

    section("Architecture",
            "Three backbones see the same image through their own normalisation, "
            "each pools to a 256-dimensional embedding, and the three are "
            "concatenated before a shared classifier head.")
    st.code(DIAGRAM, language="text")

    c = st.columns(3)
    c[0].metric("Total parameters", f"{b['card'].get('params', 0):,}")
    c[1].metric("Backbones", len(cfg.get("backbones", BACKBONES)))
    c[2].metric("Input", "224 × 224 × 3")

    st.divider()
    section("Backbones", "Each expects a different input convention. Getting that "
                         "wrong does not error — it silently starts a branch from a "
                         "worse initialisation.")
    st.dataframe(
        {"Backbone": [x[0] for x in BACKBONES],
         "Parameters": [x[1] for x in BACKBONES],
         "Contributes": [x[2] for x in BACKBONES],
         "Expects": [x[3] for x in BACKBONES]},
        hide_index=True, use_container_width=True,
    )

    st.divider()
    section("Two-stage training")
    a, bb = st.columns(2)
    with a:
        st.markdown(
            "**Stage 1 — heads only**  \n"
            "All backbones frozen. 1,182,977 trainable (2.6%). `lr = 2e-4`.  \n\n"
            "The heads learn to read fixed ImageNet features. Validation "
            "ROC-AUC plateaus near 0.885 — that is the ceiling of what those "
            "features can say about forgery."
        )
    with bb:
        st.markdown(
            "**Stage 2 — fine-tuning**  \n"
            "Top ~50 layers of each backbone unfrozen. 34,759,377 trainable "
            "(76.6%). `lr = 1e-5`.  \n\n"
            "The features themselves adapt. Validation ROC-AUC reaches 0.996."
        )

    st.info(
        "**BatchNorm stays frozen throughout stage 2.** A trainable BatchNorm "
        "layer switches to batch statistics and overwrites running estimates "
        "accumulated over a million ImageNet images — using whatever batch size "
        "you happen to be training at. At batch 32 that is a bad trade.",
        icon="🔒",
    )

    st.divider()
    section("Serving contract")
    st.markdown(
        "Resize to 224×224, scale to `[0, 1]`, shape `(N, 224, 224, 3)` float32. "
        "**Per-backbone normalisation happens inside the model** — do not apply "
        "`preprocess_input` yourself."
    )
    st.code(
        'from huggingface_hub import snapshot_download\n'
        'import tensorflow as tf, numpy as np, json\n'
        'from PIL import Image\n\n'
        'path  = snapshot_download("adarshcod30/openforensics-ensemble")\n'
        'model = tf.keras.models.load_model(f"{path}/model.keras", compile=False)\n'
        'card  = json.load(open(f"{path}/serving.json"))\n\n'
        'img = Image.open("face.jpg").convert("RGB").resize((224, 224))\n'
        'x   = np.asarray(img, dtype="float32")[None] / 255.0\n'
        'p   = float(model.predict(x)[0, 0])\n'
        'print("genuine" if p >= card["decision"]["threshold"] else "manipulated", p)',
        language="python",
    )
    st.link_button("Model card on Hugging Face", HUB_URL)
