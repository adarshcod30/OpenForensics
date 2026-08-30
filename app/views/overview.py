"""Landing page: what this is, how well it works, and what it cost."""
import streamlit as st

from shared import C_BAD, C_GOOD, HUB_URL, REPO_URL, bundle, section


def render():
    st.title("OpenForensics — Deepfake Detector")
    st.markdown(
        "A three-backbone CNN ensemble that decides whether a face image is "
        "**genuine** or **manipulated**, with a calibrated confidence score "
        "and per-backbone visual explanations."
    )

    b = bundle()
    m = b["card"].get("test_metrics", {})
    d = b["card"].get("decision", {})

    if not m:
        st.warning("Metrics unavailable — the model repository could not be reached.")
        return

    c = st.columns(4)
    c[0].metric("Accuracy", f"{m['accuracy']:.4f}", help="Held-out test split, 2,000 images, with test-time augmentation.")
    c[1].metric("ROC-AUC", f"{m['roc_auc']:.4f}", help="Ranking quality, independent of the chosen threshold.")
    c[2].metric("PR-AUC", f"{m['pr_auc']:.4f}")
    c[3].metric("False accusations", f"{m['false_accusation_rate']*100:.1f}%",
                help="Genuine images the model calls fake. The number that matters most when the output is an accusation.")

    st.divider()

    left, right = st.columns([3, 2])
    with left:
        section("What changed in v2")
        st.markdown(
            "The test split is measurably more degraded than the training "
            "split — about 22% less colour saturation and 28% more "
            "high-frequency energy. The first model trained only on clean "
            "images and paid for it on the tail: the worst tenth of genuine "
            "test images scored **0.138**, meaning it confidently called them "
            "fake, while its median sat at 0.985."
            "\n\n"
            "v2 trains against nine reproduced degradation families. The same "
            "tenth percentile is now **0.830**."
        )
        st.dataframe(
            {
                "": ["Accuracy", "ROC-AUC", "Genuine images called fake",
                     "Validation→test recall gap", "10th pct, genuine test scores"],
                "v1 — two backbones": ["0.8860", "0.9421", "175 / 1000", "0.159", "0.138"],
                "v2 — three + augmentation": ["0.9510", "0.9899", "39 / 1000", "0.025", "0.830"],
            },
            hide_index=True, use_container_width=True,
        )
        st.caption("Both at threshold 0.5 for comparability. The shipped operating point is "
                   f"{d.get('threshold', 0.5):.3f}, fitted on validation.")

    with right:
        section("At a glance")
        # A definition list rather than markdown line breaks, which collapse
        # into a run-on paragraph without trailing double-spaces.
        for k, v in [
            ("Architecture", "ResNet50 + VGG16 + EfficientNetV2-B0, late fusion"),
            ("Parameters", f"{b['card'].get('params', 0):,}"),
            ("Input", "224×224 RGB face crop"),
            ("Output", "P(genuine), calibrated"),
            ("Threshold", f"{d.get('threshold', 0.5):.3f}  ·  {d.get('criterion', 'n/a')}"),
            ("Temperature", f"{d.get('temperature', 1.0):.3f}"),
        ]:
            a, bcol = st.columns([1, 2])
            a.caption(k)
            bcol.markdown(f"**{v}**")
        st.link_button("Model on Hugging Face", HUB_URL, use_container_width=True)
        st.link_button("Source on GitHub", REPO_URL, use_container_width=True)

    st.divider()
    st.info(
        "**Research and educational use.** This is not a forensic authority. "
        "A score near the threshold is not evidence — treat the margin as part "
        "of the output, and read the Limitations section before relying on any "
        "single prediction.",
        icon="⚖️",
    )
