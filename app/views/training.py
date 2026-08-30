"""Training curves for both stages, read from the bundled histories."""
import altair as alt
import pandas as pd
import streamlit as st

from shared import C_FAKE, C_REAL, bundle, section

STAGES = {
    "stage1": ("Stage 1 — heads only", "All three backbones frozen; only the "
               "1.2M classifier-head parameters train. This is the ceiling of "
               "what fixed ImageNet features can express about forgery."),
    "stage2": ("Stage 2 — fine-tuning", "Top ~50 layers of each backbone "
               "unfrozen at a 20× lower learning rate, 34.8M trainable. "
               "BatchNorm stays frozen throughout."),
}
PAIRS = [("auc", "val_auc", "ROC-AUC"), ("accuracy", "val_accuracy", "Accuracy"),
         ("loss", "val_loss", "Loss")]


def _chart(hist: dict, train_key: str, val_key: str, label: str):
    if train_key not in hist or val_key not in hist:
        return None
    rows = []
    for i, (tr, va) in enumerate(zip(hist[train_key], hist[val_key]), start=1):
        rows.append({"epoch": i, label: tr, "split": "train"})
        rows.append({"epoch": i, label: va, "split": "validation"})
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=45), strokeWidth=2.5)
        .encode(
            x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y(f"{label}:Q", title=label, scale=alt.Scale(zero=False)),
            color=alt.Color("split:N", title=None,
                            scale=alt.Scale(domain=["train", "validation"],
                                            range=[C_FAKE, C_REAL])),
            tooltip=["epoch", "split", alt.Tooltip(f"{label}:Q", format=".4f")],
        )
        .properties(height=260)
        .interactive()
    )


def render():
    st.title("Training")
    st.caption("Two stages, 30 epochs, corruption-matched augmentation throughout. "
               "Hover any point for exact values; drag to zoom.")

    b = bundle()
    if not (b["stage1"] or b["stage2"]):
        st.warning("Training histories unavailable — the model repository could not be reached.")
        return

    for key, (title, blurb) in STAGES.items():
        hist = b.get(key) or {}
        if not hist:
            continue
        section(title, blurb)

        n = len(hist.get("loss", []))
        best_i = max(range(n), key=lambda i: hist["val_auc"][i]) if "val_auc" in hist else None
        cols = st.columns(4)
        cols[0].metric("Epochs", n)
        if best_i is not None:
            cols[1].metric("Best val ROC-AUC", f"{hist['val_auc'][best_i]:.4f}",
                           help=f"Epoch {best_i + 1}")
            cols[2].metric("Best val accuracy", f"{max(hist['val_accuracy']):.4f}")
            cols[3].metric("Final train loss", f"{hist['loss'][-1]:.4f}")

        tabs = st.tabs([lbl for _, _, lbl in PAIRS])
        for tab, (tk, vk, lbl) in zip(tabs, PAIRS):
            with tab:
                ch = _chart(hist, tk, vk, lbl)
                if ch is not None:
                    st.altair_chart(ch, use_container_width=True)
                else:
                    st.caption(f"{lbl} not recorded for this stage.")
        st.divider()

    with st.expander("Why validation scores above training"):
        st.markdown(
            "Dropout is active during training and disabled at validation, and "
            "the nine corruption families are applied to training batches only. "
            "The model is therefore scored on materially easier inputs than it "
            "learns from, so validation sitting above training is the expected "
            "shape here — a model trained with heavy augmentation that did "
            "*not* show this gap would be the surprising result."
            "\n\n"
            "The jump between stages is the substantive part: stage 1 plateaus "
            "near 0.885 ROC-AUC because frozen ImageNet features can only "
            "describe so much about forgery. Unfreezing the backbones lets the "
            "features themselves adapt, and validation ROC-AUC reaches 0.996."
        )
