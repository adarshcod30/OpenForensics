"""Per-degradation accuracy — the reason v2 exists."""
import altair as alt
import pandas as pd
import streamlit as st

from shared import CORRUPTION_BLURB, C_BAD, C_FAKE, C_GOOD, C_REAL, bundle, section


def render():
    st.title("Robustness")
    st.caption("Accuracy with a single degradation family applied to the whole "
               "test set, one at a time. A single averaged accuracy tells a "
               "forensic examiner nothing about the image in front of them.")

    b = bundle()
    pc = b["report"].get("per_corruption") or {}
    if not pc:
        st.warning("Per-corruption results unavailable — the model repository could not be reached.")
        return

    base = pc.get("clean", {}).get("accuracy")
    rows = [
        {"degradation": k, "accuracy": v["accuracy"], "roc_auc": v["roc_auc"],
         "delta": (v["accuracy"] - base) if k != "clean" else 0.0,
         "what": "No degradation." if k == "clean" else CORRUPTION_BLURB.get(k, "")}
        for k, v in pc.items()
    ]
    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)

    worst = df[df.degradation != "clean"].nsmallest(1, "accuracy").iloc[0]
    cols = st.columns(3)
    cols[0].metric("Clean accuracy", f"{base:.4f}")
    cols[1].metric("Worst degradation", worst.degradation.replace("_", " "),
                   delta=f"{worst.delta:+.4f}",
                   help="Accuracy lost relative to clean images.")
    cols[2].metric("Families tested", len(df) - 1)

    st.divider()
    section("Accuracy by degradation", "Dashed line is clean accuracy. Hover for detail.")
    bars = (
        alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("degradation:N", title=None, sort="-x"),
            x=alt.X("accuracy:Q", title="Accuracy",
                    scale=alt.Scale(domain=[min(df.accuracy) - 0.03, 1.0])),
            color=alt.condition(alt.datum.degradation == "clean",
                                alt.value(C_REAL), alt.value(C_FAKE)),
            tooltip=[alt.Tooltip("degradation:N", title="Degradation"),
                     alt.Tooltip("accuracy:Q", format=".4f"),
                     alt.Tooltip("roc_auc:Q", title="ROC-AUC", format=".4f"),
                     alt.Tooltip("delta:Q", title="vs clean", format="+.4f"),
                     alt.Tooltip("what:N", title="")],
        ).properties(height=340)
    )
    rule = alt.Chart(pd.DataFrame({"a": [base]})).mark_rule(
        color=C_REAL, strokeDash=[6, 4], strokeWidth=2).encode(x="a:Q")
    st.altair_chart(bars + rule, use_container_width=True)

    st.divider()
    section("Full table")
    show = df.copy()
    show["vs clean"] = show["delta"].map(lambda d: "—" if d == 0 else f"{d:+.4f}")
    st.dataframe(
        show[["degradation", "accuracy", "roc_auc", "vs clean", "what"]]
        .rename(columns={"degradation": "Degradation", "accuracy": "Accuracy",
                         "roc_auc": "ROC-AUC", "what": "What it does"}),
        hide_index=True, use_container_width=True,
        column_config={"Accuracy": st.column_config.NumberColumn(format="%.4f"),
                       "ROC-AUC": st.column_config.NumberColumn(format="%.4f")},
    )

    st.divider()
    shift = b["report"].get("distribution_shift")
    if shift:
        section("Why this matters", "The measurement that motivated the whole approach.")
        gap = shift["recall_real_val_at_0.5"] - shift["recall_real_test_at_0.5"]
        c = st.columns(3)
        c[0].metric("Recall on genuine — validation", f"{shift['recall_real_val_at_0.5']:.3f}")
        c[1].metric("Recall on genuine — test", f"{shift['recall_real_test_at_0.5']:.3f}")
        c[2].metric("Gap", f"{gap:.3f}",
                    help="v1, trained without corruption augmentation, showed 0.159 here.")
        st.markdown(
            f"The 10th percentile of scores on genuine images is "
            f"**{shift['p10_real_scores_val']:.3f}** on validation and "
            f"**{shift['p10_real_scores_test']:.3f}** on test. Medians are nearly "
            "identical, so what differs is a *tail* of images the model reads "
            "confidently wrong — not a uniform shift. In v1 that tail sat at "
            "0.138, low enough that genuine photographs were being called "
            "forgeries with high confidence. Reproducing the degradations during "
            "training is what moved it."
        )
