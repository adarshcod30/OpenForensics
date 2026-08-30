"""Test-set evaluation: operating point, trade-off curve, abstention."""
import altair as alt
import pandas as pd
import streamlit as st

from shared import C_BAD, C_FAKE, C_GOOD, C_REAL, bundle, section


def _confusion(cm: list[list[int]]):
    tn, fp = cm[0]
    fn, tp = cm[1]
    df = pd.DataFrame(
        [{"truth": t, "predicted": p, "n": n}
         for t, p, n in (("Manipulated", "Manipulated", tn), ("Manipulated", "Genuine", fp),
                         ("Genuine", "Manipulated", fn), ("Genuine", "Genuine", tp))]
    )
    df["correct"] = df["truth"] == df["predicted"]
    return (
        alt.Chart(df)
        .mark_rect(stroke="white", strokeWidth=3)
        .encode(
            x=alt.X("predicted:N", title="Predicted", sort=["Manipulated", "Genuine"]),
            y=alt.Y("truth:N", title="Ground truth", sort=["Manipulated", "Genuine"]),
            color=alt.Color("correct:N", legend=None,
                            scale=alt.Scale(domain=[True, False], range=[C_GOOD, C_BAD])),
            tooltip=["truth", "predicted", "n"],
        )
        .properties(height=240)
        + alt.Chart(df).mark_text(fontSize=22, fontWeight="bold", color="white").encode(
            x=alt.X("predicted:N", sort=["Manipulated", "Genuine"]),
            y=alt.Y("truth:N", sort=["Manipulated", "Genuine"]),
            text="n:Q",
        )
    )


def render():
    st.title("Evaluation")
    st.caption("2,000 held-out test images, deduplicated against train and "
               "validation by content hash. Test-time augmentation applied.")

    b = bundle()
    rep, card = b["report"], b["card"]
    if not rep:
        st.warning("Evaluation report unavailable — the model repository could not be reached.")
        return

    at_thr = rep.get("test_at_threshold", {})
    op = rep.get("operating_point", {})

    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{at_thr.get('accuracy', 0):.4f}")
    cols[1].metric("ROC-AUC", f"{at_thr.get('roc_auc', 0):.4f}")
    cols[2].metric("Genuine called fake", at_thr.get("real_called_fake", "—"))
    cols[3].metric("Fake called genuine", at_thr.get("fake_called_real", "—"))

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        section("Confusion matrix", f"At the shipped threshold {op.get('threshold', 0.5):.3f}.")
        if at_thr.get("confusion_matrix"):
            st.altair_chart(_confusion(at_thr["confusion_matrix"]), use_container_width=True)

    with right:
        section("Operating point", "Chosen on validation, applied unchanged to test.")
        st.markdown(
            f"**Criterion** `{op.get('criterion', 'n/a')}`  \n"
            f"**Threshold** {op.get('threshold', 0.5):.3f}  \n"
            f"**Temperature** {rep.get('calibration', {}).get('temperature', 1.0):.3f}"
        )
        cal = rep.get("calibration", {})
        if cal:
            st.metric("Calibration error (validation)",
                      f"{cal.get('val_ece_calibrated', 0):.4f}",
                      delta=f"{cal.get('val_ece_calibrated', 0) - cal.get('val_ece_raw', 0):+.4f}",
                      delta_color="inverse",
                      help="Expected calibration error before and after temperature scaling. "
                           "Lower is better; the transform is monotone so ROC-AUC is unchanged.")
        if op.get("rationale"):
            with st.expander("Why this criterion"):
                st.write(op["rationale"])

    st.divider()
    section("Threshold trade-off",
            "Every operating point on the test set. Lowering the threshold "
            "reduces false accusations and increases missed forgeries — there "
            "is no setting that improves both.")
    sweep = rep.get("threshold_sweep") or []
    if sweep:
        df = pd.DataFrame(sweep).melt(
            id_vars="threshold", value_vars=["real_called_fake", "fake_called_real"],
            var_name="error", value_name="count",
        )
        df["error"] = df["error"].map({"real_called_fake": "Genuine called fake",
                                       "fake_called_real": "Fake called genuine"})
        chart = (
            alt.Chart(df).mark_line(strokeWidth=2.5, point=True)
            .encode(
                x=alt.X("threshold:Q", title="Threshold on P(genuine)"),
                y=alt.Y("count:Q", title="Errors out of 1,000"),
                color=alt.Color("error:N", title=None,
                                scale=alt.Scale(range=[C_FAKE, C_REAL])),
                tooltip=["threshold", "error", "count"],
            ).properties(height=300).interactive()
        )
        rule = alt.Chart(pd.DataFrame({"t": [op.get("threshold", 0.5)]})).mark_rule(
            color=C_GOOD, strokeDash=[6, 4], strokeWidth=2).encode(x="t:Q")
        st.altair_chart(chart + rule, use_container_width=True)
        st.caption("Dashed line marks the shipped threshold.")
        with st.expander("Full sweep as a table"):
            st.dataframe(pd.DataFrame(sweep), hide_index=True, use_container_width=True)

    rc = rep.get("risk_coverage") or []
    if rc:
        st.divider()
        section("Abstention",
                "Accuracy if the least-confident predictions are declined. For a "
                "forensic tool, refusing to answer on a borderline image is the "
                "honest option — this says what that refusal buys.")
        df = pd.DataFrame(rc)
        st.altair_chart(
            alt.Chart(df).mark_line(strokeWidth=2.5, point=True, color=C_GOOD).encode(
                x=alt.X("coverage:Q", title="Fraction of images answered",
                        scale=alt.Scale(reverse=True)),
                y=alt.Y("accuracy:Q", title="Accuracy on those answered",
                        scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("coverage:Q", format=".1%"),
                         alt.Tooltip("accuracy:Q", format=".4f"),
                         alt.Tooltip("min_confidence:Q", format=".3f")],
            ).properties(height=280).interactive(),
            use_container_width=True,
        )
        best = max(rc, key=lambda r: r["accuracy"])
        st.caption(f"Answering the most confident {best['coverage']:.0%} reaches "
                   f"{best['accuracy']:.4f} accuracy.")
