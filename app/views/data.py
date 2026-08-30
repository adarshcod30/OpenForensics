"""Dataset, splits, and the integrity checks run before training."""
import altair as alt
import pandas as pd
import streamlit as st

from shared import CORRUPTION_BLURB, C_FAKE, C_GOOD, C_REAL, bundle, section

SPLIT_STATS = [
    {"split": "Train / Real", "saturation": 0.3688, "greyscale": 1.5, "high_freq": 4.45},
    {"split": "Train / Fake", "saturation": 0.3722, "greyscale": 0.5, "high_freq": 4.52},
    {"split": "Validation / Real", "saturation": 0.3570, "greyscale": 1.2, "high_freq": 4.67},
    {"split": "Validation / Fake", "saturation": 0.3451, "greyscale": 6.2, "high_freq": 5.60},
    {"split": "Test / Real", "saturation": 0.2890, "greyscale": 6.2, "high_freq": 5.87},
    {"split": "Test / Fake", "saturation": 0.3351, "greyscale": 3.2, "high_freq": 5.73},
]


def render():
    st.title("Data")
    st.caption("The face-cropped OpenForensics distribution — 190,334 JPEGs at "
               "256×256, split into Train, Validation and Test with Fake and "
               "Real subfolders.")

    b = bundle()
    d = b["config"].get("data", {})

    c = st.columns(4)
    c[0].metric("Train", f"{d.get('train_per_class', 10000) * 2:,}", help="Per class, balanced.")
    c[1].metric("Validation", f"{d.get('val_per_class', 3000) * 2:,}")
    c[2].metric("Test", f"{d.get('test_per_class', 1000) * 2:,}")
    c[3].metric("Seed", d.get("seed", "—"), help="Sampling is deterministic; the exact "
                                                 "file list ships with the model.")

    st.divider()
    section("Split integrity",
            "Content-hashed across splits before training. Filenames repeat "
            "between splits, so a name-based check proves nothing — only "
            "hashes do.")
    leak = b["leakage"]
    if leak:
        pairs = leak.get("pairs", {})
        rows = [{"pair": k.replace("|", " ↔ "), "shared images": v["shared_images"]}
                for k, v in pairs.items()]
        dup = leak.get("within_split_duplicates", {})
        rows += [{"pair": f"duplicates within {k}", "shared images": v} for k, v in dup.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if not leak.get("leaking") and not any(dup.values()):
            st.success("No image appears in more than one split, and none is duplicated "
                       "within a split.", icon="✅")
        st.caption("The raw sample of this dataset carries 125 images shared between "
                   "Train and Validation and 213 duplicates inside Train. Deduplication "
                   "draws until the target count of unique hashes is reached, so split "
                   "sizes stay exact.")

    st.divider()
    section("The test split is degraded",
            "400 images sampled per split and class. Every measure moves the way "
            "degradation would move it, and only in the test split.")
    df = pd.DataFrame(SPLIT_STATS)
    df["is_test"] = df["split"].str.startswith("Test")
    metric = st.radio("Measure", ["saturation", "high_freq", "greyscale"], horizontal=True,
                      format_func=lambda m: {"saturation": "Mean saturation",
                                             "high_freq": "High-frequency energy",
                                             "greyscale": "Near-greyscale (%)"}[m])
    st.altair_chart(
        alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("split:N", title=None, sort=[r["split"] for r in SPLIT_STATS]),
            x=alt.X(f"{metric}:Q", title=None),
            color=alt.condition(alt.datum.is_test, alt.value(C_FAKE), alt.value(C_REAL)),
            tooltip=["split", alt.Tooltip(f"{metric}:Q", format=".4f")],
        ).properties(height=250),
        use_container_width=True,
    )
    st.caption("Test/Real loses 22% of its colour saturation relative to Train/Real and "
               "carries four times the near-greyscale images, while high-frequency "
               "energy — the signature of added noise — rises about 28% across both "
               "test classes.")

    st.divider()
    section("Corruption families used in training",
            "Reproduced from what is visibly present in the test split, applied "
            "to training batches with probability 0.5, stacking up to two deep.")
    st.dataframe(
        pd.DataFrame([{"Family": k.replace("_", " "), "Effect": v}
                      for k, v in CORRUPTION_BLURB.items()]),
        hide_index=True, use_container_width=True,
    )

    st.divider()
    st.markdown(
        "> Trung-Nghia Le, Huy H. Nguyen, Junichi Yamagishi, Isao Echizen, "
        "*OpenForensics: Large-Scale Challenging Dataset For Multi-Face Forgery "
        "Detection And Segmentation In-The-Wild*, ICCV 2021."
    )
