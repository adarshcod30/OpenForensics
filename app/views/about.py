"""Method, limitations, reproduction."""
import streamlit as st

from shared import HUB_URL, REPO_URL, bundle, section


def render():
    st.title("About")

    section("What this is")
    st.markdown(
        "A deepfake face detector built on the OpenForensics dataset, shipped "
        "with the evidence behind it: training curves, a full threshold "
        "trade-off, per-degradation robustness, and the integrity checks run "
        "on its own data splits."
    )

    st.divider()
    section("Limitations", "Read these before relying on any single prediction.")
    for title, body in [
        ("Face swaps only — not AI-generated images",
         "Trained on faces composited into real photographs, so it looks for the "
         "seam where one image was blended into another. An image generated whole "
         "by a diffusion model has no seam: measured on ChatGPT and Gemini output, "
         "10 of 10 were called genuine, most above 0.999. A high genuine score is "
         "not evidence that an image is not AI-generated."),
        ("Face crops only",
         "Trained on tight crops. Behaviour on full scenes, multiple faces or "
         "non-face images is undefined — there is no face detector in the pipeline."),
        ("One dataset",
         "Trained and tested entirely on OpenForensics. Cross-dataset performance "
         "is unmeasured, and the literature is consistent that it drops sharply."),
        ("Predates current generators",
         "OpenForensics was released in 2021. Manipulation methods absent from it, "
         "including recent diffusion-based ones, are out of distribution."),
        ("A margin is not a verdict",
         "A score near the threshold is inconclusive. The Detect page flags these, "
         "and the abstention curve on the Evaluation page quantifies what declining "
         "to answer buys."),
        ("Not a forensic authority",
         "Research and educational use. A prediction is evidence to weigh, not proof."),
    ]:
        with st.container(border=True):
            st.markdown(f"**{title}**  \n{body}")

    st.divider()
    section("Reproducing this")
    st.code(
        "git clone https://github.com/adarshcod30/OpenForensics\n"
        "conda create -n openforensics python=3.11 -y && conda activate openforensics\n"
        'pip install -e ".[train]"\n\n'
        "# train (two stages, ~8 h on an M4; resumable if interrupted)\n"
        "PYTHONPATH=src python -m openforensics.training.train --name v2 \\\n"
        "  --backbones resnet50 vgg16 efficientnetv2b0 --corruption_prob 0.5\n\n"
        "# evaluate, with the per-degradation sweep\n"
        "PYTHONPATH=src python -m openforensics.evaluation.evaluate \\\n"
        "  --run_dir runs/v2 --tta --per_corruption\n\n"
        "# package and publish\n"
        "PYTHONPATH=src python -m openforensics.export.package \\\n"
        "  --run_dir runs/v2 --push_to <user>/<repo>",
        language="bash",
    )
    st.caption("Every run writes its exact file list, a content-hash leakage report "
               "and its configuration alongside the weights, so a number can be "
               "traced back to the data that produced it.")

    st.divider()
    b = bundle()
    c = st.columns(2)
    c[0].link_button("Source on GitHub", REPO_URL, use_container_width=True)
    c[1].link_button("Model on Hugging Face", HUB_URL, use_container_width=True)
    st.caption(f"Model card version: {b['card'].get('params', 0):,} parameters · "
               f"threshold {b['card'].get('decision', {}).get('threshold', 0.5):.3f}")
