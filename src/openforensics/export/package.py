"""Package a trained run for serving.

Two jobs. First, shrink: a checkpoint saved mid-training carries Adam moment
estimates for every trainable weight, which is roughly two thirds of the
file and useless at inference. Loading with `compile=False` and re-saving
drops them.

Second, keep the decision parameters with the weights. A model shipped
without its temperature and threshold is a model that will be served at 0.5
by whoever deploys it next, undoing the calibration entirely.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import tensorflow as tf

from ..models.layers import PreprocessLayer

CUSTOM = {"PreprocessLayer": PreprocessLayer}
SERVING_CARD = "serving.json"


def strip_and_save(src: str | Path, dst: str | Path) -> dict:
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(str(src), compile=False, custom_objects=CUSTOM)
    model.save(str(dst))
    before, after = src.stat().st_size, dst.stat().st_size
    return {
        "source": str(src), "output": str(dst),
        "mb_before": round(before / 1e6, 1), "mb_after": round(after / 1e6, 1),
        "reduction_pct": round(100 * (1 - after / before), 1),
        "params": int(model.count_params()),
        "input_shape": list(model.input_shape[1:]),
    }


def build_card(run_dir: Path, weights_name: str, info: dict) -> dict:
    """Serving card: everything a caller needs and nothing it must guess."""
    card = {
        "weights": weights_name,
        "input": {"size": info["input_shape"][:2], "channels": 3,
                  "dtype": "float32", "range": [0, 1],
                  "note": "Resize to size, divide by 255. Per-backbone "
                          "normalisation happens inside the model."},
        "output": {"name": "probability_real", "range": [0, 1],
                   "note": "P(image is Real). Fake is 1 - p."},
        "params": info["params"],
        "decision": {"threshold": 0.5, "temperature": 1.0,
                     "calibrated": False},
        "tta": {"recommended": True, "transform": "horizontal_flip",
                "note": "Average P(Real) over the image and its mirror."},
    }
    report = run_dir / "eval" / "report.json"
    if report.exists():
        r = json.loads(report.read_text())
        card["decision"] = {
            "threshold": r["operating_point"]["threshold"],
            "temperature": r["calibration"]["temperature"],
            "criterion": r["operating_point"]["criterion"],
            "calibrated": True,
        }
        card["test_metrics"] = {
            k: r["test_at_threshold"][k]
            for k in ("accuracy", "roc_auc", "pr_auc",
                      "real_called_fake", "false_accusation_rate")
        }
        card["tta"]["used_in_eval"] = r.get("tta", False)
    else:
        card["warning"] = (
            "No eval report found — threshold and temperature are defaults, "
            "not fitted. Run evaluate.py before serving this model."
        )
    return card


def model_card(card: dict, backbones: list[str], version: str,
               report: dict | None = None) -> str:
    report = report or {}
    d = card.get("decision", {})
    m = card.get("test_metrics", {})
    metrics_block = (
        f"| Accuracy | {m['accuracy']:.4f} |\n"
        f"| ROC-AUC | {m['roc_auc']:.4f} |\n"
        f"| PR-AUC | {m['pr_auc']:.4f} |\n"
        f"| Real images called fake | {m['real_called_fake']} "
        f"({m['false_accusation_rate']*100:.1f}%) |\n"
        if m else "| _not evaluated_ | — |\n"
    )
    crit = d.get("criterion", "n/a")
    if crit == "fixed":
        threshold_prose = (
            f"The published threshold is **{d.get('threshold', 0.5):.3f}** — a neutral "
            f"default, **not** a fitted operating point. A temperature of "
            f"**{d.get('temperature', 1.0):.3f}** was fitted on validation and is applied.\n\n"
            f"Thresholds fitted on the validation split do not transfer to the test "
            f"split for this dataset (see Limitations). Pick your own operating point "
            f"from `threshold_sweep` in the evaluation report, on data resembling "
            f"your deployment."
        )
    else:
        threshold_prose = (
            f"Decision threshold **{d.get('threshold', 0.5):.3f}** and temperature "
            f"**{d.get('temperature', 1.0):.3f}** were fitted on a held-out validation "
            f"split ({crit} criterion) and are carried in `serving.json`."
        )

    corrupt_p = ((report.get("config") or {}).get("data") or {}).get("corruption_prob")
    if corrupt_p:
        augmentation_prose = (
            "Training used corruption-matched augmentation — desaturation, colour "
            "cast, noise, speckle, blur, JPEG artefacts, pixelation, brightness "
            "shift and occlusion — because the test split is measurably more "
            "degraded than train."
        )
    else:
        augmentation_prose = (
            "Training used light augmentation only: horizontal flip, small "
            "brightness and contrast jitter. No corruption-matched augmentation."
        )

    sh = report.get("distribution_shift")
    shift_prose = ""
    if sh:
        gap = sh["recall_real_val_at_0.5"] - sh["recall_real_test_at_0.5"]
        # The wording has to follow the measurement. A model whose validation
        # tracks test must not carry a warning that it does not.
        if gap > 0.08:
            shift_prose = (
                f"- **Validation does not predict test performance.** Recall on "
                f"genuine images at threshold 0.5 is "
                f"{sh['recall_real_val_at_0.5']:.3f} on validation but "
                f"{sh['recall_real_test_at_0.5']:.3f} on test — a gap of {gap:.3f}. "
                f"The 10th percentile of scores on genuine images is "
                f"{sh['p10_real_scores_val']:.3f} on validation and "
                f"{sh['p10_real_scores_test']:.3f} on test, so a subset is "
                f"confidently misread rather than the whole distribution shifting. "
                f"Re-fit the operating point on data resembling your deployment."
            )
        else:
            shift_prose = (
                f"- **Validation tracks test closely.** Recall on genuine images at "
                f"threshold 0.5 is {sh['recall_real_val_at_0.5']:.3f} on validation "
                f"and {sh['recall_real_test_at_0.5']:.3f} on test — a gap of "
                f"{gap:.3f}. The 10th percentile of scores on genuine images is "
                f"{sh['p10_real_scores_val']:.3f} and "
                f"{sh['p10_real_scores_test']:.3f} respectively, so the operating "
                f"point fitted on validation transfers. This is a property of the "
                f"corruption-matched augmentation, not of the benchmark."
            )

    pc = report.get("per_corruption") or {}
    robustness = ""
    if pc:
        base = pc.get("clean", {}).get("accuracy")
        rows = "\n".join(
            f"| {k} | {v['accuracy']:.4f} | {v['roc_auc']:.4f} | "
            f"{'—' if k == 'clean' else format(v['accuracy'] - base, '+.4f')} |"
            for k, v in pc.items()
        )
        robustness = (
            "\n## Robustness\n\n"
            "Accuracy with a single degradation family applied to the whole test "
            "set, one at a time.\n\n"
            "| Degradation | Accuracy | ROC-AUC | vs clean |\n|---|---|---|---|\n"
            + rows + "\n"
        )

    return f"""---
license: mit
tags: [deepfake-detection, image-classification, forensics, tensorflow, keras]
library_name: keras
pipeline_tag: image-classification
---

# OpenForensics Deepfake Detector ({version})

A multi-backbone CNN ensemble that classifies face crops as **Real** or
**Fake**. Backbones: {', '.join(backbones)}. Their pooled embeddings are
concatenated and read by a shared classifier head.

## Output

A single sigmoid: **P(Real)**. Fake is `1 - p`.

{threshold_prose}

## Test metrics

| Metric | Value |
|---|---|
{metrics_block}
Measured on a held-out test split with horizontal-flip test-time
augmentation. The split is content-hash deduplicated against train and
validation, so no image appears in more than one split.

## Input

Resize to 224x224, scale to `[0, 1]`, shape `(N, 224, 224, 3)` float32.
Per-backbone normalisation happens **inside** the model — do not apply
`preprocess_input` yourself.

```python
from huggingface_hub import snapshot_download
import tensorflow as tf, numpy as np, json
from PIL import Image

path = snapshot_download("adarshcod30/openforensics-ensemble")
model = tf.keras.models.load_model(f"{{path}}/model.keras", compile=False)
card = json.load(open(f"{{path}}/serving.json"))

img = Image.open("face.jpg").convert("RGB").resize((224, 224))
x = np.asarray(img, dtype="float32")[None] / 255.0
p = float(model.predict(x)[0, 0])
print("Real" if p >= card["decision"]["threshold"] else "Fake", p)
```

Loading needs the `PreprocessLayer` custom layer from
[the repo](https://github.com/adarshcod30/OpenForensics), or pass it via
`custom_objects`.

## Training data

The face-cropped OpenForensics distribution (190,334 images at 256x256).
{augmentation_prose}

{robustness}
## Limitations

- Trained on **face crops**. Behaviour on full scenes or non-face images is
  undefined.
- A score near the threshold is not evidence. Treat the margin as part of
  the output.
- Performance degrades on manipulation methods absent from OpenForensics.
- Research and educational use. Not a forensic authority.
{shift_prose}

## Citation

> Trung-Nghia Le, Huy H. Nguyen, Junichi Yamagishi, Isao Echizen,
> "OpenForensics: Large-Scale Challenging Dataset For Multi-Face Forgery
> Detection And Segmentation In-The-Wild", ICCV 2021.
"""


def main(a):
    run_dir = Path(a.run_dir)
    out = Path(a.out_dir or (run_dir / "serving"))
    out.mkdir(parents=True, exist_ok=True)

    src = Path(a.model_path or (run_dir / "final.keras"))
    info = strip_and_save(src, out / "model.keras")
    print(f"stripped {info['mb_before']} MB -> {info['mb_after']} MB "
          f"({info['reduction_pct']}% smaller, {info['params']:,} params)")

    card = build_card(run_dir, "model.keras", info)
    (out / SERVING_CARD).write_text(json.dumps(card, indent=2))
    if "warning" in card:
        print(f"WARNING: {card['warning']}")
    else:
        print(f"decision: threshold={card['decision']['threshold']:.3f} "
              f"temperature={card['decision']['temperature']:.3f}")

    backbones = ["resnet50", "vgg16"]
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        backbones = json.loads(cfg_path.read_text())["model"]["backbones"]
    report_path = run_dir / "eval" / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    if cfg_path.exists():
        report["config"] = json.loads(cfg_path.read_text())
    (out / "README.md").write_text(model_card(card, backbones, a.version, report))

    # Training curves, the evaluation report and the leakage check travel with
    # the weights so the deployed app can render them without the repository
    # or the run directory -- neither of which exists on a hosting platform.
    for extra in ("config.json", "stage1_history.json", "stage2_history.json",
                  "stage1_log.csv", "stage2_log.csv", "leakage_report.json"):
        p = run_dir / extra
        if p.exists():
            shutil.copy(p, out / extra)
    report = run_dir / "eval" / "report.json"
    if report.exists():
        shutil.copy(report, out / "evaluation_report.json")
    print(f"packaged -> {out}")

    if a.push_to:
        push(out, a.push_to, a.private, a.make_public)
    return card


def push(folder: Path, repo_id: str, private: bool = False,
         make_public: bool = False):
    """Upload to the Hugging Face Hub.

    Weights cannot live in git: the repo's own .gitignore excludes *.keras
    precisely because they exceed what GitHub accepts. The Hub is where they
    belong, and it is also where the Space will fetch them from at runtime.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model")
    if make_public:
        api.update_repo_settings(repo_id=repo_id, repo_type="model", private=False)
        print("repo set to public")
    print(f"pushed -> https://huggingface.co/{repo_id}")


def cli():
    p = argparse.ArgumentParser(description="Package a run for serving.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--model_path", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--push_to", default=None, help="HF repo id, e.g. user/openforensics")
    p.add_argument("--private", action="store_true")
    p.add_argument("--version", default="v1")
    p.add_argument("--make_public", action="store_true",
                   help="flip an existing private repo to public after upload")
    main(p.parse_args())


if __name__ == "__main__":
    cli()
