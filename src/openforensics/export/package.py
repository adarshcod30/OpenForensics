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

    for extra in ("config.json", "manifest.json"):
        p = run_dir / extra
        if p.exists():
            shutil.copy(p, out / extra)
    print(f"packaged -> {out}")

    if a.push_to:
        push(out, a.push_to, a.private)
    return card


def push(folder: Path, repo_id: str, private: bool = False):
    """Upload to the Hugging Face Hub.

    Weights cannot live in git: the repo's own .gitignore excludes *.keras
    precisely because they exceed what GitHub accepts. The Hub is where they
    belong, and it is also where the Space will fetch them from at runtime.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model")
    print(f"pushed -> https://huggingface.co/{repo_id}")


def cli():
    p = argparse.ArgumentParser(description="Package a run for serving.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--model_path", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--push_to", default=None, help="HF repo id, e.g. user/openforensics")
    p.add_argument("--private", action="store_true")
    main(p.parse_args())


if __name__ == "__main__":
    cli()
