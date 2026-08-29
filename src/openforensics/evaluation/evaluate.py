"""Evaluate a trained run and write a decision-ready report.

The operating point and the temperature are both fitted on validation and
then applied unchanged to test. Choosing either on test is the most common
way to publish a number that does not survive contact with real inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, roc_curve

from ..config import RunConfig
from ..data import manifest as manifest_mod
from ..data import pipeline
from ..data.corruptions import CORRUPTIONS
from ..gpulock import gpu_lock
from ..models.layers import PreprocessLayer
from . import metrics as M

CUSTOM = {"PreprocessLayer": PreprocessLayer}


def load(path: str | Path):
    return tf.keras.models.load_model(str(path), compile=False, custom_objects=CUSTOM)


def _curves(y, probs, out_dir: Path, tag: str):
    fpr, tpr, _ = roc_curve(y, probs)
    prec, rec, _ = precision_recall_curve(y, probs)
    for xs, ys, xl, yl, title, fname in (
        (fpr, tpr, "False positive rate", "True positive rate", "ROC", f"roc_{tag}.png"),
        (rec, prec, "Recall", "Precision", "Precision-Recall", f"pr_{tag}.png"),
    ):
        plt.figure(figsize=(5.2, 4.4))
        plt.plot(xs, ys, lw=2)
        if title == "ROC":
            plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel(xl); plt.ylabel(yl); plt.title(f"{title} ({tag})")
        plt.tight_layout(); plt.savefig(out_dir / fname, dpi=130); plt.close()


def per_corruption(model, files, labels, batch, img_size, threshold, tta):
    """Accuracy under each degradation family, applied to the clean test set.

    This is the shipped model's robustness profile: which real-world
    degradations it survives and which it does not.
    """
    rows = {}
    clean = pipeline.build_dataset(files, labels, batch=batch, img_size=img_size)
    y = np.asarray(labels, dtype=int)
    rows["clean"] = M.summarise(y, M.predict(model, clean, tta=tta), threshold)

    for name, fn in CORRUPTIONS:
        ds = pipeline.build_dataset(files, labels, batch=batch, img_size=img_size)
        # Applied deterministically at eval time: one corruption, no stacking,
        # so each row isolates a single degradation family.
        ds = ds.map(lambda x, y_, f=fn: (tf.map_fn(f, x), y_))
        rows[name] = M.summarise(y, M.predict(model, ds, tta=tta), threshold)
    return rows


def main(a):
    with gpu_lock(f"evaluate:{Path(a.run_dir).name}"):
        return _main(a)


def _main(a):
    run_dir = Path(a.run_dir)
    out = run_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)

    model = load(a.model_path or (run_dir / "final.keras"))

    mf_path = run_dir / "manifest.json"
    if mf_path.exists():
        mf = manifest_mod.Manifest.load(mf_path)
        print(f"using committed manifest ({mf.digest()})")
    else:
        cfg = RunConfig.load(run_dir / "config.json")
        mf = manifest_mod.build(
            cfg.data.base_dir, ("Fake", "Real"),
            {"Validation": cfg.data.val_per_class, "Test": cfg.data.test_per_class},
            seed=cfg.data.seed,
        )

    img_size = tuple(model.input_shape[1:3])
    val = pipeline.from_manifest(mf, "Validation", batch=a.batch, img_size=img_size)
    test = pipeline.from_manifest(mf, "Test", batch=a.batch, img_size=img_size)
    y_val = np.asarray(mf.splits["Validation"]["labels"], dtype=int)
    y_test = np.asarray(mf.splits["Test"]["labels"], dtype=int)

    # ---- fit on validation ----
    p_val = M.predict(model, val, tta=a.tta)
    temperature = M.fit_temperature(y_val, p_val)
    p_val_cal = M.apply_temperature(p_val, temperature)
    threshold, val_at_thr = M.select_threshold(y_val, p_val_cal, a.criterion, a.target)
    print(f"\nfitted on validation: temperature={temperature:.3f}  "
          f"threshold={threshold:.3f} ({a.criterion})")
    print(f"  val ECE {M.expected_calibration_error(y_val, p_val):.4f} "
          f"-> {M.expected_calibration_error(y_val, p_val_cal):.4f} after calibration")

    # ---- apply to test ----
    p_test = M.apply_temperature(M.predict(model, test, tta=a.tta), temperature)
    at_half = M.summarise(y_test, p_test, 0.5)
    at_thr = M.summarise(y_test, p_test, threshold)
    _curves(y_test, p_test, out, "test")

    print(f"\ntest @0.5        acc {at_half['accuracy']:.4f}  AUC {at_half['roc_auc']:.4f}  "
          f"real-called-fake {at_half['real_called_fake']}")
    print(f"test @{threshold:.3f}      acc {at_thr['accuracy']:.4f}  AUC {at_thr['roc_auc']:.4f}  "
          f"real-called-fake {at_thr['real_called_fake']}")

    report = {
        "model": str(a.model_path or (run_dir / "final.keras")),
        "manifest_digest": mf.digest(),
        "tta": a.tta,
        "calibration": {"temperature": temperature,
                        "val_ece_raw": M.expected_calibration_error(y_val, p_val),
                        "val_ece_calibrated": M.expected_calibration_error(y_val, p_val_cal)},
        "operating_point": {"criterion": a.criterion, "target": a.target,
                            "threshold": threshold, "validation": val_at_thr},
        "test_at_0.5": at_half,
        "test_at_threshold": at_thr,
        "risk_coverage": M.risk_coverage(y_test, p_test),
    }

    if a.per_corruption:
        print("\nper-corruption accuracy (test set, one family at a time):")
        rows = per_corruption(model, mf.splits["Test"]["files"], y_test.tolist(),
                              a.batch, img_size, threshold, a.tta)
        report["per_corruption"] = rows
        base = rows["clean"]["accuracy"]
        for name, r in rows.items():
            delta = r["accuracy"] - base
            mark = "" if name == "clean" else f"  ({delta:+.4f})"
            print(f"  {name:<18} acc {r['accuracy']:.4f}  AUC {r['roc_auc']:.4f}{mark}")

    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwritten -> {out/'report.json'}")
    return report


def cli():
    p = argparse.ArgumentParser(description="Evaluate a trained run.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--model_path", default=None)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--tta", action="store_true", help="average over horizontal flip")
    p.add_argument("--criterion", default="youden",
                   choices=["youden", "f1", "target_recall", "min_fpr"])
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--per_corruption", action="store_true")
    main(p.parse_args())


if __name__ == "__main__":
    cli()
