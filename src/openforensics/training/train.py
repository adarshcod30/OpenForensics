"""Two-stage training.

Stage 1 trains the classifier heads with every backbone frozen. Stage 2
unfreezes the top of each backbone at a much lower learning rate.

Both stages write into one run directory together with the config, the split
manifest and the leakage report, so the run is self-describing.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from ..config import CLASSES, RunConfig
from ..gpulock import gpu_lock
from ..data import manifest as manifest_mod
from ..data import pipeline
from ..models.ensemble import build_ensemble, set_backbone_trainable, summarise


def _metrics():
    return [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.AUC(name="pr_auc", curve="PR"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]


def _callbacks(cfg: RunConfig, stage: str, ckpt: Path):
    t = cfg.train
    return [
        keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor=t.monitor, mode=t.monitor_mode, save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor=t.monitor, mode=t.monitor_mode,
            patience=t.early_stop_patience, restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=t.monitor, mode=t.monitor_mode,
            factor=0.5, patience=t.reduce_lr_patience, min_lr=1e-8,
        ),
        keras.callbacks.CSVLogger(str(cfg.run_dir / f"{stage}_log.csv")),
        keras.callbacks.TensorBoard(log_dir=str(cfg.run_dir / "tb" / stage)),
    ]


def prepare_data(cfg: RunConfig):
    d = cfg.data
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    mf = manifest_mod.build(
        d.base_dir, CLASSES,
        {"Train": d.train_per_class, "Validation": d.val_per_class, "Test": d.test_per_class},
        seed=d.seed,
    )
    mf.save(run_dir / "manifest.json")

    # Leakage is checked before training, not after a surprising result.
    report = manifest_mod.check_leakage(mf)
    (run_dir / "leakage_report.json").write_text(json.dumps(report, indent=2))

    train_ds = pipeline.from_manifest(
        mf, "Train", batch=d.batch_size, shuffle=True, augment=True,
        corruption_prob=d.corruption_prob, max_corruptions=d.max_corruptions,
        img_size=d.img_size, seed=d.seed,
    )
    # Validation and test are never augmented and are re-read every epoch, so
    # caching the decoded tensors is free speed.
    val_ds = pipeline.from_manifest(
        mf, "Validation", batch=d.batch_size, img_size=d.img_size, cache=True
    )
    test_ds = pipeline.from_manifest(
        mf, "Test", batch=d.batch_size, img_size=d.img_size, cache=True
    )
    return mf, train_ds, val_ds, test_ds


def main(cfg: RunConfig, stage2: bool = True):
    with gpu_lock(f"train:{cfg.name}"):
        return _main(cfg, stage2)


def _main(cfg: RunConfig, stage2: bool = True):
    keras.utils.set_random_seed(cfg.data.seed)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save()

    mf, train_ds, val_ds, test_ds = prepare_data(cfg)
    print(f"\nrun dir: {cfg.run_dir}   manifest digest: {mf.digest()}")

    model = build_ensemble(
        input_shape=(*cfg.data.img_size, 3),
        backbones=cfg.model.backbones,
        head_units=cfg.model.head_units,
        dropout_branch=cfg.model.dropout_branch,
        dropout_merge=cfg.model.dropout_merge,
        dropout_final=cfg.model.dropout_final,
    )

    # ---------------- stage 1: heads only ----------------
    print(f"\n[stage 1] {summarise(model)}")
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.train.lr),
        loss="binary_crossentropy",
        metrics=_metrics(),
    )
    ckpt1 = cfg.run_dir / "stage1_best.keras"
    t0 = time.time()
    h1 = model.fit(
        train_ds, validation_data=val_ds, epochs=cfg.train.epochs,
        callbacks=_callbacks(cfg, "stage1", ckpt1), verbose=2,
    )
    print(f"[stage 1] done in {(time.time()-t0)/60:.1f} min")
    (cfg.run_dir / "stage1_history.json").write_text(
        json.dumps({k: [float(x) for x in v] for k, v in h1.history.items()}, indent=2)
    )

    if not stage2:
        model.save(cfg.run_dir / "final.keras")
        return model

    # ---------------- stage 2: fine-tune backbones ----------------
    changed = set_backbone_trainable(
        model, backbones=cfg.model.backbones,
        unfreeze_last=cfg.train.unfreeze_last,
        freeze_batchnorm=cfg.train.freeze_batchnorm,
    )
    print(f"\n[stage 2] unfrozen (non-BN) per backbone: {changed}")
    print(f"[stage 2] {summarise(model)}")
    # Recompiling is required for the new trainable set to take effect.
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.train.finetune_lr),
        loss="binary_crossentropy",
        metrics=_metrics(),
    )
    ckpt2 = cfg.run_dir / "stage2_best.keras"
    t0 = time.time()
    h2 = model.fit(
        train_ds, validation_data=val_ds, epochs=cfg.train.finetune_epochs,
        callbacks=_callbacks(cfg, "stage2", ckpt2), verbose=2,
    )
    print(f"[stage 2] done in {(time.time()-t0)/60:.1f} min")
    (cfg.run_dir / "stage2_history.json").write_text(
        json.dumps({k: [float(x) for x in v] for k, v in h2.history.items()}, indent=2)
    )

    model.save(cfg.run_dir / "final.keras")
    res = model.evaluate(test_ds, return_dict=True, verbose=0)
    (cfg.run_dir / "test_quick.json").write_text(
        json.dumps({k: float(v) for k, v in res.items()}, indent=2)
    )
    print("\ntest:", {k: round(float(v), 4) for k, v in res.items()})
    return model


def cli():
    p = argparse.ArgumentParser(description="Train the OpenForensics ensemble.")
    p.add_argument("--name", default="v2")
    p.add_argument("--base_dir", default="./Dataset")
    p.add_argument("--out_dir", default="./runs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--finetune_lr", type=float, default=1e-5)
    p.add_argument("--train_per_class", type=int, default=10000)
    p.add_argument("--val_per_class", type=int, default=3000)
    p.add_argument("--test_per_class", type=int, default=1000)
    p.add_argument("--corruption_prob", type=float, default=0.5)
    p.add_argument("--backbones", nargs="+",
                   default=["resnet50", "vgg16", "efficientnetv2b0"])
    p.add_argument("--no_stage2", action="store_true")
    a = p.parse_args()

    cfg = RunConfig(name=a.name, out_dir=a.out_dir)
    cfg.data.base_dir = a.base_dir
    cfg.data.train_per_class = a.train_per_class
    cfg.data.val_per_class = a.val_per_class
    cfg.data.test_per_class = a.test_per_class
    cfg.data.batch_size = a.batch
    cfg.data.corruption_prob = a.corruption_prob
    cfg.model.backbones = tuple(a.backbones)
    cfg.train.epochs = a.epochs
    cfg.train.finetune_epochs = a.finetune_epochs
    cfg.train.lr = a.lr
    cfg.train.finetune_lr = a.finetune_lr
    main(cfg, stage2=not a.no_stage2)


if __name__ == "__main__":
    cli()
