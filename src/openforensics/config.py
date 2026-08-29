"""Central configuration.

Every tunable the pipeline reads lives here so a run is described by one
object, and that object is serialised into the run directory alongside the
weights. A result you cannot reconstruct is not a result.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

SEED = 12345
IMG_SIZE = (224, 224)
CLASSES = ("Fake", "Real")   # index 0 = Fake, 1 = Real; label 1 means "Real"


@dataclass
class DataConfig:
    base_dir: str = "./Dataset"
    train_per_class: int = 10_000
    val_per_class: int = 3_000
    test_per_class: int = 1_000
    batch_size: int = 32
    seed: int = SEED
    img_size: tuple[int, int] = IMG_SIZE

    # Corruption-matched augmentation. The test split carries desaturation,
    # colour cast, noise, blocking, blur and occlusion that Train/Validation
    # do not; training without them costs ~8 points val->test.
    corruption_prob: float = 0.5      # P(any corruption applied to a sample)
    max_corruptions: int = 2          # how many may stack on one image


@dataclass
class ModelConfig:
    # Third branch is off by default so the two-branch baseline stays
    # reproducible; the shipped model turns it on.
    backbones: tuple[str, ...] = ("resnet50", "vgg16", "efficientnetv2b0")
    head_units: int = 256
    dropout_branch: float = 0.4
    dropout_merge: float = 0.4
    dropout_final: float = 0.3


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 2e-4
    # Fine-tuning
    finetune_epochs: int = 10
    finetune_lr: float = 1e-5
    unfreeze_last: int = 50
    # BatchNorm layers stay frozen when unfreezing a backbone. Trainable BN at
    # batch=16 overwrites ImageNet running statistics from tiny batches.
    freeze_batchnorm: bool = True
    monitor: str = "val_auc"
    monitor_mode: str = "max"
    early_stop_patience: int = 7
    reduce_lr_patience: int = 3


@dataclass
class RunConfig:
    name: str = "v2"
    out_dir: str = "./runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else self.run_dir / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=list))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        raw = json.loads(Path(path).read_text())
        return cls(
            name=raw["name"],
            out_dir=raw["out_dir"],
            data=DataConfig(**{**raw["data"], "img_size": tuple(raw["data"]["img_size"])}),
            model=ModelConfig(**{**raw["model"], "backbones": tuple(raw["model"]["backbones"])}),
            train=TrainConfig(**raw["train"]),
        )
