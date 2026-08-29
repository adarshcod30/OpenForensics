"""Multi-backbone ensemble.

Each backbone sees the same image through its own preprocessing, produces a
pooled embedding, and the embeddings are concatenated before the classifier
head. Adding EfficientNetV2-B0 to the original ResNet50 + VGG16 pair costs
~6M parameters and brings a far stronger feature extractor than either.
"""
from __future__ import annotations

import warnings

import keras
from keras import layers

from .layers import PreprocessLayer

# name -> (constructor, preprocessing mode)
BACKBONES = {
    "resnet50": (keras.applications.ResNet50, "resnet50"),
    "vgg16": (keras.applications.VGG16, "vgg16"),
    "efficientnetv2b0": (keras.applications.EfficientNetV2B0, "efficientnetv2"),
    "efficientnetv2s": (keras.applications.EfficientNetV2S, "efficientnetv2"),
}


def _load_backbone(ctor, name: str, **kwargs):
    """Build a backbone and give it a stable name.

    The name is the handle used to unfreeze one backbone and to target
    Grad-CAM at a single branch, so it must be predictable. Two Keras quirks
    make this fiddly:
      * `EfficientNetV2B0` treats its `name` argument as a lookup key into an
        internal block-args table, so passing a custom name raises KeyError.
      * `Model._name` is not the backing field in Keras 3; assigning it is a
        silent no-op. The public `name` property is settable and is what
        `get_layer` reads.
    So: construct with defaults, then rename.
    """
    try:
        model = ctor(**kwargs)
    except Exception as exc:  # noqa: BLE001 - network/cache failures vary
        warnings.warn(
            f"{ctor.__name__}: could not load ImageNet weights ({exc}). "
            "Falling back to random initialisation — results will be much worse."
        )
        model = ctor(**{**kwargs, "weights": None})
    model.name = name
    return model


def _branch_head(x, units: int, dropout: float, name: str):
    x = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    x = layers.Dense(units, activation="relu", name=f"{name}_dense")(x)
    return layers.BatchNormalization(name=f"{name}_bn")(x)


def build_ensemble(
    input_shape=(224, 224, 3),
    backbones=("resnet50", "vgg16", "efficientnetv2b0"),
    head_units: int = 256,
    dropout_branch: float = 0.4,
    dropout_merge: float = 0.4,
    dropout_final: float = 0.3,
    name: str = "openforensics_ensemble",
) -> keras.Model:
    unknown = set(backbones) - set(BACKBONES)
    if unknown:
        raise ValueError(f"Unknown backbone(s): {sorted(unknown)}")

    inp = layers.Input(shape=input_shape, name="input_image")

    embeddings = []
    for key in backbones:
        ctor, mode = BACKBONES[key]
        pre = PreprocessLayer(mode=mode, name=f"preprocess_{key}")(inp)
        # Build the backbone standalone and *call* it on the preprocessed
        # tensor. Passing `input_tensor=` instead inlines every backbone layer
        # into the parent graph, so the backbone stops being addressable as a
        # unit -- get_layer("resnet50") raises, per-backbone unfreezing
        # silently degrades to "last N layers of the whole model", and
        # Grad-CAM cannot find a branch to attribute to.
        base = _load_backbone(
            ctor, key, weights="imagenet", include_top=False, input_shape=input_shape
        )
        base.trainable = False                # stage 1 trains heads only
        embeddings.append(_branch_head(base(pre), head_units, dropout_branch, key))

    merged = layers.Concatenate(name="fusion")(embeddings) if len(embeddings) > 1 else embeddings[0]
    x = layers.Dropout(dropout_merge, name="fusion_drop")(merged)
    x = layers.Dense(head_units, activation="relu", name="fusion_dense")(x)
    x = layers.BatchNormalization(name="fusion_bn")(x)
    x = layers.Dropout(dropout_final, name="head_drop")(x)
    out = layers.Dense(1, activation="sigmoid", name="pred")(x)

    return keras.Model(inputs=inp, outputs=out, name=name)


def set_backbone_trainable(
    model: keras.Model,
    backbones=("resnet50", "vgg16", "efficientnetv2b0"),
    unfreeze_last: int = 50,
    freeze_batchnorm: bool = True,
) -> dict[str, int]:
    """Unfreeze the top `unfreeze_last` layers of each backbone for fine-tuning.

    BatchNorm stays frozen by default. A trainable BN layer switches to batch
    statistics and overwrites its ImageNet running estimates from whatever
    batch size you happen to be using; at batch=16 that discards calibrated
    statistics accumulated over a million images.
    """
    changed: dict[str, int] = {}
    for key in backbones:
        try:
            sub = model.get_layer(key)
        except ValueError:
            continue
        sub.trainable = True
        inner = [l for l in sub.layers if hasattr(l, "trainable")]
        n = 0
        for i, layer in enumerate(inner):
            top = i >= len(inner) - unfreeze_last
            if isinstance(layer, layers.BatchNormalization) and freeze_batchnorm:
                layer.trainable = False
                continue
            layer.trainable = top
            n += int(top)
        changed[key] = n
    return changed


def summarise(model: keras.Model) -> str:
    total = model.count_params()
    trainable = sum(int(w.numpy().size) for w in model.trainable_weights)
    return (f"{model.name}: {total:,} params, {trainable:,} trainable "
            f"({100*trainable/max(total,1):.1f}%)")
