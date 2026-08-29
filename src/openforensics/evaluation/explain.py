"""Grad-CAM for the nested ensemble.

The original implementation walked `reversed(model.layers)` for the first
4-D output and built `Model([model.inputs], [conv_output, model.output])`.
Two problems: it attributed the decision to whichever single backbone
happened to appear last, and with the backbones inlined into the parent graph
it could not reliably find one at all.

Because backbones are now nested and named, the tensor feeding each branch's
pooling layer is addressable in the outer graph -- so a CAM can be built per
branch and the three compared. There is no single "the ensemble's CAM": the
branches are fused by concatenation into a dense layer, so no scalar weight
per branch exists. The mean is offered as a summary, clearly labelled.
"""
from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf


def branch_names(model) -> list[str]:
    return [l.name for l in model.layers
            if isinstance(l, keras.Model) and f"{l.name}_gap" in
            {x.name for x in model.layers}]


def _cam_model(model, branch: str):
    """Outer input -> (that branch's final feature map, prediction).

    `get_layer(f"{branch}_gap").input` is the tensor the backbone produced
    inside the parent graph, so gradients flow through the real forward pass
    rather than a reconstructed one.
    """
    feature = model.get_layer(f"{branch}_gap").input
    return keras.Model(model.inputs, [feature, model.outputs[0]])


def grad_cam(model, x: np.ndarray, branch: str) -> np.ndarray:
    """Return a [0,1] heatmap at the branch's native resolution."""
    cam_model = _cam_model(model, branch)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv, pred = cam_model(x, training=False)
        tape.watch(conv)
        # Gradient of the *fake* score: a heatmap for "why did you call this
        # forged" is the forensically meaningful one. pred is P(Real).
        target = 1.0 - pred[:, 0]
    grads = tape.gradient(target, conv)
    if grads is None:
        raise RuntimeError(f"no gradient path to branch {branch!r}")
    weights = tf.reduce_mean(grads, axis=(1, 2))          # (B, C)
    cam = tf.einsum("bhwc,bc->bhw", conv, weights)
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
    return cam.numpy()


def all_branches(model, x: np.ndarray, size=(224, 224)) -> dict[str, np.ndarray]:
    """Per-branch heatmaps plus their mean, all resized to `size`."""
    out: dict[str, np.ndarray] = {}
    for b in branch_names(model):
        cam = grad_cam(model, x, b)[0]
        out[b] = tf.image.resize(cam[..., None], size).numpy()[..., 0]
    if out:
        out["mean"] = np.mean(list(out.values()), axis=0)
    return out


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap over an image, both in [0,1]. Returns uint8 RGB."""
    # matplotlib.cm.get_cmap was removed in 3.9; colormaps is the current API.
    import matplotlib
    hm = matplotlib.colormaps["inferno"](np.clip(heatmap, 0, 1))[..., :3]
    blend = (1 - alpha) * np.clip(image, 0, 1) + alpha * hm
    return (np.clip(blend, 0, 1) * 255).astype(np.uint8)
