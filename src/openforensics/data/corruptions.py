"""Corruption-matched augmentation.

The OpenForensics test split is systematically degraded relative to Train and
Validation: measured over 400 sampled images per split, Test/Real loses 22% of
its mean saturation, carries 4x the near-greyscale images, and both test
classes show ~28% more high-frequency energy. Training on clean images and
evaluating on those costs roughly 8 accuracy points.

These transforms reproduce the degradation families visible in that split so
the model sees them during training. All operate on float32 images in [0, 1]
with shape (H, W, 3) and are graph-safe for use inside a tf.data pipeline.

Note on determinism: the data *split* is reproducible via the run manifest
(see manifest.py). Augmentation draws are seeded globally but not per-example,
so two runs see the same images in the same order with different noise. That
is the usual trade-off and it does not affect split integrity.
"""
from __future__ import annotations

import tensorflow as tf

# Fixed levels rather than continuous draws: several TF image ops take their
# strength as a graph attribute, not an input, so it cannot be a tensor.
_JPEG_QUALITIES = (15, 25, 40, 60)
_PIXELATE_FACTORS = (2, 4, 6, 8)


def _clip(x: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(x, 0.0, 1.0)


def desaturate(image: tf.Tensor) -> tf.Tensor:
    """Pull colour toward grey. Matches the near-greyscale test images."""
    factor = tf.random.uniform([], 0.0, 0.45)
    grey = tf.image.rgb_to_grayscale(image)
    return _clip(image * factor + tf.tile(grey, [1, 1, 3]) * (1.0 - factor))


def colour_cast(image: tf.Tensor) -> tf.Tensor:
    """Per-channel gain and offset. Matches the false-colour test images."""
    gain = tf.random.uniform([3], 0.55, 1.45)
    offset = tf.random.uniform([3], -0.12, 0.12)
    return _clip(image * gain + offset)


def gaussian_noise(image: tf.Tensor) -> tf.Tensor:
    """Additive sensor-style noise. Matches the raised high-frequency energy."""
    sigma = tf.random.uniform([], 0.02, 0.12)
    return _clip(image + tf.random.normal(tf.shape(image), stddev=sigma))


def speckle(image: tf.Tensor) -> tf.Tensor:
    """Sparse bright/dark pixels — the 'snow' overlay seen in the test split."""
    amount = tf.random.uniform([], 0.005, 0.04)
    # tf.shape(...)[:2] is a tensor: `+ [1]` would add 1 elementwise
    # (giving [225, 225]) rather than appending a channel axis.
    draw = tf.random.uniform(tf.concat([tf.shape(image)[:2], [1]], axis=0))
    salt = tf.cast(draw < amount / 2.0, tf.float32)
    pepper = tf.cast(draw > 1.0 - amount / 2.0, tf.float32)
    keep = 1.0 - tf.maximum(salt, pepper)
    return _clip(image * tf.tile(keep, [1, 1, 3]) + tf.tile(salt, [1, 1, 3]))


def _gaussian_kernel(size: int, sigma: float) -> tf.Tensor:
    ax = tf.cast(tf.range(size), tf.float32) - (size - 1) / 2.0
    k = tf.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    k = k / tf.reduce_sum(k)
    kernel_2d = tf.einsum("i,j->ij", k, k)
    return tf.tile(kernel_2d[:, :, None, None], [1, 1, 3, 1])


def blur(image: tf.Tensor) -> tf.Tensor:
    """Defocus / resize blur."""
    sigma = tf.random.uniform([], 0.6, 2.2)
    kernel = _gaussian_kernel(7, sigma)
    out = tf.nn.depthwise_conv2d(
        image[None, ...], kernel, strides=[1, 1, 1, 1], padding="SAME"
    )[0]
    return _clip(out)


def _jpeg_at(image: tf.Tensor, quality: int) -> tf.Tensor:
    byte_img = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
    encoded = tf.io.encode_jpeg(byte_img, quality=quality)
    return tf.image.convert_image_dtype(tf.io.decode_jpeg(encoded, channels=3), tf.float32)


def jpeg_artifact(image: tf.Tensor) -> tf.Tensor:
    """Low-quality recompression. Matches the blocking in the test split."""
    idx = tf.random.uniform([], 0, len(_JPEG_QUALITIES), dtype=tf.int32)
    return tf.switch_case(
        idx, [lambda q=q: _jpeg_at(image, q) for q in _JPEG_QUALITIES]
    )


def _pixelate_by(image: tf.Tensor, factor: int) -> tf.Tensor:
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    small = tf.image.resize(image, (h // factor, w // factor), method="nearest")
    return tf.image.resize(small, (h, w), method="nearest")


def pixelate(image: tf.Tensor) -> tf.Tensor:
    """Downscale-upscale blocking."""
    idx = tf.random.uniform([], 0, len(_PIXELATE_FACTORS), dtype=tf.int32)
    return tf.switch_case(
        idx, [lambda f=f: _pixelate_by(image, f) for f in _PIXELATE_FACTORS]
    )


def brightness_shift(image: tf.Tensor) -> tf.Tensor:
    """Heavy under/over-exposure, well beyond the +/-0.08 of the old pipeline."""
    image = tf.image.adjust_brightness(image, tf.random.uniform([], -0.35, 0.35))
    return _clip(tf.image.adjust_contrast(image, tf.random.uniform([], 0.5, 1.6)))


def occlusion(image: tf.Tensor) -> tf.Tensor:
    """Random erasing — the blobs and overlays present in the test split."""
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    fh = tf.cast(tf.random.uniform([], 0.10, 0.35) * tf.cast(h, tf.float32), tf.int32)
    fw = tf.cast(tf.random.uniform([], 0.10, 0.35) * tf.cast(w, tf.float32), tf.int32)
    top = tf.random.uniform([], 0, tf.maximum(h - fh, 1), dtype=tf.int32)
    left = tf.random.uniform([], 0, tf.maximum(w - fw, 1), dtype=tf.int32)

    ys = tf.range(h)[:, None]
    xs = tf.range(w)[None, :]
    inside = (ys >= top) & (ys < top + fh) & (xs >= left) & (xs < left + fw)
    mask = tf.cast(inside, tf.float32)[:, :, None]
    fill = tf.random.uniform([], 0.0, 1.0)
    return _clip(image * (1.0 - mask) + fill * mask)


# Order is fixed so a corruption index means the same thing across runs and
# can be reported in the per-corruption evaluation breakdown.
CORRUPTIONS = (
    ("desaturate", desaturate),
    ("colour_cast", colour_cast),
    ("gaussian_noise", gaussian_noise),
    ("speckle", speckle),
    ("blur", blur),
    ("jpeg_artifact", jpeg_artifact),
    ("pixelate", pixelate),
    ("brightness_shift", brightness_shift),
    ("occlusion", occlusion),
)
CORRUPTION_NAMES = tuple(name for name, _ in CORRUPTIONS)


def apply_one(image: tf.Tensor) -> tf.Tensor:
    """Apply exactly one corruption, chosen uniformly."""
    idx = tf.random.uniform([], 0, len(CORRUPTIONS), dtype=tf.int32)
    return tf.switch_case(idx, [lambda f=fn: f(image) for _, fn in CORRUPTIONS])


def random_corrupt(image: tf.Tensor, prob: float = 0.5, max_ops: int = 2) -> tf.Tensor:
    """Apply between 0 and `max_ops` corruptions, each gated on `prob`.

    Stacking matters: real degraded images are usually recompressed *after*
    being noised or resized, so single-corruption training under-represents
    what the test split actually contains.
    """
    for _ in range(max_ops):
        image = tf.cond(
            tf.random.uniform([]) < prob,
            lambda im=image: apply_one(im),
            lambda im=image: im,
        )
    return image
