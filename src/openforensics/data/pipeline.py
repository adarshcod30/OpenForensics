"""tf.data input pipeline.

Replaces the original `dataset_utils.py`. Differences that matter:

  * Decoding uses `decode_image`, not `decode_jpeg`. The old loader accepted
    `.png` in its file filter but decoded as JPEG, so any PNG in the tree was
    a hard crash waiting to happen.
  * Augmentation includes the corruption families present in the test split
    (see corruptions.py), not just flip / +-0.08 brightness.
  * File lists come from a saved manifest, so a run's exact inputs are
    recoverable after the fact.
  * `.cache()` on the validation and test sets: they are re-read every epoch
    and never augmented, so decoding them once is free speed.
"""
from __future__ import annotations

import tensorflow as tf

from ..config import IMG_SIZE
from .corruptions import random_corrupt

AUTOTUNE = tf.data.AUTOTUNE


def decode_and_resize(path: tf.Tensor, label: tf.Tensor, img_size=IMG_SIZE):
    raw = tf.io.read_file(path)
    # expand_animations=False keeps the static shape known, which resize needs.
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape((*img_size, 3))
    return img, label


def geometric_augment(image: tf.Tensor, label: tf.Tensor):
    """Label-preserving geometry. Only horizontal flip is safe for faces:
    vertical flips and large rotations produce images unlike anything in the
    distribution, which costs accuracy rather than adding robustness."""
    return tf.image.random_flip_left_right(image), label


def corruption_augment(image, label, prob: float, max_ops: int):
    return random_corrupt(image, prob=prob, max_ops=max_ops), label


def build_dataset(
    filepaths,
    labels,
    batch: int = 32,
    shuffle: bool = False,
    augment: bool = False,
    corruption_prob: float = 0.5,
    max_corruptions: int = 2,
    img_size=IMG_SIZE,
    cache: bool = False,
    seed: int | None = None,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(filepaths), tf.constant(labels, dtype=tf.float32))
    )
    ds = ds.map(lambda p, l: decode_and_resize(p, l, img_size),
                num_parallel_calls=AUTOTUNE)

    # Cache the decoded tensors *before* augmentation so every epoch still
    # gets fresh corruption draws. Caching after would freeze the noise.
    if cache:
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(filepaths), 8192),
                        seed=seed, reshuffle_each_iteration=True)

    if augment:
        ds = ds.map(geometric_augment, num_parallel_calls=AUTOTUNE)
        if corruption_prob > 0:
            ds = ds.map(
                lambda x, y: corruption_augment(x, y, corruption_prob, max_corruptions),
                num_parallel_calls=AUTOTUNE,
            )

    return ds.batch(batch).prefetch(AUTOTUNE)


def from_manifest(manifest, split: str, **kwargs) -> tf.data.Dataset:
    payload = manifest.splits[split]
    return build_dataset(payload["files"], payload["labels"], **kwargs)
