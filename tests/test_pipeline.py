"""The loader must handle every extension its own filter accepts.

The original decoded with `tf.io.decode_jpeg` while accepting `.png`.
"""
import numpy as np
from PIL import Image

from openforensics.data import pipeline


def test_decodes_png_and_jpeg(tmp_path):
    arr = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype("uint8")
    paths = []
    for ext in ("png", "jpg"):
        p = tmp_path / f"img.{ext}"
        Image.fromarray(arr).save(p)
        paths.append(str(p))

    ds = pipeline.build_dataset(paths, [0, 1], batch=2)
    x, y = next(iter(ds))
    assert tuple(x.shape) == (2, 224, 224, 3)
    assert float(x.numpy().min()) >= 0.0 and float(x.numpy().max()) <= 1.0


def test_augmentation_changes_pixels(tmp_path):
    arr = (np.random.default_rng(1).random((64, 64, 3)) * 255).astype("uint8")
    p = tmp_path / "a.jpg"; Image.fromarray(arr).save(p)
    files, labels = [str(p)] * 8, [0] * 8
    plain = next(iter(pipeline.build_dataset(files, labels, batch=8)))[0].numpy()
    aug = next(iter(pipeline.build_dataset(
        files, labels, batch=8, augment=True, corruption_prob=1.0)))[0].numpy()
    assert not np.allclose(plain, aug)
