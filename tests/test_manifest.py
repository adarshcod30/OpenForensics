"""Splits must be reproducible, balanced, and free of cross-split leakage."""
import numpy as np
import pytest
from PIL import Image

from openforensics.data import manifest as M


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """Small dataset with a deliberate duplicate in every split."""
    root = tmp_path_factory.mktemp("ds")
    rng = np.random.default_rng(0)
    dup = (rng.random((16, 16, 3)) * 255).astype("uint8")
    for split in ("Train", "Validation", "Test"):
        for cls in ("Fake", "Real"):
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(40):
                arr = dup if i < 3 else (rng.random((16, 16, 3)) * 255).astype("uint8")
                Image.fromarray(arr).save(d / f"{cls.lower()}_{i}.jpg")
    return root


def test_sampling_is_deterministic(tree):
    a = M.sample_split(tree, "Train", ("Fake", "Real"), 10, seed=1)[0]
    b = M.sample_split(tree, "Train", ("Fake", "Real"), 10, seed=1)[0]
    assert a == b


def test_sampling_is_balanced_and_shuffled(tree):
    files, labels = M.sample_split(tree, "Train", ("Fake", "Real"), 10, seed=1)
    assert labels.count(0) == labels.count(1) == 10
    assert labels != sorted(labels), "labels are class-blocked, not shuffled"


def test_dedup_removes_all_leakage(tree):
    mf = M.build(tree, ("Fake", "Real"),
                 {"Train": 10, "Validation": 5, "Test": 5}, seed=1, deduplicate=True)
    rep = M.check_leakage(mf, verbose=False)
    assert rep["leaking"] is False
    assert all(v == 0 for v in rep["within_split_duplicates"].values())
    assert len(mf.splits["Train"]["files"]) == 20


def test_without_dedup_leakage_is_detected(tree):
    mf = M.build(tree, ("Fake", "Real"),
                 {"Train": 20, "Validation": 20}, seed=1, deduplicate=False)
    rep = M.check_leakage(mf, verbose=False)
    assert rep["leaking"] is True, "planted duplicates were not detected"


def test_digest_changes_with_seed(tree):
    kw = dict(classes=("Fake", "Real"), per_class={"Train": 10})
    d1 = M.build(tree, seed=1, **kw).digest()
    d2 = M.build(tree, seed=2, **kw).digest()
    assert d1 != d2
