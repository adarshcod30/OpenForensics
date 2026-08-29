"""Split manifests.

The old pipeline sampled files at import time and recorded nothing, so the
exact test set behind a reported metric could not be reconstructed. A manifest
pins the file list, hashes it, and stores it next to the weights.

It also runs a content-level leakage check across splits. The three split
directories reuse filenames (`fake_0.jpg` exists in Train, Validation and
Test), so a name-based check proves nothing; only content hashes do.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("Train", "Validation", "Test")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _list_class_files(base_dir: str | Path, split: str, cls: str) -> list[str]:
    d = Path(base_dir) / split / cls
    if not d.is_dir():
        raise FileNotFoundError(f"Missing split directory: {d}")
    return sorted(
        str(d / f) for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTS)
    )


def sample_split(
    base_dir: str | Path,
    split: str,
    classes: tuple[str, ...],
    per_class: int,
    seed: int,
) -> tuple[list[str], list[int]]:
    """Deterministic stratified sample. Label 1 = Real, 0 = Fake.

    Sorting before shuffling matters: os.listdir order is filesystem-dependent,
    so without the sort the "seeded" sample differs between machines.
    """
    files: list[str] = []
    labels: list[int] = []
    for cls in classes:
        pool = _list_class_files(base_dir, split, cls)
        if len(pool) < per_class:
            raise ValueError(
                f"{split}/{cls}: need {per_class}, found {len(pool)}"
            )
        rng = random.Random(f"{seed}:{split}:{cls}")
        rng.shuffle(pool)
        chosen = pool[:per_class]
        files.extend(chosen)
        labels.extend([1 if cls == "Real" else 0] * len(chosen))

    order = list(range(len(files)))
    random.Random(f"{seed}:{split}:order").shuffle(order)
    return [files[i] for i in order], [labels[i] for i in order]


def _file_digest(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass
class Manifest:
    seed: int
    base_dir: str
    splits: dict[str, dict]

    def digest(self) -> str:
        payload = json.dumps(
            {s: v["files"] for s, v in self.splits.items()}, sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "seed": self.seed,
                    "base_dir": self.base_dir,
                    "digest": self.digest(),
                    "counts": {s: len(v["files"]) for s, v in self.splits.items()},
                    "splits": self.splits,
                },
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        raw = json.loads(Path(path).read_text())
        return cls(seed=raw["seed"], base_dir=raw["base_dir"], splits=raw["splits"])


# Splits are filled in this order so that when the same image appears in two
# of them, the earlier one keeps it. Training data is the most valuable and
# evaluation must stay unseen, so Train wins and Validation beats Test.
_FILL_ORDER = ("Train", "Validation", "Test")


def build(
    base_dir: str | Path,
    classes: tuple[str, ...],
    per_class: dict[str, int],
    seed: int,
    deduplicate: bool = True,
) -> Manifest:
    """Sample each split, optionally rejecting images already claimed.

    With `deduplicate`, files are content-hashed as they are drawn and any
    hash already seen is skipped -- which removes duplicates inside a split
    and collisions across splits in one pass. The result is exactly
    `per_class` images per class with no image appearing twice anywhere.

    Without it, the raw sample of this dataset carries 125 images shared
    between Train and Validation and 213 duplicates within Train, which makes
    validation quietly optimistic.
    """
    if not deduplicate:
        splits = {}
        for split, n in per_class.items():
            files, labels = sample_split(base_dir, split, classes, n, seed)
            splits[split] = {"files": files, "labels": labels}
        return Manifest(seed=seed, base_dir=str(base_dir), splits=splits)

    seen: set[str] = set()
    splits: dict[str, dict] = {}
    order = [s for s in _FILL_ORDER if s in per_class]
    order += [s for s in per_class if s not in order]

    for split in order:
        n = per_class[split]
        files: list[str] = []
        labels: list[int] = []
        for cls in classes:
            pool = _list_class_files(base_dir, split, cls)
            random.Random(f"{seed}:{split}:{cls}").shuffle(pool)
            kept, skipped = [], 0
            for path in pool:
                if len(kept) == n:
                    break
                digest = _file_digest(path)
                if digest in seen:
                    skipped += 1
                    continue
                seen.add(digest)
                kept.append(path)
            if len(kept) < n:
                raise ValueError(
                    f"{split}/{cls}: only {len(kept)} unique images available "
                    f"after dedup (needed {n}, skipped {skipped})"
                )
            files.extend(kept)
            labels.extend([1 if cls == "Real" else 0] * len(kept))

        idx = list(range(len(files)))
        random.Random(f"{seed}:{split}:order").shuffle(idx)
        splits[split] = {
            "files": [files[i] for i in idx],
            "labels": [labels[i] for i in idx],
        }

    return Manifest(seed=seed, base_dir=str(base_dir), splits=splits)


def check_leakage(manifest: Manifest, verbose: bool = True) -> dict:
    """Content-hash every sampled file and report cross-split collisions.

    Train/Validation leakage inflates validation and is a candidate
    explanation for a large validation-to-test gap, so this runs before
    any training and its result is stored with the run.
    """
    digests: dict[str, dict[str, str]] = {}
    for split, payload in manifest.splits.items():
        digests[split] = {p: _file_digest(p) for p in payload["files"]}

    report: dict[str, object] = {"pairs": {}, "leaking": False}
    names = list(manifest.splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa = set(digests[a].values())
            sb = set(digests[b].values())
            overlap = sa & sb
            report["pairs"][f"{a}|{b}"] = {
                "shared_images": len(overlap),
                "pct_of_smaller": round(
                    100.0 * len(overlap) / max(min(len(sa), len(sb)), 1), 3
                ),
            }
            if overlap:
                report["leaking"] = True

    dupes = {}
    for split, mapping in digests.items():
        seen: dict[str, int] = {}
        for d in mapping.values():
            seen[d] = seen.get(d, 0) + 1
        dupes[split] = sum(c - 1 for c in seen.values() if c > 1)
    report["within_split_duplicates"] = dupes

    if verbose:
        print(f"leakage check  (digest {manifest.digest()})")
        for pair, v in report["pairs"].items():
            flag = "  <-- LEAK" if v["shared_images"] else ""
            print(f"  {pair:<26} shared={v['shared_images']:<6} "
                  f"({v['pct_of_smaller']}% of smaller){flag}")
        print(f"  within-split duplicates: {dupes}")
    return report
