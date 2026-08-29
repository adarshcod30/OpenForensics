"""Single-writer lock for GPU work.

Two TensorFlow processes sharing the Apple Metal plugin interfere at the
signal level: starting a second one raises `Interrupted system call` on
blocking reads inside the first one's tf.data pipeline, and tf.data does not
retry on EINTR. The first process dies mid-epoch.

This cost a training run once. The fix is not a warning in the README --
it is making the second process refuse to start.

Usage:

    with gpu_lock("train"):
        ...

Set OF_NO_GPU_LOCK=1 to bypass, e.g. on a machine with real multi-GPU
isolation where concurrent jobs are actually safe.
"""
from __future__ import annotations

import errno
import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

LOCK_PATH = Path(tempfile.gettempdir()) / "openforensics-gpu.lock"


class GPUBusy(RuntimeError):
    pass


@contextmanager
def gpu_lock(purpose: str = "job", path: Path | None = None):
    if os.environ.get("OF_NO_GPU_LOCK"):
        yield None
        return

    path = path or LOCK_PATH
    fh = open(path, "a+")
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            fh.seek(0)
            holder = fh.read().strip() or "an unknown process"
            raise GPUBusy(
                f"Another OpenForensics GPU job is running: {holder}.\n"
                f"Starting a second one crashes the first with EINTR in its "
                f"input pipeline. Wait for it to finish, or set "
                f"OF_NO_GPU_LOCK=1 if you know concurrent access is safe here."
            ) from None

        fh.seek(0)
        fh.truncate()
        fh.write(f"pid={os.getpid()} purpose={purpose}")
        fh.flush()
        yield fh
    finally:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()
