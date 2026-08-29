"""The second GPU job must refuse to start, not crash the first."""
import multiprocessing as mp
import pytest

from openforensics.gpulock import GPUBusy, gpu_lock


def _hold(path, started, release):
    with gpu_lock("holder", path=path):
        started.set()
        release.wait(timeout=10)


def test_second_acquirer_is_refused(tmp_path):
    lock = tmp_path / "gpu.lock"
    ctx = mp.get_context("spawn")
    started, release = ctx.Event(), ctx.Event()
    proc = ctx.Process(target=_hold, args=(lock, started, release))
    proc.start()
    try:
        assert started.wait(timeout=10), "holder never acquired the lock"
        with pytest.raises(GPUBusy) as exc:
            with gpu_lock("second", path=lock):
                pass
        assert "already" in str(exc.value) or "running" in str(exc.value)
    finally:
        release.set(); proc.join(timeout=10)


def test_lock_is_released_after_use(tmp_path):
    lock = tmp_path / "gpu.lock"
    with gpu_lock("first", path=lock):
        pass
    with gpu_lock("second", path=lock):
        pass  # must not raise


def test_env_var_bypasses(tmp_path, monkeypatch):
    lock = tmp_path / "gpu.lock"
    with gpu_lock("first", path=lock):
        monkeypatch.setenv("OF_NO_GPU_LOCK", "1")
        with gpu_lock("second", path=lock):
            pass
