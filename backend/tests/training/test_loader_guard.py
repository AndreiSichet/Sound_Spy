"""DataLoader watchdog: a stalled or dead worker becomes a clear error, never a silent hang."""

from __future__ import annotations

import pytest

from soundspy.training.datasets import (
    THREAD_ENV_VARS,
    LoaderStalled,
    guarded,
    limit_worker_threads,
)


class FakeLoader:
    """Iterable with the DataLoader attributes ``guarded`` reads."""

    def __init__(self, items, error: Exception | None = None, num_workers: int = 4,
                 timeout: float = 120) -> None:
        self.items, self.error = items, error
        self.num_workers, self.timeout = num_workers, timeout

    def __iter__(self):
        yield from self.items
        if self.error is not None:
            raise self.error


def test_batches_pass_through_and_first_batch_is_reported() -> None:
    log: list[str] = []
    assert list(guarded(FakeLoader([1, 2, 3]), log=log.append)) == [1, 2, 3]
    assert log[0].startswith("waiting for the first batch from 4 worker(s) (timeout 120 s)")
    assert log[1].startswith("first batch after")


@pytest.mark.parametrize("message", [
    "DataLoader timed out after 120 seconds",
    "DataLoader worker (pid(s) 1234) exited unexpectedly",
])
def test_timeout_and_dead_worker_become_loader_stalled(message: str) -> None:
    with pytest.raises(LoaderStalled, match="--workers 0") as exc:
        list(guarded(FakeLoader([1], RuntimeError(message), num_workers=4)))
    assert "4 worker(s)" in str(exc.value) and "120 s" in str(exc.value)


def test_other_errors_are_not_hidden() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        list(guarded(FakeLoader([], RuntimeError("boom"))))


def test_worker_thread_limit_respects_user_settings(monkeypatch) -> None:
    for var in THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    limit_worker_threads()
    import os

    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "3"
