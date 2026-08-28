"""Worker exit-status classification must survive ``subprocess._cleanup()``.

Regression coverage for the dominant kanban failure mode on the ops board
(task t_199766f0): 159 runs in 7 days recorded ``pid NNNN not alive`` even
though the worker log showed the worker had called ``kanban_complete`` and
exited rc=0.

Root cause: ``_default_spawn`` abandoned the ``Popen`` handle. CPython's
``Popen.__del__`` then parks the un-waited instance in the module-global
``subprocess._active`` list, and the NEXT ``Popen(...)`` anywhere in the
dispatcher process calls ``subprocess._cleanup()``, which reaps it via
``_internal_poll(_deadstate=sys.maxsize)`` — bypassing
``kanban_db._record_worker_exit`` entirely. The status is lost, so
``_classify_worker_exit`` degrades to ``("unknown", None)`` and
``detect_crashed_workers`` books a SUCCESSFUL worker as a crash.

The race is not exotic: ``_pid_alive`` shells out to ``ps`` via ``Popen`` on
macOS, so the liveness probe that leads to the classifier is itself a
``_cleanup()`` trigger.

Fix under test: spawned handles are retained in ``kb._worker_handles``,
``reap_worker_zombies`` polls them (recording the real status), and
``_classify_worker_exit`` falls back to polling a still-owned handle before
answering ``unknown``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture(autouse=True)
def _clean_registries():
    """Each test starts with empty exit/handle registries."""
    kb._recent_worker_exits.clear()
    kb._worker_handles.clear()
    yield
    kb._recent_worker_exits.clear()
    kb._worker_handles.clear()


def _spawn_child(code: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _wait_gone(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    """Wait for the child to actually exit WITHOUT reaping via this handle.

    Uses ``os.waitid`` with ``WNOWAIT`` where available so the status stays
    pending for the code under test; falls back to a bounded sleep loop.
    """
    waitid = getattr(os, "waitid", None)
    if waitid is not None and hasattr(os, "P_PID") and hasattr(os, "WEXITED"):
        waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)
        return
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not kb._pid_alive(proc.pid):
            return
        time.sleep(0.05)


@pytest.mark.skipif(os.name == "nt", reason="POSIX reap semantics")
def test_register_worker_handle_records_pid():
    proc = _spawn_child("import sys; sys.exit(0)")
    try:
        kb._register_worker_handle(proc)
        assert proc.pid in kb._worker_handles
    finally:
        proc.wait()


@pytest.mark.skipif(os.name == "nt", reason="POSIX reap semantics")
def test_reap_captures_clean_exit_status():
    """An owned handle's rc=0 must classify as ``clean_exit``, not unknown."""
    proc = _spawn_child("import sys; sys.exit(0)")
    kb._register_worker_handle(proc)
    pid = proc.pid
    _wait_gone(proc)

    reaped = kb.reap_worker_zombies()

    assert pid in reaped
    assert kb._classify_worker_exit(pid) == ("clean_exit", 0)
    # Handle dropped after reap — no unbounded growth.
    assert pid not in kb._worker_handles


@pytest.mark.skipif(os.name == "nt", reason="POSIX reap semantics")
def test_reap_captures_nonzero_exit_status():
    proc = _spawn_child("import sys; sys.exit(3)")
    kb._register_worker_handle(proc)
    pid = proc.pid
    _wait_gone(proc)

    kb.reap_worker_zombies()

    assert kb._classify_worker_exit(pid) == ("nonzero_exit", 3)


@pytest.mark.skipif(os.name == "nt", reason="POSIX reap semantics")
def test_reap_captures_rate_limit_sentinel():
    """The quota-wall sentinel must survive the reap path unchanged."""
    proc = _spawn_child(f"import sys; sys.exit({kb.KANBAN_RATE_LIMIT_EXIT_CODE})")
    kb._register_worker_handle(proc)
    pid = proc.pid
    _wait_gone(proc)

    kb.reap_worker_zombies()

    kind, code = kb._classify_worker_exit(pid)
    assert kind == "rate_limited"
    assert code == kb.KANBAN_RATE_LIMIT_EXIT_CODE


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
def test_reap_captures_signalled_exit():
    import signal

    proc = _spawn_child("import time; time.sleep(30)")
    kb._register_worker_handle(proc)
    pid = proc.pid
    os.kill(pid, signal.SIGKILL)
    _wait_gone(proc)

    kb.reap_worker_zombies()

    assert kb._classify_worker_exit(pid) == ("signaled", int(signal.SIGKILL))


@pytest.mark.skipif(os.name == "nt", reason="POSIX reap semantics")
def test_subprocess_cleanup_cannot_steal_owned_status():
    """THE regression: ``subprocess._cleanup()`` must not lose the status.

    Simulates the real sequence — worker exits, then any other ``Popen`` in
    the process (e.g. the ``ps`` probe inside ``_pid_alive``) fires
    ``_cleanup()``. Before the fix the handle had already been GC'd into
    ``subprocess._active`` and ``_cleanup()`` reaped it silently, leaving
    ``_classify_worker_exit`` at ``unknown`` and turning a finished worker
    into ``pid NNNN not alive``.
    """
    proc = _spawn_child("import sys; sys.exit(0)")
    kb._register_worker_handle(proc)
    pid = proc.pid
    _wait_gone(proc)

    # Drop our local reference: only kb._worker_handles keeps it alive now.
    del proc
    import gc

    gc.collect()

    # Any new Popen triggers subprocess._cleanup(). _pid_alive does exactly
    # this on macOS (it runs `ps`), so use the real probe.
    kb._pid_alive(pid)
    subprocess.run(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    kb.reap_worker_zombies()

    kind, code = kb._classify_worker_exit(pid)
    assert (kind, code) == ("clean_exit", 0), (
        f"exit status lost to subprocess._cleanup(): got {kind!r}/{code!r}. "
        "A successful worker would be recorded as 'pid not alive'."
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX reap semantics")
def test_classify_polls_owned_handle_when_registry_is_cold():
    """Exit between the reap sweep and the liveness check must not read unknown.

    ``detect_crashed_workers`` calls ``_pid_alive`` then
    ``_classify_worker_exit``; a worker can exit inside that window, after
    the tick's reap sweep already ran. The classifier polls the owned handle
    directly so this never degrades to ``unknown``.
    """
    proc = _spawn_child("import sys; sys.exit(0)")
    kb._register_worker_handle(proc)
    pid = proc.pid
    _wait_gone(proc)

    # NOTE: no reap_worker_zombies() call — registry is deliberately cold.
    assert pid not in kb._recent_worker_exits

    assert kb._classify_worker_exit(pid) == ("clean_exit", 0)


@pytest.mark.skipif(os.name == "nt", reason="POSIX reap semantics")
def test_classify_unknown_for_unowned_pid():
    """Pids we never spawned still answer ``unknown`` (no false positives)."""
    assert kb._classify_worker_exit(999999) == ("unknown", None)


@pytest.mark.skipif(os.name == "nt", reason="POSIX reap semantics")
def test_reap_still_handles_unowned_children():
    """The bare ``waitpid`` fallback stays live for children we don't own."""
    proc = _spawn_child("import sys; sys.exit(0)")
    pid = proc.pid
    # Deliberately NOT registered. Reap must still collect + record it.
    _wait_gone(proc)

    reaped = kb.reap_worker_zombies()

    assert pid in reaped
    assert kb._classify_worker_exit(pid) == ("clean_exit", 0)


@pytest.mark.skipif(os.name == "nt", reason="POSIX reap semantics")
def test_handle_registry_is_bounded(monkeypatch):
    """A lost handle can never grow the registry without limit."""
    monkeypatch.setattr(kb, "_WORKER_HANDLES_MAX", 4)

    class _FakeProc:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None  # never exits → eviction must fall back to FIFO

    for i in range(1, 40):
        kb._register_worker_handle(_FakeProc(i))

    assert len(kb._worker_handles) <= kb._WORKER_HANDLES_MAX
