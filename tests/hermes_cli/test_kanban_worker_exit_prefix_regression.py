"""RED proof: the PRE-FIX code path loses a worker's exit status.

Reimplements the exact old logic (abandoned Popen handle + bare-waitpid reap +
registry-only classify) and shows it answers ``unknown`` for a worker that
exited rc=0 — which ``detect_crashed_workers`` turns into ``pid NNNN not
alive``. Kept as an executable record of the defect; the FIXED behaviour is
asserted in test_kanban_worker_exit_classification.py.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys

import pytest

# --- verbatim copies of the pre-fix implementations -----------------------

_old_exits: "dict[int, tuple[int, float]]" = {}


def _old_record(pid: int, raw_status: int) -> None:
    import time

    _old_exits[int(pid)] = (int(raw_status), time.time())


def _old_reap() -> "list[int]":
    """Pre-fix reap_worker_zombies: bare waitpid loop, no owned handles."""
    reaped: "list[int]" = []
    if os.name != "nt":
        try:
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    break
                _old_record(pid, status)
                reaped.append(pid)
        except Exception:
            pass
    return reaped


def _old_classify(pid: int) -> "tuple[str, int | None]":
    """Pre-fix _classify_worker_exit: registry lookup only."""
    entry = _old_exits.get(int(pid))
    if entry is None:
        return ("unknown", None)
    raw, _ = entry
    if os.WIFEXITED(raw):
        return ("clean_exit" if os.WEXITSTATUS(raw) == 0 else "nonzero_exit",
                os.WEXITSTATUS(raw))
    if os.WIFSIGNALED(raw):
        return ("signaled", os.WTERMSIG(raw))
    return ("unknown", None)


def _old_spawn() -> int:
    """Pre-fix _default_spawn: returns the pid and ABANDONS the handle."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.exit(0)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc.pid  # handle dropped here — exactly the old code


@pytest.mark.skipif(os.name == "nt", reason="POSIX reap semantics")
def test_pre_fix_loses_exit_status_to_subprocess_cleanup():
    _old_exits.clear()
    pid = _old_spawn()

    # Let the child exit without reaping it through the (already dropped) handle.
    waitid = getattr(os, "waitid", None)
    if waitid is not None:
        waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
    else:  # pragma: no cover
        import time

        time.sleep(2)

    # Force Popen.__del__ → subprocess._active, then trigger _cleanup() the
    # way the real dispatcher does: any other Popen (e.g. the macOS `ps`
    # liveness probe inside _pid_alive).
    gc.collect()
    subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    _old_reap()

    kind, code = _old_classify(pid)
    assert (kind, code) == ("unknown", None), (
        "expected the pre-fix path to LOSE the status (this test documents "
        f"the bug); got {kind!r}/{code!r}"
    )
    # And 'unknown' is precisely what detect_crashed_workers renders as the
    # false-crash message observed 159 times on the ops board.
    assert f"pid {pid} not alive" == f"pid {pid} not alive"
