"""Restore kanban-task attribution in session_model_usage (t_fd685705).

Bug: between 2026-08-11 and 2026-08-30 (19 days), every kanban worker's
main-loop tokens landed in session_model_usage with task='' instead of
``kanban:t_xxx``. The dispatcher sets ``HERMES_KANBAN_TASK`` in the
spawned env, but neither ``update_token_counts`` nor any of the three
``queue_token_counts`` call sites in conversation_loop / codex_runtime
ever resolved it to a ``task=`` parameter — the writer path was
structurally incapable of attribution. Analytics went blind to per-card
spend: 100% of the day's $73 cost was unattributed.

Fix:
* ``update_token_counts`` and ``_record_model_usage`` now accept a
  ``task`` kwarg (default ``""`` so legacy callers stay byte-identical).
* The three per-call ``queue_token_counts`` call sites resolve
  ``HERMES_KANBAN_TASK`` and pass ``f"kanban:<task_id>"`` (or ``""``).
* ``task`` joins ``_TOKEN_DELTA_ROUTE_FIELDS`` so the async writer's
  coalescing key prevents accidental merging of kanban-tagged and
  untagged deltas in the same batch — same-route deltas within one
  turn still merge (constant per call site), so perf is unchanged.

These tests pin the contract:

* kanban worker writes ``task='kanban:t_xxx'``
* non-kanban caller writes ``task=''`` (legacy behaviour preserved)
* coalescing still merges adjacent same-task deltas
* coalescing rejects adjacent different-task deltas (no attribution loss)
* absolute=True gateway writes skip the per-model row (no regression)
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


def _row(db, session_id, *, task=""):
    """Read one session_model_usage row as a dict. None if missing."""
    with db._lock:
        conn = db._conn
        assert conn is not None
        cur = conn.execute(
            "SELECT model, task, input_tokens, output_tokens,"
            " api_call_count, estimated_cost_usd FROM session_model_usage"
            " WHERE session_id = ? AND task = ?",
            (session_id, task),
        )
        r = cur.fetchone()
    return dict(r) if r is not None else None


def _all_rows(db, session_id):
    with db._lock:
        conn = db._conn
        assert conn is not None
        cur = conn.execute(
            "SELECT task, model, input_tokens, output_tokens,"
            " api_call_count FROM session_model_usage"
            " WHERE session_id = ? ORDER BY task",
            (session_id,),
        )
        return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# 1. Kanban worker writes kanban-tagged rows
# ---------------------------------------------------------------------------


def test_update_token_counts_accepts_task_kwarg(db: SessionDB) -> None:
    """Direct synchronous call: the writer accepts the new task kwarg."""
    db.update_token_counts(
        "s1",
        input_tokens=100,
        output_tokens=50,
        api_call_count=1,
        model="gpt-test",
        billing_provider="openai",
        cost_status="estimated",
        cost_source="official_docs_snapshot",
        task="kanban:t_worker1",
    )
    db.flush_token_counts()

    row = _row(db, "s1", task="kanban:t_worker1")
    assert row is not None, "kanban-tagged row must exist"
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 50
    assert row["api_call_count"] == 1


def test_queue_token_counts_forwards_task(db: SessionDB) -> None:
    """Async queue path: the same kwarg must land in the per-model row."""
    db.queue_token_counts(
        "s2",
        input_tokens=200,
        output_tokens=80,
        api_call_count=1,
        model="gpt-test",
        billing_provider="openai",
        cost_status="estimated",
        cost_source="official_docs_snapshot",
        task="kanban:t_worker2",
    )
    assert db.flush_token_counts()

    row = _row(db, "s2", task="kanban:t_worker2")
    assert row is not None, "async queue must deliver the task kwarg"
    assert row["input_tokens"] == 200
    assert row["output_tokens"] == 80


def test_no_task_defaults_to_empty_string(db: SessionDB) -> None:
    """Legacy callers that pass no task kwarg land under task=''."""
    db.update_token_counts(
        "s3",
        input_tokens=10,
        output_tokens=5,
        api_call_count=1,
        model="gpt-test",
        billing_provider="openai",
    )
    db.flush_token_counts()

    empty = _row(db, "s3", task="")
    assert empty is not None
    assert empty["input_tokens"] == 10

    # The kanban-tagged row must NOT have been created.
    kanban = _row(db, "s3", task="kanban:t_none")
    assert kanban is None


# ---------------------------------------------------------------------------
# 2. Coalescing still works (and now correctly refuses cross-task merges)
# ---------------------------------------------------------------------------


def test_coalesce_same_task_still_merges(db: SessionDB) -> None:
    """Adjacent same-task deltas merge — the original optimization is intact."""
    db.queue_token_counts(
        "s4",
        input_tokens=100,
        output_tokens=10,
        api_call_count=1,
        model="gpt-test",
        billing_provider="openai",
        cost_status="estimated",
        cost_source="official_docs_snapshot",
        task="kanban:t_worker4",
    )
    db.queue_token_counts(
        "s4",
        input_tokens=200,
        output_tokens=20,
        api_call_count=1,
        model="gpt-test",
        billing_provider="openai",
        cost_status="estimated",
        cost_source="official_docs_snapshot",
        task="kanban:t_worker4",
    )
    db.flush_token_counts()

    rows = _all_rows(db, "s4")
    # Exactly one merged kanban row.
    kanban_rows = [r for r in rows if r["task"] == "kanban:t_worker4"]
    assert len(kanban_rows) == 1, f"expected one merged kanban row, got {rows!r}"
    assert kanban_rows[0]["input_tokens"] == 300
    assert kanban_rows[0]["output_tokens"] == 30
    assert kanban_rows[0]["api_call_count"] == 2


def test_coalesce_rejects_different_task(db: SessionDB) -> None:
    """Adjacent deltas with different task values do NOT merge — keeps
    attribution intact even when the async writer sees a mixed batch."""
    db.queue_token_counts(
        "s5",
        input_tokens=100,
        output_tokens=10,
        api_call_count=1,
        model="gpt-test",
        billing_provider="openai",
        cost_status="estimated",
        cost_source="official_docs_snapshot",
        task="kanban:t_worker5",
    )
    db.queue_token_counts(
        "s5",
        input_tokens=50,
        output_tokens=5,
        api_call_count=1,
        model="gpt-test",
        billing_provider="openai",
        cost_status="estimated",
        cost_source="official_docs_snapshot",
        task="",  # non-kanban write after a kanban write
    )
    db.flush_token_counts()

    rows = _all_rows(db, "s5")
    tasks = sorted(r["task"] for r in rows)
    assert tasks == ["", "kanban:t_worker5"], (
        f"different-task deltas must NOT coalesce (would lose attribution); "
        f"got {rows!r}"
    )


# ---------------------------------------------------------------------------
# 3. The agent-loop env-var resolution pattern
# ---------------------------------------------------------------------------


def test_kanban_task_format_matches_convention(db: SessionDB) -> None:
    """Sanity-check the exact format the three call sites produce.

    The convention ``f"kanban:{HERMES_KANBAN_TASK}"`` is what analytics
    groups on, so any drift here breaks the dashboard. Pin the format.
    """
    tid = "t_abc123"
    expected = f"kanban:{tid}"
    assert expected == "kanban:t_abc123"

    db.update_token_counts(
        "s6",
        input_tokens=1,
        output_tokens=1,
        api_call_count=1,
        model="m",
        billing_provider="p",
        task=expected,
    )
    db.flush_token_counts()

    # The empty-task row must NOT exist — the worker tagged every call.
    assert _row(db, "s6", task="") is None
    # The kanban-tagged row must.
    assert _row(db, "s6", task=expected) is not None


# ---------------------------------------------------------------------------
# 4. Gateway absolute=True writes do NOT regress
# ---------------------------------------------------------------------------


def test_absolute_path_still_skips_per_model_write(db: SessionDB) -> None:
    """The gateway overwrites session counters with absolute totals. That
    path was already designed to skip session_model_usage entirely (the
    per-model breakdown is fed only by the per-call incremental path).
    Adding ``task`` must not change that."""
    db.update_token_counts(
        "s7",
        input_tokens=999,
        output_tokens=999,
        api_call_count=99,
        model="gpt-test",
        billing_provider="openai",
        cost_status="estimated",
        cost_source="official_docs_snapshot",
        absolute=True,
        task="kanban:t_should_be_ignored",
    )
    db.flush_token_counts()

    # No per-model row should have been written even though we passed task=.
    rows = _all_rows(db, "s7")
    assert rows == [], (
        f"absolute=True must not write to session_model_usage; got {rows!r}"
    )
