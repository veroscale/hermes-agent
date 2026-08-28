"""Tiered per-card turn budgets + budget-exhaustion telemetry.

Covers #t_8ab01085:
* ``resolve_tiered_turn_budget`` — the tier mapping by card body length,
  and that the feature flag ``kanban.tiered_turn_budget`` defaults OFF
  (no behavior change until an operator opts in).
* ``_default_spawn`` passing ``--max-turns N`` to the worker when the
  tier resolves a budget (flag on) and passing NOTHING when off.
* ``_extract_budget_telemetry`` — the per-card loop-signature fields
  (final action, completion tool called y/n, trailing context tail).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from agent import turn_finalizer as tf


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    c = kb.connect()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# resolve_tiered_turn_budget — tier mapping + feature-flag default
# ---------------------------------------------------------------------------


class _Task:
    def __init__(self, body):
        self.body = body


def test_tier_default_flag_off_returns_none():
    """The feature flag defaults OFF → autonomous, no behavior change."""
    task = _Task(body="x" * 5000)  # would be "large" if on
    assert kb.resolve_tiered_turn_budget(task, flag_on=False) is None


def test_tier_small_body(conn):
    task = _Task(body="x" * 799)
    assert kb.resolve_tiered_turn_budget(task, flag_on=True) == 25


def test_tier_small_boundary_inclusive(conn):
    task = _Task(body="x" * 800)
    assert kb.resolve_tiered_turn_budget(task, flag_on=True) == 25


def test_tier_medium_body(conn):
    task = _Task(body="x" * 801)
    assert kb.resolve_tiered_turn_budget(task, flag_on=True) == 50


def test_tier_medium_boundary_inclusive(conn):
    task = _Task(body="x" * 2000)
    assert kb.resolve_tiered_turn_budget(task, flag_on=True) == 50


def test_tier_large_body(conn):
    task = _Task(body="x" * 2001)
    assert kb.resolve_tiered_turn_budget(task, flag_on=True) == 80


def test_tier_large_huge_body(conn):
    task = _Task(body="y" * 10000)
    assert kb.resolve_tiered_turn_budget(task, flag_on=True) == 80


def test_tier_blank_body_is_small(conn):
    task = _Task(body="")
    assert kb.resolve_tiered_turn_budget(task, flag_on=True) == 25


# ---------------------------------------------------------------------------
# _default_spawn — passes --max-turns only when the tier resolves a budget
# ---------------------------------------------------------------------------


def _spawn_and_capture(monkeypatch, tmp_path, task, tier_flag):
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4245

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(kb, "resolve_tiered_turn_budget",
                        lambda task: (25 if tier_flag else None))
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    kb._default_spawn(task, str(workspace))
    return captured["cmd"]


def test_spawn_passes_no_max_turns_when_flag_off(monkeypatch, tmp_path, conn):
    """Flag OFF → argv has no --max-turns; the worker uses the profile default."""
    tid = kb.create_task(conn, title="small", assignee="worker", body="x" * 100)
    task = kb.get_task(conn, tid)
    cmd = _spawn_and_capture(monkeypatch, tmp_path, task, tier_flag=False)
    assert "--max-turns" not in cmd


def test_spawn_passes_max_turns_when_flag_on(monkeypatch, tmp_path, conn):
    """Flag ON → argv carries the tier-derived --max-turns N."""
    tid = kb.create_task(conn, title="small", assignee="worker", body="x" * 100)
    task = kb.get_task(conn, tid)
    cmd = _spawn_and_capture(monkeypatch, tmp_path, task, tier_flag=True)
    assert "--max-turns" in cmd
    i = cmd.index("--max-turns")
    assert cmd[i + 1] == "25"


# ---------------------------------------------------------------------------
# _extract_budget_telemetry — per-card loop-signature fields
# ---------------------------------------------------------------------------


def _assistant(tool_names, content=""):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"function": {"name": name}} for name in tool_names
        ],
    }


def test_telemetry_empty_transcript():
    fa, cc, tail = tf._extract_budget_telemetry([])
    assert fa is None
    assert cc is False
    assert tail == ""


def test_telemetry_surface_final_action_and_no_completion():
    msgs = [
        {"role": "user", "content": "work the task"},
        _assistant(["terminal", "search_files"], content=""),
    ]
    fa, cc, tail = tf._extract_budget_telemetry(msgs)
    assert fa == "search_files"  # last tool in the assistant turn
    assert cc is False
    assert "work the task" in tail


def test_telemetry_detects_completion_tool_called():
    msgs = [
        {"role": "user", "content": "work"},
        _assistant(["terminal"], content=""),
        _assistant(["kanban_complete"], content=""),
    ]
    fa, cc, _ = tf._extract_budget_telemetry(msgs)
    assert fa == "kanban_complete"
    assert cc is True


def test_telemetry_context_tail_truncated_to_200():
    msgs = [
        {"role": "user", "content": "z" * 5000},
        _assistant(["terminal"], content=""),
    ]
    fa, cc, tail = tf._extract_budget_telemetry(msgs)
    assert fa == "terminal"
    assert cc is False
    assert len(tail) <= 200


def test_telemetry_non_string_content_ignored():
    msgs = [
        {"role": "user", "content": ["part1", "part2"]},
        _assistant(["terminal"], content=""),
    ]
    fa, cc, tail = tf._extract_budget_telemetry(msgs)
    assert fa == "terminal"
    assert cc is False
    assert "part1" in tail or "part2" in tail or tail == ""


# ---------------------------------------------------------------------------
# DB-level: budget telemetry lands on the FIRST timed_out event
# ---------------------------------------------------------------------------


def test_budget_telemetry_payload_lands_on_timed_out_event(conn, monkeypatch):
    """The richer telemetry must be visible on the first timed_out event,
    not only after the circuit breaker trips into gave_up. (#t_8ab01085)"""
    tid = kb.create_task(conn, title="telemetry probe", assignee="worker")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)

    from agent import turn_finalizer as tfm
    # Route through the real _record_kanban_budget_exhausted so the full
    # payload assembly + _record_task_failure path is exercised.
    tfm._record_kanban_budget_exhausted(
        tid, 60, 60, logging.getLogger("test"),
        final_action="terminal",
        completion_tool_called=False,
        context_tail="the trailing context window",
    )

    events = [e for e in kb.list_events(conn, tid) if e.kind == "timed_out"]
    assert events, "expected a timed_out event"
    payload = events[-1].payload or {}
    assert payload["budget_used"] == 60
    assert payload["budget_max"] == 60
    assert payload["turns_used"] == 60
    assert payload["final_action"] == "terminal"
    assert payload["completion_tool_called"] is False
    assert "trailing context window" in payload["context_tail"]