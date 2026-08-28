"""Tests for signal-termination exit code interpretation.

Ported from Kilo-Org/kilocode#12698 ("settle signal-terminated shell
commands as 128 + signum"): the model must see a human-readable note for
signal deaths instead of a bare exit_code=-9 / 137 it burns turns
mis-diagnosing.
"""

import pytest

from tools.terminal_tool import _interpret_exit_code, _interpret_signal_exit


class TestInterpretSignalExit:
    # ---- negative codes: subprocess -signum semantics (definite) ----

    @pytest.mark.parametrize("code,expect", [
        (-9, "SIGKILL"),
        (-11, "SIGSEGV"),
        (-15, "SIGTERM"),
        (-6, "SIGABRT"),
        (-8, "SIGFPE"),
        (-13, "SIGPIPE"),
    ])
    def test_negative_known_signals(self, code, expect):
        note = _interpret_signal_exit(code)
        assert note is not None
        assert expect in note
        assert "terminated by" in note.lower()

    def test_negative_oom_mentions_oom(self):
        note = _interpret_signal_exit(-9)
        assert "OOM" in note

    def test_negative_unknown_signal_still_reports(self):
        # signum without a curated note still yields a generic note.
        note = _interpret_signal_exit(-31)
        assert note is not None
        assert "31" in note

    # ---- 128+signum band: shell convention (hedged) ----

    @pytest.mark.parametrize("code,expect", [
        (137, "SIGKILL"),
        (139, "SIGSEGV"),
        (143, "SIGTERM"),
        (134, "SIGABRT"),
        (141, "SIGPIPE"),
    ])
    def test_shell_band_known_signals(self, code, expect):
        note = _interpret_signal_exit(code)
        assert note is not None
        assert expect in note
        # Hedged: a program can legitimately exit with these codes.
        assert "usually" in note

    def test_shell_band_uncurated_signum_returns_none(self):
        # 128+signum for a signum outside the curated table must stay
        # silent — we never guess on ambiguous application exit codes.
        assert _interpret_signal_exit(128 + 40) is None

    # ---- exclusions ----

    def test_sigint_130_excluded(self):
        # rc=130 has bespoke interrupt-marker handling in the executor.
        assert _interpret_signal_exit(130) is None
        assert _interpret_signal_exit(-2) is None

    @pytest.mark.parametrize("code", [0, 1, 2, 42, 100, 127, 128])
    def test_normal_codes_return_none(self, code):
        # 128 was promoted out of this set by t_0cea7247 — bash convention
        # for "shell could not execute the command" (set -e abort, bad
        # redirection, builtin error) now surfaces a hedged note instead of
        # leaving the [frx] observer with a bare "exit 128".
        if code == 128:
            pytest.skip("128 is now surfaced — see test_exit_128_surfaces_hedged_note")
        assert _interpret_signal_exit(code) is None


class TestSignalExitWiring:
    """_interpret_exit_code surfaces signal notes and keeps old semantics."""

    def test_signal_note_via_interpret_exit_code(self):
        note = _interpret_exit_code("python3 crash.py", -11)
        assert note is not None and "SIGSEGV" in note

    def test_shell_band_via_interpret_exit_code(self):
        note = _interpret_exit_code("./run_build.sh", 137)
        assert note is not None and "SIGKILL" in note

    def test_signal_note_wins_over_command_semantics(self):
        # A grep killed by SIGKILL must report the signal, not "no matches".
        note = _interpret_exit_code("grep foo huge.log", 137)
        assert "SIGKILL" in note

    def test_grep_exit_1_unchanged(self):
        note = _interpret_exit_code("grep foo bar.txt", 1)
        assert note is not None and "no matches" in note.lower()

    def test_success_unchanged(self):
        assert _interpret_exit_code("ls", 0) is None


class TestExit128Band:
    """Regression for kanban t_0cea7247 — bare ``exit 128`` reaching the
    [frx] observer with no human-readable note. Bash uses 128 (signum 0 in
    the shell band) for "shell could not execute the command" — see
    https://www.gnu.org/software/bash/manual/html_node/Exit-Status.html
    and https://tldp.org/LDP/abs/html/exitcodes.html. Before the fix the
    `_interpret_signal_exit` branch only fired for ``exit_code > 128`` and
    the synthesis block at line ~3766 emitted a bare
    ``Command exited 128 producing no output.`` to the model.
    """

    def test_exit_128_surfaces_hedged_note(self):
        # The headline fix: exit_code == 128 must return a non-None note
        # so the [frx] observer has something to forward and the model has
        # something to read. Without this, the bare "exit 128" is the
        # entire signal — exactly what the user reported.
        note = _interpret_signal_exit(128)
        assert note is not None
        assert "128" in note
        # Hedged: programs can legitimately ``exit 128`` themselves.
        assert "usually" in note

    def test_exit_128_does_not_call_it_a_signal(self):
        # 128 is NOT a signal death (signum 0 means "no signal"). The fix
        # explicitly does NOT label it as a signal — bash uses 128 to
        # flag a control-flow error, so signal vocabulary would mislead.
        note: str = _interpret_signal_exit(128) or ""
        assert "SIGKILL" not in note
        assert "SIGTERM" not in note
        assert "SIGSEGV" not in note
        assert "terminated by signal" not in note.lower()

    def test_exit_128_via_interpret_exit_code(self):
        # End-to-end: the function the executor actually calls returns a
        # note for rc=128 so it lands in ``result_dict["exit_code_meaning"]``.
        note = _interpret_exit_code("set -e; false", 128)
        assert note is not None
        assert "128" in note

    def test_exit_129_still_silent_when_signum_uncurated(self):
        # Edge: 128+1 == 129. Signum 1 (SIGHUP) is not in the curated table
        # and is too ambiguous to label — the pre-existing rule that
        # uncurated 128+N stays silent must keep applying. This guards
        # against the fix accidentally over-firing.
        assert _interpret_signal_exit(129) is None

    def test_normal_codes_under_128_unchanged(self):
        # Belt-and-suspenders: the fix only touches the >= 128 branch.
        # Codes 0, 1, 2, 42, 100, 127 must still return None.
        for code in (0, 1, 2, 42, 100, 127):
            assert _interpret_signal_exit(code) is None, f"{code} regressed"

    def test_no_signal_for_raw_128(self):
        # Regression guard: the note for 128 must not mention a signal
        # number — signum 0 is not a signal, so mentioning "signal 0" or
        # "SIGUSR0" or anything signal-shaped would be wrong.
        note = _interpret_signal_exit(128)
        assert note is not None
        # The note should describe bash/control-flow, not a signal.
        assert "shell" in note.lower() or "command" in note.lower()
