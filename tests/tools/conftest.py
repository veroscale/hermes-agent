"""Shared fixtures for tests/tools/ web-provider tests.

Per-file subprocess isolation means each test file gets a fresh interpreter,
so module-level state (like the web-search-provider registry) is empty when
a file starts.  The ``web_registry_populated`` fixture registers all bundled
providers before each test and resets the registry afterwards — tests that
depend on the registry being populated should use it explicitly or via
``@pytest.mark.usefixtures("web_registry_populated")``.
"""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_host_browser_use_cli():
    """Keep the host's browser-use/uvx install out of tests.

    Browser Use mode is default-on when the CLI is runnable, so a developer
    machine with uvx on PATH would silently flip every built-in-browser test
    into CLI mode. Pin discovery to "not installed"; tests that exercise the
    CLI path monkeypatch ``bu_cli._find_cli`` themselves.
    """
    try:
        import tools.browser_use_cli as bu_cli
    except Exception:
        yield
        return
    # Keep a handle to the real discovery function so TestFindCli (and any
    # test that wants genuine PATH probing) can restore it explicitly.
    if not hasattr(bu_cli, "_find_cli_unpatched"):
        bu_cli._find_cli_unpatched = bu_cli._find_cli
    with patch.object(bu_cli, "_find_cli", lambda: None):
        yield


@pytest.fixture(autouse=True)
def _materialize_mcp_sdk_symbols():
    """Materialize the lazily-imported MCP SDK before each tools test.

    ``tools/mcp_tool.py`` defers the ~260ms ``mcp`` SDK import until first
    real use (CLI startup perf). Tests in this directory patch SDK symbols
    (``ClientSession``, ``stdio_client``, ``_MCP_HTTP_AVAILABLE``, ...) on
    the module and expect the pre-lazy eager-import world: symbols bound,
    availability flags reflecting the installed SDK. Ensure that state up
    front so ``mock.patch`` sees real originals and ``_ensure_mcp_sdk()``
    can never clobber a patched flag mid-test (it no-ops once attempted).
    """
    try:
        from tools import mcp_tool
        mcp_tool._ensure_mcp_sdk()
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True)
def _clear_web_result_cache():
    """Reset the web_search TTL memo between tests.

    The memo is module-global state in tools/web_result_cache.py; without
    this, a test that exercised web_search_tool leaves a cached response
    that a later test with the same query would receive instead of its own
    mocked provider result.
    """
    from tools.web_result_cache import search_memo
    search_memo.clear()
    yield
    search_memo.clear()


def register_all_web_providers():
    """Register all bundled web-search providers into the global registry.

    This is the single source of truth for the provider list used by
    test classes that need the registry populated for dispatch checks.
    """
    from agent.web_search_registry import register_provider, _reset_for_tests
    from plugins.web.brave_free.provider import BraveFreeWebSearchProvider
    from plugins.web.ddgs.provider import DDGSWebSearchProvider
    from plugins.web.exa.provider import ExaWebSearchProvider
    from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider
    from plugins.web.parallel.provider import ParallelWebSearchProvider
    from plugins.web.searxng.provider import SearXNGWebSearchProvider
    from plugins.web.tavily.provider import TavilyWebSearchProvider
    from plugins.web.xai.provider import XAIWebSearchProvider

    _reset_for_tests()
    for cls in (
        BraveFreeWebSearchProvider,
        DDGSWebSearchProvider,
        ExaWebSearchProvider,
        FirecrawlWebSearchProvider,
        ParallelWebSearchProvider,
        SearXNGWebSearchProvider,
        TavilyWebSearchProvider,
        XAIWebSearchProvider,
    ):
        register_provider(cls())


@pytest.fixture
def web_registry_populated():
    """Populate the web-search-provider registry for one test, then reset."""
    register_all_web_providers()
    yield
    from agent.web_search_registry import _reset_for_tests
    _reset_for_tests()


@pytest.fixture
def disable_lazy_stt_install():
    """Disarm the runtime lazy-install probe so static ``_HAS_FASTER_WHISPER``
    patches accurately simulate 'faster-whisper not installed'.

    Without this, ``_try_lazy_install_stt()`` calls
    ``importlib.util.find_spec("faster_whisper")``, which returns truthy
    whenever the package is installed in the dev / CI environment —
    defeating the test's ``_HAS_FASTER_WHISPER=False`` patch.

    Opt in at module scope with
    ``pytestmark = pytest.mark.usefixtures("disable_lazy_stt_install")``.
    """
    with patch("tools.transcription_tools._try_lazy_install_stt", return_value=False):
        yield


# Session-context markers that change approval / cron / gateway behavior.
# Tests under tests/tools/ assert specific marker states; if any of these
# is set in the parent process (developer shell, kanban worker, cron),
# pytest inherits it via os.environ and the approval gates
# (check_dangerous_command / check_execute_code_guard) silently take the
# wrong branch. 31 approval tests fail when pytest is launched from inside
# a ``hermes chat -q`` worker because HERMES_SINGLE_QUERY_SESSION leaks
# through. Clearing these markers here gives every test a clean baseline;
# tests that want a marker set it explicitly via ``monkeypatch.setenv``
# in their own body, which runs after this fixture.
_HERMES_SESSION_MARKERS = (
    "HERMES_SINGLE_QUERY_SESSION",
    "HERMES_CRON_SESSION",
    "HERMES_GATEWAY_SESSION",
    "HERMES_EXEC_ASK",
    "HERMES_YOLO_MODE",
    "HERMES_INTERACTIVE",
    "HERMES_SESSION_PLATFORM",
)


@pytest.fixture(autouse=True)
def _clear_ambient_hermes_session_markers(monkeypatch):
    """Strip ambient HERMES_* session markers inherited from the launcher.

    A developer shell, a ``hermes chat -q`` kanban worker, or a cron tick
    can each export one or more of the markers in :data:`_HERMES_SESSION_MARKERS`
    before pytest ever runs.  Once pytest inherits them, every approval-gate
    test under tests/tools/ silently takes the wrong branch — the single-query
    branch short-circuits ahead of the cron branch, the gateway branch wins
    over a headless CLI test, and ``_YOLO_MODE_FROZEN`` is whatever the
    parent process had at import time.  31 tests in
    test_execute_code_approval_cluster / test_cron_approval_mode /
    test_approval_mode_parity / test_approval_outcome_parity /
    test_approval_config_readonly / test_code_execution_modes fail
    identically on untouched main when launched from a worker (they all
    pass under ``env -u HERMES_SINGLE_QUERY_SESSION``).

    This fixture only clears ambient inheritance.  Tests that *want* a
    marker — most of the suite already do via ``monkeypatch.setenv`` — set
    it explicitly in their own body after this fixture runs, so coverage is
    preserved.

    Two layers need clearing, not one:

    1. ``os.environ``: most markers read straight off ``os.getenv`` via
       ``env_var_enabled`` / ``is_truthy_value``.  ``monkeypatch.delenv``
       removes them for the duration of this test and restores them on
       teardown.

    2. ``gateway.session_context`` ContextVars: ``_VAR_MAP`` covers
       ``HERMES_CRON_SESSION`` and ``HERMES_SESSION_PLATFORM``, and
       ``get_session_env`` prefers the ContextVar over ``os.environ`` if it
       is set (even to ``""``).  Calling ``reset_session_vars()`` here puts
       every mapped ContextVar back to ``_UNSET`` so the env fallback
       (now cleared in step 1) is what production code reads.

    ``_YOLO_MODE_FROZEN`` is module-import-time state (``tools/approval.py``
    snapshots ``os.getenv("HERMES_YOLO_MODE")`` at first import).  The env
    clear above prevents new subprocesses from seeing it, but the module
    constant still reflects the parent's value.  Patch it to ``False`` to
    match the post-clear env state — tests that want YOLO on set it via
    ``monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)``
    themselves, exactly as they already do.
    """
    for marker in _HERMES_SESSION_MARKERS:
        monkeypatch.delenv(marker, raising=False)

    try:
        from gateway.session_context import reset_session_vars

        reset_session_vars()
    except Exception:
        # gateway may not be importable in a stripped-down test environment.
        # The env clear above is still effective for the env-only markers.
        pass

    try:
        import tools.approval as _approval_mod

        monkeypatch.setattr(_approval_mod, "_YOLO_MODE_FROZEN", False)
    except Exception:
        pass

    yield
