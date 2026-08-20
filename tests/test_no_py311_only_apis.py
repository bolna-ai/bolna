"""Guard: no Python 3.11+-only `asyncio.timeout()` in runtime code.

The runtime is 3.10, where it raises AttributeError mid-call. Checks source text rather than
behaviour, because the dev venv may be newer and would never reach the failing path.
"""

import pathlib
import re

BOLNA_ROOT = pathlib.Path(__file__).resolve().parents[1] / "bolna"

# `asyncio.timeout(` — the 3.11+ timeout context manager. `asyncio.wait_for(` is fine (all versions).
FORBIDDEN = re.compile(r"\basyncio\.timeout\s*\(")


def _runtime_sources():
    return sorted(BOLNA_ROOT.rglob("*.py"))


def test_guard_actually_reads_the_package():
    """Without this, a moved test file makes the scan below pass while inspecting nothing."""
    assert BOLNA_ROOT.is_dir(), f"{BOLNA_ROOT} is not the bolna package"
    assert len(_runtime_sources()) > 50


def test_no_asyncio_timeout_context_manager():
    offenders = []
    for path in _runtime_sources():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]  # ignore comments — flag the call, not a mention
            if FORBIDDEN.search(code):
                offenders.append(f"{path.relative_to(BOLNA_ROOT.parent)}:{i}: {line.strip()}")
    assert not offenders, (
        "asyncio.timeout() is Python 3.11+ and crashes on the 3.10 runtime — use asyncio.wait_for():\n"
        + "\n".join(offenders)
    )
