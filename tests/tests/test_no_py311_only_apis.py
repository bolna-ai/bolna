"""Guard: no Python 3.11+-only `asyncio.timeout()` in runtime code — prod is 3.10, where it
raises AttributeError and cut live calls (QA 6525d51a). Checks source text, since the dev venv
is 3.11+ and wouldn't hit the path."""

import pathlib
import re

BOLNA_ROOT = pathlib.Path(__file__).resolve().parents[2] / "bolna"

# `asyncio.timeout(` — the 3.11+ timeout context manager. `asyncio.wait_for(` is fine (all versions).
FORBIDDEN = re.compile(r"\basyncio\.timeout\s*\(")


def test_no_asyncio_timeout_context_manager():
    offenders = []
    for path in BOLNA_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]  # ignore comments — flag the call, not a mention
            if FORBIDDEN.search(code):
                offenders.append(f"{path.relative_to(BOLNA_ROOT.parent)}:{i}: {line.strip()}")
    assert not offenders, (
        "asyncio.timeout() is Python 3.11+ and crashes on the 3.10 runtime — use asyncio.wait_for():\n"
        + "\n".join(offenders)
    )
