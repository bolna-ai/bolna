# Tests

```bash
pip install -e '.[dev]'
pytest                        # whole suite, from anywhere in the repo
pytest tests/test_router_nodes.py -q
pytest -k language_switch
```

Configuration lives in `[tool.pytest.ini_options]` in `pyproject.toml`. CI runs `pytest` on
every pull request.

## Conventions

One flat directory, one module per behaviour under test, named `test_<behaviour>.py`. Every
module opens with a one-line docstring saying what it guards.

`asyncio_mode = "auto"`, so a coroutine test needs no decorator — just `async def test_x()`.

Test doubles stay local to the module that uses them. A helper only earns a place in
`conftest.py` once a second module needs it; importing across test modules is not the way.

A test that needs a live provider credential belongs in `tests/manual/`, whose scripts pytest
does not collect. Mark a collected test `@pytest.mark.integration` if it dials out, so it can
be deselected with `-m 'not integration'`.
