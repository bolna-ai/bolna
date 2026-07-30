"""Connect-failure handling shared by the streaming synthesizers: a failure must be
identifiable in the logs, and the wait_for must own the connect deadline outright."""

import asyncio
import logging
import os

import pytest
import websockets

for _var in [
    "ELEVENLABS_API_KEY",
    "DEEPGRAM_AUTH_TOKEN",
    "CARTESIA_API_KEY",
    "SARVAM_API_KEY",
    "RIME_API_KEY",
    "SMALLEST_API_KEY",
    "PIXA_API_KEY",
]:
    os.environ.setdefault(_var, "test-key")

import bolna.synthesizer as synthesizers  # noqa: E402

PROVIDERS = [
    ("Deepgram", {"voice_id": "v", "voice": "v"}),
    ("Cartesia", {"voice_id": "v", "voice": "v"}),
    ("Sarvam", {"voice_id": "v", "model": "bulbul:v2", "language": "en"}),
    ("Rime", {"voice_id": "v", "voice": "v"}),
    ("Pixa", {"voice_id": "v", "voice": "v"}),
    ("Smallest", {"voice_id": "v"}),
    ("Elevenlabs", {"voice_id": "v", "voice": "v"}),
]
PROVIDER_IDS = [name for name, _ in PROVIDERS]


class FakeWS:
    def __init__(self):
        self.state = websockets.protocol.State.OPEN
        self.sent = []

    async def send(self, payload):
        self.sent.append(payload)


def _build(name, kwargs):
    return getattr(synthesizers, f"{name}Synthesizer")(task_manager_instance=None, **kwargs)


@pytest.fixture
def fake_connect(monkeypatch):
    """Replace websockets.connect and record the kwargs each provider passes."""
    calls = []

    def connect(*args, **kwargs):
        calls.append(kwargs)

        async def _await():
            return FakeWS()

        return _await()

    monkeypatch.setattr(websockets, "connect", connect)
    return calls


@pytest.mark.parametrize("name,kwargs", PROVIDERS, ids=PROVIDER_IDS)
def test_connect_disables_the_library_open_timeout(name, kwargs, fake_connect):
    """websockets' own open_timeout would race the wait_for wrapped around it, and on
    py3.10 it raises the builtin TimeoutError, which the timeout branch used to miss."""
    synth = _build(name, kwargs)

    assert asyncio.run(synth.establish_connection()) is not None
    assert fake_connect, f"{name} did not call websockets.connect"
    assert fake_connect[0].get("open_timeout", "absent") is None


@pytest.mark.parametrize("name,kwargs", PROVIDERS, ids=PROVIDER_IDS)
@pytest.mark.parametrize("exc", [TimeoutError(), asyncio.TimeoutError()], ids=["builtin", "asyncio"])
def test_either_timeout_type_reaches_the_timeout_branch(name, kwargs, exc, monkeypatch, caplog):
    """On py3.10 asyncio.TimeoutError is not the builtin, so a socket-level timeout fell
    through to the generic handler and logged an empty message."""

    def connect(*args, **kwargs):
        async def _raise():
            raise exc

        return _raise()

    monkeypatch.setattr(websockets, "connect", connect)
    synth = _build(name, kwargs)

    with caplog.at_level(logging.ERROR):
        assert asyncio.run(synth.establish_connection()) is None

    assert "Timeout while connecting" in caplog.text
    assert "Failed to connect" not in caplog.text


@pytest.mark.parametrize("name,kwargs", PROVIDERS, ids=PROVIDER_IDS)
def test_bare_socket_error_is_named_in_the_log(name, kwargs, monkeypatch, caplog):
    """str(ConnectionResetError()) is "", so logging {e} produced a blank line. The type
    has to reach the log or these failures cannot be told apart."""

    def connect(*args, **kwargs):
        async def _raise():
            raise ConnectionResetError()

        return _raise()

    monkeypatch.setattr(websockets, "connect", connect)
    synth = _build(name, kwargs)

    with caplog.at_level(logging.ERROR):
        assert asyncio.run(synth.establish_connection()) is None

    assert "ConnectionResetError" in caplog.text
