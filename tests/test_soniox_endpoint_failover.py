"""Soniox regions: a session refused for capacity moves to the next region instead of ending the call."""

import asyncio
import json

from bolna.constants import SONIOX_REGION_HOSTS, SONIOX_WEBSOCKET_HOST
from bolna.transcriber.soniox_transcriber import SonioxTranscriber

IN = {"region": "in", "host": SONIOX_REGION_HOSTS["in"]}
EU = {"region": "eu", "host": SONIOX_REGION_HOSTS["eu"]}
REFUSED = {
    "error_code": 429,
    "error_message": "Concurrent requests limit for real-time transcription has been exceeded",
}
WORD = {"tokens": [{"text": "hello", "start_ms": 0, "end_ms": 300, "is_final": True}]}


class FakeSocket:
    def __init__(self, responses):
        self.responses = [json.dumps(response) for response in responses]
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.responses:
            raise StopAsyncIteration
        return self.responses.pop(0)

    async def send(self, data):
        pass

    async def close(self):
        self.closed = True


def leg_with_regions(monkeypatch, regions, **kwargs):
    monkeypatch.setenv("SONIOX_IN_API_KEY", "in-key")
    monkeypatch.setenv("SONIOX_EU_API_KEY", "eu-key")
    monkeypatch.delenv("SONIOX_HOST", raising=False)
    return SonioxTranscriber("web_based_call", output_queue=asyncio.Queue(), soniox_regions=regions, **kwargs)


def run_session(leg, responses_by_host, connect_delay_by_host=None):
    """Run transcribe() against canned frames per host; returns (hosts tried, packets delivered)."""
    tried = []

    async def connect():
        tried.append((leg.soniox_host, leg.api_key))
        await asyncio.sleep((connect_delay_by_host or {}).get(leg.soniox_host, 0))
        socket = responses_by_host[leg.soniox_host]
        if not isinstance(socket, FakeSocket):
            socket = FakeSocket(socket)
        leg.websocket_connection = socket
        return socket

    leg.soniox_connect = connect
    leg.meta_info = {}
    asyncio.run(leg.transcribe())
    packets = []
    while not leg.transcriber_output_queue.empty():
        packets.append(leg.transcriber_output_queue.get_nowait())
    return tried, packets


def test_a_capacity_refusal_moves_the_session_to_the_next_region(monkeypatch):
    leg = leg_with_regions(monkeypatch, ["in", "eu"])

    tried, packets = run_session(leg, {IN["host"]: [REFUSED], EU["host"]: [WORD]})

    assert tried == [(IN["host"], "in-key"), (EU["host"], "eu-key")]
    assert leg.endpoint_label == "eu"
    assert any(isinstance(p["data"], dict) and p["data"].get("content") == "hello" for p in packets)
    assert "connection_error" not in packets[-1]["meta_info"]


def test_a_moved_session_reports_the_time_to_the_connection_that_held(monkeypatch):
    leg = leg_with_regions(monkeypatch, ["in", "eu"])

    run_session(
        leg, {IN["host"]: [REFUSED], EU["host"]: [WORD]}, connect_delay_by_host={IN["host"]: 0.02, EU["host"]: 0.05}
    )

    assert leg.connection_time >= 70


def test_every_region_refusing_ends_the_session_with_the_error(monkeypatch):
    leg = leg_with_regions(monkeypatch, ["in", "eu"])

    tried, packets = run_session(leg, {IN["host"]: [REFUSED], EU["host"]: [REFUSED]})

    assert [host for host, _ in tried] == [IN["host"], EU["host"]]
    assert packets[-1]["data"] == "transcriber_connection_closed"
    assert packets[-1]["meta_info"]["connection_error"].startswith("429")


def test_a_hangup_during_the_move_opens_no_further_session(monkeypatch):
    leg = leg_with_regions(monkeypatch, ["in", "eu"])
    refused = FakeSocket([REFUSED])

    async def hang_up_on_close():
        leg.connection_on = False
        refused.closed = True

    refused.close = hang_up_on_close
    tried, packets = run_session(leg, {IN["host"]: refused, EU["host"]: [WORD]})

    assert [host for host, _ in tried] == [IN["host"]]
    assert packets[-1]["data"] == "transcriber_connection_closed"


def test_a_provider_error_other_than_capacity_does_not_move(monkeypatch):
    leg = leg_with_regions(monkeypatch, ["in", "eu"])

    tried, packets = run_session(leg, {IN["host"]: [{"error_code": 401, "error_message": "bad key"}]})

    assert [host for host, _ in tried] == [IN["host"]]
    assert packets[-1]["meta_info"]["connection_error"].startswith("401")


def test_without_regions_the_env_host_is_used(monkeypatch):
    monkeypatch.setenv("SONIOX_API_KEY", "env-key")
    monkeypatch.delenv("SONIOX_HOST", raising=False)

    leg = SonioxTranscriber("web_based_call", output_queue=asyncio.Queue())

    assert leg.endpoints == [{"label": "default", "host": SONIOX_WEBSOCKET_HOST, "api_key": "env-key"}]


def test_a_customer_key_ignores_the_regions(monkeypatch):
    leg = leg_with_regions(monkeypatch, ["in", "eu"], transcriber_key="customer-key")

    assert [e["label"] for e in leg.endpoints] == ["default"]
    assert leg.api_key == "customer-key"


def test_a_region_without_its_key_in_env_is_left_out(monkeypatch):
    monkeypatch.setenv("SONIOX_IN_API_KEY", "in-key")
    monkeypatch.delenv("SONIOX_EU_API_KEY", raising=False)

    leg = SonioxTranscriber("web_based_call", output_queue=asyncio.Queue(), soniox_regions=["in", "eu"])

    assert [e["label"] for e in leg.endpoints] == ["in"]


def test_an_unknown_region_is_left_out(monkeypatch):
    leg = leg_with_regions(monkeypatch, ["mars", "eu"])

    assert leg.endpoints == [{"label": "eu", "host": SONIOX_REGION_HOSTS["eu"], "api_key": "eu-key"}]


def test_a_box_with_no_pool_keys_keeps_its_env_host_and_key(monkeypatch):
    monkeypatch.setenv("SONIOX_API_KEY", "env-key")
    monkeypatch.setenv("SONIOX_HOST", "stt-rt.example.test")
    monkeypatch.delenv("SONIOX_IN_API_KEY", raising=False)
    monkeypatch.delenv("SONIOX_EU_API_KEY", raising=False)

    leg = SonioxTranscriber("web_based_call", output_queue=asyncio.Queue(), soniox_regions=["in", "eu"])

    assert leg.endpoints == [{"label": "default", "host": "stt-rt.example.test", "api_key": "env-key"}]
