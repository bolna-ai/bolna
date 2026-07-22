"""Send-rate cap in TelephonyOutputHandler (fix for 87da790e: audio blasted to the
provider faster than real-time gets discarded by clearAudio on barge-in)."""
import asyncio
import time

from bolna.output_handlers.telephony_providers.vobiz import VobizOutputHandler
from bolna.output_handlers.telephony_providers.plivo import PlivoOutputHandler
from bolna.helpers.mark_event_meta_data import MarkEventMetaData

CHUNK_BYTES = 3200   # 0.4 s of mulaw @ 8 kHz  (handle() derives duration = len/8000)
N = 6                # 2.4 s of audio total
FACTOR = 1.5


class MockWS:
    async def send_text(self, _):
        pass


def _handler(factor):
    h = VobizOutputHandler(websocket=MockWS(), mark_event_meta_data=MarkEventMetaData(), log_dir_name=None)
    h.stream_sid = "s1"
    h.max_send_rate_factor = factor
    return h


def _pkt(i):
    return {
        "data": b"\xff" * CHUNK_BYTES,
        "meta_info": {
            "stream_sid": "s1", "sequence_id": 1, "turn_id": 1, "response_uid": "r",
            "response_group_uid": "r", "format": "mulaw", "is_first_chunk": i == 0,
            "end_of_llm_stream": i == N - 1, "end_of_synthesizer_stream": i == N - 1,
            "text_synthesized": f"w{i} ", "mark_id": f"c{i}", "cached": False, "message_category": "",
        },
    }


def _measure(factor):
    async def run():
        h = _handler(factor)
        t0 = time.monotonic()
        for i in range(N):
            await h.handle(_pkt(i))
        return time.monotonic() - t0, h.mark_event_meta_data.get_mark_tracking_summary()["total_sent"]

    return asyncio.run(run())


def test_send_rate_cap_paces_burst():
    # 2.4 s of audio blasted through handle(); cap 1.5x -> must take >= ~2.4/1.5 = 1.6 s.
    elapsed, sent = _measure(FACTOR)
    expected_min = (N * CHUNK_BYTES) / (FACTOR * 8000)  # 1.6 s
    assert elapsed >= expected_min * 0.85, f"not paced: {elapsed:.2f}s < {expected_min:.2f}s"
    assert sent == N  # pacing must not drop audio


def test_disabled_is_instant():
    elapsed, sent = _measure(0)  # cap off -> current behaviour, no pacing
    assert elapsed < 0.3, f"unexpected delay with cap disabled: {elapsed:.2f}s"
    assert sent == N


def test_cap_scoped_to_vobiz_only():
    # VoBiz opts into pacing; other telephony providers inherit the disabled base default.
    vobiz = VobizOutputHandler(websocket=MockWS(), mark_event_meta_data=MarkEventMetaData(), log_dir_name=None)
    plivo = PlivoOutputHandler(websocket=MockWS(), mark_event_meta_data=MarkEventMetaData(), log_dir_name=None)
    assert vobiz.max_send_rate_factor == 1.5
    assert plivo.max_send_rate_factor == 0.0
