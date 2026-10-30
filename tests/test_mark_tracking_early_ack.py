"""An acked mark counts as acked in mark_tracking, even when the ack lands before sent_ts + duration."""

import time

import pytest

from bolna.helpers.mark_event_meta_data import MarkEventMetaData
from bolna.input_handlers.default import DefaultInputHandler


def _ack(sent_ago, duration):
    mark = MarkEventMetaData()
    mark.update_data(
        "m-1",
        {
            "type": "agent_response",
            "text_synthesized": "hello",
            "sequence_id": 3,
            "sent_ts": time.time() - sent_ago,
            "duration": duration,
        },
    )
    handler = DefaultInputHandler(mark_event_meta_data=mark, observable_variables={})
    handler.process_mark_message({"name": "m-1"})
    return mark.get_mark_tracking_summary()


@pytest.mark.parametrize(
    "sent_ago, duration",
    [
        (2.0, 1.0),  # ack after the estimated playout end
        (0.0, 1.0),  # ack before it: the duration estimate ran long, or the client acks on receipt
    ],
)
def test_ack_is_counted(sent_ago, duration):
    summary = _ack(sent_ago, duration)
    assert summary["total_sent"] == 1
    assert summary["total_acked"] == 1
    assert summary["total_missed"] == 0
    assert summary["per_sequence"][0]["acked"] == 1


def test_early_ack_adds_no_delay_sample():
    summary = _ack(0.0, 1.0)
    assert summary["max_delay_s"] == 0
    assert summary["per_sequence"][0]["chunk_delays"] == []
