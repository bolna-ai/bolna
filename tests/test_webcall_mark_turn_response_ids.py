"""Web-call audio marks must carry turn/response identifiers.

On telephony, each post-mark records turn_id/response_uid/response_group_uid, and both
the input handler's ack path (record_heard_text / response_heard_by_turn) and TaskManager's
sync_history rely on them to trim an interrupted response to what the caller actually heard.
The web (default) output handler used to omit these, so on a barge-in during a web call the
whole turn was trimmed blindly and heard-text-by-turn was never populated.
"""

from bolna.helpers.mark_event_meta_data import MarkEventMetaData
from bolna.output_handlers.default import DefaultOutputHandler


class FakeWebSocket:
    def __init__(self):
        self.text_messages = []
        self.json_messages = []

    async def send_text(self, message):
        self.text_messages.append(message)

    async def send_json(self, message):
        self.json_messages.append(message)


def _audio_packet():
    return {
        "data": b"\x01\x02\x03\x04\x05\x06\x07\x08",
        "meta_info": {
            "type": "audio",
            "format": "pcm",
            "sequence_id": 1,
            "turn_id": 7,
            "response_uid": "resp-1",
            "response_group_uid": "group-1",
            "text_synthesized": "hello there",
            "message_category": "",
        },
    }


async def test_web_post_mark_carries_turn_and_response_ids():
    mark_store = MarkEventMetaData()
    handler = DefaultOutputHandler(io_provider="default", websocket=FakeWebSocket(), mark_event_meta_data=mark_store)

    await handler.handle(_audio_packet())

    # The post-mark (content audio) is the one kept in _mark_history; the pre-mark is excluded.
    post_marks = list(mark_store._mark_history.values())
    assert len(post_marks) == 1
    post_mark = post_marks[0]
    assert post_mark["turn_id"] == 7
    assert post_mark["response_uid"] == "resp-1"
    assert post_mark["response_group_uid"] == "group-1"


async def test_heard_text_is_attributed_to_the_turn_on_web_calls():
    """With the ids present, an incoming ack credits heard text to the right turn/response."""
    mark_store = MarkEventMetaData()
    handler = DefaultOutputHandler(io_provider="default", websocket=FakeWebSocket(), mark_event_meta_data=mark_store)

    await handler.handle(_audio_packet())
    post_mark = list(mark_store._mark_history.values())[0]

    mark_store.record_heard_text(post_mark, post_mark["text_synthesized"])

    assert mark_store.get_heard_text_for_turn(7) == "hello there"
    assert mark_store.get_heard_text_for_response("resp-1") == "hello there"
