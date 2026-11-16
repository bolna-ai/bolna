"""TTS alignment can drop characters (Devanagari on ElevenLabs v3). When the heard text covers about the
whole reply, sync_history must keep the LLM text instead of the lossy alignment copy."""

from bolna.agent_manager.task_manager import TaskManager
from bolna.helpers.conversation_history import ConversationHistory

LLM_TEXT = (
    "ठीक है जी, आपकी payment की date 5 November note कर ली है। Link आपके registered number पर भेज दिया जाएगा। शुक्रिया जी।"
)
ALIGNMENT_TEXT = (
    "क है ी, आपकी payment की date 5 November note कर ली है। Link आपके registered number पर भेज दिया जाएगा। शुक्रिया जी।"
)


class InputStub:
    last_heard_turn_id = 8
    last_heard_response_uid = "r8"
    response_heard_by_user = ""

    def __init__(self, heard):
        self.heard = heard

    def get_response_heard_for_response(self, uid):
        return self.heard if uid == "r8" else ""

    def get_response_heard_for_turn(self, turn_id):
        return ""


class MarkMetaStub:
    def get_heard_text_for_response(self, uid):
        return ""

    def get_heard_text_for_turn(self, turn_id):
        return ""


def make_task_manager(history, heard):
    tm = TaskManager.__new__(TaskManager)
    tm.conversation_history = history
    tm.tools = {"input": InputStub(heard)}
    tm.mark_event_meta_data = MarkMetaStub()
    tm._turn_msg_map = {}
    tm._last_spoken_assistant = None
    tm._pending_assistant_history = {}
    return tm


def assistant_contents(history):
    return [m.get("content") for m in history.messages if m.get("role") == "assistant"]


async def test_fully_played_reply_keeps_llm_text_over_lossy_alignment():
    history = ConversationHistory()
    history.append_user(" हां हो जाएगा")
    history.append_assistant(" " + LLM_TEXT, turn_id=8, response_uid="r8")
    tm = make_task_manager(history, ALIGNMENT_TEXT)

    await tm.sync_history([], interruption_processed_at=1000.0)

    assert assistant_contents(history) == [LLM_TEXT]


async def test_staged_reply_keeps_llm_text_over_lossy_alignment():
    history = ConversationHistory()
    history.append_user(" हां हो जाएगा")
    tm = make_task_manager(history, ALIGNMENT_TEXT)
    tm._pending_assistant_history = {8: {"content": " " + LLM_TEXT, "response_uid": "r8", "node": None}}

    await tm.sync_history([], interruption_processed_at=1000.0)

    assert assistant_contents(history) == [LLM_TEXT]


async def test_barge_in_with_lossy_alignment_stores_clean_llm_prefix():
    llm_text = "जी, मैं Woxsen University से call कर रही हूँ।"
    history = ConversationHistory()
    history.append_assistant(" " + llm_text, turn_id=8, response_uid="r8")
    tm = make_task_manager(history, "ज ी, मैं")

    await tm.sync_history([], interruption_processed_at=1000.0)

    assert assistant_contents(history) == ["जी, मैं"]


async def test_fully_played_reply_with_many_drops_stores_clean_llm_prefix():
    llm_text = "मैं Woxsen University से Law admissions में आपकी help के लिए call कर रही हूँ, क्या मैं आपसे दो मिनट बात कर सकती हूँ?"
    alignment = "ं Woxsen University से Law ad में आपकी help के लिए call कर रही हूँ, क्या मैं आपसे दो मिनट बात कर सकती हूँ?"
    history = ConversationHistory()
    history.append_assistant(" " + llm_text, turn_id=8, response_uid="r8")
    tm = make_task_manager(history, alignment)

    await tm.sync_history([], interruption_processed_at=1000.0)

    assert assistant_contents(history) == [
        "मैं Woxsen University से Law admissions में आपकी help के लिए call कर रही हूँ, क्या मैं आपसे दो मिनट बात कर"
    ]


async def test_cut_mid_word_drops_the_partial_word():
    llm_text = "आपकी EMI pending है जी।"
    history = ConversationHistory()
    history.append_assistant(" " + llm_text, turn_id=8, response_uid="r8")
    tm = make_task_manager(history, "आपी EMI pen")

    await tm.sync_history([], interruption_processed_at=1000.0)

    assert assistant_contents(history) == ["आपकी EMI"]


async def test_barge_in_still_trims_to_heard_text():
    heard = "ठीक है जी, आपकी payment की date"
    history = ConversationHistory()
    history.append_assistant(" " + LLM_TEXT, turn_id=8, response_uid="r8")
    tm = make_task_manager(history, heard)

    await tm.sync_history([], interruption_processed_at=1000.0)

    assert assistant_contents(history) == [heard]
