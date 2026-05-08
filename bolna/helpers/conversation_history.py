import copy
from bolna.enums import ChatRole
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

_UNHEARD_ROLES = frozenset({ChatRole.ASSISTANT, ChatRole.TOOL})


class ConversationHistory:
    def __init__(self, initial_history: list[dict] | None = None):
        self._messages: list[dict] = initial_history or []
        self._interim: list[dict] = copy.deepcopy(self._messages)

    def setup_system_prompt(self, system_prompt: dict):
        if system_prompt.get("content", ""):
            if not self._messages:
                self._messages = [system_prompt]
            else:
                self._messages = [system_prompt] + self._messages

        self._interim = copy.deepcopy(self._messages)

    def append_welcome_message(self, content: str):
        if content:
            self._messages.append({"role": ChatRole.ASSISTANT, "content": content})

        self._interim = copy.deepcopy(self._messages)

    def append_user(self, content: str):
        self._messages.append({"role": ChatRole.USER, "content": content})

    def append_assistant(self, content: str, tool_calls: list | None = None, **kwargs):
        if self._messages:
            last = self._messages[-1]
            if (
                last.get("role") == ChatRole.ASSISTANT
                and last.get("content") == content
                and last.get("turn_id") == kwargs.get("turn_id")
                and last.get("response_uid") == kwargs.get("response_uid")
            ):
                if tool_calls is not None:
                    last["tool_calls"] = tool_calls
                return
        msg = {"role": ChatRole.ASSISTANT, "content": content, **kwargs}
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        self._messages.append(msg)

    def upsert_assistant_for_turn(self, turn_id: int | None, content: str, interim: bool = False):
        msgs = self._interim if interim else self._messages
        if turn_id is not None:
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("role") == ChatRole.ASSISTANT and msgs[i].get("turn_id") == turn_id:
                    msgs[i]["content"] = content
                    return
        msgs.append({"role": ChatRole.ASSISTANT, "content": content, "turn_id": turn_id})

    def upsert_assistant_for_response(
        self, response_uid: str | None, content: str, interim: bool = False, turn_id: int | None = None
    ):
        msgs = self._interim if interim else self._messages
        if response_uid is not None:
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("role") == ChatRole.ASSISTANT and msgs[i].get("response_uid") == response_uid:
                    msgs[i]["content"] = content
                    if turn_id is not None and msgs[i].get("turn_id") is None:
                        msgs[i]["turn_id"] = turn_id
                    return
        msgs.append({"role": ChatRole.ASSISTANT, "content": content, "turn_id": turn_id, "response_uid": response_uid})

    def append_tool_result(self, tool_call_id: str, content: str):
        self._messages.append(
            {
                "role": ChatRole.TOOL,
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )

    def attach_tool_calls_to_last_response(self, tool_calls: list):
        if self._messages and self._messages[-1].get("role") == ChatRole.ASSISTANT:
            self._messages[-1]["tool_calls"] = tool_calls
        else:
            logger.warning("attach_tool_calls_to_last_response: last message is not assistant, appending new")
            self.append_assistant(None, tool_calls=tool_calls)

    def attach_tool_calls_to_turn(self, turn_id: int | None, tool_calls: list):
        if turn_id is None:
            self.attach_tool_calls_to_last_response(tool_calls)
            return

        for i in range(len(self._messages) - 1, -1, -1):
            if self._messages[i].get("role") == ChatRole.ASSISTANT and self._messages[i].get("turn_id") == turn_id:
                self._messages[i]["tool_calls"] = tool_calls
                return

        logger.warning(
            f"attach_tool_calls_to_turn: no assistant found for turn_id={turn_id}, appending placeholder assistant"
        )
        self.append_assistant(None, tool_calls=tool_calls, turn_id=turn_id)

    def update_system_prompt(self, content: str):
        if self._messages and self._messages[0].get("role") == ChatRole.SYSTEM:
            self._messages[0]["content"] = content

    def update_welcome_message(self, content: str):
        if len(self._messages) >= 2 and self._messages[1].get("role") == ChatRole.ASSISTANT:
            self._messages[1]["content"] = content

    def pop_unheard_responses(self) -> list[dict]:
        popped = []
        while self._messages and self._messages[-1].get("role") in _UNHEARD_ROLES:
            popped.append(self._messages.pop())
        return popped

    def pop_and_merge_user(self, new_content: str) -> str:
        if self._messages and self._messages[-1].get("role") == ChatRole.USER:
            prev_user = self._messages.pop()
            return prev_user["content"] + " " + new_content
        return new_content

    def sync_after_interruption(self, response_heard: str | None, update_fn):
        self._trim_last_assistant(self._messages, response_heard, update_fn)

    def sync_interim_after_interruption(self, response_heard: str | None, update_fn):
        self._trim_last_assistant(self._interim, response_heard, update_fn)

    def sync_turn_after_interruption(self, turn_id: int | None, response_heard: str | None, update_fn):
        self._trim_assistant_for_turn(self._messages, turn_id, response_heard, update_fn)

    def sync_interim_turn_after_interruption(self, turn_id: int | None, response_heard: str | None, update_fn):
        self._trim_assistant_for_turn(self._interim, turn_id, response_heard, update_fn)

    def sync_response_after_interruption(self, response_uid: str | None, response_heard: str | None, update_fn):
        self._trim_assistant_for_response(self._messages, response_uid, response_heard, update_fn)

    def sync_interim_response_after_interruption(self, response_uid: str | None, response_heard: str | None, update_fn):
        self._trim_assistant_for_response(self._interim, response_uid, response_heard, update_fn)

    @staticmethod
    def _trim_last_assistant(msgs: list[dict], response_heard: str | None, update_fn):
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i]["role"] == ChatRole.ASSISTANT:
                ConversationHistory._trim_assistant_at_index(msgs, i, response_heard, update_fn)
                break

    @staticmethod
    def _trim_assistant_for_turn(msgs: list[dict], turn_id: int | None, response_heard: str | None, update_fn):
        if turn_id is None:
            logger.info("Skipping assistant trim because turn_id is None; refusing to trim an older assistant blindly")
            return

        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i]["role"] == ChatRole.ASSISTANT and msgs[i].get("turn_id") == turn_id:
                ConversationHistory._trim_assistant_at_index(msgs, i, response_heard, update_fn)
                return

        logger.info(
            f"No assistant message found for turn_id={turn_id}; skipping trim to avoid removing a different assistant turn"
        )

    @staticmethod
    def _trim_assistant_for_response(msgs: list[dict], response_uid: str | None, response_heard: str | None, update_fn):
        if response_uid is None:
            logger.info("Skipping assistant trim because response_uid is None; refusing to trim an older assistant blindly")
            return

        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i]["role"] == ChatRole.ASSISTANT and msgs[i].get("response_uid") == response_uid:
                ConversationHistory._trim_assistant_at_index(msgs, i, response_heard, update_fn)
                return

        logger.info(
            f"No assistant message found for response_uid={response_uid}; skipping trim to avoid removing a different assistant turn"
        )

    @staticmethod
    def _trim_assistant_at_index(msgs: list[dict], index: int, response_heard: str | None, update_fn):
        original = msgs[index]["content"]
        if original is None:
            return
        updated = update_fn(original, response_heard)
        logger.info(
            f"Trimming assistant message. Original (last 10 chars): {str(original)[-10:]} | Updated: {updated[-10:] if updated else '<empty>'}"
        )
        if not updated or not updated.strip():
            has_tool_calls = bool(msgs[index].get("tool_calls"))
            logger.info(
                f"Removing assistant message (last 10 chars): {str(original)[-10:]} | has_tool_calls={has_tool_calls} from transcript"
            )
            msgs.pop(index)
            if has_tool_calls:
                while index < len(msgs) and msgs[index].get("role") == ChatRole.TOOL:
                    msgs.pop(index)
            else:
                while index < len(msgs) and msgs[index].get("role") in _UNHEARD_ROLES:
                    msgs.pop(index)
        else:
            msgs[index]["content"] = updated

    @property
    def messages(self) -> list[dict]:
        return self._messages

    @property
    def last_role(self) -> str | None:
        return self._messages[-1].get("role") if self._messages else None

    @property
    def last_content(self) -> str | None:
        return self._messages[-1].get("content") if self._messages else None

    def get_copy(self) -> list[dict]:
        self._sanitize_tool_messages(self._messages)
        return copy.deepcopy([m for m in self._messages if not m.get("exclude_from_llm")])

    @staticmethod
    def _sanitize_tool_messages(msgs: list[dict]):
        """Repair assistant/tool-call pairs before sending history to OpenAI.

        OpenAI requires every assistant message with ``tool_calls`` to be
        followed by tool messages for each call id, and every tool message to
        have a matching assistant parent. Interruptions and end-of-call cleanup
        can leave either side behind, so normalize both directions here.
        """
        sanitized = []
        i = 0
        while i < len(msgs):
            msg = msgs[i]
            role = msg.get("role")

            if role in (ChatRole.ASSISTANT, ChatRole.ASSISTANT.value) and msg.get("tool_calls"):
                expected_ids = [
                    tc.get("id", "")
                    for tc in msg.get("tool_calls", [])
                    if isinstance(tc, dict) and tc.get("id")
                ]
                expected_set = set(expected_ids)

                j = i + 1
                following_tools = []
                while j < len(msgs) and msgs[j].get("role") in (ChatRole.TOOL, ChatRole.TOOL.value):
                    following_tools.append(msgs[j])
                    j += 1

                matched_tools = []
                matched_ids = set()
                for tool_msg in following_tools:
                    tool_call_id = tool_msg.get("tool_call_id", "")
                    if tool_call_id in expected_set and tool_call_id not in matched_ids:
                        matched_tools.append(tool_msg)
                        matched_ids.add(tool_call_id)
                    else:
                        logger.warning(
                            f"Removing orphaned tool message after assistant at index {i} "
                            f"(tool_call_id={tool_call_id}): no matching pending tool_call"
                        )

                if expected_set and expected_set.issubset(matched_ids):
                    sanitized.append(msg)
                    sanitized.extend(matched_tools)
                else:
                    missing_ids = expected_set - matched_ids
                    logger.warning(
                        f"Removing incomplete assistant tool_calls at index {i}: "
                        f"missing tool responses for {sorted(missing_ids)}"
                    )
                    msg.pop("tool_calls", None)
                    if msg.get("content") and str(msg.get("content")).strip():
                        sanitized.append(msg)
                    else:
                        logger.warning(f"Removing assistant tool-call placeholder at index {i}: no content to keep")
                    for tool_msg in following_tools:
                        logger.warning(
                            f"Removing orphaned tool message at index {i} "
                            f"(tool_call_id={tool_msg.get('tool_call_id', '')}): assistant tool_calls incomplete"
                        )

                i = j
                continue

            if role in (ChatRole.TOOL, ChatRole.TOOL.value):
                logger.warning(
                    f"Removing orphaned tool message at index {i} "
                    f"(tool_call_id={msg.get('tool_call_id', '')}): no matching assistant with tool_calls"
                )
                i += 1
                continue

            sanitized.append(msg)
            i += 1

        msgs[:] = sanitized

    def is_duplicate_user(self, content: str) -> bool:
        if not self._messages:
            return False
        last = self._messages[-1]
        return last.get("role") == ChatRole.USER and last.get("content", "").strip() == content.strip()

    def __len__(self) -> int:
        return len(self._messages)

    def sync_interim(self, messages: list[dict] | None = None):
        self._interim = copy.deepcopy(messages if messages is not None else self._messages)

    @property
    def interim(self) -> list[dict]:
        return self._interim
