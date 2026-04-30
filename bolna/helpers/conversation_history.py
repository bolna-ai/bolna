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
        """Remove orphaned tool-role messages that have no preceding assistant with tool_calls.

        OpenAI requires every message with role='tool' to follow an assistant
        message that contains a matching 'tool_calls' entry.  Interruptions can
        cause the assistant message to be popped/trimmed while the tool result
        stays, producing an invalid sequence.  This method walks the list and
        removes any tool messages whose pairing is broken.
        """
        i = 0
        while i < len(msgs):
            if msgs[i].get("role") == ChatRole.TOOL:
                # Walk backwards to find the nearest assistant with tool_calls
                tool_call_id = msgs[i].get("tool_call_id", "")
                found_parent = False
                for j in range(i - 1, -1, -1):
                    if msgs[j].get("role") == ChatRole.ASSISTANT and msgs[j].get("tool_calls"):
                        # Check if this assistant message contains a matching tool_call id
                        for tc in msgs[j]["tool_calls"]:
                            tc_id = tc.get("id", "") if isinstance(tc, dict) else ""
                            if tc_id == tool_call_id:
                                found_parent = True
                                break
                        break  # stop at the nearest assistant with tool_calls
                    if msgs[j].get("role") == ChatRole.USER:
                        break  # crossed a user boundary, no parent
                if not found_parent:
                    logger.warning(
                        f"Removing orphaned tool message at index {i} "
                        f"(tool_call_id={tool_call_id}): no matching assistant with tool_calls"
                    )
                    msgs.pop(i)
                    continue
            i += 1

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
