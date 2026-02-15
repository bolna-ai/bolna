import copy
from bolna.constants import ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

_UNHEARD_ROLES = frozenset({ROLE_ASSISTANT, ROLE_TOOL})


class ConversationHistory:
    def __init__(self, initial_history: list[dict] | None = None):
        self._messages: list[dict] = initial_history or []
        self._interim: list[dict] = copy.deepcopy(self._messages)

    def setup_system_prompt(self, system_prompt: dict, welcome_message: str = ""):
        if system_prompt.get("content", ""):
            if not self._messages:
                self._messages = [system_prompt]
            else:
                self._messages = [system_prompt] + self._messages

        if welcome_message and len(self._messages) == 1:
            self._messages.append({"role": ROLE_ASSISTANT, "content": welcome_message})

        self._interim = copy.deepcopy(self._messages)

    def append_user(self, content: str):
        self._messages.append({"role": ROLE_USER, "content": content})

    def append_assistant(self, content: str, tool_calls: list | None = None):
        msg = {"role": ROLE_ASSISTANT, "content": content}
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        self._messages.append(msg)

    def append_tool_result(self, tool_call_id: str, content: str):
        self._messages.append({
            "role": ROLE_TOOL,
            "tool_call_id": tool_call_id,
            "content": content,
        })

    def update_system_prompt(self, content: str):
        if self._messages and self._messages[0].get("role") == ROLE_SYSTEM:
            self._messages[0]["content"] = content

    def update_welcome_message(self, content: str):
        if len(self._messages) >= 2 and self._messages[1].get("role") == ROLE_ASSISTANT:
            self._messages[1]["content"] = content

    def pop_unheard_responses(self) -> list[dict]:
        popped = []
        while self._messages and self._messages[-1].get("role") in _UNHEARD_ROLES:
            popped.append(self._messages.pop())
        return popped

    def pop_and_merge_user(self, new_content: str) -> str:
        if self._messages and self._messages[-1].get("role") == ROLE_USER:
            prev_user = self._messages.pop()
            return prev_user["content"] + " " + new_content
        return new_content

    def sync_after_interruption(self, response_heard: str | None, update_fn):
        self._trim_last_assistant(self._messages, response_heard, update_fn)

    def sync_interim_after_interruption(self, response_heard: str | None, update_fn):
        self._trim_last_assistant(self._interim, response_heard, update_fn)

    @staticmethod
    def _trim_last_assistant(msgs: list[dict], response_heard: str | None, update_fn):
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i]["role"] == ROLE_ASSISTANT:
                original = msgs[i]["content"]
                if original is None:
                    continue
                updated = update_fn(original, response_heard)
                if not updated or not updated.strip():
                    msgs.pop(i)
                else:
                    msgs[i]["content"] = updated
                break

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
        return copy.deepcopy(self._messages)

    def is_duplicate_user(self, content: str) -> bool:
        if not self._messages:
            return False
        last = self._messages[-1]
        return last.get("role") == ROLE_USER and last.get("content", "").strip() == content.strip()

    def __len__(self) -> int:
        return len(self._messages)

    def sync_interim(self, messages: list[dict] | None = None):
        self._interim = copy.deepcopy(messages if messages is not None else self._messages)

    @property
    def interim(self) -> list[dict]:
        return self._interim
