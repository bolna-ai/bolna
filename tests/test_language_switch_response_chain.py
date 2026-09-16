"""A language switch must reach the model on the OpenAI Responses API path.

Once `previous_response_id` is set, `_extract_new_input` sends `instructions=""` and the
system prompt lives on OpenAI's server-side chain. Rewriting the local system message is
then invisible to the model, so the agent flips voice and transcriber to the new language
while the LLM keeps reasoning against the old-language prompt for the rest of the call
(prod: aedd97c3, 8a9d82f0 — Marathi voice quoting Hindi-only scripted lines).
"""

import inspect
from unittest.mock import MagicMock

from bolna.agent_manager.task_manager import TaskManager
from bolna.llms.openai_base import OpenAICompatibleLLM

APPLY = TaskManager._TaskManager__apply_language_directive


def _make_tm(multilingual_prompts=None):
    tm = MagicMock()
    tm.multilingual_prompts = multilingual_prompts if multilingual_prompts is not None else {}
    tm.system_prompt = {"role": "system", "content": "OLD HINDI PROMPT"}
    tm.conversation_history = MagicMock()
    tm._TaskManager__language_directive = TaskManager._TaskManager__language_directive.__get__(tm, TaskManager)
    tm._invalidate_response_chain = MagicMock()
    return tm


class TestDirectiveDropsTheChain:
    def test_switch_with_a_variant_invalidates(self):
        tm = _make_tm({"mr": "MARATHI PROMPT"})
        APPLY(tm, "mr")
        tm._invalidate_response_chain.assert_called_once_with()

    def test_switch_without_a_variant_still_invalidates(self):
        # No per-language prompt, but the directive text itself changed language.
        tm = _make_tm({})
        APPLY(tm, "mr")
        tm._invalidate_response_chain.assert_called_once_with()

    def test_new_prompt_reaches_local_state_too(self):
        tm = _make_tm({"mr": "MARATHI PROMPT"})
        APPLY(tm, "mr")
        assert tm.system_prompt["content"].startswith("MARATHI PROMPT")
        assert "Marathi" in tm.system_prompt["content"]
        tm.conversation_history.update_system_prompt.assert_called_once_with(tm.system_prompt["content"])

    def test_invalidation_happens_after_the_rewrite(self):
        # Dropping the chain before the rewrite would resend the prompt being replaced.
        tm = _make_tm({"mr": "MARATHI PROMPT"})
        seen = {}
        tm._invalidate_response_chain = MagicMock(side_effect=lambda: seen.update(at_call=tm.system_prompt["content"]))
        APPLY(tm, "mr")
        assert seen["at_call"].startswith("MARATHI PROMPT")


class TestCallSites:
    def test_apply_language_directive_invalidates(self):
        src = inspect.getsource(TaskManager._TaskManager__apply_language_directive)
        assert "_invalidate_response_chain" in src, (
            "a language switch must drop the Responses chain or the model never sees the new prompt"
        )


class TestChainedRequestOmitsSystemPrompt:
    """Why the fix is needed at all — guards the assumption it rests on."""

    def test_chained_input_sends_no_instructions(self):
        llm = MagicMock(spec=OpenAICompatibleLLM)
        llm.previous_response_id = "resp_abc"
        messages = [
            {"role": "system", "content": "MARATHI PROMPT"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "next"},
        ]
        instructions, items = OpenAICompatibleLLM._extract_new_input(llm, messages)
        assert instructions == ""
        assert not any("MARATHI PROMPT" in str(i) for i in items)

    def test_unchained_input_carries_the_system_prompt(self):
        llm = MagicMock(spec=OpenAICompatibleLLM)
        llm.previous_response_id = None
        llm._interruption_hint = None
        llm._pending_call_ids = set()
        messages = [
            {"role": "system", "content": "MARATHI PROMPT"},
            {"role": "user", "content": "hello"},
        ]
        # The system prompt travels as a role=system input item, not as `instructions`.
        _, items = OpenAICompatibleLLM._build_responses_input(llm, messages)
        assert any("MARATHI PROMPT" in str(i) for i in items)
