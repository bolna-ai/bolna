import copy
import pytest
from bolna.helpers.conversation_history import ConversationHistory


class TestBasicCRUD:
    def test_append_user(self):
        ch = ConversationHistory()
        ch.append_user("hello")
        assert ch.messages == [{"role": "user", "content": "hello"}]

    def test_append_assistant(self):
        ch = ConversationHistory()
        ch.append_assistant("hi there")
        assert ch.messages == [{"role": "assistant", "content": "hi there"}]

    def test_append_assistant_with_tool_calls(self):
        tool_calls = [{"id": "call_1", "function": {"name": "get_order"}}]
        ch = ConversationHistory()
        ch.append_assistant("checking", tool_calls=tool_calls)
        assert ch.messages[0]["tool_calls"] == tool_calls
        assert ch.messages[0]["content"] == "checking"

    def test_append_tool_result(self):
        ch = ConversationHistory()
        ch.append_tool_result("call_1", "order shipped")
        assert ch.messages == [{"role": "tool", "tool_call_id": "call_1", "content": "order shipped"}]

    def test_setup_system_prompt_empty_history(self):
        ch = ConversationHistory()
        ch.setup_system_prompt({"role": "system", "content": "You are helpful"})
        assert len(ch) == 1
        assert ch.messages[0]["content"] == "You are helpful"

    def test_setup_system_prompt_with_existing_history(self):
        ch = ConversationHistory([{"role": "user", "content": "hi"}])
        ch.setup_system_prompt({"role": "system", "content": "prompt"})
        assert len(ch) == 2
        assert ch.messages[0]["content"] == "prompt"
        assert ch.messages[1]["content"] == "hi"

    def test_setup_system_prompt_empty_content_noop(self):
        ch = ConversationHistory()
        ch.setup_system_prompt({"role": "system", "content": ""})
        assert len(ch) == 0

    def test_setup_welcome_message(self):
        ch = ConversationHistory()
        ch.setup_system_prompt({"role": "system", "content": "prompt"}, welcome_message="Hello!")
        assert len(ch) == 2
        assert ch.messages[1] == {"role": "assistant", "content": "Hello!"}

    def test_update_system_prompt(self):
        ch = ConversationHistory([{"role": "system", "content": "old"}])
        ch.update_system_prompt("new")
        assert ch.messages[0]["content"] == "new"

    def test_update_system_prompt_noop_when_first_not_system(self):
        ch = ConversationHistory([{"role": "user", "content": "hi"}])
        ch.update_system_prompt("new system")
        assert ch.messages[0]["content"] == "hi"

    def test_update_system_prompt_noop_when_empty(self):
        ch = ConversationHistory()
        ch.update_system_prompt("new system")
        assert len(ch) == 0

    def test_update_welcome_message(self):
        ch = ConversationHistory([
            {"role": "system", "content": "prompt"},
            {"role": "assistant", "content": "old welcome"}
        ])
        ch.update_welcome_message("new welcome")
        assert ch.messages[1]["content"] == "new welcome"

    def test_update_welcome_message_noop_when_second_not_assistant(self):
        ch = ConversationHistory([
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "hi"}
        ])
        ch.update_welcome_message("new welcome")
        assert ch.messages[1]["content"] == "hi"

    def test_update_welcome_message_noop_when_only_one_message(self):
        ch = ConversationHistory([{"role": "system", "content": "prompt"}])
        ch.update_welcome_message("new welcome")
        assert len(ch) == 1
        assert ch.messages[0]["content"] == "prompt"

    def test_get_copy_is_deep(self):
        ch = ConversationHistory([{"role": "user", "content": "hi"}])
        copy_msgs = ch.get_copy()
        copy_msgs[0]["content"] = "modified"
        assert ch.messages[0]["content"] == "hi"

    def test_duplicate_user_detection(self):
        ch = ConversationHistory([{"role": "user", "content": "hello"}])
        assert ch.is_duplicate_user("hello") is True
        assert ch.is_duplicate_user("  hello  ") is True
        assert ch.is_duplicate_user("bye") is False

    def test_duplicate_user_empty_history(self):
        ch = ConversationHistory()
        assert ch.is_duplicate_user("hello") is False

    def test_last_role_and_content(self):
        ch = ConversationHistory([{"role": "user", "content": "hi"}])
        assert ch.last_role == "user"
        assert ch.last_content == "hi"

    def test_last_role_empty(self):
        ch = ConversationHistory()
        assert ch.last_role is None
        assert ch.last_content is None

    def test_len(self):
        ch = ConversationHistory([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
        assert len(ch) == 2


class TestPopAndMerge:
    def test_pop_unheard_single_assistant(self):
        ch = ConversationHistory([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        popped = ch.pop_unheard_responses()
        assert len(ch) == 1
        assert ch.messages[0]["role"] == "user"
        assert len(popped) == 1

    def test_pop_unheard_assistant_and_tool(self):
        ch = ConversationHistory([
            {"role": "user", "content": "check"},
            {"role": "assistant", "content": "let me check", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
            {"role": "assistant", "content": "your order shipped"},
        ])
        popped = ch.pop_unheard_responses()
        assert len(ch) == 1
        assert ch.messages[0]["role"] == "user"
        assert len(popped) == 3

    def test_pop_unheard_filler_and_response(self):
        ch = ConversationHistory([
            {"role": "user", "content": "check"},
            {"role": "assistant", "content": "hmm"},
            {"role": "assistant", "content": "real response"},
        ])
        ch.pop_unheard_responses()
        assert len(ch) == 1

    def test_pop_stops_at_user(self):
        ch = ConversationHistory([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ])
        ch.pop_unheard_responses()
        assert len(ch) == 4
        assert ch.messages[-1]["role"] == "user"

    def test_pop_empty_history(self):
        ch = ConversationHistory()
        popped = ch.pop_unheard_responses()
        assert popped == []
        assert len(ch) == 0

    def test_pop_history_only_system(self):
        ch = ConversationHistory([{"role": "system", "content": "sys"}])
        ch.pop_unheard_responses()
        assert len(ch) == 1

    def test_merge_basic(self):
        ch = ConversationHistory([{"role": "user", "content": "part A"}])
        merged = ch.pop_and_merge_user("part B")
        assert merged == "part A part B"
        assert len(ch) == 0

    def test_merge_preserves_earlier_turns(self):
        ch = ConversationHistory([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "part A"},
        ])
        merged = ch.pop_and_merge_user("part B")
        assert merged == "part A part B"
        assert len(ch) == 3

    def test_triple_merge(self):
        ch = ConversationHistory([{"role": "user", "content": "A"}])
        merged = ch.pop_and_merge_user("B")
        ch.append_user(merged)
        merged = ch.pop_and_merge_user("C")
        assert merged == "A B C"

    def test_merge_no_previous_user(self):
        ch = ConversationHistory([{"role": "system", "content": "sys"}])
        result = ch.pop_and_merge_user("new message")
        assert result == "new message"


class TestInterruptionSync:
    @staticmethod
    def _trim_fn(original, heard):
        if heard and original and heard in original:
            idx = original.find(heard)
            return original[:idx + len(heard)]
        return heard if heard else ""

    def test_sync_trims_to_heard_text(self):
        ch = ConversationHistory([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello how are you"},
        ])
        ch.sync_after_interruption("Hello how", self._trim_fn)
        assert ch.messages[-1]["content"] == "Hello how"

    def test_sync_removes_empty_assistant(self):
        ch = ConversationHistory([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello"},
        ])
        ch.sync_after_interruption("", self._trim_fn)
        assert len(ch) == 1

    def test_sync_with_none_content_skips(self):
        ch = ConversationHistory([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None},
        ])
        ch.sync_after_interruption("test", self._trim_fn)
        assert len(ch) == 2  # nothing changed, None content skipped

    def test_sync_updates_interim_too(self):
        ch = ConversationHistory([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello how are you"},
        ])
        ch.sync_interim()
        ch.sync_after_interruption("Hello how", self._trim_fn)
        ch.sync_interim_after_interruption("Hello how", self._trim_fn)
        assert ch.messages[-1]["content"] == "Hello how"
        assert ch.interim[-1]["content"] == "Hello how"

    def test_sync_only_touches_last_assistant(self):
        ch = ConversationHistory([
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1 full response"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2 Hello how are you"},
        ])
        ch.sync_after_interruption("a2 Hello", self._trim_fn)
        assert ch.messages[1]["content"] == "a1 full response"
        assert ch.messages[3]["content"] == "a2 Hello"


class TestInterimSync:
    def test_sync_interim_deep_copies(self):
        ch = ConversationHistory([{"role": "user", "content": "hi"}])
        ch.sync_interim()
        ch.interim[0]["content"] = "modified"
        assert ch.messages[0]["content"] == "hi"

    def test_sync_interim_from_messages_list(self):
        ch = ConversationHistory()
        external_msgs = [{"role": "user", "content": "ext"}]
        ch.sync_interim(external_msgs)
        assert ch.interim == [{"role": "user", "content": "ext"}]
        external_msgs[0]["content"] = "changed"
        assert ch.interim[0]["content"] == "ext"

    def test_interim_independent_of_main(self):
        ch = ConversationHistory([{"role": "user", "content": "hi"}])
        ch.sync_interim()
        ch.append_user("new")
        assert len(ch.interim) == 1
        assert len(ch) == 2


class TestFullConversationFlows:
    def test_five_turn_normal_conversation(self):
        ch = ConversationHistory([{"role": "system", "content": "sys"}])
        for i in range(5):
            ch.append_user(f"user msg {i}")
            ch.append_assistant(f"assistant reply {i}")
        assert len(ch) == 11
        roles = [m["role"] for m in ch.messages[1:]]
        expected = ["user", "assistant"] * 5
        assert roles == expected

    def test_normal_then_merge_then_normal(self):
        ch = ConversationHistory()
        # Turn 1 normal
        ch.append_user("hi")
        ch.append_assistant("hello!")
        # Turn 2 split speech
        ch.append_user("yes madam")
        ch.append_assistant("unheard")
        ch.pop_unheard_responses()
        merged = ch.pop_and_merge_user("I am prem")
        ch.append_user(merged)
        # Turn 2 response
        ch.append_assistant("Hello Prem!")
        # Turn 3 normal
        ch.append_user("check order")
        ch.append_assistant("Order shipped")
        roles = [m["role"] for m in ch.messages]
        assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]

    def test_function_call_in_history(self):
        ch = ConversationHistory()
        ch.append_user("check order")
        ch.append_assistant("Let me check", tool_calls=[{"id": "c1", "function": {"name": "get_order"}}])
        ch.append_tool_result("c1", "Order #789 shipped")
        ch.append_assistant("Your order #789 has been shipped.")
        assert len(ch) == 4
        assert ch.messages[1]["tool_calls"] is not None
        assert ch.messages[2]["role"] == "tool"

    def test_merge_pops_function_call_chain(self):
        ch = ConversationHistory([
            {"role": "user", "content": "check"},
            {"role": "assistant", "content": "checking", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
            {"role": "assistant", "content": "Your order shipped"},
        ])
        ch.pop_unheard_responses()
        merged = ch.pop_and_merge_user("and order 456")
        ch.append_user(merged)
        assert len(ch) == 1
        assert ch.messages[0]["content"] == "check and order 456"

    def test_get_copy_removes_orphaned_tool_after_pop(self):
        """Reproduces the exact bug: interruption pops assistant with tool_calls,
        then function execution appends orphaned tool result.  get_copy() must
        sanitize this before sending to OpenAI."""
        ch = ConversationHistory([
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "welcome"},
            {"role": "user", "content": "send me the link"},
            {"role": "assistant", "content": "Sure, sending now"},
        ])
        # User interrupts → pop unheard assistant
        ch.pop_unheard_responses()
        merged = ch.pop_and_merge_user("ok ma'am ok ok")
        ch.append_user(merged)

        # New LLM turn returns tool_call, function executes, result appended
        ch.append_assistant(
            "Sorry, small issue sending the link",
            tool_calls=[{"id": "call_abc", "function": {"name": "send_link"}}],
        )
        ch.append_tool_result("call_abc", '{"status": "sent"}')

        # Simulate an interruption that pops the assistant+tool from this turn
        ch.pop_unheard_responses()
        # Now a race: function execution re-appends an orphaned tool result
        ch.append_tool_result("call_abc", '{"status": "sent"}')

        # get_copy must remove the orphaned tool message
        messages = ch.get_copy()
        roles = [m["role"] for m in messages]
        assert "tool" not in roles, (
            f"Orphaned tool message should have been removed, got roles: {roles}"
        )

    def test_get_copy_keeps_valid_tool_chain(self):
        """A valid assistant→tool chain should NOT be removed by sanitization."""
        ch = ConversationHistory()
        ch.append_user("check order")
        ch.append_assistant(
            "Let me check",
            tool_calls=[{"id": "c1", "function": {"name": "get_order"}}],
        )
        ch.append_tool_result("c1", "Order #789 shipped")
        ch.append_assistant("Your order #789 has been shipped.")

        messages = ch.get_copy()
        assert len(messages) == 4
        assert messages[2]["role"] == "tool"

    def test_get_copy_removes_tool_with_no_assistant_at_all(self):
        """Tool message with no preceding assistant anywhere → removed."""
        ch = ConversationHistory()
        ch.append_user("hi")
        ch.append_tool_result("orphan_id", "some result")

        messages = ch.get_copy()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_get_copy_removes_tool_with_mismatched_id(self):
        """Tool message where the preceding assistant has different tool_call ids."""
        ch = ConversationHistory()
        ch.append_user("hi")
        ch.append_assistant(
            "checking",
            tool_calls=[{"id": "call_A", "function": {"name": "fn_a"}}],
        )
        ch.append_tool_result("call_B", "wrong id result")  # mismatched id

        messages = ch.get_copy()
        # The tool message with call_B has no matching parent
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant"]

    def test_trim_last_assistant_removes_dependent_tools(self):
        """When _trim_last_assistant removes an assistant with tool_calls,
        it should also remove the following tool messages."""
        def _trim_fn(original, heard):
            return ""  # force removal

        ch = ConversationHistory([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "let me check",
             "tool_calls": [{"id": "c1", "function": {"name": "fn"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ])
        ch.sync_after_interruption("", _trim_fn)
        assert len(ch) == 1
        assert ch.messages[0]["role"] == "user"

    def test_trim_last_assistant_keeps_tools_when_content_survives(self):
        """If _trim_last_assistant trims but doesn't remove, tools stay."""
        def _trim_fn(original, heard):
            if heard and heard in original:
                return original[:original.find(heard) + len(heard)]
            return heard

        ch = ConversationHistory([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "let me check your order",
             "tool_calls": [{"id": "c1", "function": {"name": "fn"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ])
        ch.sync_after_interruption("let me check", _trim_fn)
        assert len(ch) == 3
        assert ch.messages[1]["content"] == "let me check"
        assert ch.messages[2]["role"] == "tool"

    def test_exact_production_scenario(self):
        """Reproduces the exact scenario from the production logs:
        1. Multi-turn conversation with tool calls
        2. User interrupts during LLM response
        3. sync_history trims assistant, pop_unheard_responses cleans up
        4. New LLM returns tool_call, function executes
        5. Race condition leaves orphaned tool message
        6. get_copy() must sanitize before sending to OpenAI"""
        ch = ConversationHistory()
        ch.setup_system_prompt({"role": "system", "content": "You are a sales agent"})

        # Turn 1: normal
        ch.append_user("hello")
        ch.append_assistant("Hi! How can I help?")

        # Turn 2: user asks for link, LLM calls function
        ch.append_user("send me the checkout link")
        ch.append_assistant(
            "Sure, let me send that",
            tool_calls=[{"id": "call_1", "function": {"name": "send_checkout_link"}}],
        )
        ch.append_tool_result("call_1", '{"status":"sent"}')
        ch.append_assistant("I've sent the checkout link to your WhatsApp!")

        # Turn 3: user says "not now", LLM responds
        ch.append_user("अभी नहीं बाद में purchase करेंगे")
        ch.append_assistant("No problem! I'll follow up later.")

        # User interrupts with "ok ma'am ok ok" → pop unheard
        ch.pop_unheard_responses()
        merged = ch.pop_and_merge_user("ok ma'am ok ok")
        ch.append_user(merged)

        # New LLM turn returns tool_call again
        ch.append_assistant(
            "Sorry, small issue sending the link, let me try again",
            tool_calls=[{"id": "call_2", "function": {"name": "send_checkout_link"}}],
        )
        ch.append_tool_result("call_2", '{"status":"sent"}')

        # Now simulate the race: interruption pops the assistant+tool
        ch.pop_unheard_responses()
        # But function execution already completed and appends orphaned tool
        ch.append_tool_result("call_2", '{"status":"sent"}')

        # get_copy must return clean messages
        messages = ch.get_copy()
        for i, msg in enumerate(messages):
            if msg.get("role") == "tool":
                # Verify this tool has a valid parent
                found = False
                for j in range(i - 1, -1, -1):
                    if msg_j := messages[j]:
                        if msg_j.get("role") == "assistant" and msg_j.get("tool_calls"):
                            for tc in msg_j["tool_calls"]:
                                if tc.get("id") == msg.get("tool_call_id"):
                                    found = True
                            break
                        if msg_j.get("role") == "user":
                            break
                assert found, (
                    f"Tool message at index {i} (tool_call_id={msg.get('tool_call_id')}) "
                    f"has no matching assistant with tool_calls"
                )

    def test_exclude_from_llm_filtered_in_get_copy(self):
        """Messages with exclude_from_llm=True should not appear in get_copy()
        (which feeds the LLM), but should remain in .messages (used for transcript)."""
        ch = ConversationHistory()
        ch.setup_system_prompt({"role": "system", "content": "You are a sales agent"})
        ch.append_user("hello")
        ch.append_assistant("Hi! How can I help?")
        ch.append_user("I want to export")
        ch.append_assistant("Great, what products?")
        # Simulate check_user_online message
        ch.messages.append({"role": "assistant", "content": "Hey, can you hear me?", "exclude_from_llm": True})
        ch.append_user("yes")

        # get_copy (for LLM) should NOT contain the connectivity check
        llm_messages = ch.get_copy()
        llm_contents = [m.get("content") for m in llm_messages]
        assert "Hey, can you hear me?" not in llm_contents
        # But user's response "yes" should be there
        assert "yes" in llm_contents
        # And normal assistant messages should be there
        assert "Great, what products?" in llm_contents

        # .messages (for transcript) should still contain everything
        all_contents = [m.get("content") for m in ch.messages]
        assert "Hey, can you hear me?" in all_contents
        assert "yes" in all_contents

    def test_exclude_from_llm_does_not_break_pop_unheard(self):
        """pop_unheard_responses should still pop exclude_from_llm messages
        since they are assistant messages at the end of history."""
        ch = ConversationHistory()
        ch.append_user("hello")
        ch.append_assistant("response")
        ch.messages.append({"role": "assistant", "content": "Hey, can you hear me?", "exclude_from_llm": True})
        popped = ch.pop_unheard_responses()
        assert len(popped) == 2  # both assistant messages popped
        assert ch.messages == [{"role": "user", "content": "hello"}]

    def test_exclude_from_llm_preserved_across_deep_copy(self):
        """Ensure the flag survives in .messages but is stripped from get_copy()."""
        ch = ConversationHistory()
        ch.append_user("hi")
        ch.messages.append({"role": "assistant", "content": "Can you hear me?", "exclude_from_llm": True})
        ch.append_user("yes")
        ch.append_assistant("Great, let's continue")

        copy1 = ch.get_copy()
        copy2 = ch.get_copy()
        # Both copies should exclude the flagged message
        for c in [copy1, copy2]:
            assert all(m.get("content") != "Can you hear me?" for m in c)
        # Original still has it
        assert any(m.get("content") == "Can you hear me?" for m in ch.messages)

    def test_exclude_from_llm_full_conversation_flow(self):
        """Reproduces the exact production bug: after check_user_online,
        the LLM should see conversation history WITHOUT the connectivity check
        so it continues naturally instead of restarting."""
        ch = ConversationHistory()
        ch.setup_system_prompt({"role": "system", "content": "You are Pooja from Alibaba"})
        ch.append_user("hello")
        ch.append_assistant("This is Pooja calling from Alibaba dot com")
        ch.append_user("yes")
        ch.append_assistant("Great. Are you open to starting export business?")
        ch.append_user("i am new to exporting")
        ch.append_assistant("Got it. Can we arrange a quick call with our consultant?")

        # Silence detected → check_user_online fires
        ch.messages.append({"role": "assistant", "content": "Hey, can you hear me?", "exclude_from_llm": True})

        # User responds
        ch.append_user("hello")

        # What the LLM sees (get_copy) should be a clean conversation
        # ending with the consultant question → user saying hello
        llm_messages = ch.get_copy()
        # Last assistant message in LLM view should be the consultant question
        last_assistant = [m for m in llm_messages if m["role"] == "assistant"][-1]
        assert last_assistant["content"] == "Got it. Can we arrange a quick call with our consultant?"
        # Last message should be user's "hello"
        assert llm_messages[-1] == {"role": "user", "content": "hello"}
        # No "Hey, can you hear me?" anywhere in LLM messages
        assert not any(m.get("exclude_from_llm") for m in llm_messages)

    def test_history_integrity_no_consecutive_same_role(self):
        ch = ConversationHistory([{"role": "system", "content": "sys"}])
        for i in range(5):
            ch.append_user(f"u{i}")
            ch.append_assistant(f"a{i}")
        # Simulate merge
        ch.append_user("split A")
        ch.append_assistant("unheard")
        ch.pop_unheard_responses()
        merged = ch.pop_and_merge_user("split B")
        ch.append_user(merged)
        ch.append_assistant("merged response")

        for i in range(1, len(ch)):
            prev = ch.messages[i-1]["role"]
            curr = ch.messages[i]["role"]
            if prev == "system":
                continue
            assert prev != curr or prev == "tool", f"Consecutive {prev} at index {i-1},{i}"
