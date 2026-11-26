"""Gemini gets every system message, not only the last one.

Gemini takes a single system_instruction, while Bolna can send several system messages in
one request (the node prompt plus RAG context or an event hint from the graph agent).
Keeping only the last one silently dropped the agent's whole prompt.
"""

from bolna.llms.gemini_llm import GeminiLLM

PROMPT = "You are Priya from Acme Clinic. Only book appointments. Never discuss prices."
RAG = "Knowledge base for the latest user message:\nCleaning: Rs 1500\n\nUse this information naturally."


def _llm():
    return GeminiLLM(model="gemini-2.5-flash", llm_key="k")


def test_all_system_messages_are_kept_in_order():
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": "What does a cleaning cost?"},
        {"role": "system", "content": RAG},
    ]
    system_instruction, _ = _llm()._prepare_history(messages)

    assert system_instruction == f"{PROMPT}\n\n{RAG}"


def test_single_system_message_is_unchanged():
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": "hi"}]
    system_instruction, _ = _llm()._prepare_history(messages)

    assert system_instruction == PROMPT


def test_empty_system_messages_are_skipped():
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "system", "content": ""},
        {"role": "system", "content": None},
        {"role": "user", "content": "hi"},
    ]
    system_instruction, _ = _llm()._prepare_history(messages)

    assert system_instruction == PROMPT


def test_no_system_message_gives_empty_instruction():
    system_instruction, history = _llm()._prepare_history([{"role": "user", "content": "hi"}])

    assert system_instruction == ""
    assert len(history) == 1


def test_system_messages_stay_out_of_history():
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": "What does a cleaning cost?"},
        {"role": "system", "content": RAG},
    ]
    _, history = _llm()._prepare_history(messages)

    assert len(history) == 1
    assert history[0].role == "user"
    assert history[0].parts[0].text == "What does a cleaning cost?"
