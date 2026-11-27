"""Gemini: every system message must reach system_instruction, not just the last one."""

from bolna.llms.gemini_llm import GeminiLLM

PERSONA = "You are Priya from Acme Clinic. Only book appointments. Never discuss prices."
RAG = "Knowledge base for the latest user message:\nCleaning: Rs 1500\n\nUse this information naturally."


def _llm():
    return GeminiLLM(model="gemini-2.5-flash", llm_key="test-key")


def test_all_system_messages_kept_in_order():
    messages = [
        {"role": "system", "content": PERSONA},
        {"role": "user", "content": "What does a cleaning cost?"},
        {"role": "system", "content": RAG},
    ]

    system_instruction, history = _llm()._prepare_history(messages)

    assert PERSONA in system_instruction
    assert RAG in system_instruction
    assert system_instruction.index(PERSONA) < system_instruction.index(RAG)
    assert len(history) == 1
    assert history[0].role == "user"
