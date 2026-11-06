"""Earlier turns as prompt text, for the first replies after a language switch.

Some models keep the language of the chat history over every language instruction, so after a switch
they reply in the old language. Sending the earlier turns as a transcript inside the system prompt,
with only the caller's latest turn as a message, makes them follow the switch.
"""

from bolna.enums import ChatRole

TRANSCRIPT_HEADING = "## Transcript:\n"
# Headings that open the active language order. The transcript goes above them so the order stays last.
LANGUAGE_ORDER_HEADINGS = ("\n\n## Language note:", "\n\nLANGUAGE GUIDELINES")
# Without it, models given the history as text restart the script (greet again, re-ask who they are speaking to).
CONTINUE_LINE = (
    "The conversation so far is in the Transcript above. Continue it from exactly where it left off: "
    "do not greet again, re-introduce yourself or repeat a step already done."
)
SPEAKERS = {ChatRole.ASSISTANT: "Agent", ChatRole.USER: "Caller"}


def transcript_lines(messages: list) -> list[str]:
    """One line per spoken turn, agent tool call and tool result."""
    lines = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == ChatRole.TOOL:
            lines.append(f"Tool result: {content}")
            continue
        if isinstance(content, str) and content.strip():
            lines.append(f"{SPEAKERS.get(role, 'Context')}: {content.strip()}")
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            lines.append(f"Agent called: {function.get('name', '')}({function.get('arguments') or ''})")
    return lines


def with_prompt_transcript(messages: list) -> list:
    """Move every turn before the caller's latest one into the system prompt as a transcript.

    The caller's latest turn and anything after it (tool calls and results, a trailing knowledge-base
    message) stay messages. Unchanged when there is no text system prompt or no text caller turn.
    Returns a new list; the input is not modified.
    """
    if not messages or messages[0].get("role") != ChatRole.SYSTEM or not isinstance(messages[0].get("content"), str):
        return messages
    last_user = next((i for i in range(len(messages) - 1, 0, -1) if messages[i].get("role") == ChatRole.USER), None)
    if last_user is None or not isinstance(messages[last_user].get("content"), str):
        return messages
    transcript = "\n".join([*transcript_lines(messages[1:last_user]), CONTINUE_LINE]) + "\n"
    prompt = messages[0]["content"]
    if TRANSCRIPT_HEADING in prompt:
        prompt = prompt.replace(TRANSCRIPT_HEADING, f"{TRANSCRIPT_HEADING}{transcript}", 1)
    else:
        order_at = min((i for i in (prompt.find(h) for h in LANGUAGE_ORDER_HEADINGS) if i != -1), default=len(prompt))
        prompt = f"{prompt[:order_at]}\n\n{TRANSCRIPT_HEADING}{transcript}{prompt[order_at:]}"
    return [{**messages[0], "content": prompt}, *messages[last_user:]]
