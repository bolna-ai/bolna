"""
Provider-specific SSML tag mapping and prompt injection.

Strategy: The LLM outputs simple TEXT MARKERS (e.g. [pause:0.5s], [slow]...[/slow])
which are easy for LLMs to generate. A post-processing step then converts these
markers into provider-specific XML tags before the text reaches the TTS engine.

This avoids the well-known problem of LLMs being reluctant to output literal XML.
"""

import re
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

# ---------------------------------------------------------------------------
# Marker patterns — provider support varies, see _PROVIDER_CONVERTERS
# ---------------------------------------------------------------------------
# ElevenLabs: [pause] only
# Cartesia:   [pause], [slow], [fast], [loud], [soft], [spell]

_PAUSE_RE = re.compile(r'\[pause(?::(\d+(?:\.\d+)?)\s*(s|ms))?\]')
_WRAP_RE = re.compile(r'\[(slow|fast|loud|soft|spell)\](.*?)\[/\1\]', re.DOTALL)

# ---------------------------------------------------------------------------
# Provider-specific converters — each provider handles ALL its markers
# ---------------------------------------------------------------------------

_ALL_MARKERS = re.compile(r'\[/?(?:pause|slow|fast|loud|soft|spell)(?::[^\]]+)?\]')

# Safety net: catch raw <break> XML the LLM might generate despite instructions
_RAW_BREAK_RE = re.compile(r'<\s*break\s+time="([^"]+)"\s*/?>')

def _normalize_raw_breaks(text: str) -> str:
    """Convert any raw <break> XML back to [pause] markers before conversion."""
    return _RAW_BREAK_RE.sub(lambda m: f'[pause:{m.group(1)}]', text)

def _make_break_tag(match, default_duration: str) -> str:
    """Build <break> tag from a [pause] regex match."""
    duration_val = match.group(1)
    duration_unit = match.group(2)
    if duration_val and duration_unit:
        return f'<break time="{duration_val}{duration_unit}" />'
    return f'<break time="{default_duration}" />'


def _convert_elevenlabs(text: str) -> str:
    """ElevenLabs: only <break> is supported via SSML."""
    text = _normalize_raw_breaks(text)
    text = _PAUSE_RE.sub(lambda m: _make_break_tag(m, "0.5s"), text)
    return text


def _convert_cartesia(text: str) -> str:
    """Cartesia: break, speed, volume, spell."""
    text = _normalize_raw_breaks(text)
    text = _PAUSE_RE.sub(lambda m: _make_break_tag(m, "0.5s"), text)

    def _wrap(match) -> str:
        tag = match.group(1)
        content = match.group(2)
        if tag == "slow":
            return f'<speed ratio="0.8"/>{content}'
        elif tag == "fast":
            return f'<speed ratio="1.3"/>{content}'
        elif tag == "loud":
            return f'<volume ratio="1.5"/>{content}'
        elif tag == "soft":
            return f'<volume ratio="0.7"/>{content}'
        elif tag == "spell":
            return f'<spell>{content}</spell>'
        return content

    text = _WRAP_RE.sub(_wrap, text)
    return text


_PROVIDER_CONVERTERS = {
    "elevenlabs": _convert_elevenlabs,
    "cartesia": _convert_cartesia,
}

PROVIDER_SUPPORTED_MARKERS = {
    "elevenlabs": {"pause"},
    "cartesia": {"pause", "slow", "fast", "loud", "soft", "spell"},
}

PROVIDER_ALLOWED_TAGS = {
    "elevenlabs": {"break"},
    "cartesia": {"break", "speed", "volume", "spell", "emotion"},
}


def convert_markers_to_ssml(text: str, provider: str) -> str:
    """
    Convert LLM-generated text markers to provider-specific SSML tags.

    Call this BEFORE sending text to the TTS engine.
    """
    if not text:
        return text

    original = text

    converter = _PROVIDER_CONVERTERS.get(provider)
    if converter:
        text = converter(text)

    # Clean up any remaining unconverted markers (safety net)
    text = _ALL_MARKERS.sub('', text)

    # Strip partial/split markers from streaming
    text = re.sub(r'\[pause(?::[^\]]*)?$', '', text)
    text = re.sub(r'^[^[]*?\d+\s*(?:s|ms)\]', '', text)

    if text != original:
        logger.info(f"SSML markers converted for {provider}: {original[:100]} → {text[:100]}")

    return text


# ---------------------------------------------------------------------------
# Prompt block builder — instructs LLM to use simple text markers
# ---------------------------------------------------------------------------

# Per-provider: compact marker table and example
_PROVIDER_PROMPT_CONFIG = {
    "elevenlabs": [
        {
            "syntax": "[pause] / [pause:Xs]",
            "usage": "Pause (max 3s). ALWAYS use between fee/cost figures, after greetings, between list items (dates, steps, features), and before follow-up questions. When listing 2+ facts, EVERY fact must be separated by [pause].",
            "correct": "Fee is 1,150,000. [pause:0.5s] Admission fee is 76,000. [pause:0.5s] Would you like to know more?",
            "wrong": "Fee is 1,150,000. Admission fee is 76,000. Would you like to know more?",
        },
    ],
    "cartesia": [
        {
            "syntax": "[pause] / [pause:Xs]",
            "usage": "Pause. ALWAYS use between fee/cost figures, after greetings, between list items (dates, steps, features), and before follow-up questions. When listing 2+ facts, EVERY fact must be separated by [pause].",
            "correct": "Fee is eleven lakh. [pause:0.5s] Admission fee is seventy-six thousand. [pause:0.5s] Want to know more?",
            "wrong": "Fee is eleven lakh. Admission fee is seventy-six thousand. Want to know more?",
        },
        {
            "syntax": "[slow]text[/slow]",
            "usage": "Slower speech. ALWAYS wrap fees, costs, deadlines, and important numbers in [slow] so they are spoken clearly.",
            "correct": "Fee is [slow]eleven lakh for two years[/slow]. Deadline is [slow]April 30, 2025[/slow].",
            "wrong": "Fee is eleven lakh for two years. Deadline is April 30, 2025.",
        },
        {
            "syntax": "[fast]text[/fast]",
            "usage": "Faster speech. Use for routine transitions, filler phrases, or less important details.",
            "correct": "[fast]Moving on,[/fast] let me tell you about hostel fees.",
            "wrong": "Moving on, let me tell you about hostel fees.",
        },
        {
            "syntax": "[spell]text[/spell]",
            "usage": "Spell out letter by letter. Use for IDs, course codes, emails, phone numbers. Content inside [spell] MUST be raw characters/digits exactly as written — NEVER convert digits to words.",
            "correct": "Email us at [spell]admissions@bit.edu[/spell]. Call [spell]9876543210[/spell].",
            "wrong": "Call [spell]nine eight seven six five four three two one zero[/spell]. Email A D M I S S I O N S at B I T dot E D U.",
        },
    ],
}


def build_ssml_prompt_block(provider: str) -> str | None:
    """Build compact speech-marker instructions for the LLM system prompt."""
    markers = _PROVIDER_PROMPT_CONFIG.get(provider)
    if not markers:
        return None

    marker_blocks = []
    for m in markers:
        marker_blocks.append(
            f"{m['syntax']} — {m['usage']}\n"
            f"  RIGHT: {m['correct']}\n"
            f"  WRONG: {m['wrong']}"
        )

    return (
        "### Speech Markers — YOU MUST FOLLOW THESE\n"
        "Your output is spoken aloud via TTS. You MUST use the markers below. Failing to use them produces bad audio.\n\n"
        "ABSOLUTE RULES — VIOLATION OF ANY RULE IS A CRITICAL ERROR:\n"
        "1. You MUST use ONLY [bracket] markers listed below. Not using them is an error.\n"
        "2. ZERO XML/HTML tags allowed. NEVER write <break>, <speak>, <prosody>, or ANY tag with < >.\n"
        "   Writing <break time=\"0.5s\"/> is WRONG. Writing [pause:0.5s] is RIGHT.\n"
        "3. No markdown (no **, ##, bullets). No double-spaces. No invented markers.\n"
        "4. Every response with 2+ facts MUST have [pause] between them. No exceptions.\n\n"
        + "\n\n".join(marker_blocks)
    )
