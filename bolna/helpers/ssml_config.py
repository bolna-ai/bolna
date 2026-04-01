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
# Marker patterns that the LLM outputs
# ---------------------------------------------------------------------------
# [pause]              → default 0.5s pause
# [pause:1.2s]         → specific duration (seconds)
# [pause:500ms]        → specific duration (milliseconds)
# [slow]text[/slow]    → slower speech for that segment
# [fast]text[/fast]    → faster speech for that segment
# [loud]text[/loud]    → louder speech for that segment
# [soft]text[/soft]    → softer speech for that segment
# [spell]ABC[/spell]   → spell out characters
# [number]42[/number]  → read as cardinal number
# [ordinal]3[/ordinal] → read as ordinal (3rd)
# [date]March 15[/date]→ read as date
# [phone]555-1234[/phone] → read as phone number

_PAUSE_RE = re.compile(r'\[pause(?::(\d+(?:\.\d+)?)\s*(s|ms))?\]')
_WRAP_RE = re.compile(r'\[(slow|fast|loud|soft|spell|number|ordinal|date|phone)\](.*?)\[/\1\]', re.DOTALL)

# ---------------------------------------------------------------------------
# Provider-specific conversion rules
# ---------------------------------------------------------------------------

def _default_pause_duration(provider: str) -> str:
    """Default pause when [pause] is used without a duration."""
    return {
        "elevenlabs": "0.5s",
        "cartesia": "0.5s",
    }.get(provider, "0.5s")


def _convert_pause(match, provider: str) -> str:
    """Convert [pause] or [pause:Xs] to provider-specific break tag."""
    duration_val = match.group(1)
    duration_unit = match.group(2)

    if duration_val and duration_unit:
        duration = f"{duration_val}{duration_unit}"
    else:
        duration = _default_pause_duration(provider)

    return f'<break time="{duration}" />'


_SAY_AS_MAP = {
    "spell": "characters",
    "number": "cardinal",
    "ordinal": "ordinal",
    "date": "date",
    "phone": "telephone",
}


def _convert_wrap_elevenlabs(match) -> str:
    """ElevenLabs v2: break + say-as types. Speed/volume not supported — stripped."""
    tag = match.group(1)
    content = match.group(2)
    interpret_as = _SAY_AS_MAP.get(tag)
    if interpret_as:
        return f'<say-as interpret-as="{interpret_as}">{content}</say-as>'
    return content


def _convert_wrap_cartesia(match) -> str:
    """Cartesia sonic-3: speed, volume, spell, break."""
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


_WRAP_CONVERTERS = {
    "elevenlabs": _convert_wrap_elevenlabs,
    "cartesia": _convert_wrap_cartesia,
}

# Which marker types each provider actually supports in TTS
PROVIDER_SUPPORTED_MARKERS = {
    "elevenlabs": {"pause", "spell", "number", "ordinal", "date", "phone"},
    "cartesia": {"pause", "slow", "fast", "loud", "soft", "spell"},
}

PROVIDER_ALLOWED_TAGS = {
    "elevenlabs": {"break", "say-as"},
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

    # 1. Convert [pause] / [pause:Xs] markers
    text = _PAUSE_RE.sub(lambda m: _convert_pause(m, provider), text)

    # 2. Convert [slow]...[/slow], [fast]...[/fast], etc.
    converter = _WRAP_CONVERTERS.get(provider, _convert_wrap_elevenlabs)
    text = _WRAP_RE.sub(converter, text)

    _ALL_MARKERS = r'pause|slow|fast|loud|soft|spell|number|ordinal|date|phone'

    # 3. Clean up any remaining unconverted markers (safety net)
    text = re.sub(rf'\[/?({_ALL_MARKERS})(?::[^\]]+)?\]', '', text)

    # 4. Strip partial/split markers from streaming (e.g. "[pause:0.5" or "s]")
    text = re.sub(r'\[pause(?::[^\]]*)?$', '', text)       # truncated at end
    text = re.sub(r'^[^[]*?\d+\s*(?:s|ms)\]', '', text)    # leftover from prev chunk
    text = re.sub(rf'\[/?(?:{_ALL_MARKERS})\b[^\]]*$', '', text)

    if text != original:
        logger.info(f"SSML markers converted for {provider}: {original[:100]} → {text[:100]}")

    return text


# ---------------------------------------------------------------------------
# Prompt block builder — instructs LLM to use simple text markers
# ---------------------------------------------------------------------------

# Per-provider: compact marker table and example
_PROVIDER_PROMPT_CONFIG = {
    "elevenlabs": {
        "markers": [
            ("[pause] / [pause:Xs]", "Pause (max 3s). Use between info, after greetings, between list items."),
            ("[spell]text[/spell]", "Spell out character by character. Use for IDs, codes, abbreviations."),
            ("[number]digits[/number]", "Read as spoken number. Use for amounts, fees, statistics."),
            ("[ordinal]N[/ordinal]", "Read as ordinal (3→third). Use for ranks, positions, sequences."),
            ("[date]text[/date]", "Read as date. Use for deadlines, event dates."),
            ("[phone]digits[/phone]", "Read as phone number. Use for contact/helpline numbers."),
        ],
        "example_correct": (
            'Your ID is [spell]BLN-4829[/spell]. [pause:0.5s] '
            'Fee is [number]1100000[/number] for two years. [pause:0.5s] '
            'Deadline is [date]March 15, 2025[/date]. [pause:0.5s] '
            'Call [phone]9876543210[/phone].'
        ),
        "example_wrong": (
            'Your ID is BLN-4829. '
            'Fee is 1100000 for two years. '
            'Deadline is <say-as interpret-as="date">March 15, 2025</say-as>. '
            'Call 9876543210.'
        ),
        "wrong_reason": "No markers used, raw XML tags, no pauses between info.",
    },
    "cartesia": {
        "markers": [
            ("[pause] / [pause:Xs]", "Pause. Use between details, after greetings, before questions."),
            ("[slow]text[/slow]", "Slower speech. Use for important numbers, fees, instructions."),
            ("[fast]text[/fast]", "Faster speech. Use for routine transitions."),
            ("[spell]text[/spell]", "Spell out. Use for IDs, codes, abbreviations."),
        ],
        "example_correct": (
            'Fee is [slow]eleven lakh for two years[/slow]. [pause:0.5s] '
            'Admission fee is [slow]seventy-six thousand[/slow]. [pause:0.5s] '
            'Would you like to know about scholarships?'
        ),
        "example_wrong": (
            'Fee is eleven lakh for two years. '
            'Admission fee is seventy-six thousand. '
            'Would you like to know about scholarships?'
        ),
        "wrong_reason": "No speed markers on important figures, no pauses between info.",
    },
}


def build_ssml_prompt_block(provider: str) -> str | None:
    """Build compact speech-marker instructions for the LLM system prompt."""
    cfg = _PROVIDER_PROMPT_CONFIG.get(provider)
    if not cfg:
        return None

    marker_lines = "\n".join(f"  {syn} — {desc}" for syn, desc in cfg["markers"])

    parts = [
        "### Speech Markers\n"
        "Your output is spoken aloud. Embed these markers to control delivery.\n\n"
        "STRICT RULES:\n"
        "- Use ONLY the [bracket] markers below. NEVER output XML tags like <say-as>, <break>, <prosody>, etc.\n"
        "- Put content inside markers exactly as-is. Do NOT add spaces between characters or digits.\n"
        "  WRONG: [number]1 1 0 0 0 0 0[/number]  WRONG: [spell]B L N - 4 8[/spell]\n"
        "  RIGHT: [number]1100000[/number]  RIGHT: [spell]BLN-4829[/spell]\n"
        "- No markdown, no double-spaces, no invented markers.\n\n"
        f"Markers:\n{marker_lines}\n\n"
        "Use [pause] between multiple pieces of info. Skip markers for short single-sentence replies.\n\n"
        f"CORRECT: {cfg['example_correct']}\n\n"
        f"WRONG: {cfg['example_wrong']}\n"
        f"Why wrong: {cfg['wrong_reason']}"
    ]

    return "".join(parts)
