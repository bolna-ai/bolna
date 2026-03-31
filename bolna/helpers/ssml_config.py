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
# [pause]           → default 0.5s pause
# [pause:1.2s]      → specific duration (seconds)
# [pause:500ms]     → specific duration (milliseconds)
# [slow]text[/slow] → slower speech for that segment
# [fast]text[/fast] → faster speech for that segment
# [loud]text[/loud] → louder speech for that segment
# [soft]text[/soft] → softer speech for that segment
# [spell]ABC[/spell]→ spell out characters

_PAUSE_RE = re.compile(r'\[pause(?::(\d+(?:\.\d+)?)\s*(s|ms))?\]')
_WRAP_RE = re.compile(r'\[(slow|fast|loud|soft|spell)\](.*?)\[/\1\]', re.DOTALL)

# ---------------------------------------------------------------------------
# Provider-specific conversion rules
# ---------------------------------------------------------------------------

def _default_pause_duration(provider: str) -> str:
    """Default pause when [pause] is used without a duration."""
    return {
        "elevenlabs": "0.5s",
        "cartesia": "0.5s",
        "azuretts": "500ms",
        "polly": "500ms",
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


def _convert_wrap_elevenlabs(match) -> str:
    """ElevenLabs: only break tags supported, strip speed/volume/spell wrappers."""
    tag = match.group(1)
    content = match.group(2)
    # ElevenLabs doesn't support speed/volume/spell — just return the text
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


def _convert_wrap_azure(match) -> str:
    """Azure TTS: prosody-based speed/volume, say-as for spell."""
    tag = match.group(1)
    content = match.group(2)
    if tag == "slow":
        return f'<prosody rate="slow">{content}</prosody>'
    elif tag == "fast":
        return f'<prosody rate="fast">{content}</prosody>'
    elif tag == "loud":
        return f'<prosody volume="loud">{content}</prosody>'
    elif tag == "soft":
        return f'<prosody volume="soft">{content}</prosody>'
    elif tag == "spell":
        return f'<say-as interpret-as="characters">{content}</say-as>'
    return content


def _convert_wrap_polly(match) -> str:
    """Amazon Polly: prosody-based, same as Azure."""
    return _convert_wrap_azure(match)


_WRAP_CONVERTERS = {
    "elevenlabs": _convert_wrap_elevenlabs,
    "cartesia": _convert_wrap_cartesia,
    "azuretts": _convert_wrap_azure,
    "polly": _convert_wrap_polly,
}

# Which marker types each provider actually supports in TTS
PROVIDER_SUPPORTED_MARKERS = {
    "elevenlabs": {"pause"},
    "cartesia": {"pause", "slow", "fast", "loud", "soft", "spell"},
    "azuretts": {"pause", "slow", "fast", "loud", "soft", "spell"},
    "polly": {"pause", "slow", "fast", "loud", "soft", "spell"},
}

# Tags each provider actually supports — used for stripping unsupported XML
PROVIDER_ALLOWED_TAGS = {
    "elevenlabs": {"break"},
    "cartesia": {"break", "speed", "volume", "spell", "emotion"},
    "azuretts": {"break", "prosody", "say-as", "emphasis", "phoneme", "sub"},
    "polly": {"break", "prosody", "say-as", "emphasis", "phoneme", "sub"},
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

    # 3. Clean up any remaining unconverted markers (safety net)
    text = re.sub(r'\[/?(pause|slow|fast|loud|soft|spell)(?::[^\]]+)?\]', '', text)

    # 4. Strip partial/split markers from streaming (e.g. "[pause:0.5" or "s]")
    text = re.sub(r'\[pause(?::[^\]]*)?$', '', text)       # truncated at end
    text = re.sub(r'^[^[]*?\d+\s*(?:s|ms)\]', '', text)    # leftover from prev chunk
    text = re.sub(r'\[/?(?:pause|slow|fast|loud|soft|spell)\b[^\]]*$', '', text)

    if text != original:
        logger.info(f"SSML markers converted for {provider}: {original[:100]} → {text[:100]}")

    return text


# ---------------------------------------------------------------------------
# Prompt block builder — instructs LLM to use simple text markers
# ---------------------------------------------------------------------------

# Per-provider: which markers to teach and response examples
_PROVIDER_PROMPT_CONFIG = {
    "elevenlabs": {
        "markers": {
            "pause": {
                "syntax": "[pause] or [pause:Xs] (max 3s)",
                "description": "Insert a natural pause in speech",
                "when_to_use": (
                    "- After a greeting, before the next point\n"
                    "- Between numerical figures (fees, dates, stats)\n"
                    "- After important info, before a follow-up question\n"
                    "- Between items in a list\n"
                    "- When transitioning topics"
                ),
            },
        },
        "example_correct": (
            'The total fee is eleven lakh for two years. [pause:0.5s] '
            'The admission fee is seventy-six thousand. [pause:0.5s] '
            'Semester fee is two lakh fifty-six thousand per semester. [pause:0.8s] '
            'Would you like to know about scholarships?'
        ),
        "example_wrong": (
            'The total fee is eleven lakh for two years. '
            'The admission fee is seventy-six thousand. '
            'Semester fee is two lakh fifty-six thousand per semester. '
            'Would you like to know about scholarships?'
        ),
        "wrong_reason": "No pauses between fee figures — listener cannot absorb each number.",
    },
    "cartesia": {
        "markers": {
            "pause": {
                "syntax": "[pause] or [pause:Xs]",
                "description": "Insert a pause in speech",
                "when_to_use": (
                    "- Between important details\n"
                    "- After a greeting or before a question"
                ),
            },
            "slow": {
                "syntax": "[slow]important text[/slow]",
                "description": "Speak this part more slowly for emphasis",
                "when_to_use": (
                    "- For important numbers, fees, or requirements\n"
                    "- For instructions the listener must follow"
                ),
            },
            "fast": {
                "syntax": "[fast]filler text[/fast]",
                "description": "Speak this part a bit faster",
                "when_to_use": "- For routine transitions or less important phrases",
            },
            "spell": {
                "syntax": "[spell]ABC-123[/spell]",
                "description": "Spell out letters, numbers, or codes character by character",
                "when_to_use": "- For IDs, codes, phone numbers, or abbreviations",
            },
        },
        "example_correct": (
            'The total fee is [slow]eleven lakh for two years[/slow]. [pause:0.5s] '
            'The admission fee is [slow]seventy-six thousand[/slow]. [pause:0.5s] '
            'Would you like to know about scholarships?'
        ),
        "example_wrong": (
            'The total fee is eleven lakh for two years. '
            'The admission fee is seventy-six thousand. '
            'Would you like to know about scholarships?'
        ),
        "wrong_reason": "Important figures spoken at normal speed with no pauses.",
    },
    "azuretts": {
        "markers": {
            "pause": {
                "syntax": "[pause] or [pause:500ms] or [pause:1s]",
                "description": "Insert a pause in speech",
                "when_to_use": (
                    "- Between numerical figures\n"
                    "- After important info, before a question\n"
                    "- Between list items or topic transitions"
                ),
            },
            "slow": {
                "syntax": "[slow]important text[/slow]",
                "description": "Speak more slowly for emphasis",
                "when_to_use": "- For fees, eligibility criteria, or deadlines",
            },
            "fast": {
                "syntax": "[fast]filler text[/fast]",
                "description": "Speak a bit faster",
                "when_to_use": "- For routine transitions",
            },
            "spell": {
                "syntax": "[spell]ABC-123[/spell]",
                "description": "Spell out character by character",
                "when_to_use": "- For codes, phone numbers, or abbreviations",
            },
        },
        "example_correct": (
            'The total fee is [slow]eleven lakh[/slow] for two years. '
            '[pause:500ms] The admission fee is [slow]seventy-six thousand[/slow]. '
            '[pause:500ms] Would you like to know about scholarships?'
        ),
        "example_wrong": (
            'The total fee is eleven lakh for two years. '
            'The admission fee is seventy-six thousand. '
            'Would you like to know about scholarships?'
        ),
        "wrong_reason": "No pauses or speed changes — monotone delivery.",
    },
    "polly": {
        "markers": {
            "pause": {
                "syntax": "[pause] or [pause:500ms] or [pause:1s]",
                "description": "Insert a pause in speech",
                "when_to_use": (
                    "- Between numerical figures\n"
                    "- After important info, before a question\n"
                    "- Between list items or topic transitions"
                ),
            },
            "slow": {
                "syntax": "[slow]important text[/slow]",
                "description": "Speak more slowly for emphasis",
                "when_to_use": "- For fees, eligibility criteria, or deadlines",
            },
            "fast": {
                "syntax": "[fast]filler text[/fast]",
                "description": "Speak a bit faster",
                "when_to_use": "- For routine transitions",
            },
            "spell": {
                "syntax": "[spell]ABC-123[/spell]",
                "description": "Spell out character by character",
                "when_to_use": "- For codes, phone numbers, or abbreviations",
            },
        },
        "example_correct": (
            'The total fee is [slow]eleven lakh[/slow] for two years. '
            '[pause:500ms] The admission fee is [slow]seventy-six thousand[/slow]. '
            '[pause:500ms] Would you like to know about scholarships?'
        ),
        "example_wrong": (
            'The total fee is eleven lakh for two years. '
            'The admission fee is seventy-six thousand. '
            'Would you like to know about scholarships?'
        ),
        "wrong_reason": "No pauses or speed changes — monotone delivery.",
    },
}


def build_ssml_prompt_block(provider: str) -> str | None:
    """
    Build the speech formatting instruction block for the LLM.

    Instructs the LLM to use simple text markers like [pause:0.5s]
    instead of XML tags. These are converted to provider-specific
    SSML by convert_markers_to_ssml() before reaching the TTS engine.
    """
    cfg = _PROVIDER_PROMPT_CONFIG.get(provider)
    if not cfg:
        return None

    lines = [
        "### Speech Formatting Instructions\n",
        "Your text will be spoken aloud by a text-to-speech engine.",
        "To control pacing and delivery, embed the speech markers below directly in your responses.",
        "These markers look like [pause] or [slow]...[/slow] — they are NOT displayed to the user, they control how the audio sounds.",
        "",
        "RULES:",
        "- Use the markers listed below to add pauses, speed changes, etc.",
        "- Do NOT use markdown formatting (no **, no ##, no bullet points). Your output is spoken, not displayed.",
        "- Do NOT use double spaces or line breaks for pauses — use [pause] instead.",
        "- Do NOT invent markers beyond what is listed below.\n",
        "#### Available Markers\n",
    ]

    for marker_name, info in cfg["markers"].items():
        lines.append(f"**{marker_name}**: {info['description']}")
        lines.append(f"  Syntax: `{info['syntax']}`")
        when = info.get("when_to_use")
        if when:
            lines.append(f"  When to use:\n{when}")
        lines.append("")

    lines.append(
        "#### Key Rule\n"
        "Responses with multiple pieces of information (fees, requirements, steps) "
        "MUST include [pause] markers between them. Short single-sentence answers don't need markers.\n"
    )

    lines.append(f"#### CORRECT example\n{cfg['example_correct']}\n")
    lines.append(
        f"#### WRONG example\n{cfg['example_wrong']}\n"
        f"Why wrong: {cfg['wrong_reason']}"
    )

    return "\n".join(lines)
