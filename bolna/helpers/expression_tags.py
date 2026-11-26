"""Expressive tags in reply text: neutral pause, speed, spelling, mood and sound tags.

Each voice turns the tags it supports into its own SSML or audio tags and drops the rest, so a tag is never spoken
aloud. The neutral syntax doesn't collide with any provider's own tags:
  <<warm>>              mood from here on            <<slow: 4 7 2 9>>  speed or spelling for these words
  [[pause 400]]         pause in milliseconds        [[laugh]]          a sound
  ^^word^^              stress
"""

import re
from dataclasses import dataclass, replace

_NAME = r"[a-z][a-z '\-]*?"  # one word or several: "warm", "soft tone", "clears throat"
# No two neighbouring quantifiers may match the same spaces (catastrophic backtracking on crafted text): names and
# span words keep their surrounding spaces, and parse() trims them.
_TAG = re.compile(
    rf"<<\s*(?P<span>{_NAME}):(?P<span_words>[^<>]+)>>"
    rf"|<<\s*(?P<mood>{_NAME})>>"
    r"|\[\[\s*pause(?:\s+(?P<pause>\d+(?:\.\d+)?)(?:\s*(?P<unit>ms|s))?)?\s*\]\]"
    rf"|\[\[\s*(?P<sound>{_NAME})\]\]"
    r"|\^\^(?P<stress>[^^]+?)\^\^"
    r"|(?P<malformed><<[^<>\n]*>>|\[\[[^\[\]\n]*\]\])"  # any other complete tag: dropped, never spoken
    # A provider tag written directly in the text: <break time="1s"/>, [calm]. Text like "<5" or "[1]" is not one.
    r"|(?P<raw_xml></?(?P<xml_name>[A-Za-z][\w:-]*)(?:\s[^<>]*)?/?>)"
    r"|(?P<raw_bracket>\[[A-Za-z][A-Za-z ,'-]{0,40}\])",
    re.IGNORECASE,
)
_SPACES = re.compile(r"[ \t]{2,}")
_SPELLABLE = re.compile(r"[A-Za-z0-9@._-]+")
_BREAK_TIME = re.compile(r"time\s*=\s*[\"']?\s*(\d+(?:\.\d+)?)\s*(ms|s)?", re.IGNORECASE)


@dataclass(frozen=True)
class Words:
    text: str
    delivery: str | None = None  # None, or a span: "slow", "fast", "spell", "stress"


@dataclass(frozen=True)
class Pause:
    ms: int


@dataclass(frozen=True)
class Sound:
    name: str


@dataclass(frozen=True)
class Mood:
    name: str


@dataclass(frozen=True)
class RawTag:
    text: str
    xml_name: str | None = None  # set for <name …/> tags; None for [bracket] tags


@dataclass(frozen=True)
class Capabilities:
    """What one voice can perform. Anything else is dropped before the voice sees it."""

    pause_ms: tuple[int, int] | None = None
    deliveries: frozenset[str] = frozenset()
    moods: frozenset[str] = frozenset()
    sounds: frozenset[str] = frozenset()
    mood_languages: frozenset[str] | None = None  # None: every language
    raw_xml: frozenset[str] = frozenset()  # provider <name …/> tags written in the text that this voice performs
    raw_brackets: bool = False  # whether this voice performs any provider [bracket] tag written in the text
    raw_bracket_words: frozenset[str] = frozenset()  # the [bracket] tags it performs when it doesn't perform all
    drops_brackets: bool = False  # the voice itself skips any other [bracketed] text, words included
    ends_without_sound: bool = False  # a sound as the very last thing of a reply can leave the voice silent


NO_TAGS = Capabilities()

# Words that make bracketed text an expressive tag rather than text: "[warmly]" is removed on a voice that doesn't
# perform bracket tags, while "[Gold]" keeps its word. The providers' documented tags and emotions plus common delivery
# directions; provider tags are open-ended, so new directions are added here as they come up.
EXPRESSIVE_WORDS = frozenset(
    {
        # mood and delivery
        "warm", "warmly", "calm", "calmly", "soft", "softly", "gentle", "gently", "excited", "excitedly", "happy",
        "happily", "sad", "sadly", "curious", "cheerful", "cheerfully", "thoughtful", "thoughtfully", "sympathetic",
        "empathetic", "apologetic", "reassuring", "serious", "seriously", "nervous", "nervously", "surprised", "angry",
        "sarcastic", "sarcastically", "confident", "friendly", "professional", "playful", "playfully", "hesitant",
        "hesitates", "whisper", "whispers", "whispering", "shouts", "shouting", "slowly", "quickly", "smiling",
        "smiles", "energetic", "enthusiastic", "casual", "brisk", "fast", "rapid", "slow", "steady", "quietly",
        "annoyed", "appalled", "mischievously", "deadpan", "dismissive", "questioning", "delighted", "impressed",
        "dramatically", "cautiously", "panicking", "amazed", "alarmed", "sheepishly", "desperately", "frustrated",
        "cute", "polite", "politely", "upbeat", "relaxed", "reflective", "sincere", "conversational", "measured",
        "urgent", "firm", "neutral", "bright", "kind", "kindly", "patient", "patiently", "encouraging", "concerned",
        "empathetically", "warmth", "smile", "crying", "sings", "singing", "muttering",
        # emotions a voice takes as a setting
        "elated", "euphoric", "triumphant", "flirtatious", "content", "peaceful", "serene", "grateful", "affectionate",
        "trust", "anticipation", "mysterious", "mad", "outraged", "agitated", "threatened", "disgusted", "contempt",
        "envious", "ironic", "dejected", "melancholic", "disappointed", "hurt", "guilty", "bored", "tired", "rejected",
        "nostalgic", "wistful", "insecure", "confused", "resigned", "anxious", "panicked", "scared", "proud",
        "distant", "skeptical", "contemplative", "determined",
        # pauses
        "pause", "pauses", "short pause", "long pause", "brief pause", "silence",
    }
)  # fmt: skip
SOUND_WORDS = frozenset(
    {
        "laugh", "laughs", "laughing", "laughter", "chuckle", "chuckles", "giggle", "giggles", "giggling", "sigh",
        "sighs", "sighing", "exhales", "inhales", "gasps", "gasp", "clears throat", "cough", "coughs", "sobbing",
        "sniffs", "breath", "breathes", "wheezing", "snorts", "gulps", "swallows", "applause", "clapping", "woo",
        "laughs harder", "starts laughing", "laughing hysterically", "stifling laughter", "cracking up",
        "exhales sharply", "inhales deeply",
    }
)  # fmt: skip
EXPRESSIVE_WORDS = EXPRESSIVE_WORDS | SOUND_WORDS
_TAG_TOKENS = frozenset(token for phrase in EXPRESSIVE_WORDS for token in phrase.split())
# Words that join directions without being text of their own: "[warm and friendly]", "[lightly curious]".
_JOINERS = frozenset(
    {"and", "a", "an", "with", "in", "very", "slightly", "lightly", "bit", "little", "more", "less", "tone", "voice",
     "speaking", "sounding", "then", "but", "still", "somewhat", "quite", "really", "pace", "manner", "of"}
)  # fmt: skip


_LONGEST_PAUSE_MS = 60_000  # far beyond any voice's range; it only keeps a huge written number from overflowing


def _pause_ms(value, unit):
    ms = value * 1000 if (unit or "").lower() == "s" else value
    return int(min(ms, _LONGEST_PAUSE_MS))


def is_expressive_word(text):
    """Whether bracketed text is an expressive tag ("warmly", "warm and friendly", "French accent") rather than words."""
    phrase = " ".join(text.lower().split())
    if phrase in EXPRESSIVE_WORDS or phrase.endswith(" accent"):
        return True
    words = re.findall(r"[a-z']+", phrase)
    return any(w in _TAG_TOKENS for w in words) and all(w in _TAG_TOKENS or w in _JOINERS for w in words)


def parse(text):
    """The text as Words, Pause, Sound and Mood segments, in order."""
    segments, pos = [], 0
    for m in _TAG.finditer(text):
        if m.start() > pos:
            segments.append(Words(text[pos : m.start()]))
        if m["malformed"]:
            # A broken tag like "<<spell BK4821>>": its name goes, any words after it stay.
            words = m["malformed"][2:-2].strip().split(None, 1)[1:]
            if words:
                segments.append(Words(words[0]))
        elif m["raw_xml"]:
            segments.append(RawTag(m["raw_xml"], m["xml_name"].lower()))
        elif m["raw_bracket"]:
            segments.append(RawTag(m["raw_bracket"]))
        elif m["span"]:
            segments.append(Words(m["span_words"].strip(), m["span"].strip().lower()))
        elif m["mood"]:
            segments.append(Mood(m["mood"].strip().lower()))
        elif m["stress"]:
            segments.append(Words(m["stress"], "stress"))
        elif m["sound"]:
            segments.append(Sound(m["sound"].strip().lower()))
        else:
            value = float(m["pause"]) if m["pause"] else 500
            segments.append(Pause(_pause_ms(value, m["unit"])))
        pos = m.end()
    if pos < len(text):
        segments.append(Words(text[pos:]))
    return segments


# Codes a caller must copy: spelled. Phone numbers and OTPs: slowed. One token each, so an LLM cut never splits them.
_CODE = r"[A-Z][A-Z0-9-]{4,11}"
_RULES = re.compile(
    r"(?P<phone>(?<![\w+])(?:\+91|0)?[6-9]\d{9}(?!\w))"
    rf"|(?P<id>(?<![\w@.-]){_CODE}(?![\w@-]))"
    r"|(?P<otp>(?<![\w.,])\d{4,6}(?![\w]|[.,]\d))"
)
_OTP_CONTEXT = re.compile(r"\b(?:otp|code|pin|password|passcode)\b|ओटीपी|कोड|पिन|पासवर्ड", re.IGNORECASE)


def _is_reference(code):
    """A reference like BK48217 has a long number or mixes letters and digits; SAVE10 or FESTIVE100 is a word and a
    short number, said as such."""
    return (
        re.fullmatch(_CODE, code) is not None
        and sum(c.isdigit() for c in code) >= 2
        and not re.search(r"[A-Z]{5}", code)
        and not re.fullmatch(r"[A-Z]+\d{1,3}", code)
    )


def _rule_spans(text):
    """Words(…) segments of plain text, with codes marked for spelling and phone numbers and OTPs for slowing."""
    out, pos = [], 0
    for m in _RULES.finditer(text):
        code = m["id"]
        if code and not _is_reference(code):
            continue
        if m["otp"] and not _OTP_CONTEXT.search(" ".join(text[: m.start()].split()[-3:])):
            continue
        if m.start() > pos:
            out.append(Words(text[pos : m.start()]))
        out.append(Words(m.group(), "spell" if m["id"] else "slow"))
        pos = m.end()
    if pos < len(text):
        out.append(Words(text[pos:]))
    return out


def apply_rules(segments):
    """The segments with the rules applied to plain text; text inside a tag is left as the tag wrote it, except a
    reference the LLM slowed, which is spelled: LLMs slow codes they're told to leave alone."""
    out, in_spell = [], False
    for s in segments:
        if isinstance(s, RawTag) and s.xml_name == "spell":
            in_spell = not s.text.startswith("</")  # a <spell> the reply already wrote
        if isinstance(s, Words) and s.delivery == "slow" and _is_reference(s.text.strip()):
            s = Words(s.text.strip(), "spell")
        plain = isinstance(s, Words) and s.delivery is None and not in_spell
        out.extend(_rule_spans(s.text) if plain else [s])
    return out


def _break_ms(raw):
    m = _BREAK_TIME.search(raw)
    return _pause_ms(float(m.group(1)), m.group(2)) if m else 500


MAX_PAUSES_PER_REPLY = 4  # clustered pauses tend to make synthesized speech unstable


def ends_with_pause(segments, before=False):
    """Whether the last thing spoken is a pause; `before` when the segments hold nothing but spaces."""
    for s in reversed(segments):
        if isinstance(s, Pause) or getattr(s, "xml_name", None) == "break":
            return True
        if not isinstance(s, Words) or s.text.strip():
            return False
    return before


def segments(
    text, caps, language=None, pauses_left=MAX_PAUSES_PER_REPLY, rules=False, after_pause=False, end_of_reply=False
):
    """The text's segments a voice with these capabilities can perform; its words are always kept.

    after_pause: the text continues a reply whose last piece ended on a pause. end_of_reply: the text ends the reply."""
    kept = []
    mood_ok = caps.mood_languages is None or (language or "").split("-")[0] in caps.mood_languages
    parsed = parse(text)
    for s in apply_rules(parsed) if rules else parsed:
        if isinstance(s, RawTag) and s.xml_name == "break" and "break" not in caps.raw_xml and caps.pause_ms:
            s = Pause(_break_ms(s.text))  # the voice pauses with a tag of its own
        if isinstance(s, Words):
            supported = s.delivery is None or s.delivery in caps.deliveries
            if s.delivery == "spell" and not _SPELLABLE.fullmatch(s.text):
                supported = False
            kept.append(s if supported else Words(s.text))
        elif isinstance(s, Pause):
            if caps.pause_ms and pauses_left > 0 and not ends_with_pause(kept, after_pause):
                low, high = caps.pause_ms
                kept.append(Pause(min(max(s.ms, low), high)))
                pauses_left -= 1
        elif isinstance(s, Mood):
            if mood_ok and s.name in caps.moods:
                kept.append(s)
        elif isinstance(s, RawTag):
            inner = s.text[1:-1]
            if s.xml_name is not None:
                if s.xml_name in caps.raw_xml:
                    kept.append(s)
            elif caps.raw_brackets or inner.strip().lower() in caps.raw_bracket_words:
                kept.append(s)
            elif not caps.drops_brackets and not is_expressive_word(inner):
                kept.append(Words(inner))  # "[Gold]" is real text: keep the word, drop the brackets
        elif s.name in caps.sounds:
            kept.append(s)
    return _without_trailing_sound(kept) if end_of_reply and caps.ends_without_sound else kept


def _is_sound(s):
    inner = " ".join(s.text[1:-1].lower().split()) if isinstance(s, RawTag) and s.xml_name is None else ""
    return isinstance(s, Sound) or inner in SOUND_WORDS


def _without_trailing_sound(segments):
    """The segments without the sounds that end them; spaces around them stay."""
    out, i = list(segments), len(segments) - 1
    while i >= 0 and (_is_sound(out[i]) or (isinstance(out[i], Words) and not out[i].text.strip())):
        if _is_sound(out[i]):
            del out[i]
        i -= 1
    return out


# A lone "<" opens a tag only before an SSML tag name, so text like "x<y" is never held back as one.
_SSML_NAMES = r"break|speed|volume|emotion|spell|prosody|emphasis|say-as|phoneme|sub|lang|voice|speak|audio|mstts"
_OPENER = re.compile(rf"<<|\[\[|\^\^|</?(?=(?:{_SSML_NAMES})\b)|\[(?=[A-Za-z])", re.IGNORECASE)
_CLOSER = {"<<": ">>", "[[": "]]", "^^": "^^", "<": ">", "</": ">", "[": "]"}


MAX_TAG_CHARS = 80  # an open tag longer than this was never a tag
# The head of a tag that never closed: "<<spell:" keeps the "4821" after it; only a pause's number goes with it.
# The number needs a name before it, so no two space matchers can share the same spaces.
_TAG_HEAD = re.compile(r"<<\s*[\w-]*:?|\[\[\s*(?:[\w-]+(?:\s+\d+(?:\.\d+)?\s*(?:ms|s)?)?)?|\^\^")


def split_unfinished(text, final=False):
    """(ready, held): held is a tag the text ends inside, to put in front of the next piece.

    At the end of the reply (final), or when it has grown too long to be a tag, an open tag is released instead:
    its head is removed and its words kept, so a missing ">>" never silences the rest of the reply.
    """
    done = ""
    while True:
        pos = 0
        while m := _OPENER.search(text, pos):
            end = text.find(_CLOSER[m.group()], m.end())
            if end < 0:
                break
            pos = end + len(_CLOSER[m.group()])
        else:
            return done + text, ""
        head, held = text[: m.start()], text[m.start() :]
        if not final and len(held) <= MAX_TAG_CHARS:
            return done + head, held
        done += head
        if m.group() in ("<", "</"):
            text = ""  # an SSML tag cut off inside its attributes has no words worth keeping
        elif m.group() == "[":
            text = held[1:]  # most likely a bracket in the text, not a tag: keep its words
        else:
            text = _TAG_HEAD.sub("", held, count=1)


def words_only(segments):
    """The words of the segments, with the spaces a removed tag left behind joined."""
    return _SPACES.sub(" ", "".join(s.text for s in segments if isinstance(s, Words)))


def kept_text(segments):
    """The words, plus the provider tags written in the text that this voice performs."""
    return _SPACES.sub(" ", "".join(s.text for s in segments if isinstance(s, (Words, RawTag))))


# Matched only from the start of a run of spaces, so a long run is scanned once, not once per space.
_GAP_BEFORE_PUNCTUATION = re.compile(r"(?<!\s)\s+(?=[.,!?;:।])")


def remove_tags(text):
    """The text without any expressive tag, for transcripts: bracketed text that isn't a tag keeps its words."""
    ready = split_unfinished(text, final=True)[0]  # a tag an interruption cut off loses its head too
    if ready == text and all(isinstance(s, Words) and not s.delivery for s in parse(text)):
        return text  # nothing to remove: the text stays exactly as written
    return _GAP_BEFORE_PUNCTUATION.sub("", words_only(segments(ready, NO_TAGS))).strip()


def has_tags(text):
    """Whether the voice would get anything other than the plain words: a tag, or a code or number the rules mark."""
    return any(not isinstance(s, Words) or s.delivery for s in apply_rules(parse(text)))


def _spelled(text):
    return " " + " ".join(text) + " "


def _ratio(speed):
    return round(min(max(speed, 0.6), 1.5), 2)  # Cartesia takes 0.6-1.5


def format_cartesia(segments, speed=1.0):
    """Cartesia's SSML. Its <speed> is absolute, so a slow or fast span is set against the agent's own speed and
    returns to it, rather than to 1.0."""
    out = []
    for s in segments:
        if isinstance(s, Pause):
            out.append(f' <break time="{s.ms}ms"/> ')
        elif isinstance(s, Sound):
            out.append(" [laughter] ")
        elif isinstance(s, Words) and s.delivery == "spell":
            out.append(f" <spell>{s.text}</spell> ")
        elif isinstance(s, Words) and s.delivery in ("slow", "fast"):
            factor = 0.8 if s.delivery == "slow" else 1.2
            out.append(f' <speed ratio="{_ratio(speed * factor)}"/> {s.text} <speed ratio="{_ratio(speed)}"/> ')
        elif isinstance(s, (Words, RawTag)):
            out.append(s.text)
    return _SPACES.sub(" ", "".join(out))


def format_eleven_ssml(segments):
    """ElevenLabs' SSML (with SSML parsing on): <break> pauses, and codes written out to be read character by
    character."""
    out = []
    for s in segments:
        if isinstance(s, Pause):
            out.append(f' <break time="{min(s.ms, 3000) / 1000:.1f}s" /> ')
        elif isinstance(s, Words) and s.delivery == "spell":
            out.append(_spelled(s.text))
        elif isinstance(s, (Words, RawTag)):
            out.append(s.text)
    return _SPACES.sub(" ", "".join(out))


_AUDIO_MOODS = {"warm": "[warmly]", "calm": "[calm]", "apologetic": "[apologetic]", "sympathetic": "[sympathetic]",
                "excited": "[excited]"}  # fmt: skip
_AUDIO_SOUNDS = {"laugh": "[laughs]", "sigh": "[sighs]"}
_AUDIO_SPEEDS = {"slow": "[slowly]", "fast": "[quickly]"}


def format_audio_tags(segments):
    """ElevenLabs' audio tags: moods, sounds and pauses in brackets, stress in capitals."""
    out = []
    for s in segments:
        if isinstance(s, Pause):
            out.append(" [short pause] " if s.ms < 1000 else " [long pause] ")
        elif isinstance(s, Mood):
            out.append(f"{_AUDIO_MOODS[s.name]} " if s.name in _AUDIO_MOODS else "")
        elif isinstance(s, Sound):
            out.append(f" {_AUDIO_SOUNDS[s.name]} ")
        elif isinstance(s, Words) and s.delivery == "spell":
            out.append(_spelled(s.text))
        elif isinstance(s, Words) and s.delivery in _AUDIO_SPEEDS:
            out.append(f" {_AUDIO_SPEEDS[s.delivery]} {s.text} ")
        elif isinstance(s, Words) and s.delivery == "stress":
            out.append(s.text.upper())
        elif isinstance(s, (Words, RawTag)):
            out.append(s.text)
    return _SPACES.sub(" ", "".join(out))


_MOOD_MOMENTS = {
    "apologetic": "after a delay, a mistake or a problem",
    "sympathetic": "when the caller is worried or upset",
    "calm": "when the caller is angry or stressed",
    "warm": "for a greeting, thanks or good news",
    "excited": "for an offer or something worth celebrating",
}


def prompt_guide(caps):
    """Instructions for the agent's LLM, offering only the tags this voice performs; empty when it performs none."""
    lines = []
    moments = "; ".join(f"<<{m}>> {_MOOD_MOMENTS[m]}" for m in sorted(caps.moods) if m in _MOOD_MOMENTS)
    if moments:
        lines.append(f"Start a sentence with a tone that fits the moment: {moments}. It lasts until the next tone tag.")
    if caps.pause_ms:
        lines.append("Put [[pause 400]] between two separate facts or steps, so the caller can take each one in.")
    if "slow" in caps.deliveries:
        lines.append(
            "Wrap a number, amount or date the caller must note down in <<slow: …>>. "
            "Write codes and IDs as they are; they are spelled out automatically."
        )
    if "laugh" in caps.sounds:
        lines.append("Use [[laugh]] only when the caller says something genuinely funny; at most one in a reply.")
    if not lines:
        return ""
    if "apologetic" in caps.moods:
        example = "<<apologetic>> Sorry for the wait. [[pause 400]] Your booking for Monday is confirmed."
    elif "slow" in caps.deliveries:
        example = "Your booking is confirmed. [[pause 400]] The fare is <<slow: ₹1,250>>."
    else:
        example = "Your booking is confirmed. [[pause 400]] Please carry your ID card."
    return (
        "Your replies are spoken by a voice that understands these tags. A tag changes how your words sound and "
        "is never read out. Use them wherever they fit, so you sound natural instead of flat:\n"
        + "".join(f"- {line}\n" for line in lines)
        + f"Example: {example}\n"
        "Write exactly the words you would say without tags: a tag never adds, removes or rewords anything. "
        "Short, routine replies need no tag. Never write any other tag or bracketed text."
    )


# Tags each model is known to perform without reading them aloud. A model or tag that isn't listed gets words only.
_CARTESIA = Capabilities(
    pause_ms=(100, 3000),
    deliveries=frozenset({"slow", "fast", "spell"}),
    sounds=frozenset({"laugh"}),
    raw_xml=frozenset({"break", "speed", "spell"}),
    raw_bracket_words=frozenset({"laughter"}),
    drops_brackets=True,
)
_ELEVEN_SSML = Capabilities(pause_ms=(100, 3000), deliveries=frozenset({"spell"}), raw_xml=frozenset({"break"}))
_ELEVEN_V4 = Capabilities(
    pause_ms=(100, 3000),
    deliveries=frozenset({"spell", "slow", "fast", "stress"}),
    moods=frozenset({"warm", "calm", "apologetic", "sympathetic", "excited"}),
    sounds=frozenset({"laugh", "sigh"}),
    raw_brackets=True,
    ends_without_sound=True,  # a reply that ends on [laughs] after other tags can come back silent
)

_CAPABILITIES = {
    ("cartesia", "sonic-3"): _CARTESIA,
    ("cartesia", "sonic-3.5"): _CARTESIA,
    ("cartesia", "sonic-3.6"): replace(_CARTESIA, raw_xml=_CARTESIA.raw_xml | {"volume"}),
    ("elevenlabs", "eleven_turbo_v2_5"): _ELEVEN_SSML,
    ("elevenlabs", "eleven_flash_v2_5"): _ELEVEN_SSML,
    ("elevenlabs", "eleven_multilingual_v2"): _ELEVEN_SSML,
    ("elevenlabs", "eleven_v4_turbo"): _ELEVEN_V4,
    ("elevenlabs", "eleven_v3_conversational"): replace(
        _ELEVEN_V4, deliveries=frozenset({"spell", "slow", "stress"}), sounds=frozenset({"laugh"})
    ),
}


def capabilities_for(provider, model):
    """What this provider's model can perform; words only for anything not confirmed."""
    return _CAPABILITIES.get((provider, model), NO_TAGS)
