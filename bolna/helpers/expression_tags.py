"""Expressive tags in reply text: neutral pause, speed, spelling, mood and sound tags.

Each voice turns the tags it supports into its own SSML or audio tags and drops the rest, so a tag is never spoken
aloud. The neutral syntax doesn't collide with any provider's own tags:
  <<warm>>              mood from here on            <<slow: 4 7 2 9>>  speed or spelling for these words
  [[pause 400]]         pause in milliseconds        [[laugh]]          a sound
  ^^word^^              stress
"""

import re
from dataclasses import dataclass

_NAME = r"[a-z][a-z '\-]*?"  # one word or several: "warm", "soft tone", "clears throat"
_TAG = re.compile(
    rf"<<\s*(?P<span>{_NAME})\s*:\s*(?P<span_words>[^<>]+?)\s*>>"
    rf"|<<\s*(?P<mood>{_NAME})\s*>>"
    r"|\[\[\s*pause(?:\s+(?P<pause>\d+(?:\.\d+)?)\s*(?P<unit>ms|s)?)?\s*\]\]"
    rf"|\[\[\s*(?P<sound>{_NAME})\s*\]\]"
    r"|\^\^(?P<stress>[^^]+?)\^\^"
    r"|(?P<malformed><<[^<>\n]*>>|\[\[[^\[\]\n]*\]\])"  # any other complete tag: dropped, never spoken
    # A provider tag written directly in the text: <break time="1s"/>, [calm]. Text like "<5" or "[1]" is not one.
    r"|(?P<raw_xml></?(?P<xml_name>[A-Za-z][\w:-]*)(?:\s[^<>]*)?/?>)"
    r"|(?P<raw_bracket>\[[A-Za-z][A-Za-z ,'-]{0,40}\])",
    re.IGNORECASE,
)
_SPACES = re.compile(r"[ \t]{2,}")
_SPELLABLE = re.compile(r"[A-Za-z0-9@._-]+")


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
    raw_brackets: bool = False  # whether this voice performs provider [bracket] tags written in the text


NO_TAGS = Capabilities()

# Words that, alone in square brackets, are an expressive tag rather than text: "[warmly]" is removed on a voice that
# doesn't perform bracket tags, while "[Gold]" keeps its word.
EXPRESSIVE_WORDS = frozenset(
    {
        # mood and delivery
        "warm", "warmly", "calm", "calmly", "soft", "softly", "gentle", "gently", "excited", "excitedly", "happy",
        "happily", "sad", "sadly", "curious", "cheerful", "cheerfully", "thoughtful", "thoughtfully", "sympathetic",
        "empathetic", "apologetic", "reassuring", "serious", "seriously", "nervous", "nervously", "surprised", "angry",
        "sarcastic", "sarcastically", "confident", "friendly", "professional", "playfully", "hesitant", "hesitates",
        "whisper", "whispers", "whispering", "shouts", "shouting", "slowly", "quickly",
        # sounds
        "laugh", "laughs", "laughing", "chuckle", "chuckles", "giggle", "giggles", "sigh", "sighs", "sighing",
        "exhales", "inhales", "gasps", "clears throat", "cough", "coughs", "crying", "sobbing", "sniffs", "breath",
        # pauses
        "pause", "short pause", "long pause", "silence",
    }
)  # fmt: skip


def is_expressive_word(text):
    """Whether bracketed text is an expressive tag ("warmly", "long pause", "French accent") rather than words."""
    word = " ".join(text.lower().split())
    return word in EXPRESSIVE_WORDS or word.endswith(" accent")


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
            segments.append(Words(m["span_words"], m["span"].lower()))
        elif m["mood"]:
            segments.append(Mood(m["mood"].lower()))
        elif m["stress"]:
            segments.append(Words(m["stress"], "stress"))
        elif m["sound"]:
            segments.append(Sound(m["sound"].lower()))
        else:
            value = float(m["pause"]) if m["pause"] else 500
            segments.append(Pause(int(value * 1000 if m["unit"] and m["unit"].lower() == "s" else value)))
        pos = m.end()
    if pos < len(text):
        segments.append(Words(text[pos:]))
    return segments


MAX_PAUSES_PER_REPLY = 4  # clustered pauses tend to make synthesized speech unstable


def segments(text, caps, language=None, pauses_left=MAX_PAUSES_PER_REPLY):
    """The text's segments a voice with these capabilities can perform; its words are always kept."""
    kept = []
    mood_ok = caps.mood_languages is None or (language or "").split("-")[0] in caps.mood_languages
    for s in parse(text):
        if isinstance(s, Words):
            supported = s.delivery is None or s.delivery in caps.deliveries
            if s.delivery == "spell" and not _SPELLABLE.fullmatch(s.text):
                supported = False
            kept.append(s if supported else Words(s.text))
        elif isinstance(s, Pause):
            spoken_since = [k for k in reversed(kept) if not isinstance(k, Words) or k.text.strip()]
            back_to_back = bool(spoken_since) and isinstance(spoken_since[0], Pause)
            if caps.pause_ms and pauses_left > 0 and not back_to_back:
                low, high = caps.pause_ms
                kept.append(Pause(min(max(s.ms, low), high)))
                pauses_left -= 1
        elif isinstance(s, Mood):
            if mood_ok and s.name in caps.moods:
                kept.append(s)
        elif isinstance(s, RawTag):
            if (caps.raw_xml and s.xml_name in caps.raw_xml) or (caps.raw_brackets and s.xml_name is None):
                kept.append(s)
            elif s.xml_name is None and not is_expressive_word(s.text[1:-1]):
                kept.append(Words(s.text[1:-1]))  # "[Gold]" is real text: keep the word, drop the brackets
        elif s.name in caps.sounds:
            kept.append(s)
    return kept


# A lone "<" opens a tag only before an SSML tag name, so text like "x<y" is never held back as one.
_SSML_NAMES = r"break|speed|volume|emotion|spell|prosody|emphasis|say-as|phoneme|sub|lang|voice|speak|audio|mstts"
_OPENER = re.compile(rf"<<|\[\[|\^\^|</?(?=(?:{_SSML_NAMES})\b)|\[(?=[A-Za-z])", re.IGNORECASE)
_CLOSER = {"<<": ">>", "[[": "]]", "^^": "^^", "<": ">", "</": ">", "[": "]"}


MAX_TAG_CHARS = 80  # an open tag longer than this was never a tag
# The head of a tag that never closed: "<<spell:" keeps the "4821" after it; only a pause's number goes with it.
_TAG_HEAD = re.compile(r"<<\s*[\w-]*:?|\[\[\s*[\w-]*(?:\s+\d+(?:\.\d+)?\s*(?:ms|s)?)?|\^\^")


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


def remove_tags(text):
    """The text without any expressive tag, for transcripts and billing."""
    return words_only(parse(text)).strip()


# Tags each model is known to perform without reading them aloud. A model or tag that isn't listed gets words only;
# moods and sounds are added once their effect on the voice is confirmed.
_CARTESIA = Capabilities(
    pause_ms=(100, 3000),
    deliveries=frozenset({"slow", "fast", "spell"}),
    raw_xml=frozenset({"break", "speed", "volume", "spell"}),
)
_ELEVEN_SSML = Capabilities(pause_ms=(100, 3000), deliveries=frozenset({"spell"}), raw_xml=frozenset({"break"}))
_ELEVEN_AUDIO_TAGS = Capabilities(pause_ms=(100, 3000), deliveries=frozenset({"spell"}), raw_brackets=True)

_CAPABILITIES = {
    ("cartesia", "sonic-3.6"): _CARTESIA,
    ("cartesia", "sonic-3.5"): Capabilities(
        pause_ms=(100, 3000), deliveries=frozenset({"slow", "fast"}), raw_xml=frozenset({"break", "speed"})
    ),
    ("elevenlabs", "eleven_turbo_v2_5"): _ELEVEN_SSML,
    ("elevenlabs", "eleven_flash_v2_5"): _ELEVEN_SSML,
    ("elevenlabs", "eleven_v4_turbo"): _ELEVEN_AUDIO_TAGS,
    ("elevenlabs", "eleven_v3_conversational"): _ELEVEN_AUDIO_TAGS,
}


def capabilities_for(provider, model):
    """What this provider's model can perform; words only for anything not confirmed."""
    return _CAPABILITIES.get((provider, model), NO_TAGS)
