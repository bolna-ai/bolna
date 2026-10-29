"""Jev (TypeSafe System One) as the language-switch judge.

Jev answers narrow typed questions about the caller's turn, and `decide_from_answers` applies the rules
of LANGUAGE_SWITCH_SYSTEM_PROMPT to those answers in code (`decide_explicit_from_answers` does the same for
EXPLICIT_LANGUAGE_SWITCH_SYSTEM_PROMPT). The result has the same fields as LanguageSwitcher.decide(), so the
gates and the switch path downstream are unchanged.
"""

import asyncio
import os
import string
import time
import unicodedata

import aiohttp

from bolna.helpers.language_switcher import (
    DEFAULT_LANGUAGE_SWITCH_LLM,
    RUNTIME_FALLBACK_AFTER,
    LanguageSwitcher,
    resolve_switch_llm_credentials,
)
from bolna.helpers.logger_config import configure_logger
from bolna.llms import LiteLLM

logger = configure_logger(__name__)

JEV_MODEL_PREFIX = "typesafe/"
DEFAULT_JEV_MODEL = "jev-1.13.0"
DEFAULT_JEV_API_BASE = "https://api.typesafe.ai"
DEFAULT_JEV_TIMEOUT_S = 4.0

# Decision thresholds on Jev's 0-1 answers.
REQUEST_MIN = 0.5
# bolna's default LANGUAGE_SWITCH_MIN_CONFIDENCE: an auto-mode request below it would only be gated there.
SWITCH_MIN = 0.7
UNNAMED_REQUEST_MIN = 0.85
REFUSAL_MIN = 0.5
SUBSTANTIVE_MIN = 0.5
DATA_STRONG = 0.75
DRIFT_MIN_SECONDS = 1.0
DRIFT_MIN_TURNS = 1
ENGLISH_FRAME_MIN = 0.55
SELECT_MIN = 0.5
YES_MIN = 0.6
CLEAR_MARGIN = 0.3

LANGS = {
    "en": dict(
        name="English",
        script="latin",
        names="English, इंग्लिश, अंग्रेजी, अंग्रेज़ी, इंग्रजी, ఇంగ్లీష్, இங்கிலீஷ், ಇಂಗ್ಲಿಷ್, ഇംഗ്ലീഷ്, ഇംഗ്ലീഷ, ইংরেজি, "
        "ਅੰਗਰੇਜ਼ੀ, ਇੰਗਲਿਸ਼, ਇੰਗਰੇਜ਼, ଇଂରାଜୀ, અંગ્રેજી, ઇંગ્લિશ",
        telltale="I, you, my, me, want, what, where, please, is, are, was, the, can you, do not",
    ),
    "hi": dict(
        name="Hindi",
        script="devanagari",
        names="Hindi, हिंदी, हिन्दी, హిందీ, ஹிந்தி, ಹಿಂದಿ, ഹിന്ദി, হিন্দি, ਹਿੰਦੀ, હિન્દી, ହିନ୍ଦୀ",
        telltale="main, aap, mujhe, mera, chahiye, nahi, kya, kab, kahan, hai, hoon, tha, raha hai, ka / ki / ke, ko, mein, -iye",
    ),
    "mr": dict(
        name="Marathi",
        script="devanagari",
        names="Marathi, मराठी",
        telltale="mi, tumhi, mala, majha, pahije, nahi, kay, aahe, aahet, hota, -la, -cha / -chi / -che, -aycha",
    ),
    "gu": dict(
        name="Gujarati",
        script="gujarati",
        names="Gujarati, ગુજરાતી, गुजराती",
        telltale="hu, tame, mane, maru, joie, nathi, shu, chhe, chhu, -ma, -vo chhe, kem",
    ),
    "pa": dict(
        name="Punjabi",
        script="gurmukhi",
        names="Punjabi, ਪੰਜਾਬੀ, पंजाबी",
        telltale="main, tusi, mainu, tuhanu, chahida, nahin, haan, si, nu, da / di / de, vich, ji",
    ),
    "bn": dict(
        name="Bengali",
        script="bengali",
        names="Bengali, Bangla, বাংলা, बंगाली, बांग्ला",
        telltale="ami, apni, amar, apnar, lagbe, na, ki, ache, achi, -chi / -chen, -bo / -ben, -eni, korun, bolun",
    ),
    "or": dict(
        name="Odia",
        script="odia",
        names="Odia, Oriya, ଓଡ଼ିଆ, ओड़िया, उड़िया",
        telltale="mu, apana, mote, mora, darkar, nahin, kana, achhi, achhanti, -antu, -ibi, -ku, -re",
    ),
    "kn": dict(
        name="Kannada",
        script="kannada",
        names="Kannada, ಕನ್ನಡ, कन्नड़, కన్నడ, கன்னட",
        telltale="naanu, neevu, nanage, nanna, nimma, beku, illa, enu, -ide, -alli, -ge, maadi",
    ),
    "ta": dict(
        name="Tamil",
        script="tamil",
        names="Tamil, தமிழ், தமிழ, तमिल, తమిళం, తమిళ, ತಮಿಳು, തമിഴ്",
        telltale="naan, neenga, enakku, en, unga, venum, illa, enna, -ren, -reenga, -anum, -unga, irukku",
    ),
    "te": dict(
        name="Telugu",
        script="telugu",
        names="Telugu, తెలుగు, தெலுங்கு, तेलुगु, ತೆಲುಗು",
        telltale="nenu, meeru, naaku, naa, mee, kavali, ledu, emi, -andi, -undi, -aru, -lo",
    ),
    "ml": dict(
        name="Malayalam",
        script="malayalam",
        names="Malayalam, മലയാളം, മലയാള, मलयालम, మలయాళం, மலையாளம்",
        telltale="njan, ningal, enikku, ente, venam, illa, entha, -unnu, -aanu, -um, -il, undu",
    ),
}
SIBLINGS = {
    "hi": "mr, pa, gu",
    "mr": "hi",
    "pa": "hi",
    "gu": "hi",
    "bn": "or, hi",
    "or": "bn",
    "kn": "te",
    "te": "kn, ta",
    "ta": "ml, te",
    "ml": "ta, te",
    "en": "romanized Indian languages",
}
SCRIPT_BLOCKS = [
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("odia", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
    ("latin", 0x0041, 0x024F),
]
# Devanagari and Latin are shared by several languages, so they map to none.
SCRIPT_TO_LANG = {
    "bengali": "bn",
    "gujarati": "gu",
    "gurmukhi": "pa",
    "odia": "or",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
}
NUMBER_WORDS = set("zero oh one two three four five six seven eight nine ten double triple hundred thousand".split())
WORD_EDGES = string.punctuation + "।॥…—–"

LANGUAGE_QUESTION = {
    "type": "choice",
    "instructions": {
        "question": "Which language forms the grammatical frame of `turn.transcript`?",
        "how_to_decide": [
            "Judge by pronouns, function words and verb endings, not by the script the text is written in.",
            "Borrowed words, fillers (achha, haan, okay, theek hai) and language names inside the sentence do not change its language.",
            "When most of the words and grammar are one language, a single trailing word or a short tail in another script belongs to that same language.",
            "Indian languages typed in Latin letters are still those languages, not English.",
            "English words spelled in Indian letters, with no Indian grammar around them, are English.",
            "A lone yes/no word or greeting belongs to the language it comes from.",
        ],
    },
    "criteria": {
        **{
            code: {
                "language": lang["name"],
                "telltale_words_and_endings": lang["telltale"],
                "not_to_confuse_with": SIBLINGS[code],
            }
            for code, lang in LANGS.items()
        },
        "other": {"what": "A language not listed above, or no language at all: only digits, codes, a name or noise"},
    },
}
HAS_CLAUSE = {
    "type": "noul",
    "instructions": "Does `turn.transcript` contain a clause of its own: an inflected or imperative verb, a question "
    "word, or a pronoun with a verb?",
    "criteria": {
        "true": {
            "what": "At least one real clause",
            "examples": ["where is it", "tell me", "I need help", "kab aayega"],
        },
        "false": {
            "what": "No clause: only a yes/no word, greeting, thanks, a name, a noun, or numbers",
            "examples": ["okay", "hello", "thank you", "ticket", "nine eight seven"],
        },
    },
}
FILLER_ONLY = {
    "type": "noul",
    "instructions": "Is `turn.transcript` only an acknowledgment, a greeting, a courtesy word, or one or two borrowed "
    "words, with nothing else?",
    "criteria": {
        "true": {"what": "Only yes / no / okay / hmm, hello / bye, thanks / sorry, or one or two loose borrowed nouns"},
        "false": {"what": "Says something more: a request, a question, a statement with a verb"},
    },
}
DATA_READOUT = {
    "type": "noul",
    "instructions": "Is `turn.transcript` mainly the caller reading out data: digits, a code, an ID, an OTP, or a "
    "phone, order or account number?",
    "criteria": {
        "true": {
            "what": "Mostly digits, spelled letters or number words, maybe with a short lead-in like 'my number is'"
        },
        "false": {"what": "A sentence where numbers, if any, are a small part"},
    },
}
HINDI_OR_MARATHI = {
    "type": "choice",
    "instructions": {
        "question": "Is `turn.transcript` Hindi or Marathi?",
        "how_to_decide": "Both use Devanagari letters, so look at verb endings and small words, not the letters. "
        "आहे and नाही are Marathi; है and नहीं are Hindi. A reply that is only बोला ('go ahead') is Marathi; Hindi "
        "says बोलिए or बोलो.",
    },
    "criteria": {
        "hi": {
            "language": "Hindi",
            "signs": "है, हैं, हूँ, था, रहा, नहीं, मैं, मुझे, मेरा, आप, क्या, कौन, सकते, बताइए, का / की / के "
            "(hai, hain, hoon, tha, raha, mujhe, mera, kya, nahi before the verb)",
        },
        "mr": {
            "language": "Marathi",
            "signs": "आहे, आहेत, होता, नाही, मी, मला, माझा / माझी, तुम्ही, काय, कोण, होऊ, शकत, द्या, सांगितलं, -ला, "
            "-चा / -ची / -चे, -तंय / -तोय as in बोलतंय (aahe, aahet, nahi at the end, mala, majha, tumhi, kay, -tay, "
            "-aycha)",
        },
    },
}
ENGLISH_FRAME = {
    "type": "noul",
    "instructions": "Are the grammar and most of the words of `turn.transcript` English, whatever letters it is written in?",
    "criteria": {
        "true": {
            "what": "An English sentence, even with a filler like arre, yaar or achha, or spelled in Indian letters",
            "examples": ["arre the app is not opening yaar", "वेयर इज़ माय बिल"],
        },
        "false": {
            "what": "An Indian-language sentence, even if it borrows English nouns",
            "examples": ["मेरा bill कब आएगा", "naa account block ayindi"],
        },
    },
}


def _wants_question(code: str) -> dict:
    lang = LANGS[code]
    return {
        "type": "noul",
        "instructions": {
            "question": f"Does the caller in `turn.transcript` ask to talk in {lang['name']}?",
            "language_name_can_be_written_as": lang["names"],
        },
        "criteria": {
            "true": f"Asks for {lang['name']} (talk / speak / switch to it), even right after saying they don't "
            f"understand the agent; says they prefer {lang['name']} or that it is better or easier for them; says "
            f"they will talk in {lang['name']} now; or says the name {lang['name']} two or more times with nothing else",
            "false": f"Does not ask for {lang['name']}: names it in passing, says they can't speak it, only plans to "
            "use it later, or doesn't name it",
        },
    }


def _refuses_question(code: str) -> dict:
    lang = LANGS[code]
    return {
        "type": "noul",
        "instructions": {
            "question": f"Does the caller in `turn.transcript` name {lang['name']} and say they cannot speak it, "
            "cannot understand it, or do not want it?",
            "language_name_can_be_written_as": lang["names"],
        },
        "criteria": {
            "true": f"Names {lang['name']} and rejects it: doesn't know it, doesn't understand it, or asks not to use it",
            "false": f"Does not name {lang['name']} as a language they reject. A general 'I don't understand' is not "
            f"a refusal of {lang['name']}, and neither is the sentence being written in {lang['name']}. Asking for "
            f"{lang['name']} because they can't follow the agent ('I don't understand, speak {lang['name']}') is a "
            "request, never a refusal",
        },
    }


READ_BOTH = (
    "`turn.transcript` and `turn.live_transcript` are two recognizers' text for the same speech. The live one is "
    "locked to the agent's current language and often gets short replies right when the other garbles them, so a "
    "language name or request in either one counts."
)

SAYS_YES = {
    "type": "noul",
    "instructions": "Is the caller's reply in `turn.transcript` or `turn.live_transcript` a yes or an agreement to "
    "`agent_turn`?",
    "criteria": {
        "true": {"what": "Yes, okay, sure, haan, ji, ho, avunu, aamaa, or any agreement, in any language"},
        "false": {"what": "No, a refusal, a question, or a reply about something else"},
    },
}


def _agent_offer_question(codes: list) -> dict:
    return {
        "type": "choice",
        "instructions": {
            "question": "Does `agent_turn` ask the caller to choose or confirm the language of the conversation?",
            "how_to_decide": [
                "Judge the meaning of the whole agent turn, in any language or script.",
                "Offering one language, or asking whether to continue in or switch to one language, is that language.",
                "Asking which language the caller prefers, or offering two or more languages, is several.",
                "Anything else, including a turn that is merely spoken in some language, is none.",
            ],
        },
        "criteria": {
            "none": {"what": "No question or offer about which language to use"},
            "several": {"what": "Asks which language to use, or offers two or more languages"},
            **{code: {"what": f"Offers, or asks to confirm, only {LANGS[code]['name']}"} for code in codes},
        },
    }


def _explicit_wants_question(code: str) -> dict:
    lang = LANGS[code]
    return {
        "type": "noul",
        "instructions": {
            "question": f"Does the caller ask the agent to talk in {lang['name']}, or ask whether it can speak "
            f"{lang['name']}?",
            "how_to_read": READ_BOTH,
            "language_name_can_be_written_as": lang["names"],
        },
        "criteria": {
            "true": f"Asks to talk / speak / reply / switch in {lang['name']} (even right after saying they don't "
            f"understand the agent), asks whether the agent can speak or understand {lang['name']}, says they prefer "
            f"{lang['name']} or that it is better or easier for them, says they will talk in {lang['name']} now, or "
            f"says the name {lang['name']} two or more times with nothing else",
            "false": f"Does not ask for {lang['name']}: names it in passing, quotes someone, says they can't speak "
            "it, only plans to use it later, or doesn't name it",
        },
    }


def _selects_question(code: str) -> dict:
    lang = LANGS[code]
    return {
        "type": "noul",
        "instructions": {
            "question": f"Does the caller's reply choose {lang['name']} in answer to the language question in "
            "`agent_turn`?",
            "how_to_read": READ_BOTH,
            "language_name_can_be_written_as": lang["names"],
        },
        "criteria": {
            "true": f"Answers the agent's language question by choosing {lang['name']}: the name alone is enough, "
            f"or an ordinal ('the second one') that points to {lang['name']} in `agent_turn`",
            "false": f"Does not choose {lang['name']}, picks another language, rejects it, or `agent_turn` asked "
            "no language question",
        },
    }


def _reading_both(question: dict) -> dict:
    return {**question, "instructions": {**question["instructions"], "how_to_read": READ_BOTH}}


def jev_code(label: str) -> str:
    """The LANGS code for an agent label; the backend writes Odia as `od`."""
    base = (label or "").split("-")[0].lower()
    return "or" if base == "od" else base


def supports_labels(labels) -> bool:
    return bool(labels) and all(jev_code(label) in LANGS for label in labels)


def script_of(text: str) -> str:
    counts = {}
    for ch in str(text):
        if ch.isalpha() or unicodedata.category(ch).startswith("M"):
            for name, lo, hi in SCRIPT_BLOCKS:
                if lo <= ord(ch) <= hi:
                    counts[name] = counts.get(name, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def names_language(text: str, code: str) -> bool:
    lowered = unicodedata.normalize("NFC", str(text)).lower()
    return any(name.strip().lower() in lowered for name in LANGS[code]["names"].split(",") if name.strip())


def refuses_language(a: dict, code: str, text: str) -> bool:
    # A refusal must also outscore any request for the same language: asking for a language while saying
    # you can't follow the agent scores both, and the request is what the caller means.
    refusal = a.get(f"refuses_{code}", 0)
    request = max(a.get(f"wants_{code}", 0), a.get(f"selects_{code}", 0))
    return refusal >= REFUSAL_MIN and refusal >= request and names_language(text, code)


def data_tokens(text: str) -> tuple[int, int]:
    """(data words, words). Words split on spaces so Indic vowel signs and "don't" stay whole. A digit group
    weighs up to two spoken words: "98765" outweighs one word, while a year like "2027" stays two."""
    hits = total = 0
    for token in str(text).split():
        token = token.strip(WORD_EDGES)
        if not token:
            continue
        digits = min(sum(ch.isdigit() for ch in token), 2)
        if digits:
            hits += digits
            total += digits
            continue
        total += 1
        spelled_letter = len(token) == 1 and token.isascii() and token.isalpha() and token not in ("I", "a")
        hits += token.lower() in NUMBER_WORDS or spelled_letter
    return hits, total


def build_jev_request(model_id: str, transcript: str, labels: list) -> dict:
    # State is the transcript only: TypeSafe documents that extra context distracts Jev.
    questions = {
        "language": LANGUAGE_QUESTION,
        "has_clause": HAS_CLAUSE,
        "filler_only": FILLER_ONLY,
        "data_readout": DATA_READOUT,
    }
    if script_of(transcript) == "devanagari":
        questions["hindi_or_marathi"] = HINDI_OR_MARATHI
    questions["english_frame"] = ENGLISH_FRAME
    for code in dict.fromkeys(jev_code(label) for label in labels):
        questions[f"wants_{code}"] = _wants_question(code)
        questions[f"refuses_{code}"] = _refuses_question(code)
    return {"model": model_id, "state": {"turn": {"transcript": transcript}}, "questions": questions}


def build_jev_explicit_request(
    model_id: str, transcript: str, live_transcript: str, agent_turn: str, labels: list
) -> dict:
    # Explicit requests hinge on the agent's question and on short replies the unbiased recognizer
    # can garble, so this state carries both transcripts and the agent turn.
    codes = list(dict.fromkeys(jev_code(label) for label in labels))
    questions = {"language": LANGUAGE_QUESTION, "english_frame": ENGLISH_FRAME}
    if script_of(transcript) == "devanagari":
        questions["hindi_or_marathi"] = HINDI_OR_MARATHI
    turn = {"transcript": transcript, "live_transcript": live_transcript}
    state = {"turn": turn}
    if agent_turn:
        state["agent_turn"] = agent_turn
        questions["agent_offer"] = _agent_offer_question(codes)
        questions["says_yes"] = SAYS_YES
    for code in codes:
        questions[f"wants_{code}"] = _explicit_wants_question(code)
        questions[f"refuses_{code}"] = _reading_both(_refuses_question(code))
        if agent_turn:
            questions[f"selects_{code}"] = _selects_question(code)
    return {"model": model_id, "state": state, "questions": questions}


def flatten_answers(answers: dict) -> dict:
    """Raw /v1/systemone answers → nouls as floats, choices as {choice, confidence, probabilities}."""
    flat = {}
    for key, answer in answers.items():
        if answer.get("type") == "noul":
            flat[key] = float(answer["noul"])
        else:
            flat[key] = {
                "choice": answer["choice"],
                "confidence": float(answer.get("confidence") or 0.0),
                "probabilities": answer.get("probabilities") or {},
            }
    return flat


def _confusable(a: str, b: str) -> bool:
    return b in SIBLINGS.get(a, "").split(", ") or a in SIBLINGS.get(b, "").split(", ")


def _recent_after_last_switch(recent_turns) -> list:
    """Earlier turns as (code, seconds); a firing that switched the agent makes it and everything before it stale."""
    recent = []
    for turn in recent_turns or []:
        lang, seconds = turn[0], turn[1]
        if lang:
            recent.append((jev_code(lang), float(seconds or 0.0)))
        if len(turn) > 2 and turn[2]:
            recent = []
    return recent


def _probability(choice: dict) -> float:
    # Jev rescales a choice's confidence so a guess scores 0: 68% sure between two options reads 0.36.
    # The probability is what compares with the other answers and the switch threshold.
    return choice["probabilities"].get(choice["choice"], choice["confidence"])


def _spoken_language(a: dict, transcript: str) -> tuple:
    """Jev's language answer, corrected by the English-frame check, the transcript's script and Hindi vs Marathi."""
    lang, lang_p = a["language"]["choice"], _probability(a["language"])
    script = script_of(transcript)
    if a.get("english_frame", 0) >= ENGLISH_FRAME_MIN:
        lang, lang_p = "en", a["english_frame"]
    elif lang in LANGS and lang != "en" and script in SCRIPT_TO_LANG and LANGS[lang]["script"] != script:
        lang = SCRIPT_TO_LANG[script]
    if lang in ("hi", "mr") and "hindi_or_marathi" in a:
        lang, lang_p = a["hindi_or_marathi"]["choice"], _probability(a["hindi_or_marathi"])
    return lang, lang_p


def decide_from_answers(
    a: dict, transcript: str, live_transcript: str, active_label: str, labels: list, recent_turns=None
) -> dict:
    """Apply the switch rules in order: 3a, 1, 7, 4a, 8, 6. `a` is the output of flatten_answers."""
    label_of = {jev_code(label): label for label in labels}
    active = jev_code(active_label)
    supported = list(label_of)
    lang, lang_p = _spoken_language(a, transcript)
    detected = label_of.get(lang, lang) if lang != "other" else ""

    def refused(code):
        return refuses_language(a, code, transcript)

    note = ""

    def out(target, confidence, explicit, why):
        return {
            "detected_language": detected,
            "detection_confidence": round(lang_p, 3),
            "target_language": label_of[target] if target else None,
            "target_confidence": round(confidence, 3) if target else 0.0,
            "explicit_request": explicit,
            "reasoning": f"jev: {why}{note}",
        }

    hits, total = data_tokens(transcript)
    if (total and hits / total >= 0.5) or a["data_readout"] >= DATA_STRONG:
        return out(None, 0, False, "number or code readout, stay")

    def counts(code):
        # A request that names no language must be strong: Jev scores false ones (a bare acknowledgement) at
        # 0.5-0.8, and real ones whose case ending our name lists miss at 0.94 and up.
        return names_language(transcript, code) or a[f"wants_{code}"] >= UNNAMED_REQUEST_MIN

    candidates = {code: a[f"wants_{code}"] for code in supported if not refused(code)}
    wants = {code: score for code, score in candidates.items() if counts(code)}
    ignored = [LANGS[code]["name"] for code, score in candidates.items() if score >= REQUEST_MIN and code not in wants]
    if ignored:
        note = f" (ignored an unnamed request for {', '.join(ignored)})"
    if wants and max(wants.values()) >= REQUEST_MIN:
        target = max(wants, key=wants.get)
        if target == active:
            return out(None, 0, False, f"asked for {target}, already active")
        if wants[target] >= SWITCH_MIN:
            return out(target, wants[target], True, f"caller asked for {LANGS[target]['name']}")
        note += f" (request for {LANGS[target]['name']} below {SWITCH_MIN})"

    if lang == active:
        acknowledgement = a["filler_only"] >= SUBSTANTIVE_MIN
        return out(None, 0, False, "short acknowledgement, stay" if acknowledgement else "speaking the active language")
    if lang == "other":
        return out(None, 0, False, "language unclear, stay")
    if lang not in supported:
        return out(None, 0, False, f"{LANGS[lang]['name']} not supported, stay")
    if refused(lang):
        return out(None, 0, False, f"caller refused {LANGS[lang]['name']}")

    recent = _recent_after_last_switch(recent_turns)
    live = (live_transcript or "").strip()
    tags = {code for code, seconds in recent if seconds >= DRIFT_MIN_SECONDS}
    # Rule 4a covers siblings the detector confuses; English after Hindi turns is a real change of language.
    if (
        len(tags) >= 2
        and all(code == active or _confusable(code, active) for code in tags | {lang})
        and len(live.split()) >= 2
        and script_of(live) == LANGS[active]["script"]
    ):
        return out(None, 0, False, "tags unstable, live transcript stable, stay")

    if sum(1 for code, seconds in recent if code == lang and seconds >= DRIFT_MIN_SECONDS) >= DRIFT_MIN_TURNS:
        return out(lang, lang_p, False, f"sustained {LANGS[lang]['name']} across turns")

    # Discounted by substance so acknowledgements stay: a certain language still needs substantive >= 0.7 to clear
    # bolna's 0.7 bar.
    substantive = (a["has_clause"] + (1 - a["filler_only"])) / 2
    if substantive >= SUBSTANTIVE_MIN:
        return out(lang, lang_p * substantive, False, f"caller speaking {LANGS[lang]['name']}")
    return out(None, 0, False, "filler or borrowed word, stay")


def decide_explicit_from_answers(
    a: dict, transcript: str, live_transcript: str, active_label: str, labels: list
) -> dict:
    """Explicit-only contract: switch only on a request, a capability question, or an answer to the agent's
    language question; the language being spoken never switches. Fills request_status / request_source."""
    label_of = {jev_code(label): label for label in labels}
    lang, lang_p = _spoken_language(a, transcript)
    heard = f"{transcript} {live_transcript or ''}"

    def refused(code):
        return refuses_language(a, code, heard)

    def scored(prefix, minimum):
        return {c: a[f"{prefix}_{c}"] for c in label_of if a.get(f"{prefix}_{c}", 0) >= minimum and not refused(c)}

    offer = (a.get("agent_offer") or {}).get("choice", "none")
    picked, source = {}, "none"
    if offer != "none":
        picked = scored("selects", SELECT_MIN)
        if not picked and offer in label_of and a.get("says_yes", 0) >= YES_MIN and not refused(offer):
            picked = {offer: a["says_yes"]}
        source = "agent_prompted_selection" if picked else "none"
    if not picked:
        picked = scored("wants", REQUEST_MIN)
        source = "direct_request" if picked else "none"

    def out(status, code=None, confidence=0.0, why=""):
        target = label_of[code] if status == "switch" else None
        return {
            "detected_language": label_of.get(lang, lang) if lang != "other" else "",
            "detection_confidence": round(lang_p, 3),
            "explicit_request": status in ("switch", "already_active"),
            "requested_language": label_of[code] if code else None,
            "target_language": target,
            "target_confidence": round(confidence, 3) if target else 0.0,
            "request_status": status,
            "request_source": source if code else "none",
            "reasoning": f"jev: {why}",
        }

    ranked = sorted(picked.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        if offer == "several" and a.get("says_yes", 0) >= YES_MIN:
            return out("ambiguous", why="yes to several offered languages")
        return out("no_request", why="no language request")
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < CLEAR_MARGIN:
        return out("ambiguous", why="more than one language named")
    code, score = ranked[0]
    name = LANGS[code]["name"]
    if code == jev_code(active_label):
        return out("already_active", code, why=f"asked for {name}, already active")
    verb = "chose" if source == "agent_prompted_selection" else "asked for"
    return out("switch", code, score, f"caller {verb} {name}")


class JevLanguageSwitcher(LanguageSwitcher):
    """LanguageSwitcher backed by Jev; after repeated Jev failures the call continues on the default LLM judge."""

    def __init__(self, available_labels, run_id=None, model=None, explicit_only=False):
        # super().__init__ is skipped on purpose: it builds the LiteLLM judge, needed here only on fallback.
        self.available_labels = list(available_labels or [])
        self.run_id = run_id
        self.explicit_only = bool(explicit_only)
        self.model = model or f"{JEV_MODEL_PREFIX}{DEFAULT_JEV_MODEL}"
        self._jev_model_id = self.model.removeprefix(JEV_MODEL_PREFIX)
        self.latency_ms = None
        self.hedge_won = False
        self.last_generate_errored = False
        self.usage_totals = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "requests": 0}
        self.last_usage = None
        self.models_used: list = []
        self._api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
        self._api_base = (os.getenv("TYPESAFE_API_BASE") or DEFAULT_JEV_API_BASE).rstrip("/")
        self.has_credentials = bool(self._api_key)
        self._is_bedrock = False
        self._session = None
        self._consecutive_failures = 0
        self._runtime_fallback_done = False

    def _http(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers={"Authorization": f"Bearer {self._api_key}"})
        return self._session

    def prewarm(self):
        async def _warm():
            try:
                async with self._http().get(
                    f"{self._api_base}/v1/models", timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    await response.read()
                logger.info("JevLanguageSwitcher: connection prewarmed")
            except Exception as e:
                logger.debug(f"JevLanguageSwitcher: prewarm skipped: {e}")

        return asyncio.create_task(_warm())

    async def _post(self, payload: dict) -> dict:
        timeout_s = float(os.getenv("LANGUAGE_SWITCH_JEV_TIMEOUT_S", str(DEFAULT_JEV_TIMEOUT_S)))
        async with self._http().post(
            f"{self._api_base}/v1/systemone", json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def decide(
        self,
        detector_transcript: str,
        active_transcript: str,
        active_label: str,
        recent_turns: list | None = None,
        last_agent_turn: str | None = None,
    ) -> dict | None:
        if self._runtime_fallback_done:
            return await super().decide(
                detector_transcript, active_transcript, active_label, recent_turns, last_agent_turn
            )
        if not detector_transcript or not detector_transcript.strip():
            return None
        self.last_usage = None
        transcript = detector_transcript.strip()
        live = (active_transcript or "").strip()
        also_sent = None
        if self.explicit_only:
            agent_turn = (last_agent_turn or "").strip()
            payload = build_jev_explicit_request(
                self._jev_model_id, transcript, live, agent_turn, self.available_labels
            )
            also_sent = {"live_transcript": live, **({"agent_turn": agent_turn} if agent_turn else {})}
        else:
            payload = build_jev_request(self._jev_model_id, transcript, self.available_labels)
        start_time = time.time()
        log_meta = self._log_request(detector_transcript, start_time, also_sent)
        try:
            body = await self._post(payload)
            self._tally_usage(body.get("usage"))
            answers = flatten_answers(body["answers"])
            if self.explicit_only:
                result = decide_explicit_from_answers(answers, transcript, live, active_label, self.available_labels)
            else:
                result = decide_from_answers(
                    answers, transcript, active_transcript, active_label, self.available_labels, recent_turns
                )
        except asyncio.CancelledError:
            self._log_response({"error": "cancelled"}, log_meta, start_time)
            raise
        except Exception as e:
            logger.error(f"JevLanguageSwitcher decision error: {e}")
            self._log_response({"error": str(e)}, log_meta, start_time)
            self._note_failure()
            return None

        self._consecutive_failures = 0
        self.latency_ms = (time.time() - start_time) * 1000
        logger.info(f"JevLanguageSwitcher decision: {result} (latency_ms={self.latency_ms:.0f})")
        self._log_response(result, log_meta, start_time)
        return result

    def _note_failure(self):
        if self._runtime_fallback_done:
            return super()._note_failure()
        self._consecutive_failures += 1
        if self._consecutive_failures < RUNTIME_FALLBACK_AFTER:
            return
        default_model = f"anthropic/{DEFAULT_LANGUAGE_SWITCH_LLM}"
        key, base, version = resolve_switch_llm_credentials(default_model)
        if not key.strip():
            return
        logger.warning(
            f"JevLanguageSwitcher: {self._consecutive_failures} consecutive failures on '{self.model}' — "
            f"falling back to {default_model} for the rest of this call"
        )
        self.model = default_model
        self._llm = LiteLLM(
            model=default_model, max_tokens=200, temperature=0.0, llm_key=key, base_url=base, api_version=version
        )
        self._runtime_fallback_done = True
        self._consecutive_failures = 0

    async def aclose(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()


def build_language_switcher(available_labels, run_id=None, model=None, explicit_only=False):
    """Jev judge for a `typesafe/` model on an agent whose languages are all in LANGS; else the LLM judge."""
    if model and model.startswith(JEV_MODEL_PREFIX):
        if not os.getenv("TYPESAFE_API_KEY", "").strip():
            reason = "TYPESAFE_API_KEY not set"
        elif not supports_labels(available_labels):
            reason = f"labels {list(available_labels or [])} outside {sorted(LANGS)}"
        else:
            return JevLanguageSwitcher(available_labels, run_id=run_id, model=model, explicit_only=explicit_only)
        logger.warning(f"Jev judge '{model}' not used ({reason}) — using the default LLM judge")
        model = None
    return LanguageSwitcher(available_labels=available_labels, run_id=run_id, model=model, explicit_only=explicit_only)
