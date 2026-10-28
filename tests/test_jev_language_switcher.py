"""Jev language-switch judge: the questions and rules, the decide() path, fallback and selection."""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from bolna.helpers.jev_language_switcher import (
    LANGS,
    JevLanguageSwitcher,
    build_jev_explicit_request,
    build_jev_request,
    build_language_switcher,
    data_tokens,
    decide_explicit_from_answers,
    decide_from_answers,
    flatten_answers,
)

MOD = "bolna.helpers.jev_language_switcher"
REQUEST_LOG = "bolna.helpers.language_switcher.convert_to_request_log"
JEV_MODEL = "typesafe/jev-1.13.0"


def _answers(
    lang, conf=0.95, clause=0.95, filler=0.05, data=0.02, english=0.05, wants=None, refuses=None, devanagari=None
):
    a = {
        "language": {"choice": lang, "confidence": conf, "probabilities": {lang: conf}},
        "has_clause": clause,
        "filler_only": filler,
        "data_readout": data,
        "english_frame": english,
    }
    for code in LANGS:
        a[f"wants_{code}"] = 0.9 if code == wants else 0.02
        a[f"refuses_{code}"] = 0.9 if code == refuses else 0.02
    if devanagari:
        a["hindi_or_marathi"] = {"choice": devanagari, "confidence": 0.9, "probabilities": {}}
    return a


FILLER = dict(clause=0.05, filler=0.95)

CASES = [
    ("greeting-alone-stays", "good morning", "", "hi", ["hi", "en"], _answers("en", **FILLER), None, None, "filler"),
    (
        "english-sentence-switches",
        "Can you check my order status",
        "",
        "ta",
        ["ta", "en"],
        _answers("en"),
        None,
        "en",
        "speaking English",
    ),
    (
        "named-request-switches",
        "English please",
        "",
        "hi",
        ["hi", "en"],
        _answers("en", clause=0.4, filler=0.6, wants="en"),
        None,
        "en",
        "asked for English",
    ),
    (
        "weak-request-without-a-language-name-is-ignored",
        "हाँ जी, बताइए क्या करना है",
        "",
        "en",
        ["en", "hi"],
        {**_answers("hi"), "wants_en": 0.55},
        None,
        "hi",
        "ignored an unnamed request for English",
    ),
    (
        "strong-request-with-an-inflected-name-switches",
        "இங்கிலீஷில் பேசுங்க",
        "",
        "ta",
        ["ta", "en"],
        {**_answers("ta"), "wants_en": 0.98},
        None,
        "en",
        "asked for English",
    ),
    (
        "inflected-language-name-is-a-request",
        "ദയവായി മലയാളത്തിൽ പറയൂ",
        "",
        "en",
        ["en", "ml"],
        _answers("ml", wants="ml"),
        None,
        "ml",
        "asked for Malayalam",
    ),
    (
        "asking-for-active-stays",
        "Hindi please",
        "",
        "hi",
        ["hi", "en"],
        _answers("hi", wants="hi"),
        None,
        None,
        "already active",
    ),
    (
        "digit-readout-beats-drift",
        "mera number 98765 43210",
        "",
        "hi",
        ["hi", "ta"],
        _answers("hi"),
        [("ta", 2.4, None)],
        None,
        "readout",
    ),
    (
        "number-words-readout-stays",
        "ಒಂಬತ್ತು ಎಂಟು ಏಳು ಆರು",
        "",
        "en",
        ["en", "kn"],
        _answers("kn", data=0.9),
        None,
        None,
        "readout",
    ),
    (
        "indic-lead-in-with-digits-is-a-readout",
        "मेरा नंबर है 98765 43210",
        "",
        "en",
        ["en", "hi"],
        _answers("hi", data=0.3),
        None,
        None,
        "readout",
    ),
    (
        "english-sentence-is-not-a-readout",
        "I won't go.",
        "",
        "hi",
        ["hi", "en"],
        _answers("en"),
        None,
        "en",
        "speaking English",
    ),
    (
        "sentence-with-a-few-number-words-is-not-a-readout",
        "हाँ जी, आप ये नोट कर लीजिए, two three four",
        "",
        "en",
        ["en", "hi"],
        _answers("hi", data=0.6),
        None,
        "hi",
        "speaking Hindi",
    ),
    (
        "named-refusal-stays",
        "मुझे हिंदी नहीं आती",
        "",
        "en",
        ["en", "hi"],
        _answers("hi", refuses="hi"),
        None,
        None,
        "refused Hindi",
    ),
    (
        "refusal-named-in-marathi-stays",
        "मला इंग्रजी येत नाही",
        "",
        "mr",
        ["mr", "en"],
        {**_answers("mr", devanagari="mr"), "wants_en": 0.6, "refuses_en": 0.9},
        None,
        None,
        "active language",
    ),
    (
        "request-outweighs-a-weak-refusal",
        "Can you speak Hindi? I don't understand you.",
        "",
        "en",
        ["en", "hi", "ta", "te"],
        {**_answers("hi"), "wants_hi": 0.96, "refuses_hi": 0.55},
        None,
        "hi",
        "asked for Hindi",
    ),
    (
        "refusal-without-name-ignored",
        "मुझे नहीं आती",
        "",
        "en",
        ["en", "hi"],
        _answers("hi", refuses="hi"),
        None,
        "hi",
        "speaking Hindi",
    ),
    (
        "script-overrides-hindi-label",
        "আমার order কবে আসবে?",
        "",
        "ta",
        ["ta", "bn"],
        _answers("hi"),
        None,
        "bn",
        "speaking Bengali",
    ),
    (
        "english-frame-over-hindi-fillers",
        "Theek hai, please send me the invoice",
        "",
        "hi",
        ["hi", "en"],
        _answers("hi", english=0.8),
        None,
        "en",
        "speaking English",
    ),
    (
        "english-frame-with-hindi-fillers-near-the-bar",
        "Haan, I need the invoice copy yaar",
        "",
        "hi",
        ["hi", "en"],
        _answers("hi", english=0.57),
        None,
        "en",
        "speaking English",
    ),
    (
        "marathi-not-hindi",
        "मी बरा आहे",
        "",
        "hi",
        ["hi", "mr"],
        _answers("hi", devanagari="mr"),
        None,
        "mr",
        "speaking Marathi",
    ),
    (
        "ack-follows-real-drift",
        "சரி",
        "",
        "en",
        ["en", "ta"],
        _answers("ta", **FILLER),
        [("ta", 1.4, None)],
        "ta",
        "sustained Tamil",
    ),
    (
        "ack-after-subsecond-turns-stays",
        "சரி",
        "",
        "en",
        ["en", "ta"],
        _answers("ta", **FILLER),
        [("ta", 0.4, None), ("ta", 0.3, None)],
        None,
        "filler",
    ),
    (
        "drift-before-a-switch-is-stale",
        "சரி",
        "",
        "pa",
        ["pa", "ta"],
        _answers("ta", **FILLER),
        [("ta", 1.9, None), ("ta", 1.4, "pa"), ("pa", 2.1, None)],
        None,
        "filler",
    ),
    (
        "flapping-tags-stable-live-stays",
        "ഞാൻ നന്നായിരിക്കുന്നു",
        "నేను బాగున్నాను",
        "te",
        ["te", "ml"],
        _answers("ml"),
        [("ml", 1.5, None), ("ta", 1.2, None)],
        None,
        "tags unstable",
    ),
    (
        "english-after-unrelated-tags-switches",
        "Please send the receipt to my email",
        "प्लीज़ सेंड द रिसीट टू माय ईमेल",
        "hi",
        ["hi", "en"],
        _answers("en"),
        [("hi", 2.0, None), ("pa", 1.5, None)],
        "en",
        "speaking English",
    ),
    ("unsupported-language-stays", "আমি ঠিক আছি", "", "hi", ["hi", "en"], _answers("bn"), None, None, "not supported"),
    (
        "active-language-stays",
        "मेरा order status check करो",
        "",
        "hi",
        ["hi", "en"],
        _answers("hi"),
        None,
        None,
        "active language",
    ),
    ("unclear-language-stays", "Ravi", "", "hi", ["hi", "en"], _answers("other"), None, None, "unclear"),
    (
        "acknowledgement-in-the-active-language-stays",
        "Okay",
        "Okay.",
        "en",
        ["en", "hi"],
        _answers("en", **FILLER),
        None,
        None,
        "short acknowledgement",
    ),
    (
        "weak-request-in-the-active-language-stays",
        "इंग्लिश में भी बात कर लेते हैं क्या आप",
        "",
        "hi",
        ["hi", "en"],
        {**_answers("hi"), "wants_en": 0.6},
        None,
        None,
        "request for English below 0.7",
    ),
    (
        "weak-request-for-the-active-language-still-stays",
        "बस English में ही रहने दो",
        "",
        "en",
        ["en", "hi"],
        {**_answers("hi"), "wants_en": 0.6},
        None,
        None,
        "already active",
    ),
]


@pytest.mark.parametrize(
    "transcript, live, active, labels, answers, recent_turns, target, reason",
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_rules_follow_the_switch_prompt(transcript, live, active, labels, answers, recent_turns, target, reason):
    result = decide_from_answers(answers, transcript, live, active, labels, recent_turns)
    assert result["target_language"] == target
    assert reason in result["reasoning"]


def test_named_request_is_explicit_with_its_own_confidence():
    result = decide_from_answers(_answers("hi", wants="ta"), "Tamil la pesunga", "", "hi", ["hi", "ta"])
    assert result["explicit_request"] is True
    assert result["target_confidence"] == 0.9


def test_weak_request_in_the_requested_language_switches_on_what_was_spoken():
    result = decide_from_answers({**_answers("hi"), "wants_hi": 0.6}, "हिंदी में समझाइए ना थोड़ा", "", "en", ["en", "hi"])
    assert result["target_language"] == "hi"
    assert result["target_confidence"] >= 0.7
    assert "speaking Hindi (request for Hindi below 0.7)" in result["reasoning"]


def test_language_choice_reports_its_probability():
    answers = _answers("ta")
    answers["language"] = {"choice": "ta", "confidence": 0.62, "probabilities": {"ta": 0.65, "te": 0.35}}
    result = decide_from_answers(answers, "நான் நாளைக்கு கூப்பிடுறேன்", "", "en", ["en", "ta"])
    assert result["detection_confidence"] == 0.65


def test_hindi_or_marathi_reports_the_choice_probability():
    answers = _answers("hi")
    answers["hindi_or_marathi"] = {"choice": "mr", "confidence": 0.36, "probabilities": {"hi": 0.32, "mr": 0.68}}
    result = decide_from_answers(answers, "मी उद्या परत फोन करतो.", "", "hi", ["hi", "mr"])
    assert result["detected_language"] == "mr"
    assert result["detection_confidence"] == 0.68


def test_odia_label_keeps_the_backend_code():
    result = decide_from_answers(_answers("or"), "ମୋ ଟିକେଟ କେବେ ଆସିବ", "", "en", ["en", "od"])
    assert result["target_language"] == "od"
    assert result["detected_language"] == "od"


def test_data_tokens_count_digits_number_words_and_spelled_letters():
    assert data_tokens("G P one five one two five S") == (8, 8)
    assert data_tokens("where is my order") == (0, 4)


def test_data_tokens_keep_words_whole_and_weigh_digit_groups():
    assert data_tokens("I won't go.") == (0, 3)
    assert data_tokens("मेरा नंबर है 98765 43210") == (4, 7)
    assert data_tokens("The card expires in 2027") == (2, 6)


def test_request_sends_only_the_transcript_and_asks_per_label():
    body = build_jev_request("jev-1.13.0", "where is my bill", ["hi", "en", "od"])
    questions = body["questions"]
    assert body["model"] == "jev-1.13.0"
    assert body["state"] == {"turn": {"transcript": "where is my bill"}}
    assert {"language", "has_clause", "filler_only", "data_readout", "english_frame"} <= set(questions)
    assert {"wants_hi", "refuses_hi", "wants_en", "refuses_en", "wants_or", "refuses_or"} <= set(questions)
    assert "hindi_or_marathi" not in questions
    assert set(questions["language"]["criteria"]) == set(LANGS) | {"other"}


def test_devanagari_turns_get_the_hindi_or_marathi_question():
    assert "hindi_or_marathi" in build_jev_request("jev-1.13.0", "मी बरा आहे", ["hi", "mr"])["questions"]


def test_questions_say_how_requests_refusals_and_marathi_read():
    questions = build_jev_request("jev-1.13.0", "मला इंग्रजी येत नाही", ["hi", "mr", "en"])["questions"]
    assert "request, never a refusal" in questions["refuses_en"]["criteria"]["false"]
    assert "prefer" in questions["wants_en"]["criteria"]["true"]
    assert "आहे" in questions["hindi_or_marathi"]["criteria"]["mr"]["signs"]
    assert "only बोला" in questions["hindi_or_marathi"]["instructions"]["how_to_decide"]
    assert "कोण" in questions["hindi_or_marathi"]["criteria"]["mr"]["signs"]
    explicit = build_jev_explicit_request("jev-1.13.0", "Hindi is better for me", "", "", ["hi", "en"])["questions"]
    assert "prefer" in explicit["wants_hi"]["criteria"]["true"]


def test_flatten_answers_reads_the_api_shape():
    flat = flatten_answers(
        {
            "has_clause": {"type": "noul", "noul": 0.97},
            "language": {"type": "choice", "choice": "en", "confidence": 0.9, "probabilities": {"en": 0.93}},
        }
    )
    assert flat["has_clause"] == 0.97
    assert flat["language"] == {"choice": "en", "confidence": 0.9, "probabilities": {"en": 0.93}}


def _explicit(lang="en", offer=None, yes=0.05, wants=(), selects=(), refuses=None):
    a = {"language": {"choice": lang, "confidence": 0.9, "probabilities": {lang: 0.9}}, "english_frame": 0.05}
    if offer:
        a["agent_offer"] = {"choice": offer, "confidence": 0.9, "probabilities": {}}
        a["says_yes"] = yes
    for code in LANGS:
        a[f"wants_{code}"] = 0.9 if code in wants else 0.02
        a[f"selects_{code}"] = 0.9 if code in selects else 0.02
        a[f"refuses_{code}"] = 0.9 if code == refuses else 0.02
    return a


EXPLICIT_CASES = [
    (
        "live-only-selection-switches",
        "Okay.",
        "Odia.",
        "en",
        ["en", "od"],
        _explicit(offer="several", selects=("or",)),
        ("switch", "od", "agent_prompted_selection"),
    ),
    (
        "yes-to-one-offered-language-switches",
        "Yes.",
        "yes",
        "en",
        ["en", "hi"],
        _explicit(offer="hi", yes=0.95),
        ("switch", "hi", "agent_prompted_selection"),
    ),
    (
        "yes-to-several-languages-is-ambiguous",
        "Yes.",
        "yes",
        "en",
        ["en", "hi"],
        _explicit(offer="several", yes=0.95),
        ("ambiguous", None, "none"),
    ),
    (
        "live-only-request-switches",
        "Hmm.",
        "Please talk in Hindi.",
        "en",
        ["en", "hi"],
        _explicit(wants=("hi",)),
        ("switch", "hi", "direct_request"),
    ),
    (
        "choosing-the-active-language",
        "Okay.",
        "Hindi.",
        "hi",
        ["en", "hi"],
        _explicit(offer="several", selects=("hi",)),
        ("already_active", None, "agent_prompted_selection"),
    ),
    (
        "named-refusal-stays",
        "मुझे हिंदी नहीं आती",
        "मुझे हिंदी नहीं आती",
        "en",
        ["en", "hi"],
        _explicit(lang="hi", wants=("hi",), refuses="hi"),
        ("no_request", None, "none"),
    ),
    (
        "speaking-another-language-stays",
        "मेरा order कहाँ है?",
        "",
        "en",
        ["en", "hi"],
        _explicit(lang="hi"),
        ("no_request", None, "none"),
    ),
    (
        "two-languages-is-ambiguous",
        "Hindi or English, either is fine.",
        "",
        "en",
        ["en", "hi"],
        _explicit(lang="hi", wants=("hi", "en")),
        ("ambiguous", None, "none"),
    ),
]


@pytest.mark.parametrize(
    "transcript, live, active, labels, answers, expected",
    [case[1:] for case in EXPLICIT_CASES],
    ids=[case[0] for case in EXPLICIT_CASES],
)
def test_explicit_rules_follow_the_explicit_prompt(transcript, live, active, labels, answers, expected):
    result = decide_explicit_from_answers(answers, transcript, live, active, labels)
    assert (result["request_status"], result["target_language"], result["request_source"]) == expected
    assert result["explicit_request"] is (expected[0] in ("switch", "already_active"))


def test_explicit_request_sends_both_transcripts_and_the_agent_turn():
    body = build_jev_explicit_request("jev-1.13.0", "Okay.", "Hindi.", "Hindi or English?", ["hi", "en"])
    questions = body["questions"]
    assert body["state"] == {
        "turn": {"transcript": "Okay.", "live_transcript": "Hindi."},
        "agent_turn": "Hindi or English?",
    }
    assert {"language", "agent_offer", "says_yes", "wants_hi", "selects_hi", "refuses_en"} <= set(questions)
    assert "has_clause" not in questions
    assert set(questions["agent_offer"]["criteria"]) == {"none", "several", "hi", "en"}
    assert "how_to_read" in questions["refuses_hi"]["instructions"]

    no_turn = build_jev_explicit_request("jev-1.13.0", "Hindi please", "", "", ["hi", "en"])["questions"]
    assert "agent_offer" not in no_turn and "selects_hi" not in no_turn and "wants_hi" in no_turn


def _switcher(monkeypatch, labels=("hi", "en"), explicit_only=False):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    return JevLanguageSwitcher(list(labels), model=JEV_MODEL, explicit_only=explicit_only)


def _raw(answers):
    raw = {}
    for key, value in answers.items():
        if isinstance(value, dict):
            raw[key] = {"type": "choice", **value}
        else:
            raw[key] = {"type": "noul", "noul": value}
    return raw


async def test_decide_maps_jev_answers_and_tallies_usage(monkeypatch):
    switcher = _switcher(monkeypatch)
    switcher._post = AsyncMock(
        return_value={
            "answers": _raw(_answers("hi", english=0.8)),
            "usage": {"input_tokens": 1900, "output_tokens": 60},
        }
    )
    with patch(REQUEST_LOG):
        result = await switcher.decide("Theek hai, please send me the invoice", "", "hi")
    assert result["target_language"] == "en"
    payload = switcher._post.await_args.args[0]
    assert payload["model"] == "jev-1.13.0"
    assert payload["state"] == {"turn": {"transcript": "Theek hai, please send me the invoice"}}
    assert switcher.usage_totals["input_tokens"] == 1900
    assert switcher.usage_totals["requests"] == 1
    assert switcher.models_used == [JEV_MODEL]
    assert switcher.latency_ms is not None


async def test_explicit_decide_returns_the_explicit_verdict(monkeypatch):
    switcher = _switcher(monkeypatch, labels=("en", "od"), explicit_only=True)
    switcher._post = AsyncMock(return_value={"answers": _raw(_explicit(offer="several", selects=("or",)))})
    with patch(REQUEST_LOG):
        result = await switcher.decide("Okay.", "Odia.", "en", None, "Which language would you like?")
    assert (result["request_status"], result["target_language"]) == ("switch", "od")
    assert result["explicit_request"] is True
    state = switcher._post.await_args.args[0]["state"]
    assert state["agent_turn"] == "Which language would you like?"
    assert state["turn"]["live_transcript"] == "Odia."


async def test_request_row_shows_what_jev_was_sent(monkeypatch):
    explicit = _switcher(monkeypatch, labels=("en", "hi"), explicit_only=True)
    explicit._post = AsyncMock(return_value={"answers": _raw(_explicit(offer="hi", yes=0.95))})
    with patch(REQUEST_LOG) as log:
        await explicit.decide("Yes.", "Yes, please.", "en", None, "Shall we continue in Hindi?")
    row = log.call_args_list[0].kwargs["message"]
    assert row["agent_turn"] == "Shall we continue in Hindi?"
    assert row["live_transcript"] == "Yes, please."

    auto = _switcher(monkeypatch)
    auto._post = AsyncMock(return_value={"answers": _raw(_answers("hi"))})
    with patch(REQUEST_LOG) as log:
        await auto.decide("namaste ji", "", "hi", None, "Hello?")
    assert set(log.call_args_list[0].kwargs["message"]) == {"transcript", "available_languages"}


async def test_decide_skips_jev_without_a_detector_transcript(monkeypatch):
    switcher = _switcher(monkeypatch)
    switcher._post = AsyncMock()
    assert await switcher.decide("   ", "live text", "hi") is None
    switcher._post.assert_not_awaited()


async def test_two_jev_failures_fall_back_to_the_llm_judge(monkeypatch):
    switcher = _switcher(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("LANGUAGE_SWITCH_LLM_API_KEY", raising=False)
    switcher._post = AsyncMock(side_effect=aiohttp.ClientError("unavailable"))
    fake_llm = MagicMock()
    fake_llm.generate = AsyncMock(return_value=('{"target_language": "en", "detected_language": "en"}', {}))
    with patch(f"{MOD}.LiteLLM", return_value=fake_llm), patch(REQUEST_LOG):
        assert await switcher.decide("can you speak english", "", "hi") is None
        assert await switcher.decide("can you speak english", "", "hi") is None
        result = await switcher.decide("can you speak english", "", "hi")
    assert result["target_language"] == "en"
    assert switcher._post.await_count == 2
    assert switcher.model.startswith("anthropic/")
    fake_llm.generate.assert_awaited()


async def test_aclose_closes_the_http_session(monkeypatch):
    switcher = _switcher(monkeypatch)
    session = switcher._http()
    await switcher.aclose()
    assert session.closed


def test_jev_judge_is_chosen_only_when_it_can_judge_the_call(monkeypatch):
    with patch("bolna.helpers.language_switcher.LiteLLM"):
        monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
        assert isinstance(build_language_switcher(["hi", "en", "od"], model=JEV_MODEL), JevLanguageSwitcher)
        explicit = build_language_switcher(["hi", "en"], model=JEV_MODEL, explicit_only=True)
        assert isinstance(explicit, JevLanguageSwitcher) and explicit.explicit_only

        judge = build_language_switcher(["hi", "fr"], model=JEV_MODEL)
        assert not isinstance(judge, JevLanguageSwitcher)
        assert not judge.model.startswith("typesafe/")

        monkeypatch.delenv("TYPESAFE_API_KEY")
        assert not isinstance(build_language_switcher(["hi", "en"], model=JEV_MODEL), JevLanguageSwitcher)
        assert not isinstance(build_language_switcher(["hi", "en"]), JevLanguageSwitcher)
