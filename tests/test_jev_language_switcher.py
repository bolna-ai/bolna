"""Jev language-switch judge: the questions and rules, the decide() path, fallback and selection."""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from bolna.helpers.jev_language_switcher import (
    LANGS,
    JevLanguageSwitcher,
    build_jev_request,
    build_language_switcher,
    data_tokens,
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


def test_odia_label_keeps_the_backend_code():
    result = decide_from_answers(_answers("or"), "ମୋ ଟିକେଟ କେବେ ଆସିବ", "", "en", ["en", "od"])
    assert result["target_language"] == "od"
    assert result["detected_language"] == "od"


def test_data_tokens_count_digits_number_words_and_spelled_letters():
    assert data_tokens("G P one five one two five S") == (8, 8)
    assert data_tokens("where is my order") == (0, 4)


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


def test_flatten_answers_reads_the_api_shape():
    flat = flatten_answers(
        {
            "has_clause": {"type": "noul", "noul": 0.97},
            "language": {"type": "choice", "choice": "en", "confidence": 0.9, "probabilities": {"en": 0.93}},
        }
    )
    assert flat["has_clause"] == 0.97
    assert flat["language"] == {"choice": "en", "confidence": 0.9, "probabilities": {"en": 0.93}}


def _switcher(monkeypatch, labels=("hi", "en")):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    return JevLanguageSwitcher(list(labels), model=JEV_MODEL)


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

        for kwargs in (
            {"available_labels": ["hi", "en"], "explicit_only": True},
            {"available_labels": ["hi", "fr"]},
        ):
            judge = build_language_switcher(model=JEV_MODEL, **kwargs)
            assert not isinstance(judge, JevLanguageSwitcher)
            assert not judge.model.startswith("typesafe/")

        monkeypatch.delenv("TYPESAFE_API_KEY")
        assert not isinstance(build_language_switcher(["hi", "en"], model=JEV_MODEL), JevLanguageSwitcher)
        assert not isinstance(build_language_switcher(["hi", "en"]), JevLanguageSwitcher)
