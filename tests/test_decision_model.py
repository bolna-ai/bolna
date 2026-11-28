import json
import os
from unittest.mock import patch

import httpx
import pytest

from bolna.llms.decision_model import DecisionError, DecisionModel

KEYS = {"TYPESAFE_API_KEY": "ts-key", "OPENAI_API_KEY": "oa-key", "GOOGLE_API_KEY": "g-key"}
CHOICES = {"yes": "the caller agreed", "no": "the caller declined"}


def decision_model(provider, reply, status=200):
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(status, json=reply)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with (
        patch.dict(os.environ, KEYS),
        patch("bolna.llms.decision_model.get_shared_http_client", return_value=client),
    ):
        return DecisionModel(provider), sent


def keyed(choice):
    return {"answers": {"decision": {"choice": choice, "confidence": 0.9}}, "usage": {"input_tokens": 120}}


async def test_jev_is_asked_one_choice_question_about_the_context():
    model, sent = decision_model("typesafe", keyed("yes"))

    decision = await model.choose("Agent: shall I book it?\nCaller: go ahead", "Did the caller agree?", CHOICES)

    body = json.loads(sent[0].content)
    assert str(sent[0].url) == "https://api.typesafe.ai/v1/systemone"
    assert sent[0].headers["authorization"] == "Bearer ts-key"
    assert body["state"] == "Agent: shall I book it?\nCaller: go ahead"
    assert body["questions"]["decision"] == {
        "type": "choice",
        "instructions": "Did the caller agree?",
        "criteria": CHOICES,
    }
    assert (decision.choice, decision.confidence, decision.usage["input_tokens"]) == ("yes", 0.9, 120)


async def test_gemini_takes_the_same_question_under_its_own_key_and_fields():
    reply = {**keyed("no"), "usage": {"total_input_tokens": 103, "total_output_tokens": 0}}
    model, sent = decision_model("gemini-decisions", reply)

    decision = await model.choose("ctx", "q", CHOICES)

    body = json.loads(sent[0].content)
    assert sent[0].headers["x-goog-api-key"] == "g-key"
    assert body["model"] == "gemini-decision-1" and body["input"] == "ctx" and "state" not in body
    assert decision.usage == {"input_tokens": 103, "output_tokens": 0}


async def test_openai_decisions_lists_its_questions_and_answers():
    reply = {"answers": [{"name": "decision", "type": "choice", "choice": "no", "confidence": 0.7}]}
    model, sent = decision_model("openai-decisions", reply)

    decision = await model.choose("ctx", "q", CHOICES)

    question = json.loads(sent[0].content)["questions"][0]
    assert question["choices"][0] == {"value": "yes", "description": "the caller agreed"}
    assert (decision.choice, decision.confidence) == ("no", 0.7)


@pytest.mark.parametrize(
    "provider, reply",
    [
        ("typesafe", keyed("maybe")),
        ("openai-decisions", {"answers": [{"name": "decision", "type": "refusal", "refusal": "cannot say"}]}),
    ],
)
async def test_an_answer_that_is_not_one_of_the_choices_is_an_error(provider, reply):
    model, _ = decision_model(provider, reply)

    with pytest.raises(DecisionError):
        await model.choose("ctx", "q", CHOICES)


def test_a_provider_without_its_key_is_unavailable():
    with patch.dict(os.environ, {}, clear=True):
        assert not DecisionModel("typesafe").available
        assert DecisionModel("typesafe", key="ts-key").available
