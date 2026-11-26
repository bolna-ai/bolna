from unittest.mock import patch

import pytest

from bolna.llms.azure_llm import AzureLLM, is_azure_openai_endpoint


@pytest.fixture(autouse=True)
def default_domains(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_ALLOWED_DOMAINS", raising=False)


@pytest.mark.parametrize(
    "url",
    [
        "https://acme.openai.azure.com/",
        "https://acme.openai.azure.com",
        "https://ACME.cognitiveservices.azure.com/",
        " https://acme.openai.azure.com:443/openai ",
    ],
)
def test_azure_openai_hosts_are_accepted(url):
    assert is_azure_openai_endpoint(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://acme.openai.azure.com/",
        "https://acme.openai.azure.com.attacker.example/",
        "https://attacker.example/#.openai.azure.com",
        "https://attacker.example/?x=.openai.azure.com",
        "https://acme.openai.azure.com@10.0.0.5/",
        "https://user@acme.openai.azure.com/",
        "https://acme.openai.azure.com:8080/",
        "https://acme.openai.azure.com:bad/",
        "https://169.254.169.254/",
        "",
        None,
    ],
)
def test_any_other_endpoint_is_refused(url):
    assert not is_azure_openai_endpoint(url)


def test_an_llm_on_a_non_azure_endpoint_is_never_built():
    with patch("bolna.llms.azure_llm.AsyncAzureOpenAI") as client, pytest.raises(ValueError):
        AzureLLM(model="gpt-4.1-mini", llm_key="k", base_url="https://169.254.169.254/")
    client.assert_not_called()


def test_a_non_azure_endpoint_from_the_environment_is_refused_too(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://attacker.example/")
    with patch("bolna.llms.azure_llm.AsyncAzureOpenAI") as client, pytest.raises(ValueError):
        AzureLLM(model="gpt-4.1-mini", llm_key="k")
    client.assert_not_called()


def test_an_azure_endpoint_is_dialled():
    with patch("bolna.llms.azure_llm.AsyncAzureOpenAI") as client:
        llm = AzureLLM(model="gpt-4.1-mini", llm_key="k", base_url="https://acme.openai.azure.com")
    assert client.call_args.kwargs["azure_endpoint"] == "https://acme.openai.azure.com"
    assert llm.llm_host == "acme.openai.azure.com"


def test_the_allowed_domains_can_be_replaced(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ALLOWED_DOMAINS", " .Gateway.Example , azure-api.net,")
    assert is_azure_openai_endpoint("https://gateway.example/")
    assert is_azure_openai_endpoint("https://eu.gateway.example/")
    assert is_azure_openai_endpoint("https://acme.azure-api.net/")
    assert not is_azure_openai_endpoint("https://notgateway.example/")
    assert not is_azure_openai_endpoint("https://acme.openai.azure.com/")


def test_an_empty_override_keeps_the_default_domains(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ALLOWED_DOMAINS", " , ")
    assert is_azure_openai_endpoint("https://acme.openai.azure.com/")
    assert not is_azure_openai_endpoint("https://acme.azure-api.net/")
