"""A tool's own headers replace the defaults whatever their case, as HTTP header names are case-insensitive.

A form tool configured with "content-type" went out with both that and the default JSON Content-Type,
so its endpoint got a JSON body labelled as form data.
"""

import pytest
from aiohttp import web

from bolna.helpers import function_calling_helpers as fch
from bolna.helpers.function_calling_helpers import prepare_api_request, trigger_api

FORM = "application/x-www-form-urlencoded"


def _values(headers, name):
    return [value for key, value in headers.items() if key.lower() == name]


@pytest.mark.parametrize("name", ["content-type", "CONTENT-TYPE", "Content-type"])
def test_a_form_content_type_in_any_case_sends_a_form(name):
    assert prepare_api_request('{"slot": "%(slot)s"}', None, {name: FORM}, slot="3pm")["content_type"] == "form"


def test_a_lowercase_content_type_replaces_the_default():
    headers = prepare_api_request(None, None, {"content-type": FORM})["headers"]
    assert _values(headers, "content-type") == [FORM]


def test_a_lowercase_authorization_header_replaces_the_api_token():
    headers = prepare_api_request(None, "Bearer token", {"authorization": "Bearer header"})["headers"]
    assert _values(headers, "authorization") == ["Bearer header"]


def test_a_null_content_type_falls_back_to_json():
    assert prepare_api_request(None, None, {"Content-Type": None})["content_type"] == "json"


@pytest.fixture
async def recording_server():
    """HTTP server on 127.0.0.1 answering "ok"; yields its port and the Content-Type headers and body of each request."""
    received = []

    async def handle(request):
        received.append((request.headers.getall("Content-Type", []), await request.text()))
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    yield runner.addresses[0][1], received
    await runner.cleanup()


async def test_a_form_tool_with_a_lowercase_content_type_posts_a_form_body(monkeypatch, recording_server):
    port, received = recording_server
    monkeypatch.setattr(fch, "_ALLOWLISTED_HOSTS", frozenset({"localhost"}))
    monkeypatch.setattr(fch, "convert_to_request_log", lambda *a, **k: None)

    await trigger_api(
        f"http://localhost:{port}/", "POST", '{"slot": "%(slot)s"}', None, {"content-type": FORM}, {}, None, slot="3pm"
    )

    assert received == [([FORM], "slot=3pm")]
