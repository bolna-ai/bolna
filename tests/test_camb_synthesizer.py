"""Comprehensive tests for CAMB AI Bolna TTS synthesizer.

Tests cover:
- Initialization (env key, explicit key, missing key, defaults, user_instructions)
- HTTP generation (API call, payload structure, error status codes)
- Short text handling (returns empty bytes instead of audible padding)
- Timeout and network error handling
- Session lifecycle (lazy creation, reuse, cleanup)
- user_instructions support (mars-instruct only)
- Character counting
- generate() async generator (yields audio packets with meta_info)
- push() queueing
- open_connection no-op
- Integration with real API (skipped without CAMB_API_KEY)
"""

from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest
from dotenv import load_dotenv

from bolna.synthesizer.camb_synthesizer import CambSynthesizer, CAMB_TTS_URL

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CAMB_API_KEY = os.getenv("CAMB_API_KEY")
AUDIO_SAMPLE_PATH = Path(__file__).resolve().parent.parent.parent / "yt-dlp" / "voices" / "original" / "sabrina-original-clip.mp3"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_api_key():
    return "test-api-key-12345"


@pytest.fixture
def synthesizer(mock_api_key):
    with patch.dict("os.environ", {"CAMB_API_KEY": mock_api_key}):
        return CambSynthesizer(
            voice_id=147320,
            voice="test-voice",
            model="mars-flash",
            language="en-us",
            sampling_rate=8000,
        )


@pytest.fixture
def mock_http_response():
    """Create a mock aiohttp response with configurable status and audio data."""
    def _make(status=200, audio_chunks=None):
        mock_response = AsyncMock()
        mock_response.status = status
        mock_response.text = AsyncMock(return_value="error body")

        chunks = audio_chunks or [b"\x00" * 100]

        async def mock_iter():
            for chunk in chunks:
                yield chunk

        mock_response.content = AsyncMock()
        mock_response.content.iter_chunked = lambda s: mock_iter()

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=None)

        return mock_post_ctx, mock_response

    return _make


@pytest.fixture
def mock_session(mock_http_response):
    """Create a mock aiohttp session with a successful response."""
    post_ctx, _ = mock_http_response(status=200)
    session = AsyncMock()
    session.post = MagicMock(return_value=post_ctx)
    session.closed = False
    return session


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_with_env_key(self, synthesizer):
        assert synthesizer.api_key == "test-api-key-12345"
        assert synthesizer.voice_id == 147320
        assert synthesizer.model == "mars-flash"
        assert synthesizer.language == "en-us"
        assert synthesizer.sample_rate == 8000

    def test_init_with_explicit_key(self):
        synth = CambSynthesizer(
            voice_id=147320,
            voice="test",
            synthesizer_key="explicit-key",
        )
        assert synth.api_key == "explicit-key"

    def test_init_without_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key not found"):
                CambSynthesizer(voice_id=147320, voice="test")

    def test_default_model(self):
        synth = CambSynthesizer(voice_id=1, voice="v", synthesizer_key="k")
        assert synth.model == "mars-flash"

    def test_default_language(self):
        synth = CambSynthesizer(voice_id=1, voice="v", synthesizer_key="k")
        assert synth.language == "en-us"

    def test_stream_respected(self):
        synth = CambSynthesizer(voice_id=1, voice="v", synthesizer_key="k", stream=True)
        assert synth.stream is True

    def test_stream_default_false(self):
        synth = CambSynthesizer(voice_id=1, voice="v", synthesizer_key="k")
        assert synth.stream is False

    def test_voice_id_cast_to_int(self):
        synth = CambSynthesizer(voice_id="147320", voice="v", synthesizer_key="k")
        assert synth.voice_id == 147320
        assert isinstance(synth.voice_id, int)

    def test_sampling_rate_cast_from_string(self):
        synth = CambSynthesizer(voice_id=1, voice="v", synthesizer_key="k", sampling_rate="16000")
        assert synth.sample_rate == 16000

    def test_session_initially_none(self, synthesizer):
        assert synthesizer.session is None

    def test_user_instructions_default_none(self, synthesizer):
        assert synthesizer.user_instructions is None

    def test_user_instructions_set(self):
        synth = CambSynthesizer(
            voice_id=1,
            voice="v",
            model="mars-instruct",
            synthesizer_key="k",
            user_instructions="Speak in a calm tone",
        )
        assert synth.user_instructions == "Speak in a calm tone"

    def test_synthesized_characters_starts_zero(self, synthesizer):
        assert synthesizer.get_synthesized_characters() == 0


# ---------------------------------------------------------------------------
# Short text handling
# ---------------------------------------------------------------------------


class TestShortText:
    @pytest.mark.asyncio
    async def test_empty_string_returns_empty_bytes(self, synthesizer):
        assert await synthesizer.synthesize("") == b""

    @pytest.mark.asyncio
    async def test_single_char_returns_empty_bytes(self, synthesizer):
        assert await synthesizer.synthesize(".") == b""

    @pytest.mark.asyncio
    async def test_two_chars_returns_empty_bytes(self, synthesizer):
        assert await synthesizer.synthesize("Hi") == b""

    @pytest.mark.asyncio
    async def test_three_chars_does_not_skip(self, synthesizer, mock_session):
        """Exactly 3 chars should proceed to API call, not be skipped."""
        synthesizer.session = mock_session
        result = await synthesizer.synthesize("Hey")
        assert len(result) > 0
        mock_session.post.assert_called_once()


# ---------------------------------------------------------------------------
# HTTP generation & payload
# ---------------------------------------------------------------------------


class TestHTTPGeneration:
    @pytest.mark.asyncio
    async def test_synthesize_returns_audio(self, synthesizer, mock_session):
        synthesizer.session = mock_session
        result = await synthesizer.synthesize("Hello world")
        assert isinstance(result, bytes)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_payload_structure(self, synthesizer, mock_session):
        """Verify the payload sent to the API matches expected format."""
        synthesizer.session = mock_session
        await synthesizer.synthesize("Hello world")

        call_args = mock_session.post.call_args
        assert call_args[0][0] == CAMB_TTS_URL
        payload = call_args[1]["json"]
        assert payload["text"] == "Hello world"
        assert payload["voice_id"] == 147320
        assert payload["language"] == "en-us"
        assert payload["speech_model"] == "mars-flash"
        assert payload["output_configuration"]["format"] == "pcm_s16le"
        assert payload["output_configuration"]["sample_rate"] == 8000
        assert "voice_settings" not in payload

    @pytest.mark.asyncio
    async def test_headers(self, synthesizer, mock_session):
        synthesizer.session = mock_session
        await synthesizer.synthesize("Hello world")

        headers = mock_session.post.call_args[1]["headers"]
        assert headers["x-api-key"] == "test-api-key-12345"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_long_text_truncated(self, synthesizer, mock_session):
        """Text over 3000 chars should be truncated."""
        synthesizer.session = mock_session
        long_text = "a" * 5000
        await synthesizer.synthesize(long_text)

        payload = mock_session.post.call_args[1]["json"]
        assert len(payload["text"]) == 3000

    @pytest.mark.asyncio
    async def test_character_count_increments(self, synthesizer, mock_session):
        synthesizer.session = mock_session
        await synthesizer.synthesize("Hello")
        assert synthesizer.get_synthesized_characters() == 5
        await synthesizer.synthesize("World!")
        assert synthesizer.get_synthesized_characters() == 11


# ---------------------------------------------------------------------------
# Error status codes
# ---------------------------------------------------------------------------


class TestErrorStatusCodes:
    @pytest.mark.asyncio
    async def test_401_raises_value_error(self, synthesizer, mock_http_response):
        post_ctx, _ = mock_http_response(status=401)
        session = AsyncMock()
        session.post = MagicMock(return_value=post_ctx)
        synthesizer.session = session

        with pytest.raises(ValueError, match="Invalid Camb.ai API key"):
            await synthesizer.synthesize("Hello world")

    @pytest.mark.asyncio
    async def test_403_raises_value_error(self, synthesizer, mock_http_response):
        post_ctx, _ = mock_http_response(status=403)
        session = AsyncMock()
        session.post = MagicMock(return_value=post_ctx)
        synthesizer.session = session

        with pytest.raises(ValueError, match="not accessible"):
            await synthesizer.synthesize("Hello world")

    @pytest.mark.asyncio
    async def test_404_raises_value_error(self, synthesizer, mock_http_response):
        post_ctx, _ = mock_http_response(status=404)
        session = AsyncMock()
        session.post = MagicMock(return_value=post_ctx)
        synthesizer.session = session

        with pytest.raises(ValueError, match="Invalid voice ID"):
            await synthesizer.synthesize("Hello world")

    @pytest.mark.asyncio
    async def test_429_raises_value_error(self, synthesizer, mock_http_response):
        post_ctx, _ = mock_http_response(status=429)
        session = AsyncMock()
        session.post = MagicMock(return_value=post_ctx)
        synthesizer.session = session

        with pytest.raises(ValueError, match="rate limit"):
            await synthesizer.synthesize("Hello world")

    @pytest.mark.asyncio
    async def test_500_raises_value_error(self, synthesizer, mock_http_response):
        post_ctx, _ = mock_http_response(status=500)
        session = AsyncMock()
        session.post = MagicMock(return_value=post_ctx)
        synthesizer.session = session

        with pytest.raises(ValueError, match="API error 500"):
            await synthesizer.synthesize("Hello world")


# ---------------------------------------------------------------------------
# Timeout and network errors
# ---------------------------------------------------------------------------


class TestTimeoutAndNetworkErrors:
    @pytest.mark.asyncio
    async def test_timeout_raises_asyncio_timeout_error(self, synthesizer):
        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=asyncio.TimeoutError())
        synthesizer.session = mock_session

        with pytest.raises(asyncio.TimeoutError):
            await synthesizer.synthesize("Hello world test")

    @pytest.mark.asyncio
    async def test_client_error_raises_aiohttp_error(self, synthesizer):
        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            side_effect=aiohttp.ClientError("connection failed")
        )
        synthesizer.session = mock_session

        with pytest.raises(aiohttp.ClientError):
            await synthesizer.synthesize("Hello world test")


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    @pytest.mark.asyncio
    async def test_session_created_lazily_on_first_call(self, synthesizer):
        """Session should be None until the first synthesis call."""
        assert synthesizer.session is None

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_session = AsyncMock()
            post_ctx = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200

            async def mock_iter():
                yield b"\x00" * 10

            mock_response.content = AsyncMock()
            mock_response.content.iter_chunked = lambda s: mock_iter()
            post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            post_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=post_ctx)
            mock_cls.return_value = mock_session

            await synthesizer.synthesize("Hello world")

            mock_cls.assert_called_once()
            assert synthesizer.session is mock_session

    @pytest.mark.asyncio
    async def test_session_reused_across_calls(self, synthesizer, mock_session):
        """Same session should be reused for multiple calls."""
        synthesizer.session = mock_session

        await synthesizer.synthesize("Hello")
        await synthesizer.synthesize("World")

        assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_closes_session(self, synthesizer):
        mock_session = AsyncMock()
        mock_session.closed = False
        synthesizer.session = mock_session

        await synthesizer.cleanup()

        mock_session.close.assert_called_once()
        assert synthesizer.session is None

    @pytest.mark.asyncio
    async def test_cleanup_noop_when_no_session(self, synthesizer):
        await synthesizer.cleanup()  # Should not raise

    @pytest.mark.asyncio
    async def test_cleanup_noop_when_session_already_closed(self, synthesizer):
        mock_session = AsyncMock()
        mock_session.closed = True
        synthesizer.session = mock_session

        await synthesizer.cleanup()
        mock_session.close.assert_not_called()


# ---------------------------------------------------------------------------
# user_instructions (mars-instruct)
# ---------------------------------------------------------------------------


class TestUserInstructions:
    @pytest.mark.asyncio
    async def test_included_for_mars_instruct(self, mock_api_key):
        with patch.dict("os.environ", {"CAMB_API_KEY": mock_api_key}):
            synth = CambSynthesizer(
                voice_id=147320,
                voice="test",
                model="mars-instruct",
                user_instructions="Speak calmly",
            )

        mock_session = AsyncMock()
        post_ctx = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200

        async def mock_iter():
            yield b"\x00" * 100

        mock_response.content = AsyncMock()
        mock_response.content.iter_chunked = lambda s: mock_iter()
        post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        post_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(return_value=post_ctx)
        synth.session = mock_session

        await synth.synthesize("Hello world test")

        payload = mock_session.post.call_args[1]["json"]
        assert payload["user_instructions"] == "Speak calmly"

    @pytest.mark.asyncio
    async def test_excluded_for_mars_flash(self, mock_api_key):
        with patch.dict("os.environ", {"CAMB_API_KEY": mock_api_key}):
            synth = CambSynthesizer(
                voice_id=147320,
                voice="test",
                model="mars-flash",
                user_instructions="Speak calmly",
            )

        mock_session = AsyncMock()
        post_ctx = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200

        async def mock_iter():
            yield b"\x00" * 100

        mock_response.content = AsyncMock()
        mock_response.content.iter_chunked = lambda s: mock_iter()
        post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        post_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(return_value=post_ctx)
        synth.session = mock_session

        await synth.synthesize("Hello world test")

        payload = mock_session.post.call_args[1]["json"]
        assert "user_instructions" not in payload

    @pytest.mark.asyncio
    async def test_excluded_when_none(self, mock_api_key):
        """Even with mars-instruct, user_instructions=None should not be in payload."""
        with patch.dict("os.environ", {"CAMB_API_KEY": mock_api_key}):
            synth = CambSynthesizer(
                voice_id=147320,
                voice="test",
                model="mars-instruct",
            )

        mock_session = AsyncMock()
        post_ctx = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200

        async def mock_iter():
            yield b"\x00" * 100

        mock_response.content = AsyncMock()
        mock_response.content.iter_chunked = lambda s: mock_iter()
        post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        post_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(return_value=post_ctx)
        synth.session = mock_session

        await synth.synthesize("Hello world test")

        payload = mock_session.post.call_args[1]["json"]
        assert "user_instructions" not in payload


# ---------------------------------------------------------------------------
# generate() async generator
# ---------------------------------------------------------------------------


class TestGenerateLoop:
    @pytest.mark.asyncio
    async def test_generate_yields_audio_packet(self, synthesizer):
        fake_audio = b"\x00\x01" * 50
        with patch.object(
            synthesizer,
            "_CambSynthesizer__generate_http",
            new_callable=AsyncMock,
            return_value=fake_audio,
        ), patch.object(
            synthesizer,
            "should_synthesize_response",
            return_value=True,
        ):
            synthesizer.internal_queue.put_nowait({
                "data": "Hello",
                "meta_info": {"sequence_id": None, "end_of_llm_stream": True},
            })

            packets = []
            async for packet in synthesizer.generate():
                packets.append(packet)
                break  # Only get one packet

            assert len(packets) == 1

    @pytest.mark.asyncio
    async def test_generate_sets_first_chunk_flag(self, synthesizer):
        fake_audio = b"\x00" * 50
        with patch.object(
            synthesizer,
            "_CambSynthesizer__generate_http",
            new_callable=AsyncMock,
            return_value=fake_audio,
        ), patch.object(
            synthesizer,
            "should_synthesize_response",
            return_value=True,
        ):
            synthesizer.internal_queue.put_nowait({
                "data": "Hello",
                "meta_info": {"sequence_id": None, "end_of_llm_stream": True},
            })

            async for _ in synthesizer.generate():
                break

            # After end_of_llm_stream, first_chunk_generated resets to False
            assert synthesizer.first_chunk_generated is False


# ---------------------------------------------------------------------------
# push()
# ---------------------------------------------------------------------------


class TestPush:
    @pytest.mark.asyncio
    async def test_push_adds_to_queue(self, synthesizer):
        msg = {"data": "hello", "meta_info": {}}
        await synthesizer.push(msg)
        assert not synthesizer.internal_queue.empty()
        queued = synthesizer.internal_queue.get_nowait()
        assert queued == msg


# ---------------------------------------------------------------------------
# open_connection()
# ---------------------------------------------------------------------------


class TestOpenConnection:
    @pytest.mark.asyncio
    async def test_open_connection_is_noop(self, synthesizer):
        await synthesizer.open_connection()  # Should not raise


# ---------------------------------------------------------------------------
# supports_websocket() and get_sleep_time()
# ---------------------------------------------------------------------------


class TestMethodOverrides:
    def test_supports_websocket_returns_false(self, synthesizer):
        assert synthesizer.supports_websocket() is False

    def test_get_sleep_time_non_streaming(self):
        synth = CambSynthesizer(voice_id=1, voice="v", synthesizer_key="k", stream=False)
        assert synth.get_sleep_time() == 0.2

    def test_get_sleep_time_streaming(self):
        synth = CambSynthesizer(voice_id=1, voice="v", synthesizer_key="k", stream=True)
        assert synth.get_sleep_time() == 0.01


# ---------------------------------------------------------------------------
# Streaming generate()
# ---------------------------------------------------------------------------


class TestStreamingGenerate:
    @pytest.fixture
    def streaming_synthesizer(self, mock_api_key):
        with patch.dict("os.environ", {"CAMB_API_KEY": mock_api_key}):
            return CambSynthesizer(
                voice_id=147320,
                voice="test-voice",
                model="mars-flash",
                language="en-us",
                sampling_rate=8000,
                stream=True,
            )

    @pytest.mark.asyncio
    async def test_streaming_yields_multiple_chunks(self, streaming_synthesizer):
        """Streaming generate should yield one packet per chunk plus end marker."""
        chunks = [b"\x00\x01" * 100, b"\x02\x03" * 100]

        async def fake_stream(text):
            for c in chunks:
                yield c

        with patch.object(
            streaming_synthesizer,
            "_CambSynthesizer__generate_http_stream",
            side_effect=fake_stream,
        ), patch.object(
            streaming_synthesizer,
            "should_synthesize_response",
            return_value=True,
        ):
            streaming_synthesizer.internal_queue.put_nowait({
                "data": "Hello world test",
                "meta_info": {"sequence_id": None, "end_of_llm_stream": True},
            })

            packets = []
            async for packet in streaming_synthesizer.generate():
                packets.append(packet)
                if packet.get("meta_info", {}).get("end_of_synthesizer_stream"):
                    break

            # 2 audio chunks + 1 end-of-stream marker
            assert len(packets) == 3
            assert packets[0]["meta_info"]["is_first_chunk"] is True
            assert packets[2]["meta_info"]["end_of_synthesizer_stream"] is True

    @pytest.mark.asyncio
    async def test_streaming_pcm_alignment(self, streaming_synthesizer):
        """Odd-length HTTP chunks should be aligned to 2-byte PCM boundaries."""
        # 3 + 3 = 6 total bytes (even); each HTTP chunk is odd
        odd_chunks = [b"\x01\x02\x03", b"\x04\x05\x06"]

        mock_response = AsyncMock()
        mock_response.status = 200

        async def mock_iter(size):
            for c in odd_chunks:
                yield c

        mock_response.content = AsyncMock()
        mock_response.content.iter_chunked = mock_iter

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_post_ctx)
        streaming_synthesizer.session = mock_session

        chunks = []
        async for chunk in streaming_synthesizer._CambSynthesizer__generate_http_stream("Hello world test"):
            chunks.append(chunk)

        # All yielded chunks should be even-length (2-byte PCM aligned)
        for c in chunks:
            assert len(c) % 2 == 0

        # Total bytes should equal the full input
        total = b"".join(chunks)
        assert total == b"\x01\x02\x03\x04\x05\x06"


# ---------------------------------------------------------------------------
# Integration tests (require CAMB_API_KEY and audio sample)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CAMB_API_KEY, reason="CAMB_API_KEY not set")
class TestIntegrationLive:
    """Live integration tests against the real Camb.ai API.

    These tests use the Sabrina Carpenter voice clip for voice cloning
    context and verify real TTS output.
    """

    @pytest.fixture
    def live_synthesizer(self):
        return CambSynthesizer(
            voice_id=147320,
            voice="Aria",
            model="mars-flash",
            language="en-us",
            sampling_rate=8000,
            synthesizer_key=CAMB_API_KEY,
        )

    @pytest.mark.asyncio
    async def test_live_tts_returns_audio_bytes(self, live_synthesizer):
        """Real API call should return non-empty audio bytes."""
        try:
            result = await live_synthesizer.synthesize("Hello, this is a test of the Camb AI text to speech system.")
            assert isinstance(result, bytes)
            assert len(result) > 0
        finally:
            await live_synthesizer.cleanup()

    @pytest.mark.asyncio
    async def test_live_tts_character_count(self, live_synthesizer):
        """Character count should reflect synthesized text."""
        text = "Testing character counting."
        try:
            await live_synthesizer.synthesize(text)
            assert live_synthesizer.get_synthesized_characters() == len(text)
        finally:
            await live_synthesizer.cleanup()

    @pytest.mark.asyncio
    async def test_live_short_text_returns_empty(self, live_synthesizer):
        """Short text should return empty bytes without hitting API."""
        try:
            result = await live_synthesizer.synthesize("Hi")
            assert result == b""
        finally:
            await live_synthesizer.cleanup()

    @pytest.mark.asyncio
    async def test_live_mars_instruct_with_instructions(self):
        """mars-instruct model with user_instructions should return bytes (possibly empty if model unavailable)."""
        synth = CambSynthesizer(
            voice_id=147320,
            voice="Aria",
            model="mars-instruct",
            language="en-us",
            sampling_rate=8000,
            synthesizer_key=CAMB_API_KEY,
            user_instructions="Speak in a warm, friendly tone",
        )
        try:
            result = await synth.synthesize("Hello, this is a test.")
            assert isinstance(result, bytes)
        finally:
            await synth.cleanup()

    @pytest.mark.skipif(
        not AUDIO_SAMPLE_PATH.exists(),
        reason=f"Audio sample not found at {AUDIO_SAMPLE_PATH}",
    )
    def test_audio_sample_exists(self):
        """Verify the Sabrina Carpenter audio sample is accessible."""
        assert AUDIO_SAMPLE_PATH.stat().st_size > 0
        assert AUDIO_SAMPLE_PATH.suffix == ".mp3"
