import asyncio
import json
import base64
import time
import websockets
import os
from typing import Optional, Dict, Any
from .llm import BaseLLM
from bolna.helpers.logger_config import configure_logger
from bolna.constants import OPENAI_REALTIME_SAMPLE_RATE, PCM16_SAMPLE_WIDTH

logger = configure_logger(__name__)

# Audio duration divisor for calculating duration from bytes
AUDIO_DURATION_DIVISOR = PCM16_SAMPLE_WIDTH * OPENAI_REALTIME_SAMPLE_RATE

# Mode configurations
MODE_CONFIGS = {
    # (handles_audio_input, handles_audio_output): config
    (True, True): {
        'modalities': ["audio", "text"],
        'output_format': "audio",
        'name': "Full V2V (audio input → audio output)"
    },
    (True, False): {
        'modalities': ["audio", "text"],
        'output_format': "text",
        'name': "Custom TTS (audio input → text output)"
    },
    (False, False): {
        'modalities': ["text"],
        'output_format': "text",
        'name': "Pure LLM (text-to-text)"
    },
    (False, True): {
        'modalities': ["text", "audio"],
        'output_format': "audio",
        'name': "Custom STT (text input → audio output)"
    }
}


class OpenAIRealtimeLLM(BaseLLM):
    """
    OpenAI Realtime API as an LLM provider.

    Supports 4 operational modes based on toolchain:
    1. Full V2V: ["llm"] - OpenAI handles audio input and output
    2. Custom TTS: ["llm", "synthesizer"] - OpenAI handles audio input, outputs text for synthesizer
    3. Custom STT+TTS: ["transcriber", "llm", "synthesizer"] - OpenAI as pure LLM (text-to-text)
    4. Custom STT: ["transcriber", "llm"] - Custom transcriber, OpenAI outputs audio
    """

    def __init__(self, max_tokens=4096, buffer_size=40, model="gpt-4o-realtime-preview-2024-12-17",
                 temperature=0.8, **kwargs):
        super().__init__(max_tokens, buffer_size)

        # Detect mode from toolchain context
        self.toolchain_pipeline = kwargs.get('toolchain_pipeline', [])
        self.has_transcriber = 'transcriber' in self.toolchain_pipeline
        self.has_synthesizer = 'synthesizer' in self.toolchain_pipeline

        # Determine operational mode
        self.handles_audio_input = not self.has_transcriber
        self.handles_audio_output = not self.has_synthesizer

        # Set modalities based on mode using configuration dictionary
        mode_key = (self.handles_audio_input, self.handles_audio_output)
        mode_config = MODE_CONFIGS[mode_key]
        self.modalities = mode_config['modalities']
        self.output_format = mode_config['output_format']
        logger.info(f"OpenAI Realtime Mode: {mode_config['name']}")

        # WebSocket connection
        self.ws = None
        self.connected = False
        self.session_id = None

        # OpenAI Realtime configuration
        self.model = model
        self.voice = kwargs.get('voice', 'alloy')
        self.instructions = kwargs.get('instructions', '')
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Allow modalities override from config
        modalities_override = kwargs.get('modalities', None)
        if modalities_override:
            self.modalities = modalities_override
            logger.info(f"Modalities overridden to: {self.modalities}")

        self.turn_detection = kwargs.get('turn_detection', {
            "type": "server_vad",
            "threshold": 0.4,
            "prefix_padding_ms": 200,
            "silence_duration_ms": 700
        })
        self.input_audio_transcription = kwargs.get('input_audio_transcription', True)

        # Function calling
        self.custom_tools = kwargs.get("api_tools", None)
        if self.custom_tools is not None:
            self.trigger_function_call = True
            self.api_params = self.custom_tools['tools_params']
            self.tools = self.custom_tools['tools']
            logger.info(f"Function calling enabled with {len(self.tools)} tools")
        else:
            self.trigger_function_call = False
            self.api_params = {}
            self.tools = []

        # State management
        self.conversation_history = []
        self.response_in_progress = False
        self.current_response_item_id = None
        self.current_response_output_index = 0
        self.current_response_content_index = 0
        self.pending_function_calls = {}

        # Metrics
        self.connection_time = None
        self.audio_input_duration = 0.0
        self.audio_output_duration = 0.0
        self.input_text_tokens = 0
        self.input_audio_tokens = 0
        self.output_text_tokens = 0
        self.output_audio_tokens = 0
        self.cached_text_tokens = 0
        self.cached_audio_tokens = 0

        # Background tasks
        self.background_tasks = []
        self.started_streaming = False

        # Conversation state (for TaskManager orchestration)
        self.conversation_ended = False

        # API key
        self.api_key = kwargs.get('llm_key', os.getenv('OPENAI_API_KEY'))
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set llm_key or OPENAI_API_KEY environment variable")

    async def connect(self) -> bool:
        """Establish WebSocket connection to OpenAI Realtime API."""
        try:
            url = f"wss://api.openai.com/v1/realtime?model={self.model}"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }

            logger.info(f"Connecting to OpenAI Realtime: {self.model}")
            self.ws = await websockets.connect(url, additional_headers=headers)
            self.connected = True
            self.connection_time = time.time()
            logger.info("Connected to OpenAI Realtime API")

            # Wait for session.created event
            event = await self._receive_event()
            if event.get('type') == 'session.created':
                self.session_id = event.get('session', {}).get('id')
                logger.info(f"Session created: {self.session_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to OpenAI Realtime: {e}")
            self.connected = False
            return False

    async def configure_session(self, additional_config: Dict[str, Any] = None):
        """Configure the OpenAI Realtime session."""
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": self.modalities,
                "instructions": self.instructions,
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": self.turn_detection,
                "temperature": self.temperature,
                "max_response_output_tokens": self.max_tokens if self.max_tokens != 4096 else "inf"
            }
        }

        if self.input_audio_transcription and self.handles_audio_input:
            session_config["session"]["input_audio_transcription"] = {"model": "whisper-1"}

        if self.trigger_function_call and self.tools:
            session_config["session"]["tools"] = self.tools
            session_config["session"]["tool_choice"] = "auto"

        if additional_config:
            session_config["session"].update(additional_config)

        await self._send_event(session_config)
        logger.info(f"Session configured with modalities={self.modalities}")

    async def generate_stream(self, messages, synthesize=True, request_json=False, meta_info=None):
        """
        Main generation method matching BaseLLM interface.

        For Mode 1 & 2 (handles audio input): Spawns background tasks to process audio
        For Mode 3 & 4 (text input): Sends text messages and streams response
        """
        if not messages:
            raise Exception("No messages provided")

        meta_info = meta_info or {}

        # Connect if not already connected
        if not self.connected:
            connected = await self.connect()
            if not connected:
                raise Exception("Failed to connect to OpenAI Realtime")
            await self.configure_session()

        # Store conversation history
        if messages and messages[-1].get('role') == 'user':
            self.conversation_history.append(messages[-1])

        start_time = time.time()
        first_token_time = None
        latency_data = None

        # Mode-specific handling
        if self.output_format == "text":
            # Mode 2 & 3: Output text for synthesizer
            buffer = ""
            answer = ""
            function_call_data = None

            if not self.has_transcriber:
                # Mode 2: Audio input, text output
                # Background tasks are already running to handle audio
                # We just listen for text transcript events
                async for text_chunk, is_final, latency, is_function_call, func_name, func_msg in self._stream_text_from_audio_events(meta_info):
                    if not first_token_time:
                        first_token_time = time.time()
                        latency_data = {
                            "turn_id": meta_info.get("turn_id"),
                            "model": self.model,
                            "first_token_latency_ms": round((first_token_time - start_time) * 1000),
                            "total_stream_duration_ms": None
                        }

                    if is_function_call:
                        function_call_data = text_chunk
                        continue

                    answer += text_chunk
                    buffer += text_chunk

                    if synthesize and len(buffer) >= self.buffer_size and not is_final:
                        split = buffer.rsplit(" ", 1)
                        yield split[0], False, latency_data, False, None, None
                        buffer = split[1] if len(split) > 1 else ""
                    elif is_final:
                        break

            else:
                # Mode 3: Text input, text output
                # Send text message via conversation.item.create
                user_content = messages[-1].get('content', '') if messages else ''
                await self._send_text_message(user_content)
                await self._send_event({"type": "response.create"})

                # Stream text response
                async for text_chunk, is_final, latency, is_function_call, func_name, func_msg in self._stream_text_response(meta_info):
                    if not first_token_time:
                        first_token_time = time.time()
                        latency_data = {
                            "turn_id": meta_info.get("turn_id"),
                            "model": self.model,
                            "first_token_latency_ms": round((first_token_time - start_time) * 1000),
                            "total_stream_duration_ms": None
                        }

                    if is_function_call:
                        function_call_data = text_chunk
                        continue

                    answer += text_chunk
                    buffer += text_chunk

                    if synthesize and len(buffer) >= self.buffer_size and not is_final:
                        split = buffer.rsplit(" ", 1)
                        yield split[0], False, latency_data, False, None, None
                        buffer = split[1] if len(split) > 1 else ""
                    elif is_final:
                        break

            # Set final duration
            total_duration = time.time() - start_time
            if latency_data:
                latency_data["total_stream_duration_ms"] = round(total_duration * 1000)

            # Handle function calls
            if function_call_data:
                yield function_call_data, False, latency_data, True, None, None

            # Final buffer flush
            if synthesize:
                yield buffer, True, latency_data, False, None, None
            else:
                yield answer, True, latency_data, False, None, None

        else:
            # Mode 1 & 4: Output audio
            # Audio handling is done via background tasks
            # This method just needs to signal that generation started
            # For mode 4, send text message
            if self.has_transcriber:
                user_content = messages[-1].get('content', '') if messages else ''
                await self._send_text_message(user_content)
                await self._send_event({"type": "response.create"})

            # Yield empty to maintain interface compatibility
            # Actual audio is handled by background tasks
            yield "", True, {"turn_id": meta_info.get("turn_id"), "model": self.model}, False, None, None

        self.started_streaming = False

    async def _stream_text_events(self, event_config: Dict[str, str], meta_info: Dict[str, Any]):
        """
        Generic text streaming from OpenAI events.

        Args:
            event_config: Configuration dict with event type mappings
                {
                    'delta_event': 'response.text.delta',
                    'done_event': 'response.text.done',
                    'delta_key': 'delta',
                    'done_key': 'text'
                }
            meta_info: Metadata for the request
        """
        while True:
            try:
                event = await self._receive_event()
                event_type = event.get('type')

                if event_type == event_config['delta_event']:
                    text_delta = event.get(event_config['delta_key'], '')
                    if text_delta:
                        yield text_delta, False, None, False, None, None

                elif event_type == event_config['done_event']:
                    text = event.get(event_config['done_key'], '')
                    if text:
                        self.conversation_history.append({'role': 'assistant', 'content': text})
                        yield text, True, None, False, None, None
                    break

                elif event_type == 'response.function_call_arguments.done':
                    function_name = event.get('name')
                    arguments = event.get('arguments')
                    call_id = event.get('call_id')

                    func_call_payload = {
                        'name': function_name,
                        'arguments': arguments,
                        'call_id': call_id
                    }
                    yield func_call_payload, True, None, True, function_name, None
                    break

                elif event_type == 'response.done':
                    self._update_token_usage(event)
                    break

                elif event_type == 'error':
                    logger.error(f"OpenAI error: {event.get('error', {})}")
                    break

            except Exception as e:
                logger.error(f"Error streaming text events: {e}")
                break

    async def _stream_text_from_audio_events(self, meta_info):
        """Stream text transcripts from audio input events (Mode 2)."""
        event_config = {
            'delta_event': 'response.audio_transcript.delta',
            'done_event': 'response.audio_transcript.done',
            'delta_key': 'delta',
            'done_key': 'transcript'
        }
        async for result in self._stream_text_events(event_config, meta_info):
            yield result

    async def _stream_text_response(self, meta_info):
        """Stream text response for text-only mode (Mode 3)."""
        event_config = {
            'delta_event': 'response.text.delta',
            'done_event': 'response.text.done',
            'delta_key': 'delta',
            'done_key': 'text'
        }
        async for result in self._stream_text_events(event_config, meta_info):
            yield result

    async def _send_text_message(self, content: str):
        """Send a text message to OpenAI Realtime."""
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": content
                    }
                ]
            }
        }
        await self._send_event(event)

    async def send_audio(self, audio_chunk: bytes, meta_info: Dict[str, Any]):
        """Send audio chunk to OpenAI Realtime (for Mode 1 & 2)."""
        if not self.connected:
            logger.warning("Not connected, cannot send audio")
            return

        audio_base64 = base64.b64encode(audio_chunk).decode('utf-8')
        event = {
            "type": "input_audio_buffer.append",
            "audio": audio_base64
        }

        await self._send_event(event)
        self.audio_input_duration += len(audio_chunk) / AUDIO_DURATION_DIVISOR

    async def receive_audio(self) -> Optional[Dict[str, Any]]:
        """Receive audio and events from OpenAI Realtime (for Mode 1 & 4)."""
        if not self.connected:
            return None

        try:
            event = await self._receive_event()
            event_type = event.get('type')

            if event_type == 'response.audio.delta':
                audio_base64 = event.get('delta')
                audio_bytes = base64.b64decode(audio_base64)

                self.audio_output_duration += len(audio_bytes) / AUDIO_DURATION_DIVISOR
                self.current_response_item_id = event.get('item_id')
                self.current_response_output_index = event.get('output_index', 0)
                self.current_response_content_index = event.get('content_index', 0)

                return {
                    'audio': audio_bytes,
                    'transcript': None,
                    'is_final': False,
                    'meta_info': {
                        'item_id': event.get('item_id'),
                        'output_index': event.get('output_index'),
                        'content_index': event.get('content_index')
                    }
                }

            elif event_type == 'response.audio_transcript.delta':
                return None

            elif event_type == 'response.audio_transcript.done':
                transcript = event.get('transcript', '')
                logger.info(f"Assistant transcript: {transcript}")

                self.conversation_history.append({
                    'role': 'assistant',
                    'content': transcript
                })

                return {
                    'audio': None,
                    'transcript': transcript,
                    'is_final': True,
                    'meta_info': event
                }

            elif event_type == 'response.output_item.done':
                return None

            elif event_type == 'response.done':
                self.response_in_progress = False
                self.current_response_item_id = None
                self.current_response_output_index = 0
                self.current_response_content_index = 0

                self._update_token_usage(event)

                return {
                    'audio': None,
                    'transcript': None,
                    'is_final': True,
                    'response_done': True,
                    'meta_info': event,
                    'usage': event.get('response', {}).get('usage', {})
                }

            elif event_type == 'response.function_call_arguments.done':
                function_name = event.get('name')
                arguments = event.get('arguments')
                call_id = event.get('call_id')

                logger.info(f"Function call: {function_name}({arguments})")

                self.pending_function_calls[call_id] = {
                    'name': function_name,
                    'arguments': arguments,
                    'call_id': call_id
                }

                return {
                    'audio': None,
                    'transcript': None,
                    'is_final': False,
                    'function_call': {
                        'name': function_name,
                        'arguments': arguments,
                        'call_id': call_id
                    },
                    'meta_info': event
                }

            elif event_type == 'input_audio_buffer.speech_started':
                return {
                    'audio': None,
                    'transcript': None,
                    'is_final': False,
                    'speech_started': True,
                    'interruption': True,
                    'meta_info': event
                }

            elif event_type == 'input_audio_buffer.speech_stopped':
                return {
                    'audio': None,
                    'transcript': None,
                    'is_final': False,
                    'speech_stopped': True,
                    'meta_info': event
                }

            elif event_type == 'conversation.item.input_audio_transcription.completed':
                transcript = event.get('transcript', '')
                logger.info(f"User transcript: {transcript}")

                self.conversation_history.append({
                    'role': 'user',
                    'content': transcript
                })

                return None

            elif event_type == 'response.created':
                self.response_in_progress = True
                self.current_response_item_id = None
                self.current_response_output_index = 0
                self.current_response_content_index = 0
                return None

            elif event_type == 'error':
                error_msg = event.get('error', {})
                logger.error(f"OpenAI error: {error_msg}")
                return {
                    'audio': None,
                    'transcript': None,
                    'is_final': False,
                    'error': error_msg,
                    'meta_info': event
                }

            else:
                return None

        except Exception as e:
            logger.error(f"Error receiving from OpenAI Realtime: {e}")
            return None

    async def send_function_result(self, call_id: str, result: Any):
        """Send function call result back to OpenAI Realtime."""
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result) if not isinstance(result, str) else result
            }
        }
        await self._send_event(event)
        logger.info(f"Sent function result for call_id: {call_id}")

        # Trigger response generation
        await self._send_event({"type": "response.create"})

    async def handle_interruption(self, audio_end_ms: int = None):
        """Handle user interruption and cancel ongoing response."""
        if self.response_in_progress:
            event = {"type": "response.cancel"}
            await self._send_event(event)
            logger.info("Response cancelled due to interruption")

            if self.current_response_item_id:
                truncate_event = {
                    "type": "conversation.item.truncate",
                    "item_id": self.current_response_item_id,
                    "content_index": self.current_response_content_index,
                    "audio_end_ms": audio_end_ms or 0
                }
                await self._send_event(truncate_event)
                logger.info(f"Truncated conversation item {self.current_response_item_id}")

            self.response_in_progress = False

    async def close(self):
        """Close WebSocket connection and cleanup."""
        if self.ws:
            await self.ws.close()
            self.connected = False
            logger.info("Closed OpenAI Realtime connection")

        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()

    def is_connected(self) -> bool:
        """Check if connected to OpenAI Realtime API."""
        return self.connected

    def _update_token_usage(self, event):
        """Update token usage metrics from response.done event."""
        response_data = event.get('response', {})
        usage = response_data.get('usage', {})

        if usage:
            input_details = usage.get('input_token_details', {})
            self.input_text_tokens += input_details.get('text_tokens', 0)
            self.input_audio_tokens += input_details.get('audio_tokens', 0)

            cached_details = input_details.get('cached_tokens_details', {})
            if cached_details:
                self.cached_text_tokens += cached_details.get('text_tokens', 0)
                self.cached_audio_tokens += cached_details.get('audio_tokens', 0)

            output_details = usage.get('output_token_details', {})
            self.output_text_tokens += output_details.get('text_tokens', 0)
            self.output_audio_tokens += output_details.get('audio_tokens', 0)

            logger.info(f"Token usage - In: {usage.get('input_tokens', 0)} "
                       f"({input_details.get('text_tokens', 0)}t/{input_details.get('audio_tokens', 0)}a), "
                       f"Out: {usage.get('output_tokens', 0)} "
                       f"({output_details.get('text_tokens', 0)}t/{output_details.get('audio_tokens', 0)}a)")

    async def _send_event(self, event: Dict[str, Any]):
        """Send event to OpenAI Realtime API."""
        if not self.ws:
            return
        await self.ws.send(json.dumps(event))

    async def _receive_event(self) -> Dict[str, Any]:
        """Receive event from OpenAI Realtime API."""
        if not self.ws:
            return {}
        message = await self.ws.recv()
        return json.loads(message)

    async def get_metrics(self) -> Dict[str, Any]:
        """Get performance and usage metrics."""
        return {
            'connection_time': self.connection_time,
            'audio_input_duration': self.audio_input_duration,
            'audio_output_duration': self.audio_output_duration,
            'input_text_tokens': self.input_text_tokens,
            'input_audio_tokens': self.input_audio_tokens,
            'output_text_tokens': self.output_text_tokens,
            'output_audio_tokens': self.output_audio_tokens,
            'cached_text_tokens': self.cached_text_tokens,
            'cached_audio_tokens': self.cached_audio_tokens,
            'total_input_tokens': self.input_text_tokens + self.input_audio_tokens,
            'total_output_tokens': self.output_text_tokens + self.output_audio_tokens,
        }

    async def get_conversation_transcript(self) -> list:
        """Get conversation history."""
        return self.conversation_history

    async def generate(self, messages, request_json=False):
        """Non-streaming generation (for compatibility)."""
        answer = ""
        async for chunk, is_final, _, _, _, _ in self.generate_stream(messages, synthesize=False, request_json=request_json):
            answer += chunk
            if is_final:
                break
        return answer
