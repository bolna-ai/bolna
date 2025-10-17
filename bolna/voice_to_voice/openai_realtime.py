import asyncio
import json
import base64
import time
import websockets
from typing import Optional, Dict, Any
from .base_voice_to_voice import BaseVoiceToVoice
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class OpenAIRealtimeVoiceToVoice(BaseVoiceToVoice):
    """OpenAI Realtime API implementation."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini-realtime-preview-2024-12-17",
        voice: str = "alloy",
        instructions: str = "",
        temperature: float = 0.8,
        max_tokens: int = 4096,
        modalities: Optional[list] = None,
        turn_detection: Optional[Dict] = None,
        input_audio_transcription: bool = True,
        tools: Optional[list] = None,
        task_manager_instance=None,
        **kwargs
    ):
        super().__init__(api_key, voice, instructions, task_manager_instance, **kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.modalities = modalities or ["audio", "text"]
        self.turn_detection = turn_detection or {
            "type": "server_vad",
            "threshold": 0.4,
            "prefix_padding_ms": 200,
            "silence_duration_ms": 700
        }
        self.input_audio_transcription = input_audio_transcription
        self.tools = tools or []

        self.input_audio_buffer = bytearray()
        self.pending_function_calls = {}
        self.response_in_progress = False
        self.current_response_item_id = None
        self.current_response_output_index = 0
        self.current_response_content_index = 0

    async def connect(self) -> bool:
        try:
            url = f"wss://api.openai.com/v1/realtime?model={self.model}"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }

            logger.debug(f"Connecting to OpenAI Realtime: {self.model}")
            self.ws = await websockets.connect(url, additional_headers=headers)
            self.connected = True
            self.connection_time = time.time()
            logger.info("Connected to OpenAI Realtime API")

            event = await self._receive_event()
            if event.get('type') == 'session.created':
                self.session_id = event.get('session', {}).get('id')
                logger.info(f"Session created: {self.session_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to OpenAI Realtime: {e}")
            self.connected = False
            return False

    async def configure_session(self, config: Dict[str, Any]):
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

        if self.input_audio_transcription:
            session_config["session"]["input_audio_transcription"] = {"model": "whisper-1"}

        if self.tools:
            session_config["session"]["tools"] = self.tools
            session_config["session"]["tool_choice"] = "auto"

        if config:
            session_config["session"].update(config)

        await self._send_event(session_config)
        logger.debug("Session configured")

    async def send_audio(self, audio_chunk: bytes, meta_info: Dict[str, Any]):
        if not self.connected:
            logger.warning("Not connected, cannot send audio")
            return

        audio_base64 = base64.b64encode(audio_chunk).decode('utf-8')
        event = {
            "type": "input_audio_buffer.append",
            "audio": audio_base64
        }

        await self._send_event(event)
        self.audio_input_duration += len(audio_chunk) / (2 * 24000)

    async def receive_audio(self) -> Optional[Dict[str, Any]]:
        if not self.connected:
            return None

        try:
            event = await self._receive_event()
            event_type = event.get('type')

            if event_type == 'response.audio.delta':
                audio_base64 = event.get('delta')
                audio_bytes = base64.b64decode(audio_base64)

                self.audio_output_duration += len(audio_bytes) / (2 * 24000)
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
                transcript_delta = event.get('delta', '')
                logger.debug(f"Transcript delta: {transcript_delta}")
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
                item_id = event.get('item_id')
                output_index = event.get('output_index')
                logger.debug(f"Output item {item_id} done at index {output_index}")
                return None

            elif event_type == 'response.done':
                logger.debug("Response complete")
                self.response_in_progress = False
                self.current_response_item_id = None
                self.current_response_output_index = 0
                self.current_response_content_index = 0
                return {
                    'audio': None,
                    'transcript': None,
                    'is_final': True,
                    'response_done': True,
                    'meta_info': event
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
                logger.info("User started speaking")
                return {
                    'audio': None,
                    'transcript': None,
                    'is_final': False,
                    'speech_started': True,
                    'meta_info': event
                }

            elif event_type == 'input_audio_buffer.speech_stopped':
                logger.debug("User stopped speaking")
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
                logger.debug("Response created")
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
                logger.debug(f"Unhandled event type: {event_type}")
                return None

        except Exception as e:
            logger.error(f"Error receiving from OpenAI Realtime: {e}")
            return None

    async def send_function_result(self, call_id: str, result: Any):
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
        await self._send_event({"type": "response.create"})

    async def handle_interruption(self, audio_end_ms: int = None):
        """Handle user interruption and truncate conversation."""
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
                logger.info(f"Truncated conversation item {self.current_response_item_id} at content_index={self.current_response_content_index}, audio_end_ms={audio_end_ms or 0}")
            else:
                logger.debug("No item_id for truncation")

            self.response_in_progress = False

    async def close(self):
        if self.ws:
            await self.ws.close()
            self.connected = False
            logger.info("Closed OpenAI Realtime connection")

    async def _send_event(self, event: Dict[str, Any]):
        if not self.ws:
            return
        await self.ws.send(json.dumps(event))

    async def _receive_event(self) -> Dict[str, Any]:
        if not self.ws:
            return {}
        message = await self.ws.recv()
        return json.loads(message)
