import asyncio
import json
import traceback
from starlette.websockets import WebSocketDisconnect
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.input_handlers.telephony import TelephonyInputHandler
from bolna.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)
load_dotenv()


class CallingServiceInputHandler(TelephonyInputHandler):
    """
    Input handler for Asterisk WebSocket connections using ARI ExternalMedia channel.
    Handles BINARY frames for media and TEXT frames for control events.
    
    Reference: https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/
    """
    
    def __init__(self, queues, websocket=None, input_types=None, mark_event_meta_data=None, 
                 turn_based_conversation=False, is_welcome_message_played=False, observable_variables=None,
                 ws_context_data=None):
        super().__init__(queues, websocket, input_types, mark_event_meta_data, turn_based_conversation,
                         is_welcome_message_played=is_welcome_message_played, 
                         observable_variables=observable_variables)
        self.io_provider = 'calling_service'
        self.channel_id = None
        self.connection_id = None
        self.codec = None
        self.format = None
        self.ptime = None
        self.queue_full = False
        self.bulk_media = False
        self.media_paused = False
        self.buffer = []
        self.message_count = 0
        self.media_started = False
        
        # Initialize from context data if provided (when server intercepts MEDIA_START)
        if ws_context_data and 'media_start_data' in ws_context_data:
            self._initialize_from_media_start(ws_context_data['media_start_data'])
    
    def _initialize_from_media_start(self, media_start_data):
        """Initialize channel info from pre-parsed MEDIA_START data"""
        self.channel_id = media_start_data.get('channel_id')
        self.connection_id = media_start_data.get('connection_id', self.channel_id)
        self.codec = media_start_data.get('codec', media_start_data.get('format'))
        self.format = media_start_data.get('format')
        self.ptime = int(media_start_data.get('ptime', 20))
        
        # Use connection_id as stream_sid for compatibility
        self.stream_sid = self.connection_id
        self.call_sid = self.channel_id
        
        self.media_started = True
        
        logger.info(f"Initialized from context - Channel: {self.channel_id}, "
                   f"Connection: {self.connection_id}, Codec: {self.codec}, "
                   f"Format: {self.format}, Ptime: {self.ptime}ms")
        
        
    async def disconnect_stream(self):
        """Send HANGUP command to Asterisk when disconnecting"""
        try:
            if self.websocket and self.channel_id:
                hangup_cmd = {
                    "command": "HANGUP",
                    "channel_id": self.channel_id
                }
                await self.websocket.send_text(json.dumps(hangup_cmd))
                logger.info(f"Sent HANGUP command for channel {self.channel_id}")
        except Exception as e:
            logger.error(f"Error sending HANGUP command: {e}")
    
    def _parse_control_message(self, message_text):
        """
        Parse control message which can be in plain-text or JSON format.
        
        Plain-text format: "EVENT_NAME param1 param2..."
        JSON format: {"event": "EVENT_NAME", "param1": "value1", ...}
        """
        try:
            # Try parsing as JSON first

            data = json.loads(message_text)
            event_type = data.get('event') or data.get('command')
            return event_type, data
        except json.JSONDecodeError:
            # Fall back to plain-text format
            parts = message_text.strip().split(maxsplit=1)
            event_type = parts[0] if parts else None
            
            # Parse parameters from plain text
            data = {'event': event_type}
            if len(parts) > 1:
                data['params'] = parts[1]
            
            return event_type, data
    
    async def _handle_control_event(self, event_type, event_data):
        """Handle control events received from Asterisk via TEXT frames"""
        
        if event_type == 'MEDIA_START':
            # Skip if already initialized from context data
            if not self.media_started:
                await self.call_start(event_data)
            else:
                logger.info("MEDIA_START already processed from context data, ignoring duplicate")
            
        elif event_type == 'DTMF_END':
            # Handle DTMF digit
            digit = event_data.get('digit', '')
            logger.info(f"DTMF key pressed: '{digit}' | Accumulated: '{self.dtmf_digits}'")
            
            if digit:
                is_complete = await self._handle_dtmf_digit(digit)
                if is_complete and self.dtmf_digits:
                    if self.is_dtmf_active:
                        logger.info(f"DTMF complete - Sending: '{self.dtmf_digits}'")
                        self.queues['dtmf'].put_nowait(self.dtmf_digits)
                    self.dtmf_digits = ""
                    
        elif event_type == 'MEDIA_XOFF':
            # Queue is full, Asterisk is dropping frames
            self.queue_full = True
            logger.warning(f"MEDIA_XOFF received - Asterisk queue is full for channel {self.channel_id}")
            
        elif event_type == 'MEDIA_XON':
            # Queue has space again
            self.queue_full = False
            logger.info(f"MEDIA_XON received - Asterisk queue has space for channel {self.channel_id}")
            
        elif event_type == 'STATUS':
            # Status response from GET_STATUS command
            queue_length = event_data.get('queue_length', 0)
            xon_level = event_data.get('xon_level', 0)
            xoff_level = event_data.get('xoff_level', 0)
            self.queue_full = event_data.get('queue_full', False)
            self.bulk_media = event_data.get('bulk_media', False)
            self.media_paused = event_data.get('media_paused', False)
            
            logger.info(f"STATUS: Queue={queue_length}, XON={xon_level}, XOFF={xoff_level}, "
                       f"Full={self.queue_full}, Bulk={self.bulk_media}, Paused={self.media_paused}")
            
        elif event_type == 'MEDIA_BUFFERING_COMPLETED':
            # Bulk media transfer completed
            correlation_id = event_data.get('correlation_id')
            logger.info(f"MEDIA_BUFFERING_COMPLETED received - Correlation ID: {correlation_id}")
            self.bulk_media = False
            
        elif event_type == 'QUEUE_DRAINED':
            # Queue has been drained
            logger.info(f"QUEUE_DRAINED received for channel {self.channel_id}")
            
        else:
            logger.warning(f"Unknown control event received: {event_type}")
    
    async def ingest_audio(self, audio_data, meta_info):
        """Send audio data to transcriber queue"""
        logger.info(f"Ingesting audio data of length: {len(audio_data)}")
        logger.info(f"meta_info data: {meta_info}")
        ws_data_packet = create_ws_data_packet(data=audio_data, meta_info=meta_info)
        self.queues['transcriber'].put_nowait(ws_data_packet)
    
    async def _listen(self):
        """
        Listen for messages from Asterisk WebSocket.
        - BINARY frames contain raw media data
        - TEXT frames contain control events/commands
        """
        buffer = []
        
        try:
            while self.running:
                try:
                    # Receive message from WebSocket
                    # Asterisk sends BINARY for media, TEXT for control
                    message = await self.websocket.receive()
                    # Log the websocket message to a file for debugging/audit purposes
                    try:
                        with open("asterisk_ws_messages.log", "a") as log_file:
                            log_file.write(f"{message}\n")
                    except Exception as log_exc:
                        logger.warning(f"Failed to log websocket message: {log_exc}")
                    
                    # Check for disconnect message
                    if message.get('type') == 'websocket.disconnect':
                        logger.info(f"WebSocket disconnect message received for channel {self.channel_id}")
                        break
                    
                    # Handle BINARY frames (media data)
                    if 'bytes' in message:
                        media_audio = message['bytes']
                        
                        meta_info = {
                            'io': self.io_provider,
                            'call_sid': self.call_sid,
                            'stream_sid': self.stream_sid,
                            'sequence': self.input_types.get('audio', 0)
                        }
                        
                        buffer.append(media_audio)
                        self.message_count += 1
                        
                        # Buffer audio based on ptime (default 20ms chunks)
                        # Send when we have accumulated enough chunks
                        # For ulaw/alaw at 8kHz: 160 bytes per 20ms chunk
                        # Accumulate 5 chunks (100ms) before sending
                        chunks_to_accumulate = max(1, 100 // self.ptime) if self.ptime else 5
                        
                        if self.message_count >= chunks_to_accumulate:
                            merged_audio = b''.join(buffer)
                            buffer = []
                            await self.ingest_audio(merged_audio, meta_info)
                            self.message_count = 0
                    
                    # Handle TEXT frames (control events)
                    elif 'text' in message:
                        message_text = message['text']
                        event_type, event_data = self._parse_control_message(message_text)
                        
                        if event_type:
                            await self._handle_control_event(event_type, event_data)
                        else:
                            logger.warning(f"Could not parse control message: {message_text}")
                    else:
                        logger.warning(f"Received unknown message type: {message}")
                        
                except WebSocketDisconnect as e:
                    if e.code in (1000, 1001, 1006):
                        logger.info(f"WebSocket disconnected normally: code={e.code}")
                    else:
                        logger.error(f"WebSocket disconnected unexpectedly: code={e.code}, "
                                   f"reason={getattr(e, 'reason', None)}")
                    break
                    
                except RuntimeError as e:
                    # Handle "Cannot call receive once a disconnect message has been received"
                    if "disconnect message has been received" in str(e):
                        logger.info(f"WebSocket already disconnected for channel {self.channel_id}")
                        break
                    else:
                        logger.error(f"Runtime error processing message: {e}")
                        traceback.print_exc()
                        break
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    traceback.print_exc()
                    # Break on exceptions to avoid infinite error loops
                    break
                    
        except Exception as e:
            logger.error(f"Error in _listen loop: {e}")
            traceback.print_exc()
            
        finally:
            # Send EOS to transcriber when connection ends
            ws_data_packet = create_ws_data_packet(
                data=None,
                meta_info={
                    'io': self.io_provider, 
                    'eos': True,
                    'sequence': self.input_types.get('audio', 0)  # Add required sequence field
                }
            )
            self.queues['transcriber'].put_nowait(ws_data_packet)
            logger.info(f"Asterisk WebSocket connection closed for channel {self.channel_id}")
    
    async def handle(self):
        """Start listening for Asterisk WebSocket messages"""
        if not self.websocket_listen_task:
            self.websocket_listen_task = asyncio.create_task(self._listen())
