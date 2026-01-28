import json
import uuid
import time
import asyncio
import traceback
from dotenv import load_dotenv
from bolna.helpers.logger_config import configure_logger
from bolna.output_handlers.telephony import TelephonyOutputHandler

logger = configure_logger(__name__)
load_dotenv()


class CallingServiceOutputHandler(TelephonyOutputHandler):
    """
    Output handler for Asterisk WebSocket connections using ARI ExternalMedia channel.
    Sends BINARY frames for media and TEXT frames for control commands.
    
    Reference: https://docs.asterisk.org/Configuration/Channel-Drivers/WebSocket/
    """
    
    def __init__(self, websocket=None, mark_event_meta_data=None, log_dir_name=None, input_handler=None):
        io_provider = 'calling_service'
        super().__init__(io_provider, websocket, mark_event_meta_data, log_dir_name)
        
        self.channel_id = None
        self.connection_id = None
        self.codec = None
        self.format = None
        self.ptime = None
        self.is_chunking_supported = True
        self.input_handler = input_handler  # Reference to input handler to simulate mark events
        self.control_message_format = 'plain-text' 
        self.queue_full = False
        self.answered = False
        
    def set_channel_info(self, channel_id, connection_id=None, codec=None, format_type=None, ptime=None):
        """Set channel information from MEDIA_START event"""
        self.channel_id = channel_id
        self.connection_id = connection_id or channel_id
        self.codec = codec
        self.format = format_type
        self.ptime = ptime or 20
        
        # Use connection_id as stream_sid for compatibility
        self.stream_sid = self.connection_id
        
        logger.info(f"Channel info set - ID: {self.channel_id}, Codec: {self.codec}, Format: {self.format}")
    
    async def send_control_command(self, command, params=None):
        """
        Send control command to Asterisk via TEXT frame.
        
        Supported commands:
        - ANSWER: Answer the channel
        - HANGUP: Hang up the channel
        - START_MEDIA_BUFFERING: Start buffering bulk media
        - STOP_MEDIA_BUFFERING <correlation_id>: Stop buffering
        - FLUSH_MEDIA: Clear queued media
        - PAUSE_MEDIA: Pause media processing
        - CONTINUE_MEDIA: Resume media processing
        - GET_STATUS: Request status update
        - REPORT_QUEUE_DRAINED: Request notification when queue is drained
        """
        try:
            logger.info(f"Sending control command: {command} to Asterisk and params: {params}\n\n\n")
            if params:
                param_str = ' '.join(str(v) for v in params.values())
                message = f"{command} {param_str}"
            else:
                message = command
            await self.websocket.send_text(message)
                
            logger.info(f"Sent control command: {command}")
            
        except Exception as e:
            logger.error(f"Error sending control command {command}: {e}")
            traceback.print_exc()
    
    async def answer_channel(self):
        """Send ANSWER command to Asterisk"""
        if not self.answered:
            await self.send_control_command('ANSWER')
            self.answered = True
    
    async def hangup_channel(self):
        """Send HANGUP command to Asterisk"""
        await self.send_control_command('HANGUP')
    
    async def get_status(self):
        """Request status from Asterisk"""
        await self.send_control_command('GET_STATUS')
    
    async def flush_media(self):
        """Flush queued media"""
        await self.send_control_command('FLUSH_MEDIA')
    
    async def pause_media(self):
        """Pause media processing"""
        await self.send_control_command('PAUSE_MEDIA')
    
    async def continue_media(self):
        """Resume media processing"""
        await self.send_control_command('CONTINUE_MEDIA')
    
    async def start_media_buffering(self):
        """Start bulk media buffering mode"""
        await self.send_control_command('START_MEDIA_BUFFERING')
    
    async def stop_media_buffering(self, correlation_id=None):
        """Stop bulk media buffering mode"""
        params = {'correlation_id': correlation_id} if correlation_id else None
        await self.send_control_command('STOP_MEDIA_BUFFERING', params)
    
    async def handle_interruption(self):
        """
        Handle interruption by flushing queued media.
        This clears any audio that hasn't been played yet.
        """
        logger.info("Handling interruption - flushing media queue")
        try:
            await self.flush_media()
            self.mark_event_meta_data.clear_data()
        except Exception as e:
            logger.error(f"Error handling interruption: {e}")
    
    async def form_media_message(self, audio_data, audio_format="wav"):
        """
        For Asterisk WebSocket, media is sent as raw BINARY frames.
        We don't need to base64 encode or wrap in JSON - just return the raw bytes.
        
        Note: The audio format should match what was negotiated in the channel creation.
        Common formats: 'slin' (signed linear), 'ulaw', 'alaw', 'slin16'
        """
        # Asterisk expects raw audio data, no conversion needed
        # The format should match what was specified when creating the channel
        return audio_data
    
    async def form_mark_message(self, mark_id):
        """
        Asterisk WebSocket doesn't have a built-in mark/event mechanism like Twilio.
        However, we can use this to track our own timing for debugging.
        We'll return None since we don't send mark messages to Asterisk.
        """
        return None
    
    async def handle(self, ws_data_packet):
        """
        Handle outgoing audio to Asterisk.
        Send audio as BINARY frames to Asterisk.
        """
        try:
            with open("ws_data_packet.log", "a") as log_file:
                log_file.write(f"{ws_data_packet}\n")
        except Exception as log_exc:
            logger.warning(f"Failed to log websocket message: {log_exc}")
        try:
            audio_chunk = ws_data_packet.get('data')
            meta_info = ws_data_packet.get('meta_info')
            
            # CALLING_SERVICE DEBUG: Log entry to handle function
            logger.info(f"[CALLING_SERVICE OUTPUT] handle() called: audio_size={len(audio_chunk) if audio_chunk else 0} bytes, "
                       f"format={meta_info.get('format', 'unknown')}, stream_sid={self.stream_sid}, "
                       f"sequence_id={meta_info.get('sequence_id')}, category={meta_info.get('message_category', 'N/A')}")
            
            # Set stream_sid from meta_info if not already set
            if self.stream_sid is None:
                self.stream_sid = meta_info.get('stream_sid', None)
                logger.info(f"[CALLING_SERVICE OUTPUT] Set stream_sid to: {self.stream_sid}")
            
            # Skip if queue is full to avoid overwhelming Asterisk
            if self.queue_full:
                logger.warning("[CALLING_SERVICE OUTPUT] Skipping audio send - Asterisk queue is full (XOFF state)")
                return
            
            try:
                # Handle edge cases with audio chunk size
                if len(audio_chunk) == 1:
                    logger.info(f"[CALLING_SERVICE OUTPUT] Audio chunk size is 1, padding with null byte")
                    audio_chunk += b'\x00'
                
                # Send audio if we have valid data
                if audio_chunk and self.stream_sid and len(audio_chunk) != 1:
                    if audio_chunk != b'\x00\x00':
                        logger.info(f"[CALLING_SERVICE OUTPUT] Conditions met for sending audio")
                        audio_format = meta_info.get("format", "wav")
                        
                        # Track welcome message timing
                        if meta_info.get('message_category', '') == 'agent_welcome_message' and not self.welcome_message_sent_ts:
                            self.welcome_message_sent_ts = time.time() * 1000
                        
                        # Send audio as BINARY frame to Asterisk
                        # Asterisk expects raw audio data without any encoding or wrapping
                        # Chunk large audio data to prevent websocket timeout
                        original_size = len(audio_chunk)
                        logger.info(f"[CALLING_SERVICE OUTPUT] Sending via chunked websocket.send_bytes(): {original_size} bytes")
                        bytes_sent = await self._send_audio_chunked(audio_chunk, audio_format)
                        
                        logger.info(f"[CALLING_SERVICE OUTPUT] ✓ Successfully sent {bytes_sent} bytes of audio to Asterisk (chunked from {original_size} bytes) - "
                                  f"Format: {audio_format}, Category: {meta_info.get('message_category', 'N/A')}")
                        
                        # Track mark events internally for our own timing/debugging
                        # (Asterisk doesn't have mark events, but we track for compatibility)
                        if self.mark_event_meta_data:
                            audio_duration = self._calculate_audio_duration(audio_chunk, audio_format)
                            
                            # Check if this is the final chunk
                            # For welcome messages, hangup messages (sequence_id=-1), or regular responses
                            is_final = (meta_info.get("end_of_llm_stream", False) and meta_info.get("end_of_synthesizer_stream", False)) or \
                                      meta_info.get("is_final_chunk_of_entire_response", False) or \
                                      (meta_info.get("sequence_id") == -1 and meta_info.get("end_of_llm_stream", False))
                            
                            message_category = meta_info.get('message_category', 'agent_response')
                            
                            # Debug logging for mark event detection
                            logger.info(f"[CALLING_SERVICE OUTPUT] Mark event check: is_final={is_final}, "
                                       f"seq_id={meta_info.get('sequence_id')}, "
                                       f"end_llm={meta_info.get('end_of_llm_stream')}, "
                                       f"end_synth={meta_info.get('end_of_synthesizer_stream')}, "
                                       f"is_final_chunk={meta_info.get('is_final_chunk_of_entire_response')}, "
                                       f"category={message_category}, input_handler={self.input_handler is not None}")
                            
                            mark_event_meta_data = {
                                "text_synthesized": "" if meta_info["sequence_id"] == -1 else meta_info.get("text_synthesized", ""),
                                "type": message_category,
                                "is_first_chunk": meta_info.get("is_first_chunk", False),
                                "is_final_chunk": is_final,
                                "sequence_id": meta_info["sequence_id"],
                                "duration": audio_duration,
                                "sent_ts": time.time()
                            }
                            mark_id = meta_info.get("mark_id") if (meta_info.get("mark_id") and meta_info.get("mark_id") != "") else str(uuid.uuid4())
                            self.mark_event_meta_data.update_data(mark_id, mark_event_meta_data)
                            
                            # Schedule a task to simulate the mark event after audio finishes playing
                            # Critical for hangup messages - must simulate mark event to trigger hangup observer
                            if is_final and self.input_handler:
                                asyncio.create_task(self._simulate_mark_event_after_duration(mark_id, audio_duration, message_category))
                                logger.info(f"[CALLING_SERVICE OUTPUT] ✓ Scheduled mark event simulation for {message_category} after {audio_duration:.3f}s")
                            elif not is_final:
                                logger.info(f"[CALLING_SERVICE OUTPUT] ✗ NOT scheduling mark event - is_final=False")
                            elif not self.input_handler:
                                logger.error(f"[CALLING_SERVICE OUTPUT] ✗ NOT scheduling mark event - input_handler is None!")
                    else:
                        logger.info(f"[CALLING_SERVICE OUTPUT] Skipping null audio chunk (b'\\x00\\x00')")
                else:
                    logger.info(f"[CALLING_SERVICE OUTPUT] Not sending - Conditions not met: "
                               f"audio_chunk={audio_chunk is not None}, stream_sid={self.stream_sid}, "
                               f"len={len(audio_chunk) if audio_chunk else 0}")
                    
            except Exception as e:
                logger.error(f'Error sending audio to Asterisk: {e}')
                traceback.print_exc()
                
        except Exception as e:
            logger.error(f'Error handling Asterisk output: {e}')
            traceback.print_exc()
    
    def _calculate_audio_duration(self, audio_data, audio_format):
        """
        Calculate audio duration based on format.
        
        Common sample rates:
        - ulaw/alaw: 8000 Hz, 1 byte per sample
        - slin: 8000 Hz, 2 bytes per sample  
        - slin16: 16000 Hz, 2 bytes per sample
        """
        if audio_format in ('ulaw', 'alaw', 'mulaw'):
            # 8 kHz, 1 byte per sample
            return len(audio_data) / 8000.0
        elif audio_format in ('slin', 'pcm'):
            # 8 kHz, 2 bytes per sample
            return len(audio_data) / 16000.0
        elif audio_format == 'slin16':
            # 16 kHz, 2 bytes per sample
            return len(audio_data) / 32000.0
        else:
            # Default to 8kHz 2-byte samples
            return len(audio_data) / 16000.0
    
    def _get_chunk_size(self, audio_format):
        """
        Get optimal chunk size for sending audio to Asterisk.
        Returns chunk size in bytes based on audio format.
        
        We use 100ms chunks to avoid blocking the websocket:
        - ulaw/alaw: 800 bytes (100ms at 8kHz, 1 byte/sample)
        - slin: 1600 bytes (100ms at 8kHz, 2 bytes/sample)
        - slin16: 3200 bytes (100ms at 16kHz, 2 bytes/sample)
        """
        if audio_format in ('ulaw', 'alaw', 'mulaw'):
            # 100ms at 8kHz = 800 bytes
            return 800
        elif audio_format in ('slin', 'pcm'):
            # 100ms at 8kHz = 1600 bytes
            return 1600
        elif audio_format == 'slin16':
            # 100ms at 16kHz = 3200 bytes
            return 3200
        else:
            # Default to 1600 bytes (100ms at 8kHz, 2 bytes/sample)
            return 1600
    
    async def _send_audio_chunked(self, audio_data, audio_format):
        """
        Send audio data in smaller chunks to prevent Asterisk from timing out.
        Large chunks can block the websocket and cause Asterisk to disconnect.
        """
        chunk_size = self._get_chunk_size(audio_format)
        total_bytes = len(audio_data)
        
        # Only chunk if the data is larger than chunk_size
        if total_bytes <= chunk_size:
            await self.websocket.send_bytes(audio_data)
            return total_bytes
        
        # Send in chunks with small delays to keep websocket responsive
        bytes_sent = 0
        offset = 0
        
        while offset < total_bytes:
            chunk = audio_data[offset:offset + chunk_size]
            if chunk:
                await self.websocket.send_bytes(chunk)
                bytes_sent += len(chunk)
                
                # Small delay between chunks to keep websocket responsive
                # Only delay if there's more data to send
                if offset + chunk_size < total_bytes:
                    await asyncio.sleep(0.001)  # 1ms delay between chunks
            
            offset += chunk_size
        
        return bytes_sent
    
    async def _simulate_mark_event_after_duration(self, mark_id, duration, message_category):
        """
        Simulate a mark event after the audio finishes playing.
        Asterisk doesn't send mark events back, so we simulate them based on calculated duration.
        """
        try:
            # Wait for the audio to finish playing
            await asyncio.sleep(duration)
            
            # Simulate the mark event by calling the input handler's process_mark_message
            if self.input_handler and message_category:
                logger.info(f"[CALLING_SERVICE OUTPUT] Simulating mark event after {duration:.3f}s for category: {message_category}")
                mark_packet = {
                    "name": mark_id,
                    "type": message_category
                }
                self.input_handler.process_mark_message(mark_packet)
        except Exception as e:
            logger.error(f"Error simulating mark event: {e}")
    
    def requires_custom_voicemail_detection(self):
        """Asterisk requires custom voicemail detection"""
        return True
