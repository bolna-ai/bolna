import os
import asyncio
import json
import time
import aiohttp
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from .base_transcriber import BaseTranscriber
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import create_ws_data_packet

logger = configure_logger(__name__)
load_dotenv()


class SarvamTranscriber(BaseTranscriber):
    """
    Sarvam AI Speech-to-Text Transcriber
    
    Uses Sarvam AI's REST API for transcription.
    Supports models: saarika:v2.5 (transcription) and saaras:v2.5 (translation)
    """
    
    def __init__(self, telephony_provider, input_queue=None, output_queue=None, 
                 language="hi-IN", encoding="linear16", model="saarika:v2.5", 
                 stream=False, **kwargs):
        super().__init__(input_queue)
        
        self.telephony_provider = telephony_provider
        self.output_queue = output_queue
        self.language = language if language else "hi-IN"  # Default to Hindi
        self.encoding = encoding
        self.model = model
        self.stream = stream  # Sarvam currently only supports REST API
        self.api_key = kwargs.get("transcriber_key", os.getenv('SARVAM_API_KEY'))
        
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY is required but not found in environment")
        
        # Set sampling rate based on telephony provider
        if self.telephony_provider in ("twilio", "exotel", "plivo"):
            self.sampling_rate = 8000
            # Twilio uses mulaw encoding
            if self.telephony_provider == "twilio":
                self.audio_codec = "mulaw"
            else:
                self.audio_codec = "pcm_s16le"
        else:
            self.sampling_rate = 16000
            self.audio_codec = "pcm_s16le"
            
        # API endpoints
        self.api_base_url = "https://api.sarvam.ai"
        self.transcribe_endpoint = f"{self.api_base_url}/speech-to-text"
        self.translate_endpoint = f"{self.api_base_url}/speech-to-text/translate"
        
        # Session for HTTP requests
        self.session = None
        
        # Audio buffer for accumulating chunks
        self.audio_buffer = bytearray()
        # Buffer 2 seconds of audio for better accuracy
        self.buffer_duration_seconds = 2.0
        self.bytes_per_second = self.sampling_rate * 2  # 16-bit audio = 2 bytes per sample
        self.buffer_size = int(self.bytes_per_second * self.buffer_duration_seconds)
        
        # State tracking
        self.is_running = False
        self.audio_submitted = False
        self.start_time = None
        self.total_audio_duration = 0
        self.run_id = kwargs.get("run_id", "")
        
        # For tracking request timing
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Minimum 0.5 seconds between requests
        
        logger.info(f"Initialized Sarvam transcriber with model: {self.model}, "
                   f"language: {self.language}, sampling_rate: {self.sampling_rate}")
        
    async def run(self):
        """Start the transcription process"""
        self.is_running = True
        self.start_time = time.time()
        
        try:
            logger.info(f"Starting Sarvam transcriber - Model: {self.model}, Language: {self.language}")
            
            # Create HTTP session with timeout
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Start processing audio
            await self._process_audio_chunks()
            
        except Exception as e:
            logger.error(f"Error in Sarvam transcriber: {e}")
            await self._send_error_response(str(e))
        finally:
            await self._cleanup()
            
    async def _process_audio_chunks(self):
        """Process incoming audio chunks and buffer them"""
        while self.is_running:
            try:
                # Get audio packet from queue
                ws_data_packet = await self.input_queue.get()
                
                # Initialize metadata on first packet
                if not self.audio_submitted:
                    self.meta_info = ws_data_packet.get('meta_info', {})
                    self.audio_submitted = True
                    self.current_request_id = self.generate_request_id()
                    self.meta_info['request_id'] = self.current_request_id
                    logger.info(f"Started audio stream for request_id: {self.current_request_id}")
                
                # Check for end of stream
                if ws_data_packet.get('meta_info', {}).get('eos'):
                    logger.info("End of stream received")
                    # Process any remaining audio in buffer
                    if len(self.audio_buffer) > 0:
                        await self._transcribe_buffer()
                    self.is_running = False
                    break
                    
                # Process audio data
                audio_data = ws_data_packet.get('data')
                if audio_data:
                    # Convert audio format if needed
                    processed_audio = await self._convert_audio_format(audio_data)
                    
                    # Add to buffer
                    self.audio_buffer.extend(processed_audio)
                    
                    # Update duration tracking
                    chunk_duration = len(audio_data) / self.bytes_per_second
                    self.total_audio_duration += chunk_duration
                    
                    # Check if buffer has enough audio (2 seconds)
                    if len(self.audio_buffer) >= self.buffer_size:
                        await self._transcribe_buffer()
                        
            except asyncio.CancelledError:
                logger.info("Audio processing cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
                import traceback
                traceback.print_exc()
                
    async def _convert_audio_format(self, audio_data: bytes) -> bytes:
        """Convert audio format based on telephony provider"""
        try:
            if self.telephony_provider == "twilio":
                # Convert μ-law to linear PCM
                from audioop import ulaw2lin, ratecv
                converted = ulaw2lin(audio_data, 2)
                
                # Convert sample rate from 8kHz to 16kHz for better Sarvam compatibility
                if self.sampling_rate == 8000:
                    # Convert 8kHz to 16kHz
                    converted_16k, _ = ratecv(converted, 2, 1, 8000, 16000, None)
                    logger.debug(f"Converted audio: μ-law->PCM: {len(audio_data)}->{len(converted)} bytes, 8kHz->16kHz: {len(converted)}->{len(converted_16k)} bytes")
                    return converted_16k
                else:
                    logger.debug(f"Converted μ-law audio: {len(audio_data)} -> {len(converted)} bytes")
                    return converted
            elif self.telephony_provider in ["plivo", "exotel"]:
                # These typically use linear PCM already
                return audio_data
            else:
                # Default: assume linear PCM
                return audio_data
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            return audio_data
    
    def _create_wav_file(self, raw_audio_data: bytes) -> bytes:
        """Create a proper WAV file from raw PCM data"""
        import struct
        
        # WAV file parameters - use 16kHz if we upsampled from Twilio's 8kHz
        sample_rate = 16000 if self.telephony_provider == "twilio" and self.sampling_rate == 8000 else self.sampling_rate
        channels = 1  # Mono
        bits_per_sample = 16
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        data_size = len(raw_audio_data)
        
        # WAV header
        wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
            b'RIFF',                    # Chunk ID
            36 + data_size,             # Chunk size
            b'WAVE',                    # Format
            b'fmt ',                    # Subchunk1 ID
            16,                         # Subchunk1 size (PCM)
            1,                          # Audio format (PCM)
            channels,                   # Number of channels
            sample_rate,                # Sample rate
            byte_rate,                  # Byte rate
            block_align,                # Block align
            bits_per_sample,            # Bits per sample
            b'data',                    # Subchunk2 ID
            data_size                   # Subchunk2 size
        )
        
        # Combine header with audio data
        wav_file = wav_header + raw_audio_data
        logger.debug(f"Created WAV file: {len(wav_file)} bytes (header: {len(wav_header)}, data: {len(raw_audio_data)})")
        
        return wav_file
            
    async def _transcribe_buffer(self):
        """Send buffered audio to Sarvam API for transcription"""
        if len(self.audio_buffer) == 0:
            return
            
        # Get audio data and clear buffer
        raw_audio_data = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        
        # Convert raw PCM to WAV format with proper headers
        audio_data = self._create_wav_file(raw_audio_data)
        
        # Respect rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last_request)
        
        try:
            start_time = time.time()
            
            # Determine which endpoint to use based on model
            if self.model.startswith("saaras"):
                # Translation model - always translates to English
                endpoint = self.translate_endpoint
                language_param = None
            else:
                # Transcription model
                endpoint = self.transcribe_endpoint
                language_param = self.language
            
            # Create multipart form data
            form_data = aiohttp.FormData()
            
            # Add audio file
            form_data.add_field(
                'file',
                audio_data,
                filename='audio.wav',
                content_type='audio/wav'
            )
            
            # Add model
            form_data.add_field('model', self.model)
            
            # Add language code for transcription
            if language_param:
                form_data.add_field('language_code', language_param)
            
            # Add audio codec info - always PCM for WAV files
            form_data.add_field('input_audio_codec', 'pcm_s16le')
            
            # Headers
            headers = {
                "api-subscription-key": self.api_key
            }
            
            logger.info(f"Sending {len(audio_data)} bytes to Sarvam API endpoint: {endpoint}")
            
            # Make API request
            async with self.session.post(
                endpoint,
                headers=headers,
                data=form_data
            ) as response:
                
                self.last_request_time = time.time()
                latency = self.last_request_time - start_time
                
                response_text = await response.text()
                logger.info(f"Sarvam API response status: {response.status}")
                
                if response.status == 200:
                    try:
                        result = json.loads(response_text)
                        
                        # Extract transcript based on model type
                        if self.model.startswith("saaras"):
                            # Translation response
                            transcript = result.get("translation", "")
                        else:
                            # Transcription response
                            transcript = result.get("transcript", "")
                        
                        # Get other metadata
                        processing_time = result.get("processing_time", latency)
                        
                        if transcript:
                            # Send transcript to output queue
                            await self._send_transcript(
                                transcript=transcript,
                                is_final=True,
                                latency=processing_time
                            )
                        else:
                            logger.warning("Received empty transcript from Sarvam API")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Sarvam response: {e}")
                        logger.error(f"Response text: {response_text}")
                        
                else:
                    logger.error(f"Sarvam API error {response.status}: {response_text}")
                    await self._send_error_response(f"API error {response.status}")
                    
        except aiohttp.ClientTimeout:
            logger.error("Sarvam API request timed out")
            await self._send_error_response("Request timeout")
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            import traceback
            traceback.print_exc()
            
    async def _send_transcript(self, transcript: str, is_final: bool, latency: float):
        """Send transcript to output queue"""
        # Prepare response data
        response_data = {
            "type": "transcript" if is_final else "interim_transcript_received",
            "content": transcript
        }
        
        # Update metadata
        self.meta_info.update({
            "is_final": is_final,
            "transcriber_latency": latency,
            "language": self.language,
            "model": self.model,
            "audio_duration": self.total_audio_duration
        })
        
        # Create and send packet
        ws_packet = create_ws_data_packet(response_data, self.meta_info)
        await self.output_queue.put(ws_packet)
        
        logger.info(f"Sent transcript: '{transcript}' (latency: {latency:.3f}s)")
        
    async def _send_error_response(self, error_message: str):
        """Send error response to output queue"""
        error_data = {
            "type": "error",
            "content": f"Sarvam transcriber error: {error_message}"
        }
        
        if hasattr(self, 'meta_info'):
            ws_packet = create_ws_data_packet(error_data, self.meta_info)
        else:
            ws_packet = create_ws_data_packet(error_data, {})
            
        await self.output_queue.put(ws_packet)
        
    async def _cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up Sarvam transcriber")
        
        # Close HTTP session
        if self.session and not self.session.closed:
            await self.session.close()
            
        # Calculate final stats
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"Transcriber session stats - Duration: {total_time:.2f}s, "
                       f"Audio processed: {self.total_audio_duration:.2f}s")
            
        # Send completion notification
        if hasattr(self, 'meta_info'):
            self.meta_info["transcriber_duration"] = self.total_audio_duration
            completion_packet = create_ws_data_packet(
                {"type": "transcriber_connection_closed"},
                self.meta_info
            )
            await self.output_queue.put(completion_packet)
            
    def get_meta_info(self):
        """Get current metadata"""
        return getattr(self, 'meta_info', {})