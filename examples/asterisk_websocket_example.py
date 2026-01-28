"""
Example script demonstrating Asterisk WebSocket integration with Bolna.

This example shows how to:
1. Set up an Asterisk WebSocket connection
2. Handle incoming audio from Asterisk
3. Send audio back to Asterisk
4. Process control events and commands

Requirements:
- Asterisk server with WebSocket support
- ARI configured with proper credentials
- Network connectivity to Asterisk server
"""

import asyncio
import json
from typing import Dict, Any


class AsteriskWebSocketExample:
    """Example integration with Asterisk via WebSocket"""
    
    def __init__(self, asterisk_url: str, ari_username: str, ari_password: str):
        self.asterisk_url = asterisk_url
        self.ari_username = ari_username
        self.ari_password = ari_password
        self.channel_id = None
        self.connection_id = None
        
    async def create_external_media_channel(self, codec: str = "ulaw") -> Dict[str, Any]:
        """
        Create an external media channel in Asterisk using ARI.
        
        This creates a channel that waits for an incoming WebSocket connection.
        
        Args:
            codec: Audio codec to use (ulaw, alaw, slin, slin16, etc.)
            
        Returns:
            Channel information including connection_id
        """
        import aiohttp
        
        ari_url = f"{self.asterisk_url}/ari/channels/externalMedia"
        auth = aiohttp.BasicAuth(self.ari_username, self.ari_password)
        
        params = {
            "transport": "websocket",
            "encapsulation": "none",
            "external_host": "INCOMING",  # Wait for incoming connection
            "format": codec,
            "connection_type": "server",
            "transport_data": "f(json)"  # Use JSON control message format
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(ari_url, auth=auth, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Failed to create channel: {await response.text()}")
                
                channel_data = await response.json()
                self.channel_id = channel_data["id"]
                
                # Get the connection ID from channel variables
                # This is needed to connect to the WebSocket
                self.connection_id = channel_data.get("channelvars", {}).get(
                    "MEDIA_WEBSOCKET_CONNECTION_ID"
                )
                
                print(f"Created channel: {self.channel_id}")
                print(f"Connection ID: {self.connection_id}")
                print(f"Connect WebSocket to: ws://{self.asterisk_url}/media/{self.connection_id}")
                
                return channel_data
    
    async def connect_to_asterisk(self):
        """
        Connect to Asterisk WebSocket endpoint.
        
        In a real Bolna integration, this connection would be handled by
        the CallingServiceInputHandler and CallingServiceOutputHandler.
        """
        import aiohttp
        
        ws_url = f"{self.asterisk_url}/media/{self.connection_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                print(f"Connected to Asterisk WebSocket")
                
                # Wait for MEDIA_START event
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        # Control event
                        event = json.loads(msg.data)
                        event_type = event.get('event')
                        
                        if event_type == 'MEDIA_START':
                            print(f"MEDIA_START received: {event}")
                            await self.handle_media_start(ws, event)
                            break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f"WebSocket error: {msg}")
                        break
    
    async def handle_media_start(self, ws, event: Dict[str, Any]):
        """
        Handle MEDIA_START event and begin audio streaming.
        
        Args:
            ws: WebSocket connection
            event: MEDIA_START event data
        """
        channel_id = event.get('channel_id')
        codec = event.get('codec')
        format_type = event.get('format')
        ptime = event.get('ptime', 20)
        
        print(f"Media started - Codec: {codec}, Format: {format_type}, Ptime: {ptime}ms")
        
        # Send ANSWER command if needed
        answer_cmd = {
            "command": "ANSWER",
            "channel_id": channel_id
        }
        await ws.send_str(json.dumps(answer_cmd))
        print("Sent ANSWER command")
        
        # Now you can send/receive audio
        # In Bolna, this would be handled by the input/output handlers
        
        # Example: Send some test audio (silence)
        sample_rate = 8000 if codec in ('ulaw', 'alaw') else 16000
        bytes_per_sample = 1 if codec in ('ulaw', 'alaw') else 2
        chunk_size = (sample_rate * ptime // 1000) * bytes_per_sample
        
        silence = b'\x00' * chunk_size
        
        # Send a few chunks of silence
        for _ in range(10):
            await ws.send_bytes(silence)
            await asyncio.sleep(ptime / 1000.0)  # Respect ptime
        
        # Send HANGUP command
        hangup_cmd = {
            "command": "HANGUP",
            "channel_id": channel_id
        }
        await ws.send_str(json.dumps(hangup_cmd))
        print("Sent HANGUP command")


# Example Bolna task configuration for Asterisk WebSocket
BOLNA_TASK_CONFIG_EXAMPLE = {
    "task_type": "conversation",
    "toolchain": {
        "execution": "parallel",
        "pipelines": [
            ["transcriber", "llm", "synthesizer"]
        ]
    },
    "tools_config": {
        "input": {
            "provider": "calling_service",  # Use Asterisk WebSocket
            "stream": True
        },
        "output": {
            "provider": "calling_service",  # Use Asterisk WebSocket
            "stream": True
        },
        "transcriber": {
            "model": "deepgram",
            "stream": True,
            "language": "en",
            "endpointing": 200
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "max_tokens": 200,
            "temperature": 0.7
        },
        "synthesizer": {
            "provider": "elevenlabs",
            "stream": True,
            "audio_format": "mulaw",  # Match Asterisk codec
            "sample_rate": 8000
        }
    }
}


async def main():
    """
    Example usage of Asterisk WebSocket integration.
    
    This demonstrates the flow, but in production you would use
    Bolna's built-in handlers.
    """
    
    # Configuration
    ASTERISK_URL = "http://localhost:8088"
    ARI_USERNAME = "bolna"
    ARI_PASSWORD = "your_password"
    
    example = AsteriskWebSocketExample(ASTERISK_URL, ARI_USERNAME, ARI_PASSWORD)
    
    try:
        # Step 1: Create external media channel
        print("Creating external media channel...")
        channel = await example.create_external_media_channel(codec="ulaw")
        
        # Step 2: Connect to WebSocket
        print("Connecting to WebSocket...")
        await example.connect_to_asterisk()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    """
    To run this example:
    
    1. Ensure Asterisk is running with:
       - HTTP server enabled (http.conf)
       - ARI enabled (ari.conf)
       - WebSocket support
    
    2. Update the configuration above with your Asterisk details
    
    3. Run: python asterisk_websocket_example.py
    
    For production use with Bolna:
    - Configure your task with provider: "calling_service"
    - The handlers will automatically manage the WebSocket connection
    - Audio will flow through Bolna's pipeline (transcriber -> LLM -> synthesizer)
    """
    
    print("=" * 60)
    print("Asterisk WebSocket Integration Example")
    print("=" * 60)
    print()
    print("This example demonstrates how to:")
    print("- Create an external media channel in Asterisk")
    print("- Connect via WebSocket")
    print("- Handle MEDIA_START events")
    print("- Send control commands (ANSWER, HANGUP)")
    print("- Stream audio data")
    print()
    print("Example Bolna configuration:")
    print(json.dumps(BOLNA_TASK_CONFIG_EXAMPLE, indent=2))
    print()
    print("=" * 60)
    print()
    
    # Uncomment to run the example:
    # asyncio.run(main())
    
    print("NOTE: Update configuration and uncomment asyncio.run(main()) to run")
