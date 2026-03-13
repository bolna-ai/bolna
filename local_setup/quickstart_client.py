import asyncio
import pyaudio
import websockets
import logging
import base64
import json
import time
import argparse
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
import os
import queue
load_dotenv()
# Argument parsing
parser = argparse.ArgumentParser(description="Client for WebSocket communication")

audio_queue = queue.Queue()

# Set up logging
logging.basicConfig(level=logging.INFO)

play_audio_task = None

# Audio settings
format = pyaudio.paInt16
channels = 1
rate = 16000
chunk = 8000
audio = pyaudio.PyAudio()
start_time = time.time()
chunks = []
interruption_message = 0

# WebSocket server address based on connection type
server_url = "ws://localhost:5001" #os.getenv("BOLNA_WS_SERVER_URL")
assistant_id = os.getenv("ASSISTANT_ID") 
logging.info(f"Assistant ID {os.getenv('ASSISTANT_ID') }")
uri = f"{server_url}/chat/v1/{assistant_id}"

# Audio queue to store audio frames
input_queue = asyncio.Queue() 

# Callback function to add audio frames to the queue
def _callback(input_data, frame_count, time_info, status_flags):
    input_queue.put_nowait(input_data)
    logging.debug("Audio frame added to the queue")
    return (input_data, pyaudio.paContinue)

# Coroutine to open microphone stream and start sending audio
async def microphone():
    print("Starting microphone")
    stream = audio.open(
        format=format,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
        stream_callback=_callback
    )
        
    stream.start_stream()
    print("Listening")
    while stream.is_active():
        await asyncio.sleep(0.1)
                
    stream.stop_stream()
    stream.close()

async def emitter(ws):
    while True:
        audio_frame = await input_queue.get()
        base64_audio_frame = base64.b64encode(audio_frame).decode('utf-8')
        data = json.dumps({"type": "audio", "data": base64_audio_frame})
        
        global start_time 
        start_time = time.time()
        await ws.send(data)
        logging.debug("Audio frame sent to WebSocket server")

def audio_callback(outdata, frames, time, status):
    
    try:
        data = audio_queue.get_nowait()
        data = data.reshape(-1, 1)  
    except queue.Empty:
        outdata.fill(0)
    else:
        if len(data) < len(outdata):
            outdata[:len(data)] = data
            outdata[len(data):] = 0 
        else:
            outdata[:] = data


def start_audio_stream():
    stream = sd.OutputStream(
        samplerate=24000,
        channels=1,
        dtype=np.int16,
        callback=audio_callback,
        blocksize= 8192
    )
    stream.start()
    return stream

async def play_audio():
    
    while True:
        try:
            global chunks
            if len(chunks) > 0:
                chunk = chunks.pop(0)
                audio = base64.b64decode(chunk)
                print(f"Adjusted audio length: {len(audio)}")  
                if len(audio) % 2 != 0:
                    print(f"Audio chunk length is odd: {len(audio)}")
                    audio = audio[:-1]          
                print(f"Adjusted audio length: {len(audio)}")  
                audio_data = np.frombuffer(audio, dtype=np.int16)
                audio_queue.put(audio_data)
            else:
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error in playing audio: {e}")


async def receiver(ws):
    global chunks, play_audio_task
    while True:
        try:
            response = await ws.recv()
            response = json.loads(response)
            logging.info(f"{response.keys()}")
            if response["type"] != "audio":
                print(response)

            if response["type"] == "clear":
                logging.info("Got interrupt message and clearing chunks\n\n\n")
                if len(chunks) > 0:
                    logging.info("Stopping the current frame")
                    chunks.clear()
                    play_audio_task.cancel()
                    await asyncio.sleep(0)  # Yield control to allow task cancellation to complete
                    play_audio_task = asyncio.create_task(play_audio())
                continue
            if response["data"] is None:
                continue
            b64_audio = response["data"]
            chunks.append(b64_audio)

        except Exception as e:
            logging.error(e)


stream = start_audio_stream()

async def main():
    api_key = os.getenv("BOLNA_API_KEY", None)
    if api_key is not None:
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
    else:
        headers = None
    async with websockets.connect(uri, open_timeout=None, extra_headers=headers) as ws:
        global play_audio_task
        tasks = [microphone(), emitter(ws), receiver(ws)]
        play_audio_task = asyncio.create_task(play_audio())
        await asyncio.gather(*tasks)

print("Starting with the loop")
# Run the asyncio event loop
asyncio.run(main())