import asyncio
import os
from deepgram_synthesizer import DeepgramSynthesizer  # import your class

async def main():
    # Make sure your Deepgram API key is set
    os.environ["DEEPGRAM_AUTH_TOKEN"] = "00ea3a149a4336af372a2a1e04b19fb98963791e"

    synthesizer = DeepgramSynthesizer(audio_format="mp3", model="aura-2-thalia-en")

    # Test a one-off synthesis
    audio_bytes = await synthesizer.synthesize("Hello! This is Deepgram speaking.")
    print(f"Got {len(audio_bytes)} bytes of audio")

    # Save result to a file
    with open("deepgram_test.wav", "wb") as f:
        f.write(audio_bytes)
    print("✅ Audio saved as deepgram_test.wav")

if __name__ == "__main__":
    asyncio.run(main())
