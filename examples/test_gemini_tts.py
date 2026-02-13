"""
Test script for Gemini TTS integration in Bolna
This script tests the Gemini TTS synthesizer with a simple assistant
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from bolna.assistant import Assistant
from bolna.models import Transcriber, Synthesizer, GeminiConfig, LlmAgent, SimpleLlmAgent

async def test_gemini_tts():
    """Test Gemini TTS with Bolna Assistant"""
    
    # Check if GOOGLE_API_KEY is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable is not set!")
        print("Please set it in your .env file or export it:")
        print("export GOOGLE_API_KEY=your_api_key_here")
        return
    
    print("=" * 60)
    print("Testing Gemini TTS Integration in Bolna")
    print("=" * 60)
    
    # Create assistant
    assistant = Assistant(name="gemini_tts_test_agent")
    
    # Configure Gemini TTS
    print("\n✓ Configuring Gemini TTS synthesizer...")
    print("  Model: gemini-2.5-flash-preview-tts")
    print("  Voice: Aoede")
    
    synthesizer = Synthesizer(
        provider="gemini",
        provider_config=GeminiConfig(
            voice="Aoede Voice",
            voice_name="Aoede",
            model="gemini-2.5-flash-preview-tts",
            language="en"
        ),
        stream=False,  # Gemini doesn't support streaming yet
        audio_format="wav"
    )
    
    # Simple LLM for testing (using gpt-4o-mini)
    print("\n✓ Configuring LLM agent...")
    llm_agent = LlmAgent(
        agent_type="simple_llm_agent",
        agent_flow_type="streaming",
        llm_config=SimpleLlmAgent(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.3,
        ),
    )
    
    # Configure transcriber
    print("\n✓ Configuring Deepgram transcriber...")
    transcriber = Transcriber(
        provider="deepgram",
        model="nova-2",
        stream=True,
        language="en"
    )
    
    # Add task to assistant
    print("\n✓ Adding conversation task to assistant...")
    assistant.add_task(
        task_type="conversation",
        llm_agent=llm_agent,
        transcriber=transcriber,
        synthesizer=synthesizer
    )
    
    print("\n✓ Starting assistant execution...")
    print("\nGenerating audio chunks:")
    print("-" * 60)
    
    chunk_count = 0
    total_bytes = 0
    
    try:
        async for chunk in assistant.execute():
            chunk_count += 1
            audio_data = chunk.get('data', b'')
            chunk_size = len(audio_data)
            total_bytes += chunk_size
            
            print(f"Chunk #{chunk_count}: {chunk_size} bytes")
            
            # Limit test to avoid long execution
            if chunk_count >= 5:
                print("\n✓ Test limit reached (5 chunks), stopping...")
                break
                
    except KeyboardInterrupt:
        print("\n\n✗ Test interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("-" * 60)
    print(f"\n✓ Test completed successfully!")
    print(f"  Total chunks: {chunk_count}")
    print(f"  Total bytes: {total_bytes}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(test_gemini_tts())
