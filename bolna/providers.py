from .synthesizer import PollySynthesizer, ElevenlabsSynthesizer, OPENAISynthesizer, DeepgramSynthesizer, AzureSynthesizer, CartesiaSynthesizer, SmallestSynthesizer, SarvamSynthesizer, RimeSynthesizer
from .transcriber import DeepgramTranscriber, AzureTranscriber
from .input_handlers import DefaultInputHandler, TwilioInputHandler, ExotelInputHandler, PlivoInputHandler
from .output_handlers import DefaultOutputHandler, TwilioOutputHandler, ExotelOutputHandler, PlivoOutputHandler
from .llms import OpenAiLLM, LiteLLM

# Import the new enums
from .enums.providers import (
    SynthesizerProvider,
    TranscriberProvider,
    LLMProvider,
    TelephonyProvider
)

# Updated to use enums - maintains backward compatibility
SUPPORTED_SYNTHESIZER_MODELS = {
    SynthesizerProvider.POLLY: PollySynthesizer,
    SynthesizerProvider.ELEVENLABS: ElevenlabsSynthesizer,
    SynthesizerProvider.OPENAI: OPENAISynthesizer,
    SynthesizerProvider.DEEPGRAM: DeepgramSynthesizer,
    SynthesizerProvider.AZURE_TTS: AzureSynthesizer,
    SynthesizerProvider.CARTESIA: CartesiaSynthesizer,
    SynthesizerProvider.SMALLEST: SmallestSynthesizer,
    SynthesizerProvider.SARVAM: SarvamSynthesizer,
    SynthesizerProvider.RIME: RimeSynthesizer
}

# Updated to use enums - maintains backward compatibility
SUPPORTED_TRANSCRIBER_PROVIDERS = {
    TranscriberProvider.DEEPGRAM: DeepgramTranscriber,
    TranscriberProvider.AZURE: AzureTranscriber
}

# Backwards compatibility - kept for existing code
SUPPORTED_TRANSCRIBER_MODELS = {
    TranscriberProvider.DEEPGRAM: DeepgramTranscriber
}

# Updated to use enums - maintains backward compatibility
SUPPORTED_LLM_PROVIDERS = {
    LLMProvider.OPENAI: OpenAiLLM,
    LLMProvider.COHERE: LiteLLM,
    LLMProvider.OLLAMA: LiteLLM,
    LLMProvider.DEEPINFRA: LiteLLM,
    LLMProvider.TOGETHER: LiteLLM,
    LLMProvider.FIREWORKS: LiteLLM,
    LLMProvider.AZURE_OPENAI: LiteLLM,
    LLMProvider.PERPLEXITY: LiteLLM,
    LLMProvider.VLLM: LiteLLM,
    LLMProvider.ANYSCALE: LiteLLM,
    LLMProvider.CUSTOM: OpenAiLLM,
    LLMProvider.OLA: OpenAiLLM,
    LLMProvider.GROQ: LiteLLM,
    LLMProvider.ANTHROPIC: LiteLLM,
    LLMProvider.DEEPSEEK: LiteLLM,
    LLMProvider.OPENROUTER: LiteLLM,
    LLMProvider.AZURE: LiteLLM  # Backwards compatibility
}
# Updated to use enums - maintains backward compatibility
SUPPORTED_INPUT_HANDLERS = {
    TelephonyProvider.DEFAULT: DefaultInputHandler,
    TelephonyProvider.TWILIO: TwilioInputHandler,
    TelephonyProvider.EXOTEL: ExotelInputHandler,
    TelephonyProvider.PLIVO: PlivoInputHandler
}
# Updated to use enums - maintains backward compatibility
SUPPORTED_INPUT_TELEPHONY_HANDLERS = {
    TelephonyProvider.TWILIO: TwilioInputHandler,
    TelephonyProvider.EXOTEL: ExotelInputHandler,
    TelephonyProvider.PLIVO: PlivoInputHandler
}
# Updated to use enums - maintains backward compatibility
SUPPORTED_OUTPUT_HANDLERS = {
    TelephonyProvider.DEFAULT: DefaultOutputHandler,
    TelephonyProvider.TWILIO: TwilioOutputHandler,
    TelephonyProvider.EXOTEL: ExotelOutputHandler,
    TelephonyProvider.PLIVO: PlivoOutputHandler
}
# Updated to use enums - maintains backward compatibility
SUPPORTED_OUTPUT_TELEPHONY_HANDLERS = {
    TelephonyProvider.TWILIO: TwilioOutputHandler,
    TelephonyProvider.EXOTEL: ExotelOutputHandler,
    TelephonyProvider.PLIVO: PlivoOutputHandler
}
