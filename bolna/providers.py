from bolna.transcriber.bodhi_transcriber import BodhiTranscriber
from .synthesizer import PollySynthesizer, ElevenlabsSynthesizer, OPENAISynthesizer, DeepgramSynthesizer, AzureSynthesizer
from .transcriber import DeepgramTranscriber, WhisperTranscriber
from .input_handlers import DefaultInputHandler, TwilioInputHandler, ExotelInputHandler, PlivoInputHandler, DailyInputHandler
from .output_handlers import DefaultOutputHandler, TwilioOutputHandler, ExotelOutputHandler, PlivoOutputHandler, DailyOutputHandler
from .llms import OpenAiLLM, LiteLLM

SUPPORTED_SYNTHESIZER_MODELS = {
    'polly': PollySynthesizer,
    'elevenlabs': ElevenlabsSynthesizer,
    'openai': OPENAISynthesizer,
    'deepgram': DeepgramSynthesizer,
    'azuretts': AzureSynthesizer
}

SUPPORTED_TRANSCRIBER_PROVIDERS = {
    'deepgram': DeepgramTranscriber,
    'whisper': WhisperTranscriber,
    'bodhi': BodhiTranscriber
}

#Backwards compatibility
SUPPORTED_TRANSCRIBER_MODELS = {
    'deepgram': DeepgramTranscriber,
    'whisper': WhisperTranscriber #Seperate out a transcriber for https://github.com/bolna-ai/streaming-transcriber-server or build a deepgram compatible proxy
}

SUPPORTED_LLM_PROVIDERS = {
    'openai': OpenAiLLM,
    'cohere': LiteLLM,
    'ollama': LiteLLM,
    'deepinfra': LiteLLM,
    'together': LiteLLM,
    'fireworks': LiteLLM,
    'azure-openai': LiteLLM,
    'perplexity': LiteLLM,
    'vllm': LiteLLM,
    'anyscale': LiteLLM,
    'custom': OpenAiLLM,
    'ola': OpenAiLLM,
    'groq': LiteLLM,
    'anthropic': LiteLLM,
    'azure': LiteLLM #Backwards compatibility
}
SUPPORTED_INPUT_HANDLERS = {
    'default': DefaultInputHandler,
    'twilio': TwilioInputHandler,
    'exotel': ExotelInputHandler,
    'plivo': PlivoInputHandler,
    'daily': DailyInputHandler
}
SUPPORTED_INPUT_TELEPHONY_HANDLERS = {
    'twilio': TwilioInputHandler,
    'exotel': ExotelInputHandler,
    'plivo': PlivoInputHandler,
    'daily': DailyInputHandler
}
SUPPORTED_OUTPUT_HANDLERS = {
    'default': DefaultOutputHandler,
    'twilio': TwilioOutputHandler,
    'exotel': ExotelOutputHandler,
    'plivo': PlivoOutputHandler,
    'daily': DailyOutputHandler
}
SUPPORTED_OUTPUT_TELEPHONY_HANDLERS = {
    'twilio': TwilioOutputHandler,
    'exotel': ExotelOutputHandler,
    'plivo': PlivoOutputHandler,
    'daily': DailyOutputHandler
}
