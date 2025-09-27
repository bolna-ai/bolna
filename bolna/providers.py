from .synthesizer import PollySynthesizer, ElevenlabsSynthesizer, OPENAISynthesizer, DeepgramSynthesizer, AzureSynthesizer, CartesiaSynthesizer, SmallestSynthesizer, SarvamSynthesizer, RimeSynthesizer
from .transcriber import DeepgramTranscriber, AzureTranscriber , SarvamTranscriber, AssemblyAITranscriber
from .input_handlers import DefaultInputHandler, TwilioInputHandler, ExotelInputHandler, PlivoInputHandler
from .output_handlers import DefaultOutputHandler, TwilioOutputHandler, ExotelOutputHandler, PlivoOutputHandler
from .llms import OpenAiLLM, LiteLLM

SUPPORTED_SYNTHESIZER_MODELS = {
    'polly': PollySynthesizer,
    'elevenlabs': ElevenlabsSynthesizer,
    'openai': OPENAISynthesizer,
    'deepgram': DeepgramSynthesizer,
    'azuretts': AzureSynthesizer,
    'cartesia': CartesiaSynthesizer,
    'smallest': SmallestSynthesizer,
    'sarvam': SarvamSynthesizer,
    'rime': RimeSynthesizer
}

SUPPORTED_TRANSCRIBER_PROVIDERS = {
    'deepgram': DeepgramTranscriber,
    'azure': AzureTranscriber,
    'sarvam': SarvamTranscriber,
    'assembly': AssemblyAITranscriber
}

#Backwards compatibility
SUPPORTED_TRANSCRIBER_MODELS = {
    'deepgram': DeepgramTranscriber
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
    'deepseek': LiteLLM,
    'openrouter': LiteLLM,
    'azure': LiteLLM #Backwards compatibility
}
SUPPORTED_INPUT_HANDLERS = {
    'default': DefaultInputHandler,
    'twilio': TwilioInputHandler,
    'exotel': ExotelInputHandler,
    'plivo': PlivoInputHandler
}
SUPPORTED_INPUT_TELEPHONY_HANDLERS = {
    'twilio': TwilioInputHandler,
    'exotel': ExotelInputHandler,
    'plivo': PlivoInputHandler
}
SUPPORTED_OUTPUT_HANDLERS = {
    'default': DefaultOutputHandler,
    'twilio': TwilioOutputHandler,
    'exotel': ExotelOutputHandler,
    'plivo': PlivoOutputHandler
}
SUPPORTED_OUTPUT_TELEPHONY_HANDLERS = {
    'twilio': TwilioOutputHandler,
    'exotel': ExotelOutputHandler,
    'plivo': PlivoOutputHandler
}
