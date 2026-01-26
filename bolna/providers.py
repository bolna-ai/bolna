from .synthesizer import PollySynthesizer, ElevenlabsSynthesizer, OPENAISynthesizer, DeepgramSynthesizer, AzureSynthesizer, CartesiaSynthesizer, SmallestSynthesizer, SarvamSynthesizer, RimeSynthesizer, PixaSynthesizer
from .transcriber import DeepgramTranscriber, AzureTranscriber, SarvamTranscriber, AssemblyAITranscriber, GoogleTranscriber, PixaTranscriber, GladiaTranscriber, ElevenLabsTranscriber, SmallestTranscriber
from .input_handlers import DefaultInputHandler, TwilioInputHandler, ExotelInputHandler, PlivoInputHandler, CallingServiceInputHandler
from .output_handlers import DefaultOutputHandler, TwilioOutputHandler, ExotelOutputHandler, PlivoOutputHandler, CallingServiceOutputHandler
from .llms import OpenAiLLM, LiteLLM, AzureLLM

SUPPORTED_SYNTHESIZER_MODELS = {
    'polly': PollySynthesizer,
    'elevenlabs': ElevenlabsSynthesizer,
    'openai': OPENAISynthesizer,
    'deepgram': DeepgramSynthesizer,
    'azuretts': AzureSynthesizer,
    'cartesia': CartesiaSynthesizer,
    'smallest': SmallestSynthesizer,
    'sarvam': SarvamSynthesizer,
    'rime': RimeSynthesizer,
    'pixa': PixaSynthesizer
}

SUPPORTED_TRANSCRIBER_PROVIDERS = {
    'deepgram': DeepgramTranscriber,
    'azure': AzureTranscriber,
    'sarvam': SarvamTranscriber,
    'assembly': AssemblyAITranscriber,
    'google': GoogleTranscriber,
    'pixa': PixaTranscriber,
    'gladia': GladiaTranscriber,
    'elevenlabs': ElevenLabsTranscriber,
    'smallest': SmallestTranscriber
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
    'azure-openai': AzureLLM,
    'perplexity': LiteLLM,
    'vllm': LiteLLM,
    'anyscale': LiteLLM,
    'custom': OpenAiLLM,
    'ola': OpenAiLLM,
    'groq': LiteLLM,
    'anthropic': LiteLLM,
    'deepseek': LiteLLM,
    'openrouter': LiteLLM,
    'azure': AzureLLM
}
SUPPORTED_INPUT_HANDLERS = {
    'default': DefaultInputHandler,
    'twilio': TwilioInputHandler,
    'exotel': ExotelInputHandler,
    'plivo': PlivoInputHandler,
    'calling_service': CallingServiceInputHandler
}
SUPPORTED_INPUT_TELEPHONY_HANDLERS = {
    'twilio': TwilioInputHandler,
    'exotel': ExotelInputHandler,
    'plivo': PlivoInputHandler,
    'calling_service': CallingServiceInputHandler
}
SUPPORTED_OUTPUT_HANDLERS = {
    'default': DefaultOutputHandler,
    'twilio': TwilioOutputHandler,
    'exotel': ExotelOutputHandler,
    'plivo': PlivoOutputHandler,
    'calling_service': CallingServiceOutputHandler
}
SUPPORTED_OUTPUT_TELEPHONY_HANDLERS = {
    'twilio': TwilioOutputHandler,
    'exotel': ExotelOutputHandler,
    'plivo': PlivoOutputHandler,
    'calling_service': CallingServiceOutputHandler
}
