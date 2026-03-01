from .synthesizer import PollySynthesizer, ElevenlabsSynthesizer, OPENAISynthesizer, DeepgramSynthesizer, AzureSynthesizer, CartesiaSynthesizer, SmallestSynthesizer, SarvamSynthesizer, RimeSynthesizer, PixaSynthesizer
from .transcriber import DeepgramTranscriber, AzureTranscriber, SarvamTranscriber, AssemblyAITranscriber, GoogleTranscriber, PixaTranscriber, GladiaTranscriber, ElevenLabsTranscriber, SmallestTranscriber, ShunyaTranscriber
from .input_handlers import DefaultInputHandler, TwilioInputHandler, ExotelInputHandler, PlivoInputHandler, VobizInputHandler, SipTrunkInputHandler
from .output_handlers import DefaultOutputHandler, TwilioOutputHandler, ExotelOutputHandler, PlivoOutputHandler, VobizOutputHandler, SipTrunkOutputHandler
from .llms import OpenAiLLM, LiteLLM, AzureLLM
from .enums import TelephonyProvider, SynthesizerProvider, TranscriberProvider, LLMProvider

SUPPORTED_SYNTHESIZER_MODELS = {
    SynthesizerProvider.POLLY.value: PollySynthesizer,
    SynthesizerProvider.ELEVENLABS.value: ElevenlabsSynthesizer,
    SynthesizerProvider.OPENAI.value: OPENAISynthesizer,
    SynthesizerProvider.DEEPGRAM.value: DeepgramSynthesizer,
    SynthesizerProvider.AZURETTS.value: AzureSynthesizer,
    SynthesizerProvider.CARTESIA.value: CartesiaSynthesizer,
    SynthesizerProvider.SMALLEST.value: SmallestSynthesizer,
    SynthesizerProvider.SARVAM.value: SarvamSynthesizer,
    SynthesizerProvider.RIME.value: RimeSynthesizer,
    SynthesizerProvider.PIXA.value: PixaSynthesizer
}

SUPPORTED_TRANSCRIBER_PROVIDERS = {
    TranscriberProvider.DEEPGRAM.value: DeepgramTranscriber,
    TranscriberProvider.AZURE.value: AzureTranscriber,
    TranscriberProvider.SARVAM.value: SarvamTranscriber,
    TranscriberProvider.ASSEMBLY.value: AssemblyAITranscriber,
    TranscriberProvider.GOOGLE.value: GoogleTranscriber,
    TranscriberProvider.PIXA.value: PixaTranscriber,
    TranscriberProvider.GLADIA.value: GladiaTranscriber,
    TranscriberProvider.ELEVENLABS.value: ElevenLabsTranscriber,
    TranscriberProvider.SMALLEST.value: SmallestTranscriber,
    TranscriberProvider.SHUNYA.value: ShunyaTranscriber
}

#Backwards compatibility
SUPPORTED_TRANSCRIBER_MODELS = {
    'deepgram': DeepgramTranscriber
}

SUPPORTED_LLM_PROVIDERS = {
    LLMProvider.OPENAI.value: OpenAiLLM,
    LLMProvider.COHERE.value: LiteLLM,
    LLMProvider.OLLAMA.value: LiteLLM,
    LLMProvider.DEEPINFRA.value: LiteLLM,
    LLMProvider.TOGETHER.value: LiteLLM,
    LLMProvider.FIREWORKS.value: LiteLLM,
    LLMProvider.AZURE_OPENAI.value: AzureLLM,
    LLMProvider.PERPLEXITY.value: LiteLLM,
    LLMProvider.VLLM.value: LiteLLM,
    LLMProvider.ANYSCALE.value: LiteLLM,
    LLMProvider.CUSTOM.value: OpenAiLLM,
    LLMProvider.OLA.value: OpenAiLLM,
    LLMProvider.GROQ.value: LiteLLM,
    LLMProvider.ANTHROPIC.value: LiteLLM,
    LLMProvider.DEEPSEEK.value: LiteLLM,
    LLMProvider.OPENROUTER.value: LiteLLM,
    LLMProvider.AZURE.value: AzureLLM
}
SUPPORTED_INPUT_HANDLERS = {
    TelephonyProvider.DEFAULT.value: DefaultInputHandler,
    TelephonyProvider.TWILIO.value: TwilioInputHandler,
    TelephonyProvider.EXOTEL.value: ExotelInputHandler,
    TelephonyProvider.PLIVO.value: PlivoInputHandler,
    TelephonyProvider.VOBIZ.value: VobizInputHandler,
    TelephonyProvider.SIP_TRUNK.value: SipTrunkInputHandler
}
SUPPORTED_INPUT_TELEPHONY_HANDLERS = {
    TelephonyProvider.TWILIO.value: TwilioInputHandler,
    TelephonyProvider.EXOTEL.value: ExotelInputHandler,
    TelephonyProvider.PLIVO.value: PlivoInputHandler,
    TelephonyProvider.VOBIZ.value: VobizInputHandler,
    TelephonyProvider.SIP_TRUNK.value: SipTrunkInputHandler
}
SUPPORTED_OUTPUT_HANDLERS = {
    TelephonyProvider.DEFAULT.value: DefaultOutputHandler,
    TelephonyProvider.TWILIO.value: TwilioOutputHandler,
    TelephonyProvider.EXOTEL.value: ExotelOutputHandler,
    TelephonyProvider.PLIVO.value: PlivoOutputHandler,
    TelephonyProvider.VOBIZ.value: VobizOutputHandler,
    TelephonyProvider.SIP_TRUNK.value: SipTrunkOutputHandler
}
SUPPORTED_OUTPUT_TELEPHONY_HANDLERS = {
    TelephonyProvider.TWILIO.value: TwilioOutputHandler,
    TelephonyProvider.EXOTEL.value: ExotelOutputHandler,
    TelephonyProvider.PLIVO.value: PlivoOutputHandler,
    TelephonyProvider.VOBIZ.value: VobizOutputHandler,
    TelephonyProvider.SIP_TRUNK.value: SipTrunkOutputHandler
}
