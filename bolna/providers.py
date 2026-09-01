from .synthesizer import (
    PollySynthesizer,
    ElevenlabsSynthesizer,
    ElevenlabsV3Synthesizer,
    OPENAISynthesizer,
    DeepgramSynthesizer,
    AzureSynthesizer,
    CartesiaSynthesizer,
    SmallestSynthesizer,
    SarvamSynthesizer,
    RimeSynthesizer,
    PixaSynthesizer,
    MayaSynthesizer,
    KalpaSynthesizer,
)
from .transcriber import (
    DeepgramTranscriber,
    AzureTranscriber,
    SarvamTranscriber,
    AssemblyAITranscriber,
    GoogleTranscriber,
    PixaTranscriber,
    GladiaTranscriber,
    ElevenLabsTranscriber,
    SmallestTranscriber,
    OpenAITranscriber,
    SonioxTranscriber,
    GeminiTranscriber,
)
from .input_handlers import (
    DefaultInputHandler,
    TwilioInputHandler,
    ExotelInputHandler,
    PlivoInputHandler,
    VobizInputHandler,
    SipTrunkInputHandler,
    FreeSwitchInputHandler,
)
from .output_handlers import (
    DefaultOutputHandler,
    TwilioOutputHandler,
    ExotelOutputHandler,
    PlivoOutputHandler,
    VobizOutputHandler,
    SipTrunkOutputHandler,
    FreeSwitchOutputHandler,
)
from .llms import OpenAiLLM, LiteLLM, AzureLLM, GeminiLLM
from .s2s import GeminiLiveS2S, OpenAIRealtimeS2S
from .enums import TelephonyProvider, SynthesizerProvider, TranscriberProvider, LLMProvider, S2SProvider


def elevenlabs_synthesizer(**kwargs):
    """Eleven v3 is served only from the text-to-dialogue socket; multi-stream-input 403s
    on those model ids. Everything else stays on the original synthesizer."""
    # `or ""` rather than a get() default: a stored config can carry an explicit null model.
    cls = ElevenlabsV3Synthesizer if (kwargs.get("model") or "").startswith("eleven_v3") else ElevenlabsSynthesizer
    return cls(**kwargs)


SUPPORTED_SYNTHESIZER_MODELS = {
    SynthesizerProvider.POLLY.value: PollySynthesizer,
    SynthesizerProvider.ELEVENLABS.value: elevenlabs_synthesizer,
    SynthesizerProvider.OPENAI.value: OPENAISynthesizer,
    SynthesizerProvider.DEEPGRAM.value: DeepgramSynthesizer,
    SynthesizerProvider.AZURETTS.value: AzureSynthesizer,
    SynthesizerProvider.CARTESIA.value: CartesiaSynthesizer,
    SynthesizerProvider.SMALLEST.value: SmallestSynthesizer,
    SynthesizerProvider.SARVAM.value: SarvamSynthesizer,
    SynthesizerProvider.RIME.value: RimeSynthesizer,
    SynthesizerProvider.PIXA.value: PixaSynthesizer,
    SynthesizerProvider.MAYA.value: MayaSynthesizer,
    SynthesizerProvider.KALPA.value: KalpaSynthesizer,
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
    TranscriberProvider.OPENAI.value: OpenAITranscriber,
    TranscriberProvider.SONIOX.value: SonioxTranscriber,
    TranscriberProvider.GEMINI.value: GeminiTranscriber,
}

# Backwards compatibility
SUPPORTED_TRANSCRIBER_MODELS = {"deepgram": DeepgramTranscriber}

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
    LLMProvider.AZURE.value: AzureLLM,
    LLMProvider.GOOGLE.value: GeminiLLM,
}
SUPPORTED_INPUT_HANDLERS = {
    TelephonyProvider.DEFAULT.value: DefaultInputHandler,
    TelephonyProvider.TWILIO.value: TwilioInputHandler,
    TelephonyProvider.EXOTEL.value: ExotelInputHandler,
    TelephonyProvider.PLIVO.value: PlivoInputHandler,
    TelephonyProvider.VOBIZ.value: VobizInputHandler,
    TelephonyProvider.SIP_TRUNK.value: SipTrunkInputHandler,
    TelephonyProvider.FREESWITCH.value: FreeSwitchInputHandler,
}
SUPPORTED_INPUT_TELEPHONY_HANDLERS = {
    TelephonyProvider.TWILIO.value: TwilioInputHandler,
    TelephonyProvider.EXOTEL.value: ExotelInputHandler,
    TelephonyProvider.PLIVO.value: PlivoInputHandler,
    TelephonyProvider.VOBIZ.value: VobizInputHandler,
    TelephonyProvider.SIP_TRUNK.value: SipTrunkInputHandler,
}
SUPPORTED_OUTPUT_HANDLERS = {
    TelephonyProvider.DEFAULT.value: DefaultOutputHandler,
    TelephonyProvider.TWILIO.value: TwilioOutputHandler,
    TelephonyProvider.EXOTEL.value: ExotelOutputHandler,
    TelephonyProvider.PLIVO.value: PlivoOutputHandler,
    TelephonyProvider.VOBIZ.value: VobizOutputHandler,
    TelephonyProvider.SIP_TRUNK.value: SipTrunkOutputHandler,
    TelephonyProvider.FREESWITCH.value: FreeSwitchOutputHandler,
}
SUPPORTED_OUTPUT_TELEPHONY_HANDLERS = {
    TelephonyProvider.TWILIO.value: TwilioOutputHandler,
    TelephonyProvider.EXOTEL.value: ExotelOutputHandler,
    TelephonyProvider.PLIVO.value: PlivoOutputHandler,
    TelephonyProvider.VOBIZ.value: VobizOutputHandler,
    TelephonyProvider.SIP_TRUNK.value: SipTrunkOutputHandler,
}
SUPPORTED_S2S_PROVIDERS = {
    S2SProvider.OPENAI_REALTIME.value: OpenAIRealtimeS2S,
    S2SProvider.GEMINI_LIVE.value: GeminiLiveS2S,
}
