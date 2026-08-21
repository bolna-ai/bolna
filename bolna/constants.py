from datetime import datetime, timezone
from bolna.enums import ReasoningEffort as RE

PREPROCESS_DIR = "agent_data"
PCM16_SCALE = 32768.0

# Provider label for browser web calls (raw-WS transport). Not a TelephonyProvider — it's the
# codebase-wide literal the transcribers already branch on; single home for new comparisons.
WEB_BASED_CALL_PROVIDER = "web_based_call"
# Web + FreeSWITCH webcall paths play raw PCM at this fixed rate (telephony stays 8k mulaw).
WEBCALL_TTS_SAMPLE_RATE = 24000

OPENAI_TRANSCRIBER_HEARTBEAT_INTERVAL_S = 5
OPENAI_TRANSCRIBER_UTTERANCE_TIMEOUT_S = 0.5

# ElevenLabs realtime (scribe_v2_realtime) accepts up to 50 keyterms for biasing.
ELEVENLABS_REALTIME_MAX_KEYTERMS = 50

# Deepgram Flux defaults — all overridable via agent transcriber config
DEEPGRAM_FLUX_EOT_THRESHOLD = 0.7  # confidence to declare end-of-turn
DEEPGRAM_FLUX_EAGER_EOT_THRESHOLD = 0.5  # confidence to trigger speculative LLM early
DEEPGRAM_FLUX_EOT_TIMEOUT_MS = 500  # max silence before forcing end-of-turn
# Min time a Flux turn may stay open with no transcriber events before it is force-closed.
DEEPGRAM_FLUX_TURN_STALL_FLOOR_S = 3.0
# Min idle time before the inactivity backstop hangs up; kept above hangup_after_silence.
STALL_HANGUP_FLOOR_S = 20.0

# LLM-driven language-switch defaults, all overridable by the matching LANGUAGE_SWITCH_* env.
# Read via os.getenv(..., CONSTANT) at call time, never frozen at import (load_dotenv runs later).
# Ceiling on the Switch-LLM decide. The detector buffer is drained BEFORE the decide, so a
# timeout loses that utterance — keep above the observed decide tail (~5.9s seen in QA).
LANGUAGE_SWITCH_DECIDE_TIMEOUT_S = 6.0
# Let the detector's socket deliver this turn's tail before draining its buffer.
LANGUAGE_SWITCH_SETTLE_MS = 300
# Silence between cutting audible old-language audio and the first new-language audio.
LANGUAGE_SWITCH_AUDIO_GAP_S = 0.2
# Ceiling on how long a mismatched turn's AUDIO waits for the switch decision. Generation is not
# delayed, so this is only paid when synthesis outruns the decide. Capped independently of the
# decide timeout (sized for the slow tail) because past this point a wrong-language reply the
# switch then truncates beats more dead air. Wall-clock backstop: the gate cannot wedge on it.
LANGUAGE_SWITCH_MAX_HOLD_S = 4.0
# Substance gate: a non-explicit switch needs at least one FOREIGN detector segment this long.
# Guards against one-word mis-tags switching the call. Sarvam splits real turns into sub-second
# fragments, so correct verdicts were dying just under the bar (1.0 → 0.8 → 0.7).
LANGUAGE_SWITCH_MIN_SEGMENT_AUDIO_S = 0.7
# Idle-flush suppression: while main-ASR interims say the caller is mid-utterance, defer the
# flush (the coming turn will drain the buffer). Past this buffer age the speaking flag is
# treated as stale — the detector hears the same audio and has produced nothing that long.
LANGUAGE_SWITCH_SPEAKING_STALE_CAP_S = 2.5

# Debounce for overlapped finals: one regenerate after this quiet window instead of per fragment.
LLM_REGEN_SETTLE_S = 0.7
# Class-name prefixes whose endpointing rules out an in-window final: they skip the debounce.
REGEN_SETTLE_EXCLUDED_TRANSCRIBERS = ("deepgram",)

# Past this much caller silence, callee_speaking is stale and held audio ships. Deepgram closes a
# healthy turn within utterance_end_ms (1s floor), so a real speaker stays well inside this.
STUCK_AUDIO_GATE_RELEASE_S = 3.0

# Above __await_stream_sid's own 10s timeout, so that path is what ends the call.
S2S_STREAM_SID_TIMEOUT_S = 12.0
# How long an armed goodbye gets before the call is closed without it.
S2S_GOODBYE_TIMEOUT_S = 10.0

# Soniox real-time STT
SONIOX_WEBSOCKET_HOST = "stt-rt.soniox.com"
SONIOX_ENDPOINT_TOKEN = "<end>"  # sentinel token emitted when the speaker stops
# Hinted for multilingual auto-detect: Soniox identifies and code-switches across these in one
# stream; hinting the relevant set sharpens accuracy over fully-open auto. No real-time as/od.
SONIOX_DEFAULT_MULTILINGUAL_HINTS = ["en", "hi", "ta", "te", "kn", "ml", "mr", "bn", "gu", "pa", "ur"]
SONIOX_AUTO_LANGUAGE_VALUES = {"", "multi", "auto", "multilingual", "unknown"}

# Model prefixes
GPT5_MODEL_PREFIX = "gpt-5"
GPT5_4_MODEL_PREFIX = "gpt-5.4"
GPT5_5_MODEL_PREFIX = "gpt-5.5"
GPT5_6_MODEL_PREFIX = "gpt-5.6"
# Function tools with reasoning_effort are rejected on chat completions for these models,
# so tool-using agents are routed through the Responses API.
RESPONSES_API_MODEL_PREFIXES = (GPT5_4_MODEL_PREFIX, GPT5_5_MODEL_PREFIX, GPT5_6_MODEL_PREFIX)

HIGH_LEVEL_ASSISTANT_ANALYTICS_DATA = {
    "extraction_details": {},
    "cost_details": {
        "average_transcriber_cost_per_conversation": 0,
        "average_llm_cost_per_conversation": 0,
        "average_synthesizer_cost_per_conversation": 1.0,
    },
    "historical_spread": {
        "number_of_conversations_in_past_5_days": [],
        "cost_past_5_days": [],
        "average_duration_past_5_days": [],
    },
    "conversation_details": {"total_conversations": 0, "finished_conversations": 0, "rejected_conversations": 0},
    "execution_details": {"total_conversations": 0, "total_cost": 0, "average_duration_of_conversation": 0},
    "last_updated_at": datetime.now(timezone.utc).isoformat(),
}

ACCIDENTAL_INTERRUPTION_PHRASES = [
    "stop",
    "quit",
    "bye",
    "wait",
    "no",
    "wrong",
    "incorrect",
    "hold",
    "pause",
    "break",
    "cease",
    "halt",
    "silence",
    "enough",
    "excuse",
    "hold on",
    "hang on",
    "cut it",
    "that's enough",
    "shush",
    "listen",
    "excuse me",
    "hold up",
    "not now",
    "stop there",
    "stop speaking",
]

PRE_FUNCTION_CALL_MESSAGE = {
    "en": "Just give me a moment, I'll be back with you.",
    "ge": "Geben Sie mir einen Moment Zeit, ich bin gleich wieder bei Ihnen.",
}

FILLER_PHRASES = [
    "No worries.",
    "It's fine.",
    "I'm here.",
    "No rush.",
    "Take your time.",
    "Great!",
    "Awesome!",
    "Fantastic!",
    "Wonderful!",
    "Perfect!",
    "Excellent!",
    "I get it.",
    "Noted.",
    "Alright.",
    "I understand.",
    "Understood.",
    "Got it.",
    "Sure.",
    "Okay.",
    "Right.",
    "Absolutely.",
    "Sure thing.",
    "I see.",
    "Gotcha.",
    "Makes sense.",
]

FILLER_DICT = {
    "Unsure": ["No worries.", "It's fine.", "I'm here.", "No rush.", "Take your time."],
    "Positive": ["Great!", "Awesome!", "Fantastic!", "Wonderful!", "Perfect!", "Excellent!"],
    "Negative": ["I get it.", "Noted.", "Alright.", "I understand.", "Understood.", "Got it."],
    "Neutral": ["Sure.", "Okay.", "Right.", "Absolutely.", "Sure thing."],
    "Explaining": ["I see.", "Gotcha.", "Makes sense."],
    "Greeting": ["Hello!", "Hi there!", "Hi!", "Hey!"],
    "Farewell": ["Goodbye!", "Thank you!", "Take care!", "Bye!"],
    "Thanking": ["Welcome!", "No worries!"],
    "Apology": ["I'm sorry.", "My apologies.", "I apologize.", "Sorry."],
    "Clarification": ["Please clarify.", "Can you explain?", "More details?", "Can you elaborate?"],
    "Confirmation": ["Got it.", "Okay.", "Understood."],
}

CHECKING_THE_DOCUMENTS_FILLER = "Umm, just a moment, getting details..."
TRANSFERING_CALL_FILLER = {
    "en": "Sure, I'll transfer the call for you. Please wait a moment...",
    "fr": "D'accord, je transfère l'appel. Un instant, s'il vous plaît.",
}

DEFAULT_USER_ONLINE_MESSAGE = "Hey, are you still there?"
DEFAULT_USER_ONLINE_MESSAGE_TRIGGER_DURATION = 6
DEFAULT_LANGUAGE_CODE = "en"
DEFAULT_TIMEZONE = "America/Los_Angeles"

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "de": "German",
    "it": "Italian",
    "nl": "Dutch",
    "id": "Indonesian",
    "ms": "Malay",
    "th": "Thai",
    "vi": "Vietnamese",
    "od": "Odia",
}

LLM_DEFAULT_CONFIGS = {
    "summarization": {"model": "gpt-4.1-mini", "provider": "openai"},
    "extraction": {"model": "gpt-4.1-mini", "provider": "openai"},
    "google": {"model": "gemini-2.5-flash", "provider": "google"},
}

# Legacy language-switch tool, injected into the main LLM on multilingual agents
# when the LLM-driven switch flow is NOT enabled (tools_config["llm_language_switch"]
# false/absent) — restored from master for the feature-flag fallback path.
SWITCH_LANGUAGE_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "switch_language",
        "description": "Switch the conversation language for speech recognition and synthesis. Call this when the user speaks in or requests a different language.",
        "parameters": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "description": "The language label to switch to (e.g. 'hi' for Hindi, 'en' for English)",
                }
            },
            "required": ["language"],
        },
    },
}

# Control marks carry no playback evidence and must not be used as a trim target.
NON_EVIDENCE_MARK_TYPES = ("pre_mark_message", "backchanneling")

# message_category of the "are you still there" prompt. The playout estimate and
# final_chunk_played_observable must exclude the same value or the two silence clocks disagree.
IS_USER_ONLINE_MESSAGE = "is_user_online_message"

# Formats whose byte length maps directly to playback time. Compressed audio does not.
UNCOMPRESSED_AUDIO_FORMATS = ("pcm", "wav", "mulaw", "ulaw")

# End-of-stream control signal; telephony pads the single byte to two before sending.
AUDIO_STREAM_END_SENTINELS = (b"\x00", b"\x00\x00")

END_CALL_FUNCTION_PREFIX = "end_call"

END_CALL_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "end_call",
        "description": "End the current call. Use this when the conversation is naturally complete, the user has explicitly said goodbye, or you've fulfilled the purpose of the call. Always say your goodbye message before calling this function.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief reason for ending the call (e.g. 'conversation_complete', 'user_goodbye', 'task_fulfilled')",
                }
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

SARVAM_MODEL_SAMPLING_RATE_MAPPING = {
    "bulbul:v2": 22050,
    "bulbul:v3": 22050,  # NOTE: Documentation claims 24000, but WAV header shows 22050
}

# bulbul TTS requires a concrete target_language_code (no "unknown"/auto).
SARVAM_TTS_SUPPORTED_LANGUAGES = {
    "en-IN",
    "hi-IN",
    "bn-IN",
    "ta-IN",
    "te-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "gu-IN",
    "pa-IN",
    "od-IN",
}

# Maya matches the voice case-sensitively; "ananya" is a 400.
MAYA_TTS_SUPPORTED_VOICES = {"Ananya", "Arjun"}

# "en" is Indian English; "auto" lets Maya detect per utterance.
MAYA_TTS_SUPPORTED_LANGUAGES = {
    "hi",
    "bn",
    "gu",
    "kn",
    "ml",
    "mr",
    "or",
    "pa",
    "ta",
    "te",
    "en",
    "auto",
}

MODEL_REASONING_EFFORT_MAP = {
    "gpt-5": [RE.MINIMAL, RE.LOW, RE.MEDIUM, RE.HIGH],
    "gpt-5-mini": [RE.MINIMAL, RE.LOW, RE.MEDIUM, RE.HIGH],
    "gpt-5-nano": [RE.MINIMAL, RE.LOW, RE.MEDIUM, RE.HIGH],
    "gpt-5-codex": [RE.LOW, RE.MEDIUM, RE.HIGH],
    "gpt-5-pro": [RE.HIGH],
    "gpt-5.1": [RE.NONE, RE.LOW, RE.MEDIUM, RE.HIGH],
    "gpt-5.1-codex": [RE.LOW, RE.MEDIUM, RE.HIGH],
    "gpt-5.1-codex-max": [RE.LOW, RE.MEDIUM, RE.HIGH, RE.XHIGH],
    "gpt-5.1-codex-mini": [RE.LOW, RE.MEDIUM, RE.HIGH],
    "gpt-5.2": [RE.NONE, RE.LOW, RE.MEDIUM, RE.HIGH, RE.XHIGH],
    "gpt-5.4": [RE.NONE, RE.LOW, RE.MEDIUM, RE.HIGH, RE.XHIGH],
    "gpt-5.4-mini": [RE.NONE, RE.LOW, RE.MEDIUM, RE.HIGH],
    "gpt-5.4-nano": [RE.NONE, RE.LOW, RE.MEDIUM, RE.HIGH],
    "gpt-5.5": [RE.NONE, RE.LOW, RE.MEDIUM, RE.HIGH, RE.XHIGH],
    "gpt-5.5-pro": [RE.MEDIUM, RE.HIGH, RE.XHIGH],
    "gpt-5.6-sol": [RE.NONE, RE.LOW, RE.MEDIUM, RE.HIGH, RE.XHIGH],
    "gpt-5.6-terra": [RE.NONE, RE.LOW, RE.MEDIUM, RE.HIGH, RE.XHIGH],
    "gpt-5.6-luna": [RE.NONE, RE.LOW, RE.MEDIUM, RE.HIGH, RE.XHIGH],
    # Realtime speech-to-speech. gpt-realtime-1.5 has no reasoning and is deliberately absent.
    "gpt-realtime-2": [RE.MINIMAL, RE.LOW, RE.MEDIUM, RE.HIGH, RE.XHIGH],
    "gpt-realtime-2.1": [RE.MINIMAL, RE.LOW, RE.MEDIUM, RE.HIGH, RE.XHIGH],
    "gpt-realtime-2.1-mini": [RE.MINIMAL, RE.LOW, RE.MEDIUM, RE.HIGH, RE.XHIGH],
}


def default_reasoning_effort(model: str) -> str:
    """Lowest-latency effort the model supports: minimal where available, else the lowest in its map."""
    supported = MODEL_REASONING_EFFORT_MAP.get(model)
    if not supported or RE.MINIMAL in supported:
        return RE.MINIMAL.value
    return supported[0].value


GEMINI_THINKING_LEVEL_MAP = {
    "gemini-3-flash-preview": [RE.MINIMAL, RE.LOW, RE.MEDIUM, RE.HIGH],
    "gemini-3.1-flash-lite": [RE.MINIMAL, RE.LOW, RE.MEDIUM, RE.HIGH],
    "gemini-3.1-flash-lite-preview": [RE.MINIMAL, RE.LOW, RE.MEDIUM, RE.HIGH],
    "gemini-3.1-pro-preview": [RE.LOW, RE.MEDIUM, RE.HIGH],
    "gemini-3.5-flash": [RE.MINIMAL, RE.LOW, RE.MEDIUM, RE.HIGH],
    "gemini-3.5-flash-lite": [RE.MINIMAL, RE.LOW, RE.MEDIUM, RE.HIGH],
    "gemini-3.6-flash": [RE.MINIMAL, RE.LOW, RE.MEDIUM, RE.HIGH],
    "gemini-3.7-flash": [RE.LOW, RE.MEDIUM, RE.HIGH],
}


def default_thinking_level(model: str) -> str:
    """Lowest-latency thinking level the Gemini 3.x model supports.

    Unknown models fall back to "low", the only level the whole 3.x family accepts.
    """
    supported = GEMINI_THINKING_LEVEL_MAP.get(model.rsplit("/", 1)[-1])
    if not supported:
        return RE.LOW.value
    return supported[0].value


def canonical_model(name: str) -> str:
    """The known model a deployment serves, e.g. 'ptu-gpt-5.4-mini' -> 'gpt-5.4-mini'.

    Azure deployment names are chosen freely, so model-family checks cannot read them directly.
    Longest match wins so 'gpt-5.4-mini' beats 'gpt-5.4'. Unrecognised names pass through.
    """
    bare = (name or "").rsplit("/", 1)[-1]
    known = [m for m in MODEL_REASONING_EFFORT_MAP if m in bare]
    return max(known, key=len) if known else bare
