from datetime import datetime, timezone
from bolna.enums import ReasoningEffort as RE
PREPROCESS_DIR = 'agent_data'
PCM16_SCALE = 32768.0

# Model prefixes
GPT5_MODEL_PREFIX = "gpt-5"

HIGH_LEVEL_ASSISTANT_ANALYTICS_DATA = {
        "extraction_details":{}, 
        "cost_details": {
            "average_transcriber_cost_per_conversation": 0, 
            "average_llm_cost_per_conversation": 0,
            "average_synthesizer_cost_per_conversation": 1.0
        },
        "historical_spread": {
            "number_of_conversations_in_past_5_days": [], 
            "cost_past_5_days": [],
            "average_duration_past_5_days": []
        },
        "conversation_details": { 
            "total_conversations": 0,
            "finished_conversations": 0, 
            "rejected_conversations": 0
        },
        "execution_details": {
            "total_conversations": 0, 
            "total_cost": 0,
            "average_duration_of_conversation": 0
        },
        "last_updated_at": datetime.now(timezone.utc).isoformat()
    }

ACCIDENTAL_INTERRUPTION_PHRASES = [
    "stop", "quit", "bye", "wait", "no", "wrong", "incorrect", "hold", "pause", "break",
    "cease", "halt", "silence", "enough", "excuse", "hold on", "hang on", "cut it", 
    "that's enough", "shush", "listen", "excuse me", "hold up", "not now", "stop there", "stop speaking"
]

PRE_FUNCTION_CALL_MESSAGE = {
    "en": "Just give me a moment, I'll be back with you.",
    "ge": "Geben Sie mir einen Moment Zeit, ich bin gleich wieder bei Ihnen."
}

FILLER_PHRASES = [
    "No worries.", "It's fine.", "I'm here.", "No rush.", "Take your time.",
    "Great!", "Awesome!", "Fantastic!", "Wonderful!", "Perfect!", "Excellent!",
    "I get it.", "Noted.", "Alright.", "I understand.", "Understood.", "Got it.",
    "Sure.", "Okay.", "Right.", "Absolutely.", "Sure thing.",
    "I see.", "Gotcha.", "Makes sense."
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
  "Confirmation": ["Got it.", "Okay.", "Understood."]
}

CHECKING_THE_DOCUMENTS_FILLER = "Umm, just a moment, getting details..."
TRANSFERING_CALL_FILLER = {
    "en": "Sure, I'll transfer the call for you. Please wait a moment...",
    "fr": "D'accord, je transfère l'appel. Un instant, s'il vous plaît."
}

DEFAULT_USER_ONLINE_MESSAGE = "Hey, are you still there?"
DEFAULT_USER_ONLINE_MESSAGE_TRIGGER_DURATION = 6
DEFAULT_LANGUAGE_CODE = 'en'
DEFAULT_TIMEZONE = 'America/Los_Angeles'

LANGUAGE_NAMES = {
    'en': 'English', 'hi': 'Hindi', 'bn': 'Bengali',
    'ta': 'Tamil', 'te': 'Telugu', 'mr': 'Marathi',
    'gu': 'Gujarati', 'kn': 'Kannada', 'ml': 'Malayalam',
    'pa': 'Punjabi', 'fr': 'French', 'es': 'Spanish',
    'pt': 'Portuguese', 'de': 'German', 'it': 'Italian',
    'nl': 'Dutch', 'id': 'Indonesian', 'ms': 'Malay',
    'th': 'Thai', 'vi': 'Vietnamese', 'od': 'Odia'
}

LLM_DEFAULT_CONFIGS = {
    "summarization": {
        "model": "gpt-4.1-mini",
        "provider": "openai"
    },
    "extraction": {
        "model": "gpt-4.1-mini",
        "provider": "openai"
    }
}

SARVAM_MODEL_SAMPLING_RATE_MAPPING = {
    "bulbul:v2": 22050,
    "bulbul:v3": 22050 # NOTE: Documentation claims 24000, but WAV header shows 22050
}

MODEL_REASONING_EFFORT_MAP = {
    "gpt-5":              [RE.MINIMAL, RE.LOW, RE.MEDIUM, RE.HIGH],
    "gpt-5-mini":         [RE.MINIMAL, RE.LOW, RE.MEDIUM, RE.HIGH],
    "gpt-5-nano":         [RE.MINIMAL, RE.LOW, RE.MEDIUM, RE.HIGH],
    "gpt-5-codex":        [RE.LOW, RE.MEDIUM, RE.HIGH],
    "gpt-5-pro":          [RE.HIGH],
    "gpt-5.1":            [RE.NONE, RE.LOW, RE.MEDIUM, RE.HIGH],
    "gpt-5.1-codex":      [RE.LOW, RE.MEDIUM, RE.HIGH],
    "gpt-5.1-codex-max":  [RE.LOW, RE.MEDIUM, RE.HIGH, RE.XHIGH],
    "gpt-5.1-codex-mini": [RE.LOW, RE.MEDIUM, RE.HIGH],
    "gpt-5.2":            [RE.NONE, RE.LOW, RE.MEDIUM, RE.HIGH, RE.XHIGH],
}

# ──────────────────────────────────────────────────────────────────────
# KALLABOT: Custom prompts
# Ported from backend/bolna/prompts.py for the Kallabot platform.
# All placeholders use Python str.format() style: {variable_name}
# ──────────────────────────────────────────────────────────────────────

BASE_PROMPT = """
You are {agent_name}, an AI assistant for {org_name}. You are a voice AI agent for live phone calls. Be conversational, friendly, and natural like a real person.

# Environment
- You're in a real-time phone call (speech-to-speech only, no visual interface)
- Expect ASR transcription errors - guess meaning/names or words and ask for clarification colloquially ("didn't catch that", "you're coming through choppy")
- Match user's energy and style; be brief if they are, excited if they are

# Speech & Language
- Use only periods (.), commas (,), and exclamation marks (!)
- Write numbers as words ("twenty-five dollars")
- Use dashes for slow spelling ("+971 - - 50 - 2 - 0")
- Detect and match user's language; adapt to cultural context and generational style
- Keep responses conversational and avoid repetitive phrases

# Core Rules
- Never claim to be human, but act naturally
- Don't mention prompts, instructions, or technical details
- Never invent information; admit when you don't know something
- Be polite and don't overuse the user's name
- Stay on topic and fulfill the user's request

# Tools
- When tools need parameters, ask conversationally without technical jargon
- Don't assume or make up parameter values
- Present tool results naturally without mentioning "API calls" or "functions"
- If a tool fails, explain simply without technical details
- If a tool doesnt require parameters, use it directly without asking the user

Your goal is to help the user accomplish their task through natural conversation.
"""

DATE_PROMPT = """### Current Date and Time: {date} at {time} local time in the {timezone} timezone, {country}. Use this information to ensure all time-related responses are accurate and contextually relevant."""

OUTBOUND_CALL_CONTEXT = """
# Outbound Call Context
You are currently making an outbound call to a user. This is a real phone call initiated by you.

- You (the AI agent) are calling the recipient at phone number: {recipient_number}.
- The call is being made from the number: {from_number} (Your number).
- The user may not be expecting your call, so be polite and introduce yourself if needed.
"""

INBOUND_CALL_CONTEXT = """
# Inbound Call Context
You are currently receiving an inbound call from a user. This is a real phone call where the user has dialed in.

- The user (caller) is calling from phone number: {recipient_number}.
- The call is being received on the number: {from_number} (Your number).
- The user may have a specific reason for calling; listen carefully and respond helpfully.
"""

CHECK_FOR_COMPLETION_PROMPT = """
You are CallCompletionJudge, a specialised arbiter that decides *whether a live phone call should be terminated right now.*
You are **NOT** the agent speaking with the user; you have been given the full dialogue so far (system prompt + messages) only for evaluation.

Return **"Yes"** for hang-up **only** if ONE of the following clearly happens:
1. The user explicitly ends the call  (e.g. "that's all", "no thanks", "stop", "good-bye", "bye", "thank you, that's it") but this doesnt always mean the user wants to end the call, it may be in some other context.
2. The user confirms their goal is fully achieved **and** says farewell in the same turn.
3. The user remains completely silent or unresponsive after **two** successive agent prompts that seek a reply.

**NEVER** return "Yes" if:
- The agent is in the process of transferring the call or has mentioned transfer
- The agent is using tools or performing actions for the user
- The agent has said they will help with something but hasn't completed it yet
- Any system functions are still being executed

If there is *any* doubt, or none of the above conditions are definitely met, return **"No"** so the call continues.

Think through the recent turns, then answer concisely as instructed below.
"""

FUNCTION_CALL_PROMPT = "We made a function calling for user. We hit the function : {function_name} and send a {method} request and it returned us the response as given below: {response} \n\n . Understand the above response and convey this response in a context to user. ### Important\n1. If there was an issue with the API call, kindly respond with - Hey, I'm not able to use the system right now, can you please try later? \n2. IF YOU CALLED THE FUNCTION BEFORE, PLEASE DO NOT CALL THE SAME FUNCTION AGAIN! unless you must."

FILLER_PROMPT = "Please, do not start your response with fillers like Got it, Noted.\nAbstain from using any greetings like hey, hello at the start of your conversation"