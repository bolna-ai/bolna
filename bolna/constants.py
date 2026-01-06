from datetime import datetime, timezone
PREPROCESS_DIR = 'agent_data'

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
