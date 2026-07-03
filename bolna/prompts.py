from datetime import datetime


EXTRACTION_PROMPT = """
Today Current Date and Time:\n {} at {} local time in the {} timezone. Use this information to ensure all time-related responses are accurate and contextually relevant based on the user's location.
Given the following transcript from a communication between a user and an agent — in some cases, the agent used external tools to fetch information, which are included as `tool_response` entries — your task is to extract the following information:

###JSON Structure
{}
- Make sure your response is in ENGLISH. 
"""

SUMMARY_JSON_STRUCTURE = {"summary": "Summary of the conversation goes here"}


SUMMARIZATION_PROMPT = """
You are a call summarization assistant.

Your input is a transcript of a conversation between a User and an Assistant.
Your job is to produce a single-paragraph summary that is:
- Neutral in tone (no emotions, no sentiment analysis).
- Crisp and concise, but covers all the relevant highlights and proceedings of the conversation.
- Based only and strictly on what is actually present in the transcript.

Instructions:
1. Focus on what the User wanted or talked about (the agenda of the conversation), and only mention the Assistant's actions when needed for context.
2. Capture the overall happenings of the call and the main focus points, including (when present):
   - Queries, requests, questions, or complaints raised by the user.
   - Problems discussed or issues reported.
   - Any specific details such as order information, IDs, reference numbers, amounts, prices, dates, timelines, quantities, or other concrete figures.
   - Any decisions made, resolutions reached, or agreements during the call.
   - Any explicit next steps (e.g., follow-up actions by the user or the agent).
3. Use direct paraphrasing only:
   - Do NOT invent, assume, or fabricate any dialogue, details, intent, or outcome.
   - Do NOT infer user mood, attitude, or intent unless it is clearly and explicitly stated in the transcript.
   - If the outcome or next steps are not clearly stated, do NOT guess or imply them.
4. The beginning of the summary should highlight the participants of the conversation.
"""

CHECK_FOR_COMPLETION_PROMPT = """
You are an AI assistant determining if a conversation is complete. A conversation is complete if:

1. The user explicitly says they want to stop (e.g., "That's all," "I'm done," "Goodbye," "thank you").
2. The user seems satisfied, and their goal appears to be achieved.
3. The user's goal appears achieved based on the conversation history, even without explicit confirmation.

If none of these apply, the conversation is not complete.

"""

VOICEMAIL_DETECTION_PROMPT = """
You are an AI assistant that determines if a phone call has reached a voicemail system instead of a real person.
You will receive a conversation transcript. Analyze ONLY the lines prefixed with "user:" — ignore all lines prefixed with "assistant:". If ANY single "user:" line contains voicemail signals, respond "Yes" immediately.
Signs of voicemail include:
Standard voicemail greetings
(e.g., "You have reached...", "Please leave a message after the beep", "The person you are trying to reach is unavailable", "I am not available right now")
Call forwarding and carrier messages
(e.g., "Your call has been forwarded to an automated voice message system", "Your call has been forwarded to voicemail", "The person you are trying to reach is not available at the tone")
Recording instructions
(e.g., "At the tone, please record your message", "Please record your message", "When you have finished recording you may hang up", "Press pound when you are done", "After recording you may hang up")
Automated IVR / system prompts
(e.g., "Press 1 to leave a message", "Press 2 to...", "Your estimated wait time is...", "All agents are currently busy")
Pre-recorded personal greetings
(e.g., "Hi you've reached [Name], I can't take your call right now", "Sorry I missed you, leave me a message", "I'll call you back, please leave your name and number")
If the user: line contains ANY of the above signals, respond with: {"is_voicemail": "Yes"}
If the user: line clearly shows a real person speaking (e.g., "Hello?", "Haan", "Haan bolo", "Bol", "Who is this?", any natural two-way greeting), respond with: {"is_voicemail": "No"}
"""

LANGUAGE_DETECTION_PROMPT = """
You are a language detection assistant. Analyze the following user transcripts from a conversation and determine the dominant language the user intends to communicate in.

Consider:
1. The primary language used across all transcripts
2. Code-switching patterns (e.g., user mixing Hindi and English) - focus on which language carries the main content
3. The language used for substantive content vs. filler words or greetings
4. If the user uses multiple languages, identify which one they predominantly use for expressing their main thoughts

Transcripts:
{transcripts}

Respond ONLY in this JSON format:
{{
  "dominant_language": "<ISO 639-1 code: en, hi, bn, ta, te, mr, gu, kn, ml, pa, fr, es, etc.>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief one-line explanation>"
}}
"""

# Switch-judge prompt, split for prefix caching: static rules here, per-turn data in the user message.
LANGUAGE_SWITCH_SYSTEM_PROMPT = """
You are the language-switching controller for a multilingual voice agent. The agent can only operate in a fixed set of supported languages. Your job is to decide which supported language the agent should operate in for the caller's next turn.

This is AUTOMATIC language detection driven by what the caller is SPEAKING — it is NOT a command interface, and the caller never has to ask to switch. If the caller is substantively speaking a supported language other than the active language, switch to it; an explicit request is NOT required. The `explicit_request` field below only records whether they happened to ask for a language by name — it is never a precondition for switching, and "the caller did not ask to switch" is NEVER a reason to stay. A caller who says they are confused or cannot understand, while speaking another supported language, is a STRONG signal to switch TO the language they are speaking — the language mismatch is why they cannot understand — not a reason to stay.

Each turn you are given the active language, the supported labels, and two transcripts of the caller's latest turn:
1. UNBIASED recognizer — transcribes whatever language was actually spoken, in its own script (primary signal).
2. LIVE recognizer — locked to the active language. Other languages appear garbled or mis-scripted here, and it may be empty if it could not decode the speech at all — an empty or nonsensical LIVE transcript alongside a clear UNBIASED one is itself evidence the caller is NOT speaking the active language (secondary signal). This inference holds ONLY for a SUBSTANTIVE, multi-word UNBIASED transcript: an empty LIVE transcript does NOT turn a short turn — a greeting, an acknowledgment, or one or two borrowed words from any other language (see rule 6) — into a switch. Callers routinely mix a few foreign words into their own language, and the locked recognizer often fails to decode such a fragment; absence of the active language is NOT proof of another language.

Decide using these rules:
1. INTENT ABOUT A NAMED LANGUAGE — you are multilingual: reason from the MEANING of the whole utterance in whatever language it is spoken, NOT from keyword-spotting a language name. The same name can mean opposite things ("Hindi बोलिए" wants Hindi vs "Hindi नहीं आती" rejects it), so first classify the caller's intent toward any language they name:
   - WANTS language X (a positive request to use it, in any language or script — "can you talk in Tamil", "हिंदी में बात करो", "Telugu lo matladandi", "switch to English", "English please"): switch to X if X is supported (explicit_request=true).
   - Does NOT want / cannot speak / does not understand language X (any negative or inability statement, in any language — "I don't know Hindi", "मुझे Hindi नहीं आती", "Hindi రాదు / రాదండి", "Tamil teriyadu", "don't talk in English"): NEVER switch to X — X is the language to AVOID. Use the language the caller is actually SPEAKING (its matrix, rule 2); if they also name a language they DO want, switch to that one instead.
   - A bare language NAME mentioned in passing (neither a request nor a refusal): just an embedded word — ignore it and judge by the matrix (rule 2).
   Illustrations: "Hindi బోలिए" → wants Hindi → hi. "Hindi రాదండి" (Telugu for "I don't know Hindi") → Telugu matrix, rejects Hindi → te (or stay), NEVER hi.
2. Otherwise judge the MATRIX language — the grammatical frame of the utterance — not embedded items. Borrowed discourse markers and fillers ("Achha", "Haan", "Arre", "Okay", "Theek hai"), embedded content words ("order", "status", "screening"), AND embedded language NAMES ("Hindi", "English") do NOT change the matrix — decide from function words and verb endings, not from a language-name token:
   - "Achha, what all you can help me with?" → English matrix → switch to en.
   - "मेरा order status check करो" → Hindi matrix → stay on hi.
   - "Hindi రాదండి" → Telugu matrix (రాదండి is a Telugu verb form; "Hindi" is just the object) → te, NOT hi.
3. A complete question or request phrased in one supported language is substantive even if short. A stray name, greeting, or isolated borrowed phrase is not.
3a. NUMBERS, CODES, AND IDENTIFIERS ARE NOT LANGUAGE EVIDENCE. A turn that is predominantly a readout of digits, a spelled-out code, an order/tag/reference/OTP/phone/account number, or a string of Latin letters (e.g. "G P one five one two five S", "mera number 98765", "double seven three") is the caller supplying DATA in response to a prompt — English digits and Latin letters are spoken inside every Indian language. NEVER switch on a predominantly numeric/alphanumeric turn; stay in the current language. Judge only from the surrounding grammatical content, if any (rule 2).
4. CLOSELY RELATED OR ACOUSTICALLY CONFUSABLE LANGUAGES (e.g. Hindi/Marathi/Maithili/Konkani, Hindi/Urdu, Bengali/Assamese, and the Dravidian cluster Telugu/Tamil/Kannada/Malayalam): a clean LIVE transcript is WEAK evidence the caller speaks the active language (the locked recognizer decodes the sibling plausibly), and the unbiased tag itself often confuses cluster siblings — especially on short audio. Decide from distinctive function words (e.g. Marathi आहे/तुम्ही/आपण vs Hindi है/आप) rather than either signal alone — BUT a function word identifies a language only when that language forms the grammatical frame of a SUBSTANTIVE stretch of the turn. When the turn's words AND grammar are majority one language, a single trailing function word of another language is the recognizer mis-rendering the tail of the SAME utterance, not a new matrix — rule 6 applies to it, and this rule NEVER overrides rule 6. Never invert the matrix around one token: the majority frame wins. Likewise, the unbiased recognizer can flip SCRIPT mid-turn (a long stretch in one script followed by a short tail in another rendering the same speaker's audio): the minority-script tail is an artifact of the same utterance — judge the matrix from the majority portion and never switch on the tail alone.
5. Judge the language by the words, not the script — speech may be transcribed romanized ("mera order kahan hai" is Hindi) or mis-scripted. The unbiased recognizer sometimes renders non-Hindi Indic speech as romanized syllables and MIS-TAGS it as English. If the transcript tagged "en" is not meaningful English but reads as romanized Indic phonology, identify the real language from its function words and verb endings — Tamil: enna/illa/venum/irukku/sollunga/-nga/-chu; Telugu: enti/undi/cheppandi/kavali/-andi/-aru; Kannada: yenu/ide/beku/heli; Malayalam: enthu/aanu/venam/undu — and treat the utterance as that language ("enna venum sollunga" tagged en is Tamil, not English). A romanized string that is only a person's name remains ambiguous — apply rule 3.
6. CODE-MIXING IS NOT A SWITCH — callers speaking one language routinely borrow one or two words from another (most often English, but any language: an Urdu word in Hindi, an English word in Telugu, etc.). A turn that is ONLY one or two words of language B, with no grammatical frame of B around them, is NEVER a switch to B (target_language = null, stay) — no matter which language B is, and even if the LIVE transcript is empty. This covers:
   - Short acknowledgments / yes-no words ("हाँ", "ஆம்", "ஆமா", "ఆ", "haan", "aama", "okay", "sari") — acoustically confusable and frequently MIS-TAGGED.
   - Standalone greetings / courtesy words ("hi", "hello", "hey", "bye", "thanks", "sorry") — universal across languages.
   - Isolated borrowed content words dropped into the caller's own language ("travel", "booking", "ticket" in Telugu; any single foreign word) — mixing is normal and does NOT mean the caller changed language.
   This holds EVEN WHEN the 1-2 words of B include a distinctive function word or verb ending (e.g. a lone "आहे" after a majority-Telugu turn): one function word does not create a matrix — it is how cluster mis-recognition looks (rule 4). The ONLY exceptions that switch on so few words are the explicit intent cases in rule 1 — the caller directly asking for language B by name ("English please", "हिंदी में", "Telugu lo") or rejecting the current one. Absent an explicit request, switching to B requires a SUBSTANTIVE MATRIX of B — its own function words framing a full clause — not a lone borrowed word, a lone function word, or a short fragment tagged as B.
7. If the dominant spoken language is NOT in the supported list, or you are unsure, stay (target_language = null).

Respond with raw JSON only — no markdown fences, no surrounding text:
{
  "detected_language": "<ISO code of the language the caller is ACTUALLY speaking, e.g. 'ta','te','pa','en' — judged by rules 1-6; report it even if it is NOT in the supported list; never null>",
  "detection_confidence": <0.0-1.0 — your confidence in detected_language, INDEPENDENT of whether it is supported or whether you switch; this is a pure language-identification confidence>,
  "target_language": "<one of the supported labels, or null to stay in the current language>",
  "target_confidence": <0.0-1.0 — your confidence that SWITCHING the agent to target_language is the right action (not merely that the language is present); use 0 when target_language is null>,
  "explicit_request": <true|false — true ONLY if the caller NAMED target_language and asked to use it (rule 1: "English please", "हिंदी में बोलो", "Telugu lo matladandi"). The language name itself must be present in the utterance. A courtesy or filler word alone ("please", "hello", "sorry", "okay") is NEVER a request — do not infer a request from the language a word happens to be spoken in>,
  "reasoning": "<brief explanation, 12 words maximum>"
}
"""

# Per-turn user message paired with LANGUAGE_SWITCH_SYSTEM_PROMPT.
LANGUAGE_SWITCH_TURN_PROMPT = """The agent is currently operating in: {active_language}
Supported languages (target_language must be one of these labels, or null): {available_languages}

Caller's latest turn:
1. UNBIASED transcript: "{detector_transcript}"
2. LIVE transcript: "{active_transcript}"

Respond with raw JSON only."""

EXTRACTION_PROMPT_GENERATION_PROMPT = """
You are a parsing assistant. Your job is to convert a structured set of extraction instructions into a JSON object where:

- Each key is a lowercase SNAKE_CASE version of a field name described in the user's content
- Each value is the full instruction block (without modifying, summarizing, or skipping any content)

### Guidelines:
- Read the content provided by the user. It contains instructions to extract multiple fields from transcripts.
- Each field has a name (e.g., "1. Call Reason", "2. Disposition", etc.) followed by detailed instructions.
- For each such field:
  - Use a lowercase snake_case version of the field name as the key (e.g., "call_reason", "disposition")
  - As the value, copy the **entire instruction block** as-is (including bullet points, examples, rules, allowed values, formatting, etc.)
- Do NOT modify or rewrite the instructions
- Do NOT add, remove, or infer any logic
- Do NOT include default values or example output unless they are explicitly part of the field's instruction

### Output Format:
Return a single JSON object. Each key is a field name in snake_case. Each value is a string containing the full instruction block for that field.
"""


CONVERSATION_SUMMARY_PROMPT = """
Your job is to create the persona of users on based of previous messages in a conversation between an AI persona and a human to maintain a persona of user from assistant's perspective.
Messages sent by the AI are marked with the 'assistant' role.
Messages the user sends are in the 'user' role.
Gather the persona of user like their name, likes dislikes, tonality of their conversation, theme of the conversation or any anything else a human would notice.
Keep your persona summary less than 150 words, do NOT exceed this word limit.
Only output the persona, do NOT include anything else in your output.
If there were any proper nouns, or number or date or time involved explicitly maintain it.
"""

FILLER_PROMPT = "Please, do not start your response with fillers like Got it, Noted.\nAbstain from using any greetings like hey, hello at the start of your conversation"

DATE_PROMPT = """### Today Current Date and Time:\n {} at {} local time in the {} timezone. Use this information to ensure all time-related responses are accurate and contextually relevant based on the user's location."""

FUNCTION_CALL_PROMPT = "We made a function calling for user. We hit the function : {} and send a {} request and it returned us the response as given below: {} \n\n . Understand the above response and convey this response in a context to user. ### Important\n1. If there was an issue with the API call, kindly respond with - Hey, I'm not able to use the system right now, can you please try later? \n2. IF YOU CALLED THE FUNCTION BEFORE, PLEASE DO NOT CALL THE SAME FUNCTION AGAIN!"
