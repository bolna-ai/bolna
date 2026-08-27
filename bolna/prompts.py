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
You will receive a conversation transcript. Analyse ONLY the lines prefixed with "user:" - ignore all lines prefixed with "assistant:". If ANY single "user:" line contains voicemail signals, respond "Yes" immediately.
Partial, cut-off, or mid-sentence fragments still count: if a "user:" line contains any recognisable part of a voicemail phrase below: its beginning, middle, or end then, respond "Yes", as long as the fragment clearly belongs to a voicemail message and not a normal human greeting.

Also match approximately, not just exactly: if a "user:" line is close to any voicemail phrase below in wording or meaning, even with transcription errors, missing words, or slightly different phrasing, treat it as a voicemail signal and respond "Yes". It does not need to match word-for-word, as long as it clearly resembles a voicemail message and not a normal human greeting.

Signs of voicemail include:
Standard voicemail greetings
(e.g., "You have reached...", "Please leave a message after the beep", "The person you are trying to reach..." (with or without "is unavailable", "is not available", or "at the tone"), "I am not available right now")

Call forwarding and carrier messages
(e.g., "Your call has been forwarded to an automated voice message system", "Your call has been forwarded to voicemail", "The person you are trying to reach is not available at the tone")
Recording instructions
(e.g., "At the tone, please record your message", "Please record your message", "When you have finished recording you may hang up", "Press pound when you are done", "After recording you may hang up")

Automated IVR / system prompts (e.g., "Press 1 to leave a message", "Press 2 to...", "Your estimated wait time is...", "All agents are currently busy")
Pre-recorded personal greetings
(e.g., "Hi you've reached [Name], I can't take your call right now", "Sorry I missed you, leave me a message", "I'll call you back, please leave your name and number")
If the user: line contains ANY of the above signals, respond with: {"is_voicemail": "Yes"}
If the user: line clearly shows a real person speaking (e.g., "Hello?", "Haan", "Haan bolo", "Bol", "Who is this?", any natural two-way greeting), respond with: {"is_voicemail": "No"}


Respond only in this JSON format:
{
"is_voicemail": "Yes" or "No"
}
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
2. LIVE recognizer — locked to the active language. Other languages appear garbled or mis-scripted here, and it may be empty if it could not decode the speech at all — an empty or nonsensical LIVE transcript alongside a clear UNBIASED one is itself evidence the caller is NOT speaking the active language (secondary signal). This inference holds ONLY for a SUBSTANTIVE, multi-word UNBIASED transcript: an empty LIVE transcript does NOT turn a short turn — a greeting, an acknowledgment, or one or two borrowed words from any other language (see rule 6) — into a switch. Callers routinely mix a few foreign words into their own language, and the locked recognizer often fails to decode such a fragment; absence of the active language is NOT proof of another language. When the LIVE line reads "(no turn from the language-locked recognizer — idle flush)", no LIVE transcript was ever produced for this audio — that absence carries NO evidential weight in either direction; never cite it as confirmation of a mismatch. Judge from the UNBIASED transcript alone, with extra suspicion of transliteration (rule 5).

Decide using these rules:
1. INTENT ABOUT A NAMED LANGUAGE — you are multilingual: reason from the MEANING of the whole utterance in whatever language it is spoken, NOT from keyword-spotting a language name. The same name can mean opposite things ("Hindi बोलिए" wants Hindi vs "Hindi नहीं आती" rejects it), so first classify the caller's intent toward any language they name:
   - WANTS language X (a positive request to use it, in any language or script — "can you talk in Tamil", "हिंदी में बात करो", "Telugu lo matladandi", "switch to English", "English please"): switch to X if X is supported (explicit_request=true). A bare language NAME REPEATED — the caller saying the name of language X two or more times with nothing else of substance ("Hindi Hindi", "English English", "Marathi Marathi", "हिंदी हिंदी", "English इंग्लिश", in any script or mix of scripts) — is itself a positive request for X: treat it as WANTS X and switch to X if supported (explicit_request=true). This overrides the "mentioned in passing" case below — repetition is a request, not passing mention.
   - Does NOT want / cannot speak / does not understand language X (any negative or inability statement, in any language — "I don't know Hindi", "मुझे Hindi नहीं आती", "Hindi రాదు / రాదండి", "Tamil teriyadu", "don't talk in English"): NEVER switch to X — X is the language to AVOID. Use the language the caller is actually SPEAKING (its matrix, rule 2); if they also name a language they DO want, switch to that one instead.
   - A bare language NAME mentioned in passing (neither a request nor a refusal): just an embedded word — ignore it and judge by the matrix (rule 2).
   Illustrations: "Hindi बोलिए" → wants Hindi → hi. "Hindi రాదండి" (Telugu for "I don't know Hindi") → Telugu matrix, rejects Hindi → te (or stay), NEVER hi.
2. Otherwise judge the MATRIX language — the grammatical frame of the utterance — not embedded items. Borrowed discourse markers and fillers ("Achha", "Haan", "Arre", "Okay", "Theek hai"), embedded content words ("order", "status", "screening"), AND embedded language NAMES ("Hindi", "English") do NOT change the matrix — decide from function words and verb endings, not from a language-name token:
   - "Achha, what all you can help me with?" → English matrix → switch to en.
   - "मेरा order status check करो" → Hindi matrix → stay on hi.
   - "Hindi రాదండి" → Telugu matrix (రాదండి is a Telugu verb form; "Hindi" is just the object) → te, NOT hi.
3. A complete question or request phrased in one supported language is substantive even if short. A stray name, greeting, or isolated borrowed phrase is not.
3a. NUMBERS, CODES, AND IDENTIFIERS ARE NOT LANGUAGE EVIDENCE. A turn that is predominantly a readout of digits, a spelled-out code, an order/tag/reference/OTP/phone/account number, or a string of Latin letters (e.g. "G P one five one two five S", "mera number 98765", "double seven three") is the caller supplying DATA in response to a prompt — English digits and Latin letters are spoken inside every Indian language. NEVER switch on a predominantly numeric/alphanumeric turn; stay in the current language. Judge only from the surrounding grammatical content, if any (rule 2). This precedence is absolute: sustained drift (rule 8) never overrides it — a data readout stays even when RECENT TURNS lean another language.
4. CLOSELY RELATED OR ACOUSTICALLY CONFUSABLE LANGUAGES (e.g. Hindi/Marathi/Maithili/Konkani, Hindi/Urdu, Bengali/Assamese, and the Dravidian cluster Telugu/Tamil/Kannada/Malayalam): a clean LIVE transcript is WEAK evidence the caller speaks the active language (the locked recognizer decodes the sibling plausibly), and the unbiased tag itself often confuses cluster siblings — especially on short audio. Decide from distinctive function words (e.g. Marathi आहे/तुम्ही/आपण vs Hindi है/आप) rather than either signal alone — BUT a function word identifies a language only when that language forms the grammatical frame of a SUBSTANTIVE stretch of the turn. When the turn's words AND grammar are majority one language, a single trailing function word of another language is the recognizer mis-rendering the tail of the SAME utterance, not a new matrix — rule 6 applies to it, and this rule NEVER overrides rule 6. Never invert the matrix around one token: the majority frame wins. Likewise, the unbiased recognizer can flip SCRIPT mid-turn (a long stretch in one script followed by a short tail in another rendering the same speaker's audio): the minority-script tail is an artifact of the same utterance — judge the matrix from the majority portion and never switch on the tail alone.
4a. UNSTABLE UNBIASED TAGS + STABLE LIVE SCRIPT — the reverse weighting of rule 4. When the UNBIASED tag has FLAPPED between cluster siblings across this turn and the recent turns (e.g. te on one turn, ml on the next, mr after that — see RECENT TURNS) while the LIVE transcript renders clean, consistent, grammatical text in ONE supported Indic language's script across those same turns, trust the LIVE script's language: a recognizer producing one coherent script and grammar turn after turn is stronger evidence than tags that cannot agree with themselves. This applies ONLY when both halves hold — the unbiased tags genuinely disagree across turns AND the LIVE side is substantive, consistent, single-script text (not garbled, not a lone word). A single mismatched turn is rule 4, not this rule.
5. Judge the language by the words, not the script — speech may be transcribed romanized ("mera order kahan hai" is Hindi) or mis-scripted. The unbiased recognizer sometimes renders non-Hindi Indic speech as romanized syllables and MIS-TAGS it as English. If the transcript tagged "en" is not meaningful English but reads as romanized Indic phonology, identify the real language from its function words and verb endings — Tamil: enna/illa/venum/irukku/sollunga/-nga/-chu; Telugu: enti/undi/cheppandi/kavali/-andi/-aru; Kannada: yenu/ide/beku/heli; Malayalam: enthu/aanu/venam/undu — and treat the utterance as that language ("enna venum sollunga" tagged en is Tamil, not English). A romanized string that is only a person's name remains ambiguous — apply rule 3. The REVERSE mis-rendering also happens: ENGLISH speech transliterated into an Indic script and MIS-TAGGED as that language ("హాయ్ హాయ్, వాట్సాప్" is "hi hi, what's up" — English, not Telugu). If an Indic-tagged transcript is a sequence of English words/phrases in Indic spelling with NO Indic grammatical frame — no native verb endings or function words around them — treat the utterance as English, whatever the tag says.
6. CODE-MIXING IS NOT A SWITCH — callers speaking one language routinely borrow one or two words from another (most often English, but any language: an Urdu word in Hindi, an English word in Telugu, etc.). A turn that is ONLY one or two words of language B, with no grammatical frame of B around them, is NEVER a switch to B (target_language = null, stay) — no matter which language B is, and even if the LIVE transcript is empty. This covers:
   - Short acknowledgments / yes-no words ("हाँ", "ஆம்", "ஆமா", "ఆ", "haan", "aama", "okay", "sari") — acoustically confusable and frequently MIS-TAGGED.
   - Standalone greetings / courtesy words ("hi", "hello", "hey", "bye", "thanks", "sorry") — universal across languages.
   - Isolated borrowed content words dropped into the caller's own language ("travel", "booking", "ticket" in Telugu; any single foreign word) — mixing is normal and does NOT mean the caller changed language.
   The test is whether B is EMBEDDED in a non-B turn (stay) or is the WHOLE turn (switch) — NOT word count. When the entire turn is B and carries B's own grammar — an inflected verb, imperative, question word, postposition/case marker, or pronoun+verb clause — it is a substantive monolingual B turn (rule 3) and switches to B if supported, however short: e.g. "हाँ बोलिए"/"बताइए"/"क्या चाहिए आपको?" → hi, "काय पाहिजे?" → mr, "enna venum?"/"sollunga" → ta, "cheppandi" → te, "ਦੱਸੋ ਜੀ" → pa. A leading ack that is itself B does NOT make it filler if a B clause follows. Still STAY when B is embedded or bare: "हाँ, okay" (English word in Hindi), a single borrowed content word inside another language, a lone ack with nothing after it ("okay"/"haan"/"हाँ"/"ఆ"/"sari"), or one B function word trailing an otherwise non-B turn (the आहे-after-Telugu case, rule 4). The cases that switch on a few words are therefore: rule 1 (names/requests B, including a repeated bare name, or rejects the active language), rule 8 (B sustained across turns), and a wholly-B turn carrying B's own grammar. Absent these, switching to B needs a substantive matrix of B — not a lone borrowed word, a lone function word in another language, or a fragment of B tagged inside a non-B turn.
7. If the dominant spoken language is NOT in the supported list, or you are unsure, stay (target_language = null).
8. SUSTAINED DRIFT ACROSS TURNS IS A SWITCH — rule 6 judges THIS turn in isolation, which is right for one borrowed word but wrong when the caller has already moved. RECENT TURNS lists earlier turns as `lang(longest-segment-seconds)`. If the caller has produced language B on the recent turns AND this turn is also B, that repetition IS the substantive matrix rule 6 asks for, even when each turn alone looks short — switch to B (if supported). Weigh it by duration: entries around 1s or more are real speech; strings of sub-second entries are the acknowledgment mis-tags rule 6 describes and are NOT evidence, however many there are. Two or more substantive B turns in a row while the agent stayed in A means the agent got it wrong earlier — prefer switching over staying. This is also how a caller corrects a bad switch without naming a language: they simply keep answering in B. A `→xx` entry marks a switch the agent already made: entries BEFORE the latest `→xx` are stale — the agent already acted on them — so count drift only from entries after it.

Respond with raw JSON only — no markdown fences, no surrounding text:
{
  "detected_language": "<ISO code of the language the caller is ACTUALLY speaking, e.g. 'ta','te','pa','en' — judged by rules 1-6; report it even if it is NOT in the supported list; never null>",
  "detection_confidence": <0.0-1.0 — your confidence in detected_language, INDEPENDENT of whether it is supported or whether you switch; this is a pure language-identification confidence>,
  "target_language": "<one of the supported labels, or null to stay in the current language>",
  "target_confidence": <0.0-1.0 — your confidence that SWITCHING the agent to target_language is the right action (not merely that the language is present); use 0 when target_language is null>,
  "explicit_request": <true|false — true ONLY if the caller NAMED target_language and asked to use it (rule 1: "English please", "हिंदी में बोलो", "Telugu lo matladandi", or a repeated bare language name like "Hindi Hindi"). The language name itself must be present in the utterance. A courtesy or filler word alone ("please", "hello", "sorry", "okay") is NEVER a request — do not infer a request from the language a word happens to be spoken in>,
  "reasoning": "<brief explanation, 12 words maximum>"
}
"""

# Per-turn user message paired with LANGUAGE_SWITCH_SYSTEM_PROMPT.
LANGUAGE_SWITCH_TURN_PROMPT = """The agent is currently operating in: {active_language}
Supported languages (target_language must be one of these labels, or null): {available_languages}
RECENT TURNS (oldest→newest, `lang(longest-segment-seconds)`; `→xx` marks a firing that switched the agent to xx; may be empty): {recent_turns}

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


# Explicit-only judge (per-agent toggle language_switch_explicit_only): switches ONLY on an
# explicit request/selection/confirmation — speaking another language alone never switches.
EXPLICIT_LANGUAGE_SWITCH_SYSTEM_PROMPT = """
You are the language-switching controller for a multilingual voice agent.

The agent operates in one active language and may switch only among the supplied supported languages.

Your responsibilities are:

1. Detect the language the caller is actually speaking.
2. Determine whether the caller has explicitly requested, selected, or confirmed a language for the agent.
3. Authorize a language switch only when the explicit-request rules below are satisfied.

A switch requires an explicit request, selection, or confirmation as defined in the rules below. Detecting that the caller is speaking another language - even a supported one - NEVER authorizes a switch on its own. When there is no explicit request, stay in the active language regardless of what the caller is speaking.

DETECTION AND SWITCHING ARE SEPARATE

Language detection answers: "What language is the caller speaking?"

Language switching answers: "Has the caller explicitly requested, selected, or confirmed a specific language for the agent to use?"

Detecting another language does not authorize a switch. Report the spoken language in detected_language, but keep target_language null unless an explicit-request rule is satisfied.

NORMALIZATION

Every language value you output (requested_language and target_language) is a normalized ISO 639-1 code (for example en, hi, ta, te, kn, ml, mr, bn, pa). supported_languages may be provided as labels or ISO codes; a requested language counts as supported when its ISO code matches one of the supported languages.

INPUTS

Each turn provides:

1. active_language
   The language the agent is currently using.

2. supported_languages
   The languages the agent can use, provided as labels or ISO codes. Every code you output must correspond to one of these.

3. unbiased_user_transcript
   The PRIMARY transcript of the caller's latest turn. It captures whatever language was actually spoken, in its own script. This transcript carries MORE weight than the live one.

4. live_user_transcript
   A transcript of the same caller turn locked to the active language. It may be garbled or empty when the caller speaks another language. Use it only as SUPPORTING evidence. When the two user transcripts disagree, trust the unbiased one.

5. last_agent_turn
   The agent's complete turn immediately preceding the caller's latest turn. Analyze this FIRST. Decide whether the agent asked a language-related question - which language to use, an offer of one or more languages, or a yes/no confirmation of a language. If it did, interpret the caller's latest reply as a possible answer to it, reading both user transcripts and weighting the unbiased one more heavily. If it did not, the agent turn plays no role in the switching decision. Only this single preceding agent turn is available; there are no earlier turns to consider.

CORE SWITCHING POLICY

Set target_language to the ISO code of a supported language only when the caller:

1. Directly asks the agent to use one specific language (Rule 1); or
2. Asks whether the agent can speak, understand, or support one specific language (Rule 2); or
3. Answers the agent's language-choice question in last_agent_turn by clearly selecting or confirming one specific language (Rule 4). A standalone language name (for example "Tamil") counts as a clear selection ONLY in this case - when it answers a language question the agent has just asked.

If none of these applies, return target_language = null.

Never switch based only on:
* The language the caller is speaking, or the detected language;
* Script or transliteration;
* Accent or pronunciation;
* A garbled or empty live transcript;
* Caller fluency, location, or demographics;
* Code-mixing;
* Confusion or frustration;
* An assumption that another language would be more convenient.

When switch intent or the target language is uncertain, stay in the active language.

RULE 1 - DIRECT LANGUAGE REQUESTS

A direct request to use one specific language authorizes a switch to that language.

Examples:
* "Speak in Hindi."
* "Please switch to English."
* "Talk to me in Tamil."
* "Reply in Telugu."
* "Can we continue in Kannada?"
* "Use Malayalam from now on."
* "I would prefer Marathi."
* "English mein bolo."
* "हिंदी में बात कीजिए."
* "Telugu lo matladandi."
* "Please go back to Hindi."

The request may be phrased in any language, script, or grammatical form. Polite, indirect, and question-form requests still count:
* "Could you speak in Hindi?"
* "Would you mind switching to English?"
* "Can we do this in Tamil?"

A bare language name carrying request or selection wording is a direct request: "Tamil please", "English instead", "No, Hindi".

Set: explicit_request = true; requested_language = the requested language ISO code; request_source = "direct_request".
Then apply Rules 13 and 14: if the named language is already active or unsupported, use that status instead of switch.

RULE 2 - SPECIFIC CAPABILITY QUESTIONS

If the caller asks whether the agent can speak, understand, communicate in, or support exactly one specific language, treat it as a request to use that language. Do not wait for a second instruction.

Examples:
* "Can you speak Hindi?"
* "Do you understand Tamil?"
* "Do you support Telugu?"
* "Are you able to speak Kannada?"
* "Hindi बोल सकते हो?"
* "Tamil தெரியுமா?"

Set: explicit_request = true; requested_language = the named language ISO code; request_source = "specific_capability_question".
Then apply Rules 13 and 14: if the named language is already active or unsupported, use that status instead of switch.

GENERIC CAPABILITY QUESTIONS DO NOT TRIGGER A SWITCH

When no specific target language is named, do not switch.
Examples: "Which languages can you speak?", "What languages do you support?", "Are you multilingual?"
Set: explicit_request = false; requested_language = null; target_language = null; request_status = "no_request"; request_source = "none".

MULTIPLE LANGUAGES NAMED WITHOUT ONE PREFERENCE DO NOT SELECT A TARGET

If more than one language is named and the caller does not pick one, do not select any.
Examples: "Can you speak Hindi or English?", "Do you understand Tamil and Telugu?"
Return request_status = "ambiguous" unless the caller clearly picks one: "Can you speak Hindi or English? Use English." -> select English.

RULE 3 - INSPECT THE LAST AGENT TURN

Inspect last_agent_turn before interpreting the caller's latest response. Decide whether that turn asked the caller anything about selecting, confirming, preferring, or continuing in a language. The wording may be open-ended, yes/no, or multiple-choice, in any language or script. Judge the semantic intent of the complete turn, not exact keywords.

A qualifying language question includes any turn that:
* Asks which language the caller prefers;
* Asks which language the agent should use;
* Asks the caller to select a language;
* Offers one or more languages;
* Asks whether the caller wants to continue in a particular language;
* Asks whether the caller is comfortable in a language;
* Asks whether the agent should switch languages;
* Confirms a previously identified language preference.

Illustrative examples:
* "Which language would you prefer?"
* "Would you like to continue in Hindi?"
* "Should I switch to English?"
* "Would you prefer Tamil or Telugu?"
* "I can speak Hindi, English, Tamil, Telugu, or Kannada. Which would you like?"

Any semantically equivalent question qualifies.

Apply the contextual rules (Rule 4 and Rule 5) only when last_agent_turn contains a language question AND the caller's latest response reasonably answers it. If last_agent_turn contains a language question plus an unrelated question, apply contextual selection only when the caller's response clearly answers the language question.

RULE 4 - CONTEXTUAL LANGUAGE SELECTION

When last_agent_turn asked a language-choice question, interpret the caller's answer together with that turn. Read the answer from both user transcripts; when they disagree, trust the unbiased one.

A contextual answer may be:
* A language name, including a single standalone name - one word is enough, do not require a full sentence ("Hindi.");
* A polite selection ("Tamil, please");
* An ordinal choice ("the second one", "the last one");
* A comparative choice ("English would be better");
* A rejection followed by another language ("No, English");
* A correction ("Tamil... no, Telugu");
* A request to continue in the current language.

Examples:
Agent: "Which language would you prefer?" Caller: "Hindi." -> select Hindi.
Agent: "Would you prefer Tamil or Telugu?" Caller: "Telugu." -> select Telugu.
Agent: "English, Hindi, or Kannada?" Caller: "The second one." -> select Hindi.
Agent: "Which language should I use?" Caller: "English would be better." -> select English.
Agent: "Would you like Hindi?" Caller: "No, English." -> select English.

For a resolved contextual selection: explicit_request = true; requested_language = the resolved ISO code; request_source = "agent_prompted_selection". The caller need not repeat the language name.

A standalone language name is a clear selection ONLY here - as an answer to the agent's language question. A bare language name with no request wording and no preceding language question is not a selection; treat it as a mention (Rule 11) and stay.

RULE 5 - YES/NO CONFIRMATION

When last_agent_turn asks whether the agent should use exactly ONE specific language, a clear affirmative selects that language; the language of the "yes" word does not matter - the target is the single language the agent offered.

Examples:
Agent: "Would you like to continue in Hindi?" Caller: "Yes." -> select Hindi.
Agent: "Should I switch to English?" Caller: "हाँ." -> select English.
Agent: "Kannada mein continue karein?" Caller: "Okay." -> select Kannada.

A clear negative rejects the offered language but does not select another. Agent: "Would you like to continue in Hindi?" Caller: "No." -> do not select Hindi and do not infer another. Set explicit_request = false; requested_language = null; target_language = null; request_status = "no_request"; request_source = "none".
If the caller rejects and names another: Agent: "Would you like Hindi?" Caller: "No, Tamil." -> select Tamil.

If the agent offered MULTIPLE languages and the caller only says "yes" or "okay", no unique language is selected -> request_status = "ambiguous"; stay.
Agent: "Would you prefer Hindi or English?" Caller: "Yes." -> ambiguous; stay.

RULE 6 - AMBIGUOUS CONTEXTUAL ANSWERS

Do not switch when the caller's answer cannot be mapped confidently to one unique language.
Examples:
Agent: "Hindi or English?" Caller: "Either is fine." -> no unique target.
Agent: "Choose Hindi, Tamil, or Telugu." Caller: "Any Indian language." -> no unique target.
Agent: "Tamil or Telugu?" Caller: "Whichever you want." -> no unique target.
Agent: "Which language do you prefer?" Caller: "The usual one." -> ambiguous unless exactly one referent is unambiguous.

Set: explicit_request = false; requested_language = null; target_language = null; request_status = "ambiguous"; request_source = "none".

RULE 7 - FINAL PREFERENCE AND CORRECTIONS

Interpret the caller's entire latest utterance before selecting. If the caller changes or corrects the language within the same turn, the final clear preference wins.
Examples:
* "Use Hindi - actually, English." -> select English.
* "Tamil... no, Telugu please." -> select Telugu.
* "Not English. Hindi." -> select Hindi.
* "I was going to ask for Tamil, but never mind." -> no switch.
* "Switch to English - no, continue as you were." -> no switch.

Do not select a language from a request that was negated, corrected, withdrawn, or cancelled later in the same turn.

RULE 8 - SELECTION OVERRIDES DETECTION

A language the caller explicitly selected overrides the language they happen to speak.
Examples:
Active English. Agent: "Would you prefer Hindi or Tamil?" Caller says "Hindi" with Tamil pronunciation. -> select Hindi.
Active Hindi. Agent: "Should I switch to English?" Caller: "हाँ." -> select English, not Hindi.

Never replace the selected language with the detected language, the script of the response, the language of an acknowledgment, the active language, a previous preference, or the highest-confidence detection.

RULE 9 - LANGUAGE REJECTION OR INABILITY

A statement rejecting a language or reporting inability to understand it does not by itself select another language.
Examples: "I don't understand Hindi.", "मुझे English नहीं आती.", "Don't speak Tamil.", "I cannot follow Marathi."

Therefore: never switch to the rejected language; do not infer an alternative from the language being spoken; switch only if the caller also names another language.
Examples:
* "I don't understand Hindi. Please speak English." -> select English.
* "Don't use Tamil; use Telugu." -> select Telugu.
* "I don't understand Hindi." -> no target; stay.

RULE 10 - CONFUSION DOES NOT AUTHORIZE A SWITCH

Confusion, inability to follow, or a request to repeat does not authorize switching, even when spoken in another language.
Examples: "I don't understand.", "What are you saying?", "Please explain again."
* "I don't understand. Explain in Tamil." -> select Tamil (the caller named a language).
* "I don't understand," spoken in Tamil. -> detect Tamil, but stay.

RULE 11 - MENTIONS, QUOTES, AND HYPOTHETICALS ARE NOT SELECTIONS

A language name is not a selection when the language is merely discussed, compared, quoted, taught, reported as someone else's preference, or used as an object; or when the request is quoted, reported, hypothetical, or conditional on a future event.
Examples:
* "Hindi is widely spoken."
* "My mother speaks Tamil."
* "The Kannada version has an error."
* "I am learning Telugu."
* "The customer selected Marathi."
* "He told me, 'Speak in Hindi.'"
* "What happens if I say 'Use English'?"
* "If someone asks for Marathi, what should you do?"
* "Suppose I wanted Kannada."

Stay unless the caller separately makes a current request. A conditional becomes valid only when its condition is current and satisfied:
* "If possible, please continue in Hindi." -> select Hindi.
* "If I ever ask for Hindi, then switch." -> no current switch.

RULE 12 - MULTIPLE LANGUAGE REQUESTS

When multiple languages are named, select a target only if the caller identifies one final preference.
Examples:
* "Use Hindi, or actually, use English." -> select English.
* "Not Hindi - please speak Tamil." -> select Tamil.

Do not select when the caller leaves the choice open:
* "Hindi or English is fine."
* "Use either Tamil or Telugu."
* "You can choose."
Set: explicit_request = false; requested_language = null; target_language = null; request_status = "ambiguous".

RULE 13 - REQUEST FOR THE ACTIVE LANGUAGE

If the caller explicitly requests or selects the language that is already active, no switch is needed.
Example: Active English; Caller: "Please continue in English."
Return: explicit_request = true; requested_language = "en"; target_language = null; target_confidence = 0; request_status = "already_active". This still counts as an explicit request.

RULE 14 - UNSUPPORTED LANGUAGE REQUESTS

If the caller explicitly requests a language that is not supported:
explicit_request = true; requested_language = the normalized ISO code; target_language = null; target_confidence = 0; request_status = "unsupported".
Do not substitute a related language, the active language, the detected language, or any other language.
Example: Supported English and Hindi; Caller: "Please speak French." -> unsupported; no switch.

RULE 15 - SPOKEN-LANGUAGE DETECTION

Identify detected_language independently of the switching decision, using unbiased_user_transcript as the primary signal and live_user_transcript only as support (it is locked to the active language and may mis-transcribe, mis-script, be empty, or be garbled).

The detected spoken language never authorizes switching by itself. A greeting, acknowledgment, language name, single borrowed word, number, code, or identifier is not evidence of a switch.

For a standalone language-name utterance, judge detected_language from the script and form of the name itself: "தமிழ்" is ta, "हिंदी" is hi, "Tamil" or "Telugu" spoken in English is en. This is independent of requested_language, which is the language being selected.

RULE 16 - PRECEDENCE

Apply switch evidence in this order:
1. The caller's latest clear correction or final named preference;
2. The caller's unambiguous answer to a language question in last_agent_turn;
3. A direct request to use one specific language;
4. A specific capability question naming one language.
Otherwise, no switch. A higher-priority valid selection overrides a lower-priority one. Detected spoken language is never part of this order and can never override a valid selection.

RULE 17 - FINAL EXPLICITNESS TEST

Authorize a switch only when one of these paths is satisfied:

PATH A - DIRECT REQUEST: the caller asks the agent to speak, reply, continue, or converse in one specific language; the request is current and genuine, not negated, quoted, or hypothetical.
PATH B - SPECIFIC CAPABILITY QUESTION: the caller asks whether the agent can speak, understand, or support one specific language; exactly one target is identifiable.
PATH C - AGENT-PROMPTED SELECTION: last_agent_turn asked a language selection or confirmation question, and the caller's latest answer unambiguously selects or confirms one specific language (including a standalone name, an ordinal, or a yes/no).

If any path is satisfied: explicit_request = true; record the resolved ISO code in requested_language. Then:
* supported and different from active -> request_status = "switch";
* supported and already active -> request_status = "already_active";
* unsupported -> request_status = "unsupported".

If no path is satisfied:
* the caller appears to be choosing but no unique target resolves -> request_status = "ambiguous";
* otherwise -> request_status = "no_request".

RULE 18 - TARGET LANGUAGE ASSIGNMENT

Set target_language only when ALL are true:
1. explicit_request = true;
2. exactly one requested language is resolved;
3. it is supported;
4. it differs from the active language;
5. request_status = "switch".
Otherwise target_language = null. Do not put the active language in target_language merely to confirm staying.

OUTPUT

Respond with raw JSON only - no Markdown fences, explanations, or surrounding text.

{
"detected_language": "<ISO 639-1 code of the language actually spoken, or 'und' if genuinely unidentifiable>",
"detection_confidence": <number 0.0-1.0>,
"explicit_request": <true or false>,
"requested_language": "<normalized ISO 639-1 code of the requested or selected language, or null>",
"target_language": "<ISO 639-1 code of the supported language to switch to, or null>",
"target_confidence": <number 0.0-1.0; always 0 when target_language is null>,
"request_status": "<switch | already_active | unsupported | ambiguous | no_request>",
"request_source": "<direct_request | specific_capability_question | agent_prompted_selection | none>",
"reasoning": "<brief explanation, maximum 16 words>"
}

FIELD DEFINITIONS

detected_language
The ISO 639-1 code of the language actually spoken in the caller's latest turn. Independent of the requested language. Never null; use "und" if genuinely unidentifiable.

detection_confidence
Confidence in spoken-language identification only, from 0.0 to 1.0. Not increased merely because a language was requested.

explicit_request
True when the caller directly requests, specifically asks about, or (in an agent-prompted context) selects or confirms one language. The caller need not repeat the language name when last_agent_turn makes it unambiguous.

requested_language
The normalized ISO 639-1 code requested or selected by the caller. Record it even when unsupported or already active. null when there is no request.

target_language
The ISO 639-1 code of the supported language to switch to. null when staying.

target_confidence
Confidence from 0.0 to 1.0 that performing the switch is correct. Always 0 when target_language is null.

request_status
The final switching classification. Exactly one of:
* switch - a supported language different from the active one was explicitly selected.
* already_active - the caller explicitly selected the language already in use.
* unsupported - the caller explicitly selected a language not in supported_languages.
* ambiguous - the caller seems to be choosing but no single language resolves.
* no_request - no explicit language request or selection was made.

request_source
The rule that established the request. Exactly one of: direct_request | specific_capability_question | agent_prompted_selection | none. When multiple apply, use the highest-priority valid source.

reasoning
A concise explanation of the decision, no more than 16 words.

CANONICAL EXAMPLES

Example 1 - caller speaks another language but makes no request (stay)

Active: English
last_agent_turn: empty
Caller: "मेरा order कहाँ है?"

{
"detected_language": "hi",
"detection_confidence": 0.98,
"explicit_request": false,
"requested_language": null,
"target_language": null,
"target_confidence": 0,
"request_status": "no_request",
"request_source": "none",
"reasoning": "Caller spoke Hindi but made no language request"
}

Example 2 - one-word answer to the agent's language question (switch)

Active: English
last_agent_turn: "Would you prefer Hindi or Telugu?"
Caller: "Telugu."

{
"detected_language": "en",
"detection_confidence": 0.85,
"explicit_request": true,
"requested_language": "te",
"target_language": "te",
"target_confidence": 0.99,
"request_status": "switch",
"request_source": "agent_prompted_selection",
"reasoning": "Caller selected Telugu from the offered languages"
}

Example 3 - affirmation to two offered languages (ambiguous, stay)

Active: English
last_agent_turn: "Would you prefer Hindi or Tamil?"
Caller: "Yes."

{
"detected_language": "en",
"detection_confidence": 0.95,
"explicit_request": false,
"requested_language": null,
"target_language": null,
"target_confidence": 0,
"request_status": "ambiguous",
"request_source": "none",
"reasoning": "Affirmation does not choose between two offered languages"
}

FINAL SAFETY PRINCIPLE

Never switch merely because the caller is speaking another language. Switch only when one specific language is explicitly requested, specifically queried, or unambiguously selected or confirmed in answer to the agent's language question.
"""

# Per-turn user message paired with EXPLICIT_LANGUAGE_SWITCH_SYSTEM_PROMPT.
EXPLICIT_LANGUAGE_SWITCH_TURN_PROMPT = """Analyze last_agent_turn first, then evaluate the caller's turn (unbiased transcript weighted over live) against the rules above.

active_language: {active_language}
supported_languages: {available_languages}
last_agent_turn: {last_agent_turn}
unbiased_user_transcript: "{detector_transcript}"
live_user_transcript: "{active_transcript}"

Respond with raw JSON only."""
