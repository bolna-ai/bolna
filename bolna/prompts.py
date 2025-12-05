from datetime import datetime


EXTRACTION_PROMPT = """
Today Current Date and Time:\n {} at {} local time in the {} timezone. Use this information to ensure all time-related responses are accurate and contextually relevant based on the user's location.
Given the following transcript from a communication between a user and an agent — in some cases, the agent used external tools to fetch information, which are included as `tool_response` entries — your task is to extract the following information:

###JSON Structure
{}
- Make sure your response is in ENGLISH. 
"""

SUMMARY_JSON_STRUCTURE = {"summary": "Summary of the conversation goes here"}

# SUMMARIZATION_PROMPT = """
# You are an AI agent that summarizes phone call transcripts between an assistant and a user. Your job is to give a short, factual brief of what actually happened in the call.
#
# Instructions:
# - Read the entire transcript.
# - Write one short paragraph that describes the call in chronological order.
# - Do not create or assume information, strictly stick to the conversation that took place in the transcript.
#
# Make sure your brief covers:
# - Why the call happened (if it's clear from the transcript).
# - What the user said, asked, or wanted.
# - What the assistant explained, offered, or asked.
# - Any decisions, agreements, or key information (e.g., booked time, confirmed details, said not interested).
# - How the call ended and outline the next steps if any (e.g., agreed to something, refused, asked to call later, call dropped, no answer).
#
# Style rules:
# - Be short, concise and to the point.
# - The output generated should be in a paragraphical format.
# - Avoid inserting any vague or unclear responses in the summary unless it effects the outcome of the call.
# - The beginning of the summary should highlight the participants of the conversation.
# - Mention all numbers/figures and currencies in the summary if they were addressed in the conversation.
# """

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

DATE_PROMPT = '''### Today Current Date and Time:\n {} at {} local time in the {} timezone. Use this information to ensure all time-related responses are accurate and contextually relevant based on the user's location.'''

FUNCTION_CALL_PROMPT = "We made a function calling for user. We hit the function : {} and send a {} request and it returned us the response as given below: {} \n\n . Understand the above response and convey this response in a context to user. ### Important\n1. If there was an issue with the API call, kindly respond with - Hey, I'm not able to use the system right now, can you please try later? \n2. IF YOU CALLED THE FUNCTION BEFORE, PLEASE DO NOT CALL THE SAME FUNCTION AGAIN!"