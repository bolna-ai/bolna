from litellm import token_counter
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from dateutil import parser
import copy
from .utils import format_messages
from .logger_config import configure_logger
from bolna.prompts import CHECK_FOR_COMPLETION_PROMPT
from bolna.constants import HIGH_LEVEL_ASSISTANT_ANALYTICS_DATA


load_dotenv()
logger = configure_logger(__name__)


def calculate_total_cost_of_llm_from_transcript(
    messages,
    cost_per_input_token,
    cost_per_output_token,
    model="gpt-3.5-turbo",
    check_for_completion=False,
    ended_by_assistant=False,
    completion_input_token_cost=0.000001,
    completion_output_token_cost=0.000002,
):
    total_input_tokens = 0
    total_output_tokens = 0
    completion_check_input_tokens = 0
    completion_check_output_tokens = 0
    completion_model = os.getenv("CHECK_FOR_COMPLETION_LLM")
    completion_wrong_answer_tokens = token_counter(model=model, text="{'answer': 'No'}")
    completion_right_answer_tokens = token_counter(model=model, text="{'answer': 'Yes'}")
    llm_token_usage = dict()
    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            total_input_tokens += token_counter(model=model, messages=messages[:i])
            total_output_tokens += token_counter(model=model, text=message["content"])

            # Check for conversation completion
            completion_check_prompt = [
                {"role": "system", "content": CHECK_FOR_COMPLETION_PROMPT},
                {"role": "user", "content": format_messages(messages[: i + 1])},
            ]
            completion_check_input_tokens += token_counter(model=completion_model, messages=completion_check_prompt)
            if i == len(messages) - 1 and ended_by_assistant:
                completion_check_output_tokens += completion_right_answer_tokens
            else:
                completion_check_output_tokens += completion_wrong_answer_tokens

    total_cost = (total_input_tokens * cost_per_input_token) + (total_output_tokens * cost_per_output_token)
    llm_token_usage[model] = {
        "input": total_input_tokens,
        "output": total_output_tokens,
    }

    if check_for_completion:
        if completion_model not in llm_token_usage:
            llm_token_usage[completion_model] = {"input": 0, "output": 0}
        llm_token_usage[completion_model]["input"] += completion_check_input_tokens
        llm_token_usage[completion_model]["output"] += completion_check_output_tokens
        check_for_completion_cost = (completion_check_input_tokens * completion_input_token_cost) + (
            completion_check_output_tokens * completion_output_token_cost
        )
        logger.info(f"Cost to check completion = {check_for_completion_cost}")
        total_cost += check_for_completion_cost

    return round(total_cost, 5), llm_token_usage


def update_extraction_details(current_high_level_assistant_analytics_data, run_details):
    if "extracted_data" not in run_details or not run_details["extracted_data"]:
        return None
    extraction_data = run_details["extracted_data"]
    for index, key in enumerate(extraction_data.keys()):
        if key not in current_high_level_assistant_analytics_data["extraction_details"]:
            logger.info(
                f"current_high_level_assistant_analytics_data['extraction_details'] {current_high_level_assistant_analytics_data['extraction_details']} extraction_data[key] {extraction_data[key]}"
            )
            current_high_level_assistant_analytics_data["extraction_details"][key] = {extraction_data[key]: 0}
        elif extraction_data[key] not in current_high_level_assistant_analytics_data["extraction_details"]:
            current_high_level_assistant_analytics_data["extraction_details"][key] = {extraction_data[key]: 0}
        current_high_level_assistant_analytics_data["extraction_details"][key][extraction_data[key]] += 1
    return current_high_level_assistant_analytics_data


def update_execution_details(current_high_level_assistant_analytics_data, run_details):
    total_duration_till_now = (
        current_high_level_assistant_analytics_data["execution_details"]["average_duration_of_conversation"]
        * current_high_level_assistant_analytics_data["execution_details"]["total_conversations"]
    )
    current_high_level_assistant_analytics_data["execution_details"]["total_conversations"] += 1
    current_high_level_assistant_analytics_data["execution_details"]["total_cost"] += run_details["total_cost"]
    current_high_level_assistant_analytics_data["execution_details"]["average_duration_of_conversation"] = (
        total_duration_till_now + run_details["conversation_time"]
    ) / current_high_level_assistant_analytics_data["execution_details"]["total_conversations"]


def update_historical_values(
    arr, current_run_val, last_updated_at, should_increment, multiplier=0, interval_minutes=1440
):
    now = datetime.now(timezone.utc)
    last_updated_datetime = parser.isoparse(last_updated_at)
    difference_in_minutes = (now - last_updated_datetime).total_seconds() / 60

    if not arr or len(arr) == 0:
        return [0, 0, 0, 0, current_run_val]

    if difference_in_minutes < interval_minutes:
        if should_increment:
            arr[-1] += current_run_val
        else:
            arr[-1] = round((arr[-1] * multiplier + current_run_val) / (multiplier + 1), 5)
    else:
        days_missed = int(difference_in_minutes // interval_minutes) - 1
        if days_missed > 0:
            arr = arr[-min(len(arr), days_missed) :] + [0] * min(len(arr), days_missed)
        if len(arr) < 5:
            arr.append(current_run_val)
        else:
            arr.pop(0)
            arr.append(current_run_val)

    return arr


def update_historical_spread(current_high_level_assistant_analytics_data, run_details):
    current_high_level_assistant_analytics_data["historical_spread"]["number_of_conversations_in_past_5_days"] = (
        update_historical_values(
            current_high_level_assistant_analytics_data["historical_spread"]["number_of_conversations_in_past_5_days"],
            1,
            current_high_level_assistant_analytics_data["last_updated_at"],
            should_increment=True,
        )
    )
    current_high_level_assistant_analytics_data["historical_spread"]["cost_past_5_days"] = update_historical_values(
        current_high_level_assistant_analytics_data["historical_spread"]["cost_past_5_days"],
        run_details["total_cost"],
        current_high_level_assistant_analytics_data["last_updated_at"],
        should_increment=True,
    )
    logger.info(f"Before updating average duratio {current_high_level_assistant_analytics_data['historical_spread']}")
    current_high_level_assistant_analytics_data["historical_spread"]["average_duration_past_5_days"] = (
        update_historical_values(
            current_high_level_assistant_analytics_data["historical_spread"]["average_duration_past_5_days"],
            run_details["conversation_time"],
            current_high_level_assistant_analytics_data["last_updated_at"],
            should_increment=False,
            multiplier=current_high_level_assistant_analytics_data["historical_spread"][
                "number_of_conversations_in_past_5_days"
            ][-1],
        )
    )


def update_cost_details(current_high_level_assistant_analytics_data, run_details):
    if run_details["cost_breakdown"]["transcriber"] > 0:
        current_high_level_assistant_analytics_data["cost_details"]["average_transcriber_cost_per_conversation"] = (
            round(
                (
                    (
                        current_high_level_assistant_analytics_data["cost_details"][
                            "average_transcriber_cost_per_conversation"
                        ]
                        * (current_high_level_assistant_analytics_data["execution_details"]["total_conversations"] - 1)
                        + run_details["cost_breakdown"]["transcriber"]
                    )
                    / current_high_level_assistant_analytics_data["execution_details"]["total_conversations"]
                ),
                5,
            )
        )
    if run_details["cost_breakdown"]["synthesizer"] > 0:
        current_high_level_assistant_analytics_data["cost_details"]["average_synthesizer_cost_per_conversation"] = (
            round(
                (
                    (
                        current_high_level_assistant_analytics_data["cost_details"][
                            "average_synthesizer_cost_per_conversation"
                        ]
                        * (current_high_level_assistant_analytics_data["execution_details"]["total_conversations"] - 1)
                        + run_details["cost_breakdown"]["synthesizer"]
                    )
                    / current_high_level_assistant_analytics_data["execution_details"]["total_conversations"]
                ),
                5,
            )
        )
    current_high_level_assistant_analytics_data["cost_details"]["average_llm_cost_per_conversation"] = round(
        (
            (
                current_high_level_assistant_analytics_data["cost_details"]["average_llm_cost_per_conversation"]
                * (current_high_level_assistant_analytics_data["execution_details"]["total_conversations"] - 1)
                + run_details["cost_breakdown"]["llm"]
            )
            / current_high_level_assistant_analytics_data["execution_details"]["total_conversations"]
        ),
        5,
    )


def update_conversation_details(current_high_level_assistant_analytics_data, conversation_status="finished"):
    current_high_level_assistant_analytics_data["conversation_details"]["total_conversations"] += 1
    if conversation_status == "finished":
        current_high_level_assistant_analytics_data["conversation_details"]["finished_conversations"] += 1
    else:
        current_high_level_assistant_analytics_data["conversation_details"]["rejected_conversations"] += 1


def update_high_level_assistant_analytics_data(current_high_level_assistant_analytics_data, run_details):
    logger.info(f"run details {run_details}")
    if current_high_level_assistant_analytics_data is None:
        current_high_level_assistant_analytics_data = copy.deepcopy(HIGH_LEVEL_ASSISTANT_ANALYTICS_DATA)

    update_execution_details(current_high_level_assistant_analytics_data, run_details)
    update_extraction_details(current_high_level_assistant_analytics_data, run_details)
    update_historical_spread(current_high_level_assistant_analytics_data, run_details)
    update_cost_details(current_high_level_assistant_analytics_data, run_details)
    update_conversation_details(current_high_level_assistant_analytics_data)

    logger.info(f"current_high_level_assistant_analytics_data {current_high_level_assistant_analytics_data}")
    return current_high_level_assistant_analytics_data
