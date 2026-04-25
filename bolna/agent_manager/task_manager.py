import asyncio
from collections import defaultdict
from datetime import datetime
import math
import os
import random
import traceback
import time
import json
import uuid
import copy
import base64
import pytz
import websockets

import aiohttp

from bolna.constants import (
    ACCIDENTAL_INTERRUPTION_PHRASES,
    DEFAULT_USER_ONLINE_MESSAGE,
    DEFAULT_USER_ONLINE_MESSAGE_TRIGGER_DURATION,
    FILLER_DICT,
    DEFAULT_LANGUAGE_CODE,
    DEFAULT_TIMEZONE,
    LANGUAGE_NAMES,
    LLM_DEFAULT_CONFIGS,
    SWITCH_LANGUAGE_TOOL_DEFINITION,
    END_CALL_FUNCTION_PREFIX,
    END_CALL_TOOL_DEFINITION,
)
from bolna.helpers.aiohttp_session import get_shared_aiohttp_session
from bolna.helpers.function_calling_helpers import trigger_api, computed_api_response, prepare_api_request
from bolna.helpers.conversation_history import ConversationHistory
from .base_manager import BaseManager
from .interruption_manager import InterruptionManager
from bolna.agent_types import *
from bolna.providers import *
from bolna.enums import TelephonyProvider, LogComponent, LogDirection, HangupReason
from bolna.exceptions import BolnaComponentError, LLMError, SynthesizerError, TranscriberError
from bolna.prompts import *
from bolna.helpers.language_detector import LanguageDetector
from bolna.transcriber.transcriber_pool import TranscriberPool
from bolna.synthesizer.synthesizer_pool import SynthesizerPool
from bolna.helpers.utils import (
    structure_system_prompt,
    compute_function_pre_call_message,
    select_message_by_language,
    get_date_time_from_timezone,
    calculate_audio_duration,
    create_ws_data_packet,
    get_file_names_in_directory,
    get_raw_audio_bytes,
    is_valid_md5,
    get_required_input_types,
    format_messages,
    get_prompt_responses,
    resample,
    save_audio_file_to_s3,
    update_prompt_with_context,
    get_md5_hash,
    clean_json_string,
    wav_bytes_to_pcm,
    convert_to_request_log,
    yield_chunks_from_memory,
    process_task_cancellation,
    pcm_to_ulaw,
    format_error_message,
    enrich_context_with_time_variables,
)
from bolna.helpers.logger_config import configure_logger
from ..helpers.mark_event_meta_data import MarkEventMetaData
from ..helpers.observable_variable import ObservableVariable
from .models import ComponentLatencies
from .voicemail_handler import VoicemailHandler

logger = configure_logger(__name__)


class TaskManager(BaseManager):
    def __init__(
        self,
        assistant_name,
        task_id,
        task,
        ws,
        input_parameters=None,
        context_data=None,
        assistant_id=None,
        turn_based_conversation=False,
        cache=None,
        input_queue=None,
        conversation_history=None,
        output_queue=None,
        yield_chunks=True,
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs
        self.kwargs["task_manager_instance"] = self

        self.conversation_start_init_ts = time.time() * 1000
        self.llm_latencies = ComponentLatencies()
        self.transcriber_latencies = ComponentLatencies()
        self.synthesizer_latencies = ComponentLatencies()
        self.rag_latencies = {"turn_latencies": []}
        self.routing_latencies = {"turn_latencies": []}
        self.stream_sid_ts = None

        self.task_config = task

        self.timezone = pytz.timezone(DEFAULT_TIMEZONE)
        self.language = DEFAULT_LANGUAGE_CODE
        self.transfer_call_params = self.kwargs.get("transfer_call_params", None)

        if task["tools_config"].get("api_tools", None) is not None:
            self.kwargs["api_tools"] = task["tools_config"]["api_tools"]

        if (
            task["tools_config"]["llm_agent"]
            and task["tools_config"]["llm_agent"]["llm_config"].get("assistant_id", None) is not None
        ):
            self.kwargs["assistant_id"] = task["tools_config"]["llm_agent"]["llm_config"]["assistant_id"]

        logger.info(f"doing task {task}")
        self.task_id = task_id
        self.assistant_name = assistant_name
        self.tools = {}
        self.multilingual_prompts = {}
        self.websocket = ws
        self.context_data = context_data
        self.turn_based_conversation = turn_based_conversation
        self.enforce_streaming = kwargs.get("enforce_streaming", False)
        self.room_url = kwargs.get("room_url", None)
        self.is_web_based_call = kwargs.get("is_web_based_call", False)
        # self.callee_silent = True
        # TODO check if we need to toggle this based on some config
        self.yield_chunks = False
        # Set up communication queues between processes
        self.audio_queue = asyncio.Queue()
        self.llm_queue = asyncio.Queue()
        self.synthesizer_queue = asyncio.Queue()
        self.transcriber_output_queue = asyncio.Queue()
        self.dtmf_queue = asyncio.Queue()
        self.queues = {
            "dtmf": self.dtmf_queue,
            "transcriber": self.audio_queue,
            "llm": self.llm_queue,
            "synthesizer": self.synthesizer_queue,
        }
        self.pipelines = task["toolchain"]["pipelines"]
        self.textual_chat_agent = False
        if (
            task["toolchain"]["pipelines"][0] == "llm"
            and task["tools_config"]["llm_agent"]["agent_task"] == "conversation"
        ):
            self.textual_chat_agent = False

        # Assistant persistance stuff
        self.assistant_id = assistant_id
        self.run_id = kwargs.get("run_id")

        self.mark_event_meta_data = MarkEventMetaData()
        self.sampling_rate = 24000
        self.conversation_ended = False
        self.has_transfer = False
        self.hangup_triggered = False
        self.hangup_triggered_at = None
        self.hangup_message_queued = False
        self.switch_handoff_messages = {}
        self.agent_names = {}
        self._end_of_conversation_in_progress = False
        self._turn_audio_flushed = asyncio.Event()
        self._turn_audio_flushed.set()
        self.hangup_mark_event_timeout = 10

        # Prompts
        self.prompts, self.system_prompt = {}, {}
        self.input_parameters = input_parameters

        # Recording
        self.should_record = False
        self.conversation_recording = {
            "input": {"data": b"", "started": time.time()},
            "output": [],
            "metadata": {"started": 0},
        }

        self.welcome_message_audio = self.kwargs.pop("welcome_message_audio", None)

        self.welcome_message_delay = task.get("task_config", {}).get("welcome_message_delay", 0)
        # Pre-decode welcome audio for faster playback
        self.preloaded_welcome_audio = (
            base64.b64decode(self.welcome_message_audio) if self.welcome_message_audio else None
        )
        self.observable_variables = {}
        self.output_handler_set = False
        # IO HANDLERS
        if task_id == 0:
            if self.is_web_based_call:
                self.task_config["tools_config"]["input"]["provider"] = "default"
                self.task_config["tools_config"]["output"]["provider"] = "default"

            self.default_io = self.task_config["tools_config"]["output"]["provider"] == "default"
            self.observable_variables["agent_hangup_observable"] = ObservableVariable(False)
            self.observable_variables["agent_hangup_observable"].add_observer(self.agent_hangup_observer)

            self.observable_variables["final_chunk_played_observable"] = ObservableVariable(False)
            self.observable_variables["final_chunk_played_observable"].add_observer(self.final_chunk_played_observer)

            if self.is_web_based_call:
                self.observable_variables["init_event_observable"] = ObservableVariable(None)
                self.observable_variables["init_event_observable"].add_observer(self.handle_init_event)

            # TODO revert this temporary change for web based call
            if self.is_web_based_call:
                self.should_record = False
            else:
                self.should_record = (
                    self.task_config["tools_config"]["output"]["provider"] == "default" and self.enforce_streaming
                )  # In this case, this is a websocket connection and we should record

            self.__setup_input_handlers(turn_based_conversation, input_queue, self.should_record)
        self.__setup_output_handlers(turn_based_conversation, output_queue)

        self.first_message_task_new = asyncio.create_task(self.message_task_new())

        self.conversation_history = ConversationHistory(conversation_history)
        self.label_flow = []

        # Setup IO SERVICE, TRANSCRIBER, LLM, SYNTHESIZER
        self.llm_task = None
        self.llm_queue_task = None
        self.execute_function_call_task = None
        self.synthesizer_tasks = []
        self.synthesizer_task = None
        self._component_error = None
        self._error_logged = False
        self.synthesizer_monitor_task = None
        self.dtmf_task = None

        # state of conversation
        self.current_request_id = None
        self.previous_request_id = None
        self.llm_rejected_request_ids = set()
        self.llm_processed_request_ids = set()
        self.buffers = []
        self.should_respond = False
        self.last_response_time = time.time()
        self.consider_next_transcript_after = time.time()
        self.llm_response_generated = False
        self.response_in_pipeline = False

        # Language detection
        self.language_detector = LanguageDetector(self.task_config["task_config"], run_id=self.run_id)
        self.language_injection_mode = self.task_config["task_config"].get("language_injection_mode")
        self.language_instruction_template = self.task_config["task_config"].get("language_instruction_template")

        # Call conversations
        self.call_sid = None
        self.stream_sid = None

        # metering
        self.transcriber_duration = 0
        self.synthesizer_characters = 0
        self.ended_by_assistant = False
        self.start_time = time.time()

        # Tasks
        self.extracted_data = None
        self.summarized_data = None
        self.stream = (
            self.task_config["tools_config"]["synthesizer"] is not None
            and self.task_config["tools_config"]["synthesizer"]["stream"]
        ) and (self.enforce_streaming or not self.turn_based_conversation)

        self.is_local = False
        self.llm_config = None
        self.agent_type = None

        self.llm_config_map = {}
        self.llm_agent_map = {}
        if self.__is_multiagent():
            for agent, config in self.task_config["tools_config"]["llm_agent"]["llm_config"]["agent_map"].items():
                self.llm_config_map[agent] = config.copy()
                self.llm_config_map[agent]["buffer_size"] = self.task_config["tools_config"]["synthesizer"][
                    "buffer_size"
                ]
        else:
            if self.task_config["tools_config"]["llm_agent"] is not None:
                if self.__is_knowledgebase_agent():
                    self.llm_agent_config = self.task_config["tools_config"]["llm_agent"]
                    self.llm_config = {
                        "model": self.llm_agent_config["llm_config"]["model"],
                        "max_tokens": self.llm_agent_config["llm_config"]["max_tokens"],
                        "provider": self.llm_agent_config["llm_config"]["provider"],
                        "buffer_size": self.task_config["tools_config"]["synthesizer"].get("buffer_size"),
                        "temperature": self.llm_agent_config["llm_config"]["temperature"],
                    }
                elif self.__is_graph_agent():
                    self.llm_agent_config = self.task_config["tools_config"]["llm_agent"]
                    self.llm_config = {
                        "model": self.llm_agent_config["llm_config"]["model"],
                        "max_tokens": self.llm_agent_config["llm_config"]["max_tokens"],
                        "provider": self.llm_agent_config["llm_config"]["provider"],
                        "buffer_size": self.task_config["tools_config"]["synthesizer"].get("buffer_size"),
                        "temperature": self.llm_agent_config["llm_config"]["temperature"],
                    }
                else:
                    agent_type = self.task_config["tools_config"]["llm_agent"].get("agent_type", None)
                    if not agent_type:
                        self.llm_agent_config = self.task_config["tools_config"]["llm_agent"]
                    else:
                        self.llm_agent_config = self.task_config["tools_config"]["llm_agent"]["llm_config"]

                    self.llm_config = {
                        "model": self.llm_agent_config["model"],
                        "max_tokens": self.llm_agent_config["max_tokens"],
                        "provider": self.llm_agent_config["provider"],
                        "temperature": self.llm_agent_config["temperature"],
                    }

                if "reasoning_effort" in self.llm_agent_config:
                    self.llm_config["reasoning_effort"] = self.llm_agent_config["reasoning_effort"]

                if "thinking_budget" in self.llm_agent_config:
                    self.llm_config["thinking_budget"] = self.llm_agent_config["thinking_budget"]

                if self.llm_agent_config.get("use_responses_api"):
                    self.llm_config["use_responses_api"] = True

                if self.llm_agent_config.get("compact_threshold"):
                    self.llm_config["compact_threshold"] = self.llm_agent_config["compact_threshold"]

        # Output stuff
        self.output_task = None
        self.buffered_output_queue = asyncio.Queue()

        # Memory
        self.cache = cache

        # Initialize InterruptionManager with defaults (will be reconfigured for task_id == 0)
        self.interruption_manager = InterruptionManager()

        # setup request logs
        self.request_logs = []

        # Stores structured API call records for dashboard/backend persistence.
        self.function_tool_api_call_details = []
        self.hangup_task = None

        self.conversation_config = None

        if task_id == 0:
            provider_config = self.task_config["tools_config"]["synthesizer"].get("provider_config")
            self.synthesizer_voice = provider_config["voice"]
            self.hangup_detail = None
            self.shadow_end_call_enabled = False
            self.shadow_end_call_events = []

            self.handle_accumulated_message_task = None
            # self.initial_silence_task = None
            self.hangup_task = None
            self.transcriber_task = None
            self.output_chunk_size = 16384 if self.sampling_rate == 24000 else 4096  # 0.5 second chunk size for calls
            # For nitro
            self.nitro = True
            self.conversation_config = task.get("task_config", {})
            logger.info(f"Conversation config {self.conversation_config}")
            self.generate_precise_transcript = self.conversation_config.get("generate_precise_transcript", False)

            # Enable DTMF flow
            dtmf_enabled = self.conversation_config.get("dtmf_enabled", False)
            if dtmf_enabled:
                self.tools["input"].is_dtmf_active = True
                self.dtmf_task = asyncio.create_task(self.inject_digits_to_conversation())

            self.trigger_user_online_message_after = self.conversation_config.get(
                "trigger_user_online_message_after", DEFAULT_USER_ONLINE_MESSAGE_TRIGGER_DURATION
            )
            self.check_if_user_online = self.conversation_config.get("check_if_user_online", True)
            self.check_user_online_message_config = self.conversation_config.get(
                "check_user_online_message", DEFAULT_USER_ONLINE_MESSAGE
            )
            if self.check_user_online_message_config and self.context_data:
                if isinstance(self.check_user_online_message_config, dict):
                    self.check_user_online_message_config = {
                        lang: update_prompt_with_context(msg, self.context_data)
                        for lang, msg in self.check_user_online_message_config.items()
                    }
                else:
                    self.check_user_online_message_config = update_prompt_with_context(
                        self.check_user_online_message_config, self.context_data
                    )

            self.kwargs["process_interim_results"] = (
                "true" if self.conversation_config.get("optimize_latency", False) is True else "false"
            )

            # for long pauses and rushing
            if self.conversation_config is not None:
                # TODO need to get this for azure - for azure the subtraction would not happen
                self.minimum_wait_duration = self.task_config["tools_config"]["transcriber"]["endpointing"]
                self.last_spoken_timestamp = time.time() * 1000
                self.incremental_delay = self.conversation_config.get("incremental_delay", 100)

                # Cut conversation
                self.hang_conversation_after = self.conversation_config.get("hangup_after_silence", 10)
                self.last_transmitted_timestamp = 0

                self.use_fillers = self.conversation_config.get("use_fillers", False)
                self.use_llm_to_determine_hangup = self.conversation_config.get("hangup_after_LLMCall", False)
                self.check_for_completion_prompt = None
                if self.use_llm_to_determine_hangup:
                    self.check_for_completion_prompt = self.conversation_config.get("call_cancellation_prompt", None)
                    if not self.check_for_completion_prompt:
                        self.check_for_completion_prompt = CHECK_FOR_COMPLETION_PROMPT
                    self.check_for_completion_prompt += """
                        Respond only in this JSON format:
                            {{
                              "hangup": "Yes" or "No"
                            }}
                    """

                self.call_hangup_message_config = self.conversation_config.get("call_hangup_message", None)
                if self.call_hangup_message_config and self.context_data and not self.is_web_based_call:
                    if isinstance(self.call_hangup_message_config, dict):
                        self.call_hangup_message_config = {
                            lang: update_prompt_with_context(msg, self.context_data)
                            for lang, msg in self.call_hangup_message_config.items()
                        }
                    else:
                        self.call_hangup_message_config = update_prompt_with_context(
                            self.call_hangup_message_config, self.context_data
                        )
                self.check_for_completion_llm = os.getenv("CHECK_FOR_COMPLETION_LLM")

                # Shadow end_call tool: inject alongside hangup_after_LLMCall for comparison testing
                self.shadow_end_call_enabled = False
                self.shadow_end_call_events = []
                if os.getenv("SHADOW_END_CALL_TOOL", "").lower() == "true" and self.use_llm_to_determine_hangup:
                    self.shadow_end_call_enabled = True
                    api_tools = self.kwargs.get("api_tools")
                    if api_tools is None:
                        api_tools = {"tools": [], "tools_params": {}}
                        self.kwargs["api_tools"] = api_tools
                    if END_CALL_FUNCTION_PREFIX not in api_tools.get("tools_params", {}):
                        tool_def = copy.deepcopy(END_CALL_TOOL_DEFINITION)
                        # Use the agent's hangup criteria so the comparison is apples-to-apples
                        cancellation_criteria = self.conversation_config.get("call_cancellation_prompt", "")
                        if cancellation_criteria:
                            tool_def["function"]["description"] = (
                                f"End the current call. Always say your goodbye message before calling this function.\n"
                                f"Criteria for when to end: {cancellation_criteria}"
                            )
                        tools_list = api_tools.get("tools", [])
                        if isinstance(tools_list, str):
                            tools_list = json.loads(tools_list)
                        tools_list.append(tool_def)
                        api_tools["tools"] = tools_list
                        api_tools["tools_params"][END_CALL_FUNCTION_PREFIX] = {"pre_call_message": None}
                    logger.info("Shadow end_call tool injected for hangup comparison")

                # Voicemail detection (time-based)
                output_tool_available = (
                    "output" in self.tools
                    and self.tools["output"]
                    and self.tools["output"].requires_custom_voicemail_detection()
                )
                self.voicemail_handler = VoicemailHandler(self, self.conversation_config, output_tool_available)

                self.time_since_last_spoken_human_word = 0

                # Handling accidental interruption
                self.number_of_words_for_interruption = self.conversation_config.get(
                    "number_of_words_for_interruption", 3
                )
                self.asked_if_user_is_still_there = False  # Used to make sure that if user's phrase qualifies as acciedental interruption, we don't break the conversation loop
                self.started_transmitting_audio = False
                self.accidental_interruption_phrases = set(ACCIDENTAL_INTERRUPTION_PHRASES)
                # self.interruption_backoff_period = 1000 #conversation_config.get("interruption_backoff_period", 300) #this is the amount of time output loop will sleep before sending next audio
                self.allow_extra_sleep = False  # It'll help us to back off as soon as we hear interruption for a while

                # Initialize InterruptionManager to centralize interruption logic
                self.interruption_manager = InterruptionManager(
                    number_of_words_for_interruption=self.number_of_words_for_interruption,
                    accidental_interruption_phrases=ACCIDENTAL_INTERRUPTION_PHRASES,
                    incremental_delay=self.incremental_delay,
                    minimum_wait_duration=self.minimum_wait_duration,
                )

                # Backchanneling
                self.should_backchannel = self.conversation_config.get("backchanneling", False)
                self.backchanneling_task = None
                self.backchanneling_start_delay = self.conversation_config.get("backchanneling_start_delay", 5)
                self.backchanneling_message_gap = self.conversation_config.get(
                    "backchanneling_message_gap", 2
                )  # Amount of duration co routine will sleep
                if self.should_backchannel and not turn_based_conversation and task_id == 0:
                    logger.info(f"Should backchannel")
                    self.backchanneling_audios = f"{kwargs.get('backchanneling_audio_location', os.getenv('BACKCHANNELING_PRESETS_DIR'))}/{self.synthesizer_voice.lower()}"
                    # self.num_files = list_number_of_wav_files_in_directory(self.backchanneling_audios)
                    try:
                        self.filenames = get_file_names_in_directory(self.backchanneling_audios)
                        logger.info(f"Backchanneling audio location {self.backchanneling_audios}")
                    except Exception as e:
                        logger.error(f"Something went wrong an putting should backchannel to false {e}")
                        self.should_backchannel = False
                else:
                    self.backchanneling_audio_map = []
                # Agent welcome message
                if "agent_welcome_message" in self.kwargs:
                    logger.info(f"Agent welcome message: {self.kwargs['agent_welcome_message']}")
                    self.first_message_task = None
                    self.transcriber_message = ""

                # Discard pre-welcome utterance
                self.discard_pre_welcome_utterance = self.conversation_config.get(
                    "discard_pre_welcome_utterance", False
                )
                self._speech_started_before_welcome = False

        # setting transcriber and synthesizer in parallel
        self.__setup_transcriber()
        self.__setup_synthesizer(self.llm_config)
        if not self.turn_based_conversation and task_id == 0:
            self.synthesizer_monitor_task = asyncio.create_task(self.tools["synthesizer"].monitor_connection())

        # Auto-inject switch_language tool if multilingual pools are active
        self.__inject_switch_language_tool()

        # # setting llm
        # llm = self.__setup_llm(self.llm_config)
        # # Setup tasks
        # self.__setup_tasks(llm)

        # setting llm
        if self.llm_config is not None:
            llm = self.__setup_llm(self.llm_config, task_id)
            # Setup tasks
            agent_params = {"llm": llm, "agent_type": self.llm_agent_config.get("agent_type", "simple_llm_agent")}
            self.__setup_tasks(**agent_params)

        elif self.__is_multiagent():
            # Setup task for multiagent conversation
            for agent in self.task_config["tools_config"]["llm_agent"]["llm_config"]["agent_map"]:
                if "routes" in self.llm_config_map[agent]:
                    del self.llm_config_map[agent]["routes"]  # Remove routes from here as it'll create conflict ahead
                llm = self.__setup_llm(self.llm_config_map[agent])
                agent_type = self.llm_config_map[agent].get("agent_type", "simple_llm_agent")
                logger.info(f"Getting response for {llm} and agent type {agent_type} and {agent}")
                agent_params = {"llm": llm, "agent_type": agent_type}
                llm_agent = self.__setup_tasks(**agent_params)
                self.llm_agent_map[agent] = llm_agent

        elif self.task_config["task_type"] == "webhook":
            if "webhookURL" in self.task_config["tools_config"]["api_tools"]:
                webhook_url = self.task_config["tools_config"]["api_tools"]["webhookURL"]
            else:
                webhook_url = self.task_config["tools_config"]["api_tools"]["tools_params"]["webhook"]["url"]
            logger.info(f"Webhook URL {webhook_url}")
            self.tools["webhook_agent"] = WebhookAgent(webhook_url=webhook_url)

    @staticmethod
    def _sanitize_api_call_headers(headers):
        if not isinstance(headers, dict):
            return headers

        redacted_headers = {}
        sensitive_keys = {"authorization", "proxy-authorization", "x-api-key", "api-key"}
        for key, value in headers.items():
            if str(key).lower() in sensitive_keys:
                redacted_headers[key] = "<redacted>"
            else:
                redacted_headers[key] = value
        return redacted_headers

    @staticmethod
    def _extract_api_call_runtime_args(resp):
        excluded_keys = {"model_response", "textual_response"}
        return {key: copy.deepcopy(value) for key, value in resp.items() if key not in excluded_keys}

    def _start_api_call_detail(
        self,
        *,
        called_fun,
        url,
        method,
        param,
        headers,
        meta_info,
        runtime_args,
        request_body=None,
        api_params=None,
    ):
        api_call_detail = {
            "tool_name": called_fun,
            "tool_call_id": runtime_args.get("tool_call_id", ""),
            "url": url,
            "method": method.upper() if isinstance(method, str) else method,
            "request_template": copy.deepcopy(param),
            "request_body": copy.deepcopy(request_body),
            "request_params": copy.deepcopy(api_params if api_params is not None else runtime_args),
            "runtime_args": copy.deepcopy(runtime_args),
            "headers": self._sanitize_api_call_headers(copy.deepcopy(headers)),
            "meta": {
                "request_id": meta_info.get("request_id"),
                "sequence_id": meta_info.get("sequence_id"),
            },
            "started_at": datetime.now().isoformat(),
            "status": "pending",
            "response_status_code": None,
            "response_content_type": None,
            "response_body": None,
            "response_json": None,
        }
        self.function_tool_api_call_details.append(api_call_detail)
        return api_call_detail

    @staticmethod
    def _finalize_api_call_detail(api_call_detail, response=None, status_code=None, content_type=None, error=None):
        if api_call_detail is None:
            return

        completed_at = datetime.now()
        api_call_detail["completed_at"] = completed_at.isoformat()
        api_call_detail["latency_ms"] = None
        started_at = api_call_detail.get("started_at")
        if started_at:
            try:
                started_at_dt = datetime.fromisoformat(started_at)
                api_call_detail["latency_ms"] = round((completed_at - started_at_dt).total_seconds() * 1000, 2)
            except ValueError:
                logger.warning(f"Could not compute api call latency from started_at={started_at}")
        if error is not None:
            api_call_detail["status"] = "error"
            api_call_detail["error"] = str(error)
        else:
            api_call_detail["status"] = "completed"
        api_call_detail["response_status_code"] = status_code
        api_call_detail["response_content_type"] = content_type
        api_call_detail["response_body"] = copy.deepcopy(response)
        try:
            api_call_detail["response_json"] = (
                json.loads(response) if isinstance(response, str) else copy.deepcopy(response)
            )
        except (TypeError, json.JSONDecodeError):
            api_call_detail["response_json"] = None

    @property
    def history(self):
        return self.conversation_history.messages

    @history.setter
    def history(self, value):
        self.conversation_history._messages = value

    @property
    def interim_history(self):
        return self.conversation_history.interim

    @interim_history.setter
    def interim_history(self, value):
        self.conversation_history._interim = value

    @property
    def language(self) -> str:
        """Active language code.

        Returns the language-detector's result when detection has completed
        (non-multilingual path only: detection is disabled for multilingual
        pools, so dominant_language is always None). Falls back to the
        configured/switched language otherwise.
        """
        detector = getattr(self, "language_detector", None)
        if detector is not None:
            detected = detector.dominant_language
            if detected:
                return detected
        return self._language

    @language.setter
    def language(self, value: str):
        logger.info(f"Setting base language to {value}")
        self._language = value

    @property
    def call_hangup_message(self):
        return select_message_by_language(self.call_hangup_message_config, self.language)

    def __is_multiagent(self):
        if self.task_config["task_type"] == "webhook":
            return False
        agent_type = self.task_config["tools_config"]["llm_agent"].get("agent_type", None)
        return agent_type == "multiagent"

    def __is_knowledgebase_agent(self):
        if self.task_config["task_type"] == "webhook":
            return False
        agent_type = self.task_config["tools_config"]["llm_agent"].get("agent_type", None)
        return agent_type == "knowledgebase_agent"

    def __is_graph_agent(self):
        if self.task_config["task_type"] == "webhook":
            return False
        agent_type = self.task_config["tools_config"]["llm_agent"].get("agent_type", None)
        return agent_type == "graph_agent"

    # def __is_knowledge_agent(self):
    #     if self.task_config["task_type"] == "webhook":
    #         return False
    #     agent_type = self.task_config['tools_config']["llm_agent"].get("agent_type", None)
    #     return agent_type == "knowledge_agent"

    def _invalidate_response_chain(self):
        try:
            llm_agent = self.tools.get("llm_agent")
            if llm_agent and hasattr(llm_agent, "llm"):
                llm_agent.llm.invalidate_response_chain()
        except Exception as e:
            logger.debug(f"Failed to invalidate response chain: {e}")

    def _inject_language_instruction(self, messages: list) -> list:
        """Inject language instruction into messages based on detected language."""
        lang = self.language_detector.dominant_language
        if not lang or not self.language_injection_mode or not self.language_instruction_template:
            return messages

        try:
            lang_name = LANGUAGE_NAMES.get(lang, lang)
            instruction = self.language_instruction_template.format(language=lang_name) + "\n\n"

            if self.language_injection_mode == "system_only":
                for i, msg in enumerate(messages):
                    if msg.get("role") == "system":
                        messages[i]["content"] = instruction + msg["content"]
                        logger.info(f"[system_only] Injected: {lang_name} ({lang})")
                        break
            elif self.language_injection_mode == "per_turn":
                for i, msg in enumerate(messages):
                    if msg.get("role") == "user":
                        messages[i]["content"] = instruction + msg["content"]
                logger.info(
                    f"[per_turn] Injected to {sum(1 for m in messages if m.get('role') == 'user')} user messages: {lang_name} ({lang})"
                )
        except Exception as e:
            logger.error(f"Language injection error: {e}")

        return messages

    def __setup_output_handlers(self, turn_based_conversation, output_queue):
        output_kwargs = {"websocket": self.websocket}

        if self.task_config["tools_config"]["output"] is None:
            logger.info("Not setting up any output handler as it is none")
        elif self.task_config["tools_config"]["output"]["provider"] in SUPPORTED_OUTPUT_HANDLERS.keys():
            # Explicitly use default for turn based conversation as we expect to use HTTP endpoints
            if turn_based_conversation:
                logger.info("Connected through dashboard and hence using default output handler")
                output_handler_class = SUPPORTED_OUTPUT_HANDLERS.get("default")
            else:
                output_handler_class = SUPPORTED_OUTPUT_HANDLERS.get(
                    self.task_config["tools_config"]["output"]["provider"]
                )

                if self.task_config["tools_config"]["output"]["provider"] in SUPPORTED_OUTPUT_TELEPHONY_HANDLERS.keys():
                    output_kwargs["mark_event_meta_data"] = self.mark_event_meta_data
                    logger.info(f"Making sure that the sampling rate for output handler is 8000")
                    self.task_config["tools_config"]["synthesizer"]["provider_config"]["sampling_rate"] = 8000
                    # sip-trunk (Asterisk) uses ulaw; other telephony use pcm (handler converts to mulaw)
                    if self.task_config["tools_config"]["output"]["provider"] == TelephonyProvider.SIP_TRUNK.value:
                        self.task_config["tools_config"]["synthesizer"]["audio_format"] = "ulaw"
                        logger.info(f"Setting synthesizer audio format to ulaw for Asterisk sip-trunk")
                        # Pass input handler to output handler so it can simulate mark events
                        input_handler = self.tools.get("input")
                        output_kwargs["input_handler"] = input_handler
                        output_kwargs["asterisk_media_start"] = (self.context_data or {}).get("media_start_data")
                        output_kwargs["agent_config"] = {"tasks": [self.task_config]}
                        logger.info(
                            f"Passing input_handler to sip-trunk output handler for mark event simulation: {input_handler is not None}"
                        )
                    else:
                        self.task_config["tools_config"]["synthesizer"]["audio_format"] = "pcm"
                else:
                    self.task_config["tools_config"]["synthesizer"]["provider_config"]["sampling_rate"] = 24000
                    output_kwargs["queue"] = output_queue
                self.sampling_rate = self.task_config["tools_config"]["synthesizer"]["provider_config"]["sampling_rate"]

            if self.task_config["tools_config"]["output"]["provider"] == "default":
                output_kwargs["is_web_based_call"] = self.is_web_based_call
                output_kwargs["mark_event_meta_data"] = self.mark_event_meta_data

            self.tools["output"] = output_handler_class(**output_kwargs)
            self.output_handler_set = True
            logger.info("output handler set")
        else:
            raise "Other input handlers not supported yet"

    async def message_task_new(self):
        tasks = []
        if self._is_conversation_task():
            tasks.append(self.tools["input"].handle())

            if not self.turn_based_conversation and not self.is_web_based_call:
                tasks.append(self.__forced_first_message())

        if tasks:
            await asyncio.gather(*tasks)

    def __setup_input_handlers(self, turn_based_conversation, input_queue, should_record):
        if self.task_config["tools_config"]["input"]["provider"] in SUPPORTED_INPUT_HANDLERS.keys():
            input_kwargs = {
                "queues": self.queues,
                "websocket": self.websocket,
                "input_types": get_required_input_types(self.task_config),
                "mark_event_meta_data": self.mark_event_meta_data,
                "is_welcome_message_played": True
                if self.task_config["tools_config"]["output"]["provider"] == "default" and not self.is_web_based_call
                else False,
            }

            if should_record:
                input_kwargs["conversation_recording"] = self.conversation_recording

            if self.turn_based_conversation:
                input_kwargs["turn_based_conversation"] = True
                input_handler_class = SUPPORTED_INPUT_HANDLERS.get("default")
                input_kwargs["queue"] = input_queue
            else:
                input_handler_class = SUPPORTED_INPUT_HANDLERS.get(
                    self.task_config["tools_config"]["input"]["provider"]
                )

                if self.task_config["tools_config"]["input"]["provider"] == "default":
                    input_kwargs["queue"] = input_queue

                input_kwargs["observable_variables"] = self.observable_variables

                # Asterisk (sip-trunk): pass context data for pre-parsed MEDIA_START
                if (
                    self.task_config["tools_config"]["input"]["provider"] == TelephonyProvider.SIP_TRUNK.value
                    and self.context_data
                ):
                    input_kwargs["ws_context_data"] = self.context_data
                    input_kwargs["agent_config"] = {"tasks": [self.task_config]}
            self.tools["input"] = input_handler_class(**input_kwargs)
        else:
            raise "Other input handlers not supported yet"

    async def __forced_first_message(self, timeout=10.0):
        logger.info(f"Executing the first message task")
        try:
            delay_ms = int(self.welcome_message_delay or 0)
            if delay_ms > 0:
                logger.info(f"Welcome message delay set to {delay_ms} ms")
                await asyncio.sleep(delay_ms / 1000)
            start_time = asyncio.get_running_loop().time()
            while True:
                elapsed_time = asyncio.get_running_loop().time() - start_time
                if elapsed_time > timeout:
                    await self.__process_end_of_conversation()
                    logger.warning("Timeout reached while waiting for stream_sid")
                    break

                text = self.kwargs.get("agent_welcome_message", None)
                meta_info = {
                    "io": self.tools["output"].get_provider(),
                    "message_category": "agent_welcome_message",
                    "request_id": str(uuid.uuid4()),
                    "cached": True,
                    "sequence_id": -1,
                    "format": self.task_config["tools_config"]["output"]["format"],
                    "text": text,
                    "end_of_llm_stream": True,
                }
                ws_data_packet = create_ws_data_packet(text, meta_info=meta_info)

                meta_info = ws_data_packet["meta_info"]
                text = ws_data_packet["data"]
                meta_info["type"] = "audio"
                meta_info["synthesizer_start_time"] = time.time()

                audio_chunk = self.preloaded_welcome_audio if self.preloaded_welcome_audio else None
                if meta_info["text"] == "":
                    audio_chunk = None

                # Convert to ulaw for Asterisk/sip-trunk provider (cached welcome is PCM)
                if self.tools["output"].get_provider() == TelephonyProvider.SIP_TRUNK.value and audio_chunk:
                    original_size = len(audio_chunk)
                    audio_chunk = pcm_to_ulaw(audio_chunk)
                    logger.info(
                        f"[SIP-TRUNK] Converted welcome message PCM to ulaw: {original_size} bytes -> {len(audio_chunk)} bytes"
                    )
                    meta_info["format"] = "ulaw"
                else:
                    meta_info["format"] = "pcm"
                meta_info["is_first_chunk"] = True
                meta_info["end_of_synthesizer_stream"] = True
                meta_info["chunk_id"] = 1
                meta_info["is_first_chunk_of_entire_response"] = True
                meta_info["is_final_chunk_of_entire_response"] = True
                message = create_ws_data_packet(audio_chunk, meta_info)

                stream_sid = self.tools["input"].get_stream_sid()
                if stream_sid is not None and self.output_handler_set:
                    self.stream_sid_ts = time.time() * 1000
                    logger.info(f"Got stream sid and hence sending the first message {stream_sid}")
                    self.stream_sid = stream_sid
                    await self.tools["output"].set_stream_sid(stream_sid)

                    if audio_chunk is None:
                        # No welcome message to play - mark as played immediately
                        # so the system doesn't wait for a mark event that will never arrive
                        logger.info("No welcome message audio to send, marking welcome message as played")
                        self.tools["input"].is_welcome_message_played = True
                    else:
                        self.tools["input"].update_is_audio_being_played(True)
                        self.conversation_history.append_welcome_message(text)
                        convert_to_request_log(
                            message=text,
                            meta_info=meta_info,
                            component=LogComponent.SYNTHESIZER,
                            direction=LogDirection.RESPONSE,
                            model=self.synthesizer_provider,
                            is_cached=meta_info.get("is_cached", False),
                            engine=self.tools["synthesizer"].get_engine(),
                            run_id=self.run_id,
                        )
                        await self.tools["output"].handle(message)
                        try:
                            data = message.get("data")
                            if data is not None:
                                duration = calculate_audio_duration(
                                    len(data), self.sampling_rate, format=message["meta_info"]["format"]
                                )
                                if self.should_record:
                                    self.conversation_recording["output"].append(
                                        {"data": data, "start_time": time.time(), "duration": duration}
                                    )
                        except Exception as e:
                            duration = 0.256
                            logger.error(
                                "Exception in __forced_first_message for duration calculation: {}".format(str(e))
                            )
                    break
                else:
                    logger.info(
                        f"Stream id is still None ({stream_sid}) or output handler not set ({self.output_handler_set}), waiting..."
                    )
                    await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Exception in __forced_first_message {str(e)}")

        return

    def __inject_switch_language_tool(self):
        """Auto-inject switch_language tool when multilingual pools are active."""
        has_pool = isinstance(self.tools.get("transcriber"), TranscriberPool) or isinstance(
            self.tools.get("synthesizer"), SynthesizerPool
        )
        if not has_pool:
            return

        # Collect available labels from pools
        labels = set()
        if isinstance(self.tools.get("transcriber"), TranscriberPool):
            labels.update(self.tools["transcriber"].labels)
        if isinstance(self.tools.get("synthesizer"), SynthesizerPool):
            labels.update(self.tools["synthesizer"].labels)

        # Enrich the tool schema with available labels in the description
        tool_def = copy.deepcopy(SWITCH_LANGUAGE_TOOL_DEFINITION)
        custom_description = self.task_config.get("tools_config", {}).get("switch_tool_description")
        if custom_description:
            tool_def["function"]["description"] = custom_description
        lang_prop = tool_def["function"]["parameters"]["properties"]["language"]
        lang_prop["enum"] = sorted(labels)
        lang_prop["description"] = f"Language to switch to. Available: {sorted(labels)}"

        if self.kwargs.get("api_tools") is None:
            self.kwargs["api_tools"] = {"tools": [], "tools_params": {}}

        self.kwargs["api_tools"]["tools"].append(tool_def)
        # Entry must exist in tools_params so ToolCallAccumulator.build_api_payload
        # doesn't drop the call, but no pre_call_message — the switch is silent.
        self.kwargs["api_tools"]["tools_params"]["switch_language"] = {}

        self.switch_handoff_messages = self.task_config.get("tools_config", {}).get("switch_handoff_messages", {})
        self.agent_names = self.task_config.get("tools_config", {}).get("agent_names", {})

    def __setup_transcriber(self):
        try:
            if self.task_config["tools_config"]["transcriber"] is not None:
                transcriber_config = self.task_config["tools_config"]["transcriber"]

                self.transcriber_provider = transcriber_config.get("provider", transcriber_config.get("model"))

                # --- Multilingual pool path ---
                if "multilingual" in transcriber_config:
                    multilingual = transcriber_config["multilingual"]
                    active_label = transcriber_config.get("active", DEFAULT_LANGUAGE_CODE)
                    self.language = active_label
                    if hasattr(self, "language_detector"):
                        self.language_detector.set_enabled_status(
                            False
                        )  # Disable language detection when using multilingual pool

                    if self.turn_based_conversation:
                        provider = "playground"
                    elif self.is_web_based_call:
                        provider = "web_based_call"
                    else:
                        provider = self.task_config["tools_config"]["input"]["provider"]

                    is_sip = provider == TelephonyProvider.SIP_TRUNK.value

                    transcribers = {}
                    for label, cfg in multilingual.items():
                        private_queue = asyncio.Queue()
                        cfg["input_queue"] = private_queue
                        cfg["output_queue"] = self.transcriber_output_queue
                        if is_sip:
                            cfg["encoding"] = "mulaw"
                            cfg["sampling_rate"] = 8000
                        if self.turn_based_conversation:
                            cfg["stream"] = True if self.enforce_streaming else False

                        if "provider" in cfg:
                            cls = SUPPORTED_TRANSCRIBER_PROVIDERS.get(cfg["provider"])
                        else:
                            cls = SUPPORTED_TRANSCRIBER_MODELS.get(cfg["model"])
                        transcribers[label] = cls(provider, **cfg, **self.kwargs)

                        if label == active_label:
                            self.transcriber_provider = cfg.get("provider", cfg.get("model"))

                    self.tools["transcriber"] = TranscriberPool(
                        transcribers=transcribers,
                        shared_input_queue=self.audio_queue,
                        output_queue=self.transcriber_output_queue,
                        active_label=active_label,
                        multilingual_config=multilingual,
                    )
                    logger.info(
                        f"TranscriberPool created with labels={list(transcribers.keys())}, active='{active_label}'"
                    )
                    return

                # --- Single transcriber path (unchanged) ---
                self.language = transcriber_config.get("language", DEFAULT_LANGUAGE_CODE)
                if self.turn_based_conversation:
                    provider = "playground"
                elif self.is_web_based_call:
                    provider = "web_based_call"
                else:
                    provider = self.task_config["tools_config"]["input"]["provider"]

                transcriber_config["input_queue"] = self.audio_queue
                transcriber_config["output_queue"] = self.transcriber_output_queue

                # Configure encoding for Asterisk/sip-trunk (uses ulaw like Twilio)
                if provider == TelephonyProvider.SIP_TRUNK.value:
                    transcriber_config["encoding"] = "mulaw"
                    transcriber_config["sampling_rate"] = 8000
                    logger.info(f"Configured transcriber for Asterisk sip-trunk with mulaw encoding @ 8kHz")

                # Checking models for backwards compatibility
                if (
                    transcriber_config["model"] in SUPPORTED_TRANSCRIBER_MODELS.keys()
                    or transcriber_config["provider"] in SUPPORTED_TRANSCRIBER_PROVIDERS.keys()
                ):
                    if self.turn_based_conversation:
                        transcriber_config["stream"] = True if self.enforce_streaming else False
                        logger.info(
                            f"transcriber stream={transcriber_config['stream']} enforce_streaming={self.enforce_streaming}"
                        )
                    if "provider" in transcriber_config:
                        transcriber_class = SUPPORTED_TRANSCRIBER_PROVIDERS.get(transcriber_config["provider"])
                    else:
                        transcriber_class = SUPPORTED_TRANSCRIBER_MODELS.get(transcriber_config["model"])
                    self.tools["transcriber"] = transcriber_class(provider, **transcriber_config, **self.kwargs)
        except Exception as e:
            logger.error(f"Something went wrong with starting transcriber {e}")

    def __setup_synthesizer(self, llm_config=None):
        if self._is_conversation_task():
            self.kwargs["use_turbo"] = (
                self.task_config["tools_config"]["transcriber"]["language"] == DEFAULT_LANGUAGE_CODE
            )
        if self.task_config["tools_config"]["synthesizer"] is not None:
            synth_config = self.task_config["tools_config"]["synthesizer"]

            # --- Multilingual pool path ---
            if "multilingual" in synth_config:
                multilingual = synth_config["multilingual"]
                active_label = synth_config.get("active", DEFAULT_LANGUAGE_CODE)
                if hasattr(self, "language_detector"):
                    self.language_detector.set_enabled_status(
                        False
                    )  # Disable language detection when using multilingual pool

                # Telephony providers expect mulaw@8000Hz — force use_mulaw for all synths in the pool
                output_provider = self.task_config["tools_config"]["output"]["provider"]
                is_telephony = output_provider in (
                    TelephonyProvider.PLIVO.value,
                    TelephonyProvider.TWILIO.value,
                    TelephonyProvider.EXOTEL.value,
                    TelephonyProvider.VOBIZ.value,
                    TelephonyProvider.SIP_TRUNK.value,
                )
                synthesizer_kwargs = self.kwargs.copy()
                if is_telephony:
                    synthesizer_kwargs["use_mulaw"] = True

                synthesizers = {}
                for label, cfg in multilingual.items():
                    cfg = dict(cfg)  # shallow copy so pops don't mutate original
                    caching = cfg.pop("caching", True)
                    provider_name = cfg.pop("provider")
                    provider_config = cfg.pop("provider_config")

                    if self.turn_based_conversation:
                        cfg["audio_format"] = "mp3"
                        cfg["stream"] = True if self.enforce_streaming else False

                    cls = SUPPORTED_SYNTHESIZER_MODELS.get(provider_name)
                    synthesizers[label] = cls(**cfg, **provider_config, **synthesizer_kwargs, caching=caching)

                # Use active synth's provider/voice for logging metadata, and buffer_size
                # Note that in the current state, buffer_size of other synth configs is ignored
                active_cfg = synth_config
                if active_label in multilingual:
                    active_cfg = multilingual[active_label]

                self.synthesizer_provider = active_cfg.get("provider", "unknown")
                self.synthesizer_voice = active_cfg.get("provider_config", {}).get("voice", "unknown")

                self.tools["synthesizer"] = SynthesizerPool(
                    synthesizers=synthesizers, active_label=active_label, multilingual_config=multilingual
                )

                logger.info(f"SynthesizerPool created with labels={list(synthesizers.keys())}, active='{active_label}'")

                if self.task_config["tools_config"]["llm_agent"] is not None and llm_config is not None:
                    llm_config["buffer_size"] = active_cfg.get("buffer_size")
                return

            # --- Single synthesizer path (apna normal path) ---
            if "caching" in synth_config:
                caching = synth_config.pop("caching")
            else:
                caching = True

            self.synthesizer_provider = synth_config.pop("provider")
            synthesizer_class = SUPPORTED_SYNTHESIZER_MODELS.get(self.synthesizer_provider)
            provider_config = synth_config.pop("provider_config")
            self.synthesizer_voice = provider_config["voice"]
            if self.turn_based_conversation:
                synth_config["audio_format"] = "mp3"  # Hard code mp3 if we're connected through dashboard
                synth_config["stream"] = (
                    True if self.enforce_streaming else False
                )  # Hardcode stream to be False as we don't want to get blocked by a __listen_synthesizer co-routine

            # Configure use_mulaw for Asterisk/sip-trunk to ensure synthesizer outputs ulaw
            synthesizer_kwargs = self.kwargs.copy()
            if self.task_config["tools_config"]["output"]["provider"] == TelephonyProvider.SIP_TRUNK.value:
                synthesizer_kwargs["use_mulaw"] = True
                logger.info(f"[SIP-TRUNK] Configuring synthesizer with use_mulaw=True for Asterisk sip-trunk")

            self.tools["synthesizer"] = synthesizer_class(
                **synth_config, **provider_config, **synthesizer_kwargs, caching=caching
            )
            # if not self.turn_based_conversation:
            #     self.synthesizer_monitor_task = asyncio.create_task(self.tools['synthesizer'].monitor_connection())
            if self.task_config["tools_config"]["llm_agent"] is not None and llm_config is not None:
                llm_config["buffer_size"] = synth_config.get("buffer_size")

    def __setup_llm(self, llm_config, task_id=0):
        if self.task_config["tools_config"]["llm_agent"] is not None:
            if task_id and task_id > 0:
                self.kwargs.pop("llm_key", None)
                self.kwargs.pop("base_url", None)
                self.kwargs.pop("api_version", None)

                if self._is_summarization_task() or self._is_extraction_task():
                    llm_config["model"] = LLM_DEFAULT_CONFIGS["summarization"]["model"]
                    llm_config["provider"] = LLM_DEFAULT_CONFIGS["summarization"]["provider"]

            if llm_config["provider"] in SUPPORTED_LLM_PROVIDERS.keys():
                llm_class = SUPPORTED_LLM_PROVIDERS.get(llm_config["provider"])
                llm = llm_class(language=self.language, **llm_config, **self.kwargs)
                return llm
            else:
                raise Exception(f"LLM {llm_config['provider']} not supported")

    def __get_agent_object(self, llm, agent_type, assistant_config=None):
        self.agent_type = agent_type
        if agent_type == "simple_llm_agent":
            llm_agent = StreamingContextualAgent(llm)
        elif agent_type == "graph_agent":
            logger.info("Setting up graph agent with rag-proxy-server support")
            llm_config = self.task_config["tools_config"]["llm_agent"].get("llm_config", {})
            rag_server_url = self.kwargs.get("rag_server_url", os.getenv("RAG_SERVER_URL", "http://localhost:8000"))

            logger.info(f"Graph agent config: {llm_config}")
            logger.info(f"RAG server URL: {rag_server_url}")

            # Set RAG server URL in environment for GraphAgent to use
            os.environ["RAG_SERVER_URL"] = rag_server_url

            # Inject provider credentials for routing and response generation
            injected_cfg = dict(llm_config)
            if "llm_key" in self.kwargs:
                injected_cfg["llm_key"] = self.kwargs["llm_key"]
            if "base_url" in self.kwargs:
                injected_cfg["base_url"] = self.kwargs["base_url"]

            # Pass context_data for variable replacement in node prompts
            if self.context_data:
                injected_cfg["context_data"] = self.context_data

            if "api_version" in self.kwargs:
                injected_cfg["api_version"] = self.kwargs["api_version"]
            if "api_tools" in self.kwargs:
                injected_cfg["api_tools"] = self.kwargs["api_tools"]
            if "reasoning_effort" in self.kwargs:
                injected_cfg["reasoning_effort"] = self.kwargs["reasoning_effort"]
            if "service_tier" in self.kwargs:
                injected_cfg["service_tier"] = self.kwargs["service_tier"]
            if "routing_reasoning_effort" in self.kwargs:
                injected_cfg["routing_reasoning_effort"] = self.kwargs["routing_reasoning_effort"]
            if "routing_max_tokens" in self.kwargs:
                injected_cfg["routing_max_tokens"] = self.kwargs["routing_max_tokens"]
            if self.llm_config.get("use_responses_api"):
                injected_cfg["use_responses_api"] = True
            if self.llm_config.get("compact_threshold"):
                injected_cfg["compact_threshold"] = self.llm_config["compact_threshold"]
            injected_cfg["buffer_size"] = self.task_config["tools_config"]["synthesizer"].get("buffer_size")
            injected_cfg["language"] = self.language

            llm_agent = GraphAgent(injected_cfg)
            logger.info("Graph agent created with rag-proxy-server support")
        elif agent_type == "knowledgebase_agent":
            logger.info("Setting up knowledge agent with rag-proxy-server support")
            llm_config = self.task_config["tools_config"]["llm_agent"].get("llm_config", {})
            rag_server_url = self.kwargs.get("rag_server_url", os.getenv("RAG_SERVER_URL", "http://localhost:8000"))

            logger.info(f"Knowledge agent config: {llm_config}")
            logger.info(f"RAG server URL: {rag_server_url}")

            # Set RAG server URL in environment for KnowledgeAgent to use
            os.environ["RAG_SERVER_URL"] = rag_server_url

            # Inject provider credentials and endpoints into KnowledgeAgent config
            injected_cfg = dict(llm_config)
            if "llm_key" in self.kwargs:
                injected_cfg["llm_key"] = self.kwargs["llm_key"]
            if "base_url" in self.kwargs:
                injected_cfg["base_url"] = self.kwargs["base_url"]
            if "api_version" in self.kwargs:
                injected_cfg["api_version"] = self.kwargs["api_version"]
            if "api_tools" in self.kwargs:
                injected_cfg["api_tools"] = self.kwargs["api_tools"]
            if "reasoning_effort" in self.kwargs:
                injected_cfg["reasoning_effort"] = self.kwargs["reasoning_effort"]
            if "service_tier" in self.kwargs:
                injected_cfg["service_tier"] = self.kwargs["service_tier"]
            if self.llm_config.get("use_responses_api"):
                injected_cfg["use_responses_api"] = True
            if self.llm_config.get("compact_threshold"):
                injected_cfg["compact_threshold"] = self.llm_config["compact_threshold"]
            injected_cfg["buffer_size"] = self.task_config["tools_config"]["synthesizer"].get("buffer_size")
            injected_cfg["language"] = self.language

            llm_agent = KnowledgeBaseAgent(injected_cfg)
            logger.info("Knowledge agent created with rag-proxy-server support")
        else:
            raise f"{agent_type} Agent type is not created yet"
        return llm_agent

    def __setup_tasks(self, llm=None, agent_type=None, assistant_config=None):
        if self.task_config["task_type"] == "conversation" and not self.__is_multiagent():
            self.tools["llm_agent"] = self.__get_agent_object(llm, agent_type, assistant_config)
        elif self.__is_multiagent():
            return self.__get_agent_object(llm, agent_type, assistant_config)
        elif self.task_config["task_type"] == "extraction":
            logger.info("Setting up extraction agent")
            self.tools["llm_agent"] = ExtractionContextualAgent(llm, prompt=self.system_prompt)
            self.extracted_data = None
        elif self.task_config["task_type"] == "summarization":
            logger.info("Setting up summarization agent")
            self.tools["llm_agent"] = SummarizationContextualAgent(llm, prompt=self.system_prompt)
            self.summarized_data = None
        logger.info("prompt and config setup completed")

    ########################
    # Helper methods
    ########################

    def __get_final_prompt(self, prompt, today, current_time, current_timezone):
        enriched_prompt = prompt
        if self.context_data is not None:
            enrich_context_with_time_variables(self.context_data, current_timezone)
            enriched_prompt = update_prompt_with_context(enriched_prompt, self.context_data)
        notes = "### Note:\n"
        if self._is_conversation_task() and self.use_fillers:
            notes += f"1.{FILLER_PROMPT}\n"
        return f"{enriched_prompt}\n{notes}\n{DATE_PROMPT.format(today, current_time, current_timezone)}"

    async def load_prompt(self, assistant_name, task_id, local, **kwargs):
        if self.task_config["task_type"] == "webhook":
            return

        agent_type = self.task_config["tools_config"]["llm_agent"].get("agent_type", "simple_llm_agent")
        self.is_local = local
        if task_id == 0:
            if (
                self.context_data
                and "recipient_data" in self.context_data
                and self.context_data["recipient_data"]
                and self.context_data["recipient_data"].get("timezone", None)
            ):
                self.timezone = pytz.timezone(self.context_data["recipient_data"]["timezone"])
        current_date, current_time = get_date_time_from_timezone(self.timezone)

        prompt_responses = kwargs.get("prompt_responses", None)
        if not prompt_responses:
            prompt_responses = await get_prompt_responses(assistant_id=self.assistant_id, local=self.is_local)

        current_task = "task_{}".format(task_id + 1)
        if self.__is_multiagent():
            logger.info(
                f"Getting {current_task} from prompt responses of type {type(prompt_responses)}, prompt responses key {prompt_responses.keys()}"
            )
            prompts = prompt_responses.get(current_task, None)
            self.prompt_map = {}
            for agent in self.task_config["tools_config"]["llm_agent"]["llm_config"]["agent_map"]:
                prompt = prompts[agent]["system_prompt"]
                prompt = self.__prefill_prompts(self.task_config, prompt, self.task_config["task_type"])
                prompt = self.__get_final_prompt(prompt, current_date, current_time, self.timezone)
                if agent == self.task_config["tools_config"]["llm_agent"]["llm_config"]["default_agent"]:
                    self.system_prompt = {"role": "system", "content": prompt}
                self.prompt_map[agent] = prompt
            logger.info(f"Initialised prompt dict {self.prompt_map}, Set default prompt {self.system_prompt}")
        else:
            self.prompts = self.__prefill_prompts(
                self.task_config, prompt_responses.get(current_task, None), self.task_config["task_type"]
            )

        if "system_prompt" in self.prompts:
            # This isn't a graph based agent
            enriched_prompt = self.prompts["system_prompt"]
            if self.context_data and self.context_data.get("recipient_data", {}).get("call_sid"):
                self.call_sid = self.context_data["recipient_data"]["call_sid"]

            enriched_prompt = structure_system_prompt(
                self.prompts["system_prompt"],
                self.run_id,
                self.assistant_id,
                self.call_sid,
                self.context_data,
                self.timezone,
                self.is_web_based_call,
            )

            notes = ""
            if self._is_conversation_task() and self.use_fillers:
                notes = "### Note:\n"
                notes += f"1.{FILLER_PROMPT}\n"

            final_prompt = f"\n## Agent Prompt:\n\n{enriched_prompt}\n{notes}\n\n## Transcript:\n"
            self.prompts["system_prompt"] = final_prompt

            self.system_prompt = {"role": "system", "content": final_prompt}
        else:
            self.system_prompt = {"role": "system", "content": ""}

        self.conversation_history.setup_system_prompt(self.system_prompt)

        self.multilingual_prompts = {}
        raw_multilingual = prompt_responses.get(current_task, {}).get("multilingual_prompts", {})
        if raw_multilingual and not self.__is_multiagent():
            for lang_code, lang_prompt in raw_multilingual.items():
                enriched = structure_system_prompt(
                    lang_prompt,
                    self.run_id,
                    self.assistant_id,
                    self.call_sid,
                    self.context_data,
                    self.timezone,
                    self.is_web_based_call,
                )
                notes = ""
                if self._is_conversation_task() and self.use_fillers:
                    notes = "### Note:\n"
                    notes += f"1.{FILLER_PROMPT}\n"
                self.multilingual_prompts[lang_code] = f"\n## Agent Prompt:\n\n{enriched}\n{notes}\n\n## Transcript:\n"
            logger.info(f"Loaded multilingual prompts for languages: {list(self.multilingual_prompts.keys())}")

        # If using knowledge_agent, inject the prompt into agent config so agent can read it
        try:
            if self.__is_knowledgebase_agent() and "llm_agent" in self.task_config["tools_config"]:
                if "llm_config" in self.task_config["tools_config"]["llm_agent"]:
                    self.task_config["tools_config"]["llm_agent"]["llm_config"]["prompt"] = self.system_prompt[
                        "content"
                    ]
        except Exception as e:
            logger.error(f"Failed to inject prompt into knowledge agent config: {e}")

    def __prefill_prompts(self, task, prompt, task_type):
        if (
            self.context_data
            and "recipient_data" in self.context_data
            and self.context_data["recipient_data"]
            and self.context_data["recipient_data"].get("timezone", None)
        ):
            self.timezone = pytz.timezone(self.context_data["recipient_data"]["timezone"])
        current_date, current_time = get_date_time_from_timezone(self.timezone)

        if not prompt and task_type in ("extraction", "summarization"):
            if task_type == "extraction":
                extraction_json = (
                    task.get("tools_config").get("llm_agent", {}).get("llm_config", {}).get("extraction_json")
                )
                prompt = EXTRACTION_PROMPT.format(current_date, current_time, self.timezone, extraction_json)
                return {"system_prompt": prompt}
            elif task_type == "summarization":
                return {"system_prompt": SUMMARIZATION_PROMPT}
        return prompt

    def __process_stop_words(self, text_chunk, meta_info):
        # THis is to remove stop words. Really helpful in smaller 7B models
        if "end_of_llm_stream" in meta_info and meta_info["end_of_llm_stream"] and "user" in text_chunk[-5:].lower():
            if text_chunk[-5:].lower() == "user:":
                text_chunk = text_chunk[:-5]
            elif text_chunk[-4:].lower() == "user":
                text_chunk = text_chunk[:-4]

        # index = text_chunk.find("AI")
        # if index != -1:
        #     text_chunk = text_chunk[index+2:]
        return text_chunk

    def update_transcript_for_interruption(self, original_stream, heard_text):
        """Trim original response to match what was actually heard."""
        if original_stream is None:
            return heard_text.strip() if heard_text else None

        if not heard_text or not heard_text.strip():
            return ""

        heard_text = heard_text.strip()

        # Try exact match
        index = original_stream.find(heard_text)
        if index != -1:
            return original_stream[: index + len(heard_text)]

        # Try progressively shorter prefixes (handles synthesizer trailing spaces)
        if len(heard_text) > 3 and original_stream[:3] == heard_text[:3]:
            for i in range(len(heard_text), 0, -1):
                partial = heard_text[:i].strip()
                if partial and original_stream.startswith(partial):
                    return partial

        return heard_text

    async def sync_history(self, mark_events_data, interruption_processed_at):
        """Sync history to reflect only what was actually spoken. Uses confirmed text or falls back to pending marks."""
        try:
            response_heard = self.tools["input"].response_heard_by_user
            logger.info(f"sync_history: response_heard len={len(response_heard) if response_heard else 0}")
            if response_heard:
                logger.info(f"response_heard (last 10 chars): {response_heard[-10:]}")

            if not response_heard:
                pending_marks = [{"mark_id": k, "mark_data": v} for k, v in mark_events_data]
                pending_chunks = []
                for mark in pending_marks:
                    mark_data = mark.get("mark_data", {})
                    mark_type = mark_data.get("type", "")
                    text = mark_data.get("text_synthesized", "")
                    if mark_type in ["pre_mark_message", "backchanneling"] or not text:
                        continue
                    pending_chunks.append(
                        {"text": text, "duration": mark_data.get("duration", 0), "sent_ts": mark_data.get("sent_ts", 0)}
                    )

                if pending_chunks:
                    first_sent_ts = pending_chunks[0].get("sent_ts", 0)
                    if first_sent_ts > 0:
                        time_since_first_send = interruption_processed_at - first_sent_ts
                        actual_play_time = max(0, time_since_first_send)
                    else:
                        elapsed_time = interruption_processed_at - self.tools["input"].get_current_mark_started_time()
                        actual_play_time = max(0, elapsed_time)

                    played_text = []
                    cumulative_duration = 0
                    for chunk in pending_chunks:
                        if cumulative_duration >= actual_play_time:
                            break
                        chunk_duration = chunk["duration"]
                        if cumulative_duration + chunk_duration <= actual_play_time:
                            played_text.append(chunk["text"])
                        else:
                            remaining_time = actual_play_time - cumulative_duration
                            proportion = remaining_time / chunk_duration if chunk_duration > 0 else 0
                            char_count = int(len(chunk["text"]) * proportion)
                            partial_text = chunk["text"][:char_count]
                            if partial_text and char_count < len(chunk["text"]):
                                last_space = partial_text.rfind(" ")
                                if last_space > 0:
                                    partial_text = partial_text[:last_space]
                            if partial_text:
                                played_text.append(partial_text)
                        cumulative_duration += chunk_duration

                    if played_text:
                        response_heard = "".join(played_text)
                        logger.info(
                            f"Estimated played text (last 10 chars): {response_heard[-10:]}, len={len(response_heard)}"
                        )
                else:
                    logger.info("No pending content marks found to estimate played text")
                    # No pending content marks - response likely completed normally
                    return

            self.conversation_history.sync_after_interruption(response_heard, self.update_transcript_for_interruption)
            self.conversation_history.sync_interim_after_interruption(
                response_heard, self.update_transcript_for_interruption
            )
            self._invalidate_response_chain()

        except Exception as e:
            logger.error(f"sync_history failed: {e}")
            import traceback

            traceback.print_exc()

    async def __cleanup_downstream_tasks(self):
        current_ts = time.time()
        logger.info(f"Cleaning up downstream task")
        start_time = time.time()
        await self.tools["output"].handle_interruption()
        await self.tools["synthesizer"].handle_interruption()

        if self.generate_precise_transcript:
            await self.sync_history(self.mark_event_meta_data.fetch_cleared_mark_event_data().items(), current_ts)
            self.tools["input"].reset_response_heard_by_user()

        self.interruption_manager.invalidate_pending_responses()
        self.response_in_pipeline = False
        await self.tools["synthesizer"].flush_synthesizer_stream()

        # Stop the output loop first so that we do not transmit anything else
        if self.output_task is not None:
            logger.info(f"Cancelling output task")
            self.output_task.cancel()
        self.output_task = None

        if self.llm_task is not None:
            logger.info(f"Cancelling LLM Task")
            self.llm_task.cancel()
            self.llm_task = None

        if self.first_message_task is not None:
            logger.info("Cancelling first message task")
            self.first_message_task.cancel()
            self.first_message_task = None

        self.voicemail_handler.cancel_task()

        # self.synthesizer_task.cancel()
        # self.synthesizer_task = asyncio.create_task(self.__listen_synthesizer())
        for task in self.synthesizer_tasks:
            task.cancel()
        self.synthesizer_tasks = []

        logger.info(f"Synth Task cancelled seconds")
        if not self.buffered_output_queue.empty():
            logger.info(f"Output queue was not empty and hence emptying it")
            self.buffered_output_queue = asyncio.Queue()

        self._turn_audio_flushed.set()

        # restart output task
        self.output_task = asyncio.create_task(self.__process_output_loop())
        self.started_transmitting_audio = False  # Since we're interrupting we need to stop transmitting as well
        self.last_transmitted_timestamp = time.time()
        logger.info(f"Cleaning up downstream tasks. Time taken to send a clear message {time.time() - start_time}")

    def __get_updated_meta_info(self, meta_info=None):
        # This is used in case there's silence from callee's side
        if meta_info is None:
            meta_info = self.tools["transcriber"].get_meta_info()
            logger.info(f"Metainfo {meta_info}")
        meta_info_copy = meta_info.copy()

        new_sequence_id = self.interruption_manager.get_next_sequence_id()
        meta_info_copy["sequence_id"] = new_sequence_id
        meta_info_copy["turn_id"] = self.interruption_manager.get_turn_id()

        return meta_info_copy

    def _extract_sequence_and_meta(self, message):
        sequence, meta_info = None, None
        if isinstance(message, dict) and "meta_info" in message:
            self._set_call_details(message)
            meta_info = message["meta_info"]
            sequence = meta_info.get("sequence", 0)
        return sequence, meta_info

    def _is_extraction_task(self):
        return self.task_config["task_type"] == "extraction"

    def _is_summarization_task(self):
        return self.task_config["task_type"] == "summarization"

    def _is_conversation_task(self):
        return self.task_config["task_type"] == "conversation"

    def _get_next_step(self, sequence, origin):
        try:
            return next(
                (
                    self.pipelines[sequence][i + 1]
                    for i in range(len(self.pipelines[sequence]) - 1)
                    if self.pipelines[sequence][i] == origin
                ),
                "output",
            )
        except Exception as e:
            logger.error(f"Error getting next step: {e}")

    def _set_call_details(self, message):
        if (
            self.call_sid is not None
            and self.stream_sid is not None
            and "call_sid" not in message["meta_info"]
            and "stream_sid" not in message["meta_info"]
        ):
            return

        if "call_sid" in message.get("meta_info", {}):
            self.call_sid = message["meta_info"]["call_sid"]
        if "stream_sid" in message.get("meta_info", {}):
            self.stream_sid = message["meta_info"]["stream_sid"]

    async def _process_followup_task(self, message=None):
        if self.task_config["task_type"] == "webhook":
            logger.info(f"Input patrameters {self.input_parameters}")
            extraction_details = self.input_parameters.get("extraction_details", {})
            logger.info(f"DOING THE POST REQUEST TO WEBHOOK {extraction_details}")
            self.webhook_response = await self.tools["webhook_agent"].execute(extraction_details)
            logger.info(f"Response from the server {self.webhook_response}")
        else:
            message = format_messages(
                self.input_parameters["messages"], include_tools=True
            )  # Remove the initial system prompt
            self.history.append({"role": "user", "content": message})

            start_time = time.time()
            try:
                json_data = await self.tools["llm_agent"].generate(self.history)
            except BolnaComponentError:
                raise
            except Exception as e:
                raise LLMError(
                    str(e), provider=self.llm_config.get("provider"), model=self.llm_config.get("model")
                ) from e
            latency_ms = (time.time() - start_time) * 1000

            if self.task_config["task_type"] == "summarization":
                self.summarized_data = json_data["summary"]
                self.llm_latencies.other_latencies.append(
                    {
                        "type": "summarization",
                        "latency_ms": latency_ms,
                        "model": LLM_DEFAULT_CONFIGS["summarization"]["model"],
                        "provider": LLM_DEFAULT_CONFIGS["summarization"]["provider"],
                    }
                )
            else:
                json_data = clean_json_string(json_data)
                if type(json_data) is not dict:
                    json_data = json.loads(json_data)
                self.extracted_data = json_data
                self.llm_latencies.other_latencies.append(
                    {
                        "type": "extraction",
                        "latency_ms": latency_ms,
                        "model": LLM_DEFAULT_CONFIGS["extraction"]["model"],
                        "provider": LLM_DEFAULT_CONFIGS["extraction"]["provider"],
                    }
                )

    # This observer works only for messages which have sequence_id != -1
    def final_chunk_played_observer(self, is_final_chunk_played):
        logger.info(f"Updating last_transmitted_timestamp")
        self.last_transmitted_timestamp = time.time()

    async def agent_hangup_observer(self, is_agent_hangup):
        logger.info(f"agent_hangup_observer triggered with is_agent_hangup = {is_agent_hangup}")
        if is_agent_hangup:
            self.tools["output"].set_hangup_sent()
            await self.__process_end_of_conversation()

    async def wait_for_current_message(self):
        try:
            await asyncio.wait_for(self._turn_audio_flushed.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            logger.warning("wait_for_current_message: synth pipeline flush timed out after 3s")

        start_time = time.time()
        while not self.conversation_ended:
            elapsed = time.time() - start_time
            if elapsed > self.hangup_mark_event_timeout:
                mark_events = self.mark_event_meta_data.mark_event_meta_data
                logger.warning(
                    f"wait_for_current_message timed out after {self.hangup_mark_event_timeout}s with {len(mark_events)} remaining marks"
                )
                break

            mark_events = self.mark_event_meta_data.mark_event_meta_data
            mark_items_list = [{"mark_id": k, "mark_data": v} for k, v in mark_events.items()]
            logger.info(f"current_list: {mark_items_list}")

            if not mark_items_list:
                break

            first_item = mark_items_list[0]["mark_data"]
            if len(mark_items_list) == 1 and first_item.get("type") == "pre_mark_message":
                break

            # plivo mark_event bug
            if len(mark_items_list) == 2:
                second_item = mark_items_list[1]["mark_data"]
                if (
                    first_item.get("type") == "agent_hangup"
                    and first_item.get("text_synthesized") == ""
                    and second_item.get("type") == "pre_mark_message"
                ):
                    break

            if first_item.get("text_synthesized") and first_item.get("is_final_chunk") is True:
                break

            remaining = self.hangup_mark_event_timeout - elapsed
            self.mark_event_meta_data.mark_changed.clear()
            try:
                await asyncio.wait_for(self.mark_event_meta_data.mark_changed.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pass  # re-enters loop, hits timeout check at top
        return

    async def inject_digits_to_conversation(self) -> None:
        while True:
            try:
                dtmf_digits = await self.queues["dtmf"].get()
                logger.info(f"DTMF collected {dtmf_digits}")

                dtmf_message = "dtmf_number: " + dtmf_digits
                base_meta_info = {
                    "io": self.tools["input"].io_provider,
                    "type": "text",
                    "sequence": 0,
                    "origin": "dtmf",
                }
                meta_info = self.__get_updated_meta_info(base_meta_info)
                await self._handle_transcriber_output("llm", dtmf_message, meta_info)
                logger.info(f"DTMF LLM processing triggered with sequence_id={meta_info['sequence_id']}")
            except Exception as e:
                logger.info(f"DTMF LLM processing triggered with exception {e}")

    async def __process_end_of_conversation(self, web_call_timeout=False):
        if self._end_of_conversation_in_progress or self.conversation_ended:
            logger.info("__process_end_of_conversation: Already in progress or ended, skipping duplicate call")
            return

        self._end_of_conversation_in_progress = True
        logger.info("Got end of conversation. I'm stopping now")

        await self.wait_for_current_message()

        # Check completion of agent_hangup_message sent from output
        # Only wait for hangup chunk if a hangup message was actually queued
        while self.hangup_triggered and self.hangup_message_queued:
            try:
                if self.tools["output"].hangup_sent():
                    logger.info("final hangup chunk is now sent. Breaking now")
                    break
                else:
                    logger.info("final hangup chunk has not been sent yet")
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Error while checking queue: {e}", exc_info=True)
                break

        if self.hangup_message_queued and not web_call_timeout:
            self.history.append({"role": "assistant", "content": self.call_hangup_message})

        self.conversation_ended = True
        self.ended_by_assistant = True

        # Cancel any running LLM / function-call tasks so they don't add
        # phantom responses to the transcript after the call has ended.
        if self.llm_task is not None and not self.llm_task.done():
            logger.info("__process_end_of_conversation: Cancelling LLM task")
            self.llm_task.cancel()
            self.llm_task = None

        # Close output handler to prevent sends after websocket close
        if "output" in self.tools and self.tools["output"] is not None:
            self.tools["output"].close()

        await self.tools["input"].stop_handler()
        logger.info("Stopped input handler")
        if "transcriber" in self.tools and not self.turn_based_conversation:
            logger.info("Stopping transcriber")
            await self.tools["transcriber"].toggle_connection()
            await asyncio.sleep(2)  # Making sure whatever message was passed is over

        self.voicemail_handler.cancel_task()

    def __update_preprocessed_tree_node(self):
        logger.info(f"It's a preprocessed flow and hence updating current node")
        self.tools["llm_agent"].update_current_node()

    ##############################################################
    # LLM task
    ##############################################################
    async def _handle_llm_output(
        self, next_step, text_chunk, should_bypass_synth, meta_info, is_filler=False, is_function_call=False
    ):
        if "request_id" not in meta_info:
            meta_info["request_id"] = str(uuid.uuid4())

        if not self.stream and not is_filler:
            first_buffer_latency = time.time() - meta_info["llm_start_time"]
            meta_info["llm_first_buffer_generation_latency"] = first_buffer_latency

        elif is_filler:
            logger.info(f"It's a filler message and hence adding required metadata")
            meta_info["origin"] = "classifier"
            meta_info["cached"] = True
            meta_info["local"] = True
            meta_info["message_category"] = "filler"

        if next_step == "synthesizer" and not should_bypass_synth:
            if not text_chunk or not text_chunk.strip():
                return
            self._turn_audio_flushed.clear()
            task = asyncio.create_task(self._synthesize(create_ws_data_packet(text_chunk, meta_info)))
            self.synthesizer_tasks.append(asyncio.ensure_future(task))
        elif self.tools["output"] is not None:
            logger.info("Synthesizer not the next step and hence simply returning back")
            overall_time = time.time() - meta_info["llm_start_time"]
            # self.history = copy.deepcopy(self.interim_history)
            if is_function_call:
                bos_packet = create_ws_data_packet("<beginning_of_stream>", meta_info)
                await self.tools["output"].handle(bos_packet)
                await self.tools["output"].handle(create_ws_data_packet(text_chunk, meta_info))
                eos_packet = create_ws_data_packet("<end_of_stream>", meta_info)
                await self.tools["output"].handle(eos_packet)
            else:
                await self.tools["output"].handle(create_ws_data_packet(text_chunk, meta_info))

    async def _process_conversation_preprocessed_task(self, message, sequence, meta_info):
        if self.task_config["tools_config"]["llm_agent"]["agent_flow_type"] == "preprocessed":
            messages = self.conversation_history.get_copy()
            # TODO revisit this
            messages.append({"role": "user", "content": message["data"]})
            logger.info(f"Starting LLM Agent {messages}")
            # Expose get current classification_response method from the agent class and use it for the response log
            convert_to_request_log(
                message=format_messages(messages, use_system_prompt=True),
                meta_info=meta_info,
                component=LogComponent.LLM,
                direction=LogDirection.REQUEST,
                model=self.llm_agent_config["model"],
                is_cached=True,
                run_id=self.run_id,
            )
            async for next_state in self.tools["llm_agent"].generate(messages, label_flow=self.label_flow):
                if next_state == "<end_of_conversation>":
                    meta_info["end_of_conversation"] = True
                    self.buffered_output_queue.put_nowait(create_ws_data_packet("<end_of_conversation>", meta_info))
                    return

                logger.info(f"Text chunk {next_state['text']}")
                # TODO revisit this
                messages.append({"role": "assistant", "content": next_state["text"]})
                self.synthesizer_tasks.append(
                    asyncio.create_task(
                        self._synthesize(create_ws_data_packet(next_state["audio"], meta_info, is_md5_hash=True))
                    )
                )
            logger.info(f"Interim history after the LLM task {messages}")
            self.llm_response_generated = True
            self.conversation_history.sync_interim(messages)

    async def _process_conversation_formulaic_task(self, message, sequence, meta_info):
        llm_response = ""
        logger.info("Agent flow is formulaic and hence moving smoothly")
        async for text_chunk in self.tools["llm_agent"].generate(self.history):
            if is_valid_md5(text_chunk):
                self.synthesizer_tasks.append(
                    asyncio.create_task(
                        self._synthesize(create_ws_data_packet(text_chunk, meta_info, is_md5_hash=True))
                    )
                )
            else:
                # TODO Make it more modular
                llm_response += " " + text_chunk
                next_step = self._get_next_step(sequence, "llm")
                if next_step == "synthesizer":
                    self.synthesizer_tasks.append(
                        asyncio.create_task(self._synthesize(create_ws_data_packet(text_chunk, meta_info)))
                    )
                else:
                    logger.info(f"Sending output text {sequence}")
                    await self.tools["output"].handle(create_ws_data_packet(text_chunk, meta_info))
                    self.synthesizer_tasks.append(
                        asyncio.create_task(
                            self._synthesize(create_ws_data_packet(text_chunk, meta_info, is_md5_hash=False))
                        )
                    )

    async def __execute_function_call(
        self, url, method, param, api_token, headers, model_args, meta_info, next_step, called_fun, **resp
    ):
        self.check_if_user_online = False
        function_call_log = None

        if "execution_id" in resp and resp["execution_id"] != self.run_id:
            logger.warning(f"Correcting LLM-generated execution_id: '{resp['execution_id']}' -> '{self.run_id}'")
            resp["execution_id"] = self.run_id

        if called_fun.startswith(END_CALL_FUNCTION_PREFIX):
            reason = resp.get("reason", "")

            if self.shadow_end_call_enabled:
                logger.info(f"Shadow end_call invoked: reason={reason}, seq={meta_info.get('sequence_id')}")
                self.shadow_end_call_events.append(
                    {
                        "seq": meta_info.get("sequence_id"),
                        "reason": reason,
                        "timestamp": time.time(),
                    }
                )
                return

            logger.info(f"end_call tool invoked, reason: {reason}")
            convert_to_request_log(
                json.dumps({"called_fun": called_fun, "reason": reason}),
                meta_info,
                None,
                "function_call",
                direction="request",
                run_id=self.run_id,
            )

            # Tool-result flow: feed result back to LLM so it generates a clean goodbye
            textual_response = resp.get("textual_response", None)
            tool_result = json.dumps(
                {"status": "success", "message": "Call is ending now. Say a brief goodbye to the user."}
            )
            self.conversation_history.append_assistant(textual_response, tool_calls=resp["model_response"])
            self.conversation_history.append_tool_result(resp.get("tool_call_id", ""), tool_result)
            convert_to_request_log(
                tool_result, meta_info, None, "function_call", direction="response", run_id=self.run_id
            )

            messages = self.conversation_history.get_copy()
            convert_to_request_log(
                format_messages(messages, True),
                meta_info,
                self.llm_config["model"],
                "llm",
                direction="request",
                run_id=self.run_id,
            )

            # Generate goodbye with should_trigger_function_call=False to prevent recursion
            await self.__do_llm_generation(messages, meta_info, next_step, should_trigger_function_call=False)
            await self.wait_for_current_message()

            self.hangup_detail = "end_call_tool"
            self.call_hangup_message_config = None
            await self.process_call_hangup()
            return

        if called_fun.startswith("transfer_call"):
            self.has_transfer = True
            await asyncio.sleep(2)
            try:
                from_number = self.context_data["recipient_data"]["from_number"]
            except Exception as e:
                from_number = None

            call_sid = None
            call_transfer_number = None
            payload = {
                "call_sid": call_sid,
                "provider": self.tools["input"].io_provider,
                "stream_sid": self.stream_sid,
                "from_number": from_number,
                "execution_id": self.run_id,
                **(self.transfer_call_params or {}),
            }

            if self.tools["input"].io_provider != "default":
                call_sid = self.tools["input"].get_call_sid()
                payload["call_sid"] = call_sid

            if url is None:
                url = os.getenv("CALL_TRANSFER_WEBHOOK_URL")

                try:
                    json_function_call_params = copy.deepcopy(param)
                    if isinstance(param, str):
                        json_function_call_params = json.loads(param)
                    call_transfer_number = json_function_call_params["call_transfer_number"]
                    if call_transfer_number:
                        payload["call_transfer_number"] = call_transfer_number
                except Exception as e:
                    logger.error(f"Error in __execute_function_call {e}")

            if param is not None:
                logger.info(f"Gotten response {resp}")
                payload = {**payload, **resp}

            if self.tools["input"].io_provider != "default":
                payload["call_sid"] = self.tools["input"].get_call_sid()

            if self.tools["input"].io_provider == "default":
                mock_response = (
                    f"This is a mocked response demonstrating a successful transfer of call to {call_transfer_number}"
                )
                function_call_log = self._start_api_call_detail(
                    called_fun=called_fun,
                    url=url,
                    method="POST",
                    param=param,
                    headers={"Content-Type": "application/json"},
                    meta_info=meta_info,
                    runtime_args={
                        **self._extract_api_call_runtime_args(resp),
                        "tool_call_id": resp.get("tool_call_id", ""),
                    },
                    request_body=payload,
                    api_params=payload,
                )
                convert_to_request_log(
                    str(payload),
                    meta_info,
                    None,
                    LogComponent.FUNCTION_CALL,
                    direction=LogDirection.REQUEST,
                    run_id=self.run_id,
                )
                convert_to_request_log(
                    mock_response,
                    meta_info,
                    None,
                    LogComponent.FUNCTION_CALL,
                    direction=LogDirection.RESPONSE,
                    run_id=self.run_id,
                )
                self._finalize_api_call_detail(
                    function_call_log, response=mock_response, status_code=200, content_type="text/plain"
                )

                bos_packet = create_ws_data_packet("<beginning_of_stream>", meta_info)
                await self.tools["output"].handle(bos_packet)
                await self.tools["output"].handle(create_ws_data_packet(mock_response, meta_info))
                eos_packet = create_ws_data_packet("<end_of_stream>", meta_info)
                await self.tools["output"].handle(eos_packet)
                return

            session = await get_shared_aiohttp_session()
            logger.info(f"Sending the payload to stop the conversation {payload} url {url}")
            while self.tools["input"].is_audio_being_played_to_user():
                await asyncio.sleep(1)
            function_call_log = self._start_api_call_detail(
                called_fun=called_fun,
                url=url,
                method="POST",
                param=param,
                headers={"Content-Type": "application/json"},
                meta_info=meta_info,
                runtime_args={
                    **self._extract_api_call_runtime_args(resp),
                    "tool_call_id": resp.get("tool_call_id", ""),
                },
                request_body=payload,
                api_params=payload,
            )
            convert_to_request_log(
                str(payload),
                meta_info,
                None,
                LogComponent.FUNCTION_CALL,
                direction=LogDirection.REQUEST,
                is_cached=False,
                run_id=self.run_id,
            )
            async with session.post(url, json=payload) as response:
                response_text = await response.text()
                logger.info(f"Response from the server after call transfer: {response_text}")
                convert_to_request_log(
                    str(response_text),
                    meta_info,
                    None,
                    LogComponent.FUNCTION_CALL,
                    direction=LogDirection.RESPONSE,
                    is_cached=False,
                    run_id=self.run_id,
                )
                self._finalize_api_call_detail(
                    function_call_log,
                    response=response_text,
                    status_code=response.status,
                    content_type=response.headers.get("Content-Type"),
                )
                return

        if called_fun == "switch_language":
            language_label = resp.get("language", "")

            # If the requested language is already active, skip handoff and switch entirely
            if language_label == self.language:
                logger.info(
                    f"switch_language: '{language_label}' is already the active language, skipping handoff and switch"
                )
                function_response = f"Already speaking in {language_label}, no switch needed"

                textual_response = resp.get("textual_response", None)
                if not textual_response:
                    self.conversation_history.append_assistant(textual_response, tool_calls=resp["model_response"])
                else:
                    self.conversation_history.attach_tool_calls_to_last_response(resp["model_response"])
                self.conversation_history.append_tool_result(resp.get("tool_call_id", ""), function_response)
                convert_to_request_log(
                    function_response, meta_info, None, "function_call", direction="response", run_id=self.run_id
                )

                messages = self.conversation_history.get_copy()
                await self.__do_llm_generation(
                    messages, meta_info, next_step, should_bypass_synth=False, should_trigger_function_call=True
                )
                self.execute_function_call_task = None
                return

            # Only wait if audio is currently playing
            if not self._turn_audio_flushed.is_set():
                await self.wait_for_current_message()

            # Synthesize handoff message with CURRENT voice before switching
            handoff_template = self.switch_handoff_messages.get(self.language, "")
            if handoff_template:
                target_agent_name = self._get_voice_name_for_label(language_label)
                language_display = LANGUAGE_NAMES.get(language_label, language_label)
                handoff_text = handoff_template.replace("{agent_name}", target_agent_name).replace(
                    "{language}", language_display
                )
                meta_info_handoff = {
                    "io": self.tools["output"].get_provider(),
                    "request_id": str(uuid.uuid4()),
                    "cached": False,
                    "sequence_id": -1,
                    "format": "pcm",
                    "message_category": "handoff",
                    "end_of_llm_stream": True,
                    "text": handoff_text,
                }
                self._turn_audio_flushed.clear()
                await self._synthesize(create_ws_data_packet(handoff_text, meta_info=meta_info_handoff))
                await self.wait_for_current_message()
                self.conversation_history.append_assistant(handoff_text)

            try:
                await self.switch_language(language_label)
                function_response = f"Switched to {language_label}"
            except ValueError as e:
                function_response = f"Failed to switch language: {e}"

            textual_response = resp.get("textual_response", None)
            self.check_if_user_online = self.conversation_config.get("check_if_user_online", True)
            if not textual_response:
                self.conversation_history.append_assistant(textual_response, tool_calls=resp["model_response"])
            else:
                self.conversation_history.attach_tool_calls_to_last_response(resp["model_response"])
            self.conversation_history.append_tool_result(resp.get("tool_call_id", ""), function_response)
            convert_to_request_log(
                function_response, meta_info, None, "function_call", direction="response", run_id=self.run_id
            )

            messages = self.conversation_history.get_copy()
            await self.__do_llm_generation(
                messages, meta_info, next_step, should_bypass_synth=False, should_trigger_function_call=True
            )
            self.execute_function_call_task = None
            return

        await self.wait_for_current_message()

        if self.hangup_triggered or self.conversation_ended:
            logger.info(
                f"__execute_function_call: Aborting before API call — hangup_triggered={self.hangup_triggered}, conversation_ended={self.conversation_ended}"
            )
            return

        runtime_args = self._extract_api_call_runtime_args(resp)
        try:
            prepared_request = prepare_api_request(param, api_token, headers, **runtime_args)
        except Exception as exc:
            logger.warning(f"Could not prepare structured function call request for logging: {exc}")
            prepared_request = {
                "request_body": None,
                "api_params": None,
                "headers": headers,
            }
        function_call_log = self._start_api_call_detail(
            called_fun=called_fun,
            url=url,
            method=method,
            param=param,
            headers=prepared_request["headers"],
            meta_info=meta_info,
            runtime_args=runtime_args,
            request_body=prepared_request["request_body"],
            api_params=prepared_request["api_params"],
        )
        response = await trigger_api(
            url=url,
            method=method.lower(),
            param=param,
            api_token=api_token,
            headers_data=headers,
            meta_info=meta_info,
            run_id=self.run_id,
            return_response_metadata=True,
            **resp,
        )
        self._finalize_api_call_detail(
            function_call_log,
            response=response.get("body"),
            status_code=response.get("status_code"),
            content_type=response.get("content_type"),
            error=response.get("error"),
        )
        function_response = str(response.get("body"))
        get_res_keys, get_res_values = await computed_api_response(function_response)

        # Merge API response data into context_data for routing decisions
        if self.__is_graph_agent():
            try:
                response_data = (
                    json.loads(function_response) if isinstance(function_response, str) else function_response
                )
                if isinstance(response_data, dict):
                    # Update task manager's context_data
                    if self.context_data is None:
                        self.context_data = {}
                    self.context_data.update(response_data)
                    # Update graph agent's context_data for routing
                    if hasattr(self.tools.get("llm_agent"), "context_data"):
                        self.tools["llm_agent"].context_data.update(response_data)
                    logger.info(f"Merged API response into context_data: {list(response_data.keys())}")
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug(f"Could not parse API response as JSON for context merge: {e}")
        if called_fun.startswith("check_availability_of_slots") and (
            not get_res_values or (len(get_res_values) == 1 and len(get_res_values[0]) == 0)
        ):
            set_response_prompt = []
        elif called_fun.startswith("book_appointment") and "id" not in get_res_keys:
            if get_res_values and get_res_values[0] == "no_available_users_found_error":
                function_response = "Sorry, the host isn't available at this time. Are you available at any other time?"
            set_response_prompt = []
        else:
            set_response_prompt = function_response

        textual_response = resp.get("textual_response", None)
        if not textual_response:
            self.conversation_history.append_assistant(textual_response, tool_calls=resp["model_response"])
        else:
            self.conversation_history.attach_tool_calls_to_last_response(resp["model_response"])
        self.conversation_history.append_tool_result(resp.get("tool_call_id", ""), function_response)

        logger.info(f"Logging function call parameters ")
        convert_to_request_log(
            function_response,
            meta_info,
            None,
            LogComponent.FUNCTION_CALL,
            direction=LogDirection.RESPONSE,
            is_cached=False,
            run_id=self.run_id,
        )

        messages = self.conversation_history.get_copy()
        convert_to_request_log(
            format_messages(messages, use_system_prompt=True, include_tools=True),
            meta_info,
            self.llm_config["model"],
            LogComponent.LLM,
            direction=LogDirection.REQUEST,
            is_cached=False,
            run_id=self.run_id,
        )
        self.check_if_user_online = self.conversation_config.get("check_if_user_online", True)

        if not called_fun.startswith("transfer_call"):
            should_bypass_synth = meta_info.get("bypass_synth", False)
            await self.__do_llm_generation(
                messages,
                meta_info,
                next_step,
                should_bypass_synth=should_bypass_synth,
                should_trigger_function_call=True,
            )

        self.execute_function_call_task = None

    def __store_into_history(
        self,
        meta_info,
        messages,
        llm_response,
        should_trigger_function_call=False,
        input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        cached_tokens=None,
        reasoning_content=None,
    ):
        self.llm_response_generated = True
        convert_to_request_log(
            message=llm_response,
            meta_info=meta_info,
            component=LogComponent.LLM,
            direction=LogDirection.RESPONSE,
            model=self.llm_config["model"],
            run_id=self.run_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
            reasoning_content=reasoning_content,
        )
        if should_trigger_function_call:
            logger.info(f"There was a function call and need to make that work")
            self.conversation_history.append_assistant(llm_response)
        else:
            messages.append({"role": "assistant", "content": llm_response})
            self.conversation_history.append_assistant(llm_response)
            self.conversation_history.sync_interim(messages)

    async def __do_llm_generation(
        self, messages, meta_info, next_step, should_bypass_synth=False, should_trigger_function_call=False
    ):
        if self.hangup_triggered or self.conversation_ended:
            logger.info(
                f"__do_llm_generation: Skipping — hangup_triggered={self.hangup_triggered}, conversation_ended={self.conversation_ended}"
            )
            return

        # Clear stale end_of_llm_stream from previous generation so only
        # the final chunk of THIS generation carries the flag.
        meta_info.pop("end_of_llm_stream", None)

        # Reset response tracking for new turn
        if self.generate_precise_transcript:
            self.tools["input"].reset_response_heard_by_user()

        llm_response, function_tool, function_tool_message = "", "", ""
        actual_input_tokens, actual_output_tokens, actual_reasoning_tokens, actual_cached_tokens = (
            None,
            None,
            None,
            None,
        )
        actual_reasoning_content = None
        synthesize = True
        if should_bypass_synth:
            synthesize = False

        # Inject language instruction if detection complete
        messages = self._inject_language_instruction(messages)

        # Pass detected language to LLM for pre_call_message selection
        meta_info["detected_language"] = self.language

        try:
            async for llm_message in self.tools["llm_agent"].generate(
                messages, synthesize=synthesize, meta_info=meta_info
            ):
                if (
                    isinstance(llm_message, dict) and "messages" in llm_message
                ):  # custom list of messages before the llm call
                    convert_to_request_log(
                        format_messages(llm_message["messages"], use_system_prompt=True, include_tools=True),
                        meta_info,
                        self.llm_config["model"],
                        LogComponent.LLM,
                        direction=LogDirection.REQUEST,
                        is_cached=False,
                        run_id=self.run_id,
                    )
                    continue

                # Handle graph agent routing info
                if isinstance(llm_message, dict) and "routing_info" in llm_message:
                    routing_info = llm_message["routing_info"]

                    # Log routing request with tools
                    routing_messages = routing_info.get("routing_messages")
                    routing_tools = routing_info.get("routing_tools", [])
                    if routing_messages:
                        # Format tools for logging (show full descriptions with conditions)
                        tools_summary = ""
                        if routing_tools:
                            tool_lines = []
                            for t in routing_tools:
                                if "function" in t:
                                    name = t["function"]["name"]
                                    desc = t["function"].get("description", "")
                                    tool_lines.append(f"  - {name}: {desc}")
                            if tool_lines:
                                tools_summary = "\n\nAvailable transitions:\n" + "\n".join(tool_lines)

                        convert_to_request_log(
                            message=format_messages(routing_messages, use_system_prompt=True) + tools_summary,
                            meta_info=meta_info,
                            model=routing_info.get("routing_model", ""),
                            component=LogComponent.GRAPH_ROUTING,
                            direction=LogDirection.REQUEST,
                            run_id=self.run_id,
                        )

                    # Build routing response data
                    if routing_info.get("transitioned"):
                        routing_data = (
                            f"Node: {routing_info.get('previous_node', '?')} → {routing_info['current_node']}"
                        )
                    else:
                        routing_data = f"Node: {routing_info['current_node']} (no transition)"
                    if routing_info.get("extracted_params"):
                        routing_data += f" | Params: {json.dumps(routing_info['extracted_params'])}"
                    if routing_info.get("confidence") is not None:
                        routing_data += f" | Confidence: {routing_info['confidence']}"
                    if routing_info.get("reasoning"):
                        routing_data += f" | Reasoning: {routing_info['reasoning']}"
                    if routing_info.get("node_history"):
                        routing_data += f" | Flow: {' → '.join(routing_info['node_history'])}"

                    meta_info["llm_metadata"] = meta_info.get("llm_metadata") or {}
                    meta_info["llm_metadata"]["graph_routing_info"] = routing_info

                    routing_usage = routing_info.get("routing_usage") or {}
                    if routing_info.get("routing_latency_ms") is not None:
                        self.routing_latencies["turn_latencies"].append(
                            {
                                "latency_ms": routing_info["routing_latency_ms"],
                                "routing_model": routing_info.get("routing_model"),
                                "routing_provider": routing_info.get("routing_provider"),
                                "previous_node": routing_info.get("previous_node"),
                                "current_node": routing_info.get("current_node"),
                                "transitioned": routing_info.get("transitioned", False),
                                "sequence_id": meta_info.get("sequence_id"),
                                "reasoning": routing_info.get("reasoning"),
                                "confidence": routing_info.get("confidence"),
                                "input_tokens": routing_usage.get("input_tokens"),
                                "output_tokens": routing_usage.get("output_tokens"),
                                "reasoning_tokens": routing_usage.get("reasoning_tokens"),
                                "cached_tokens": routing_usage.get("cached_tokens"),
                            }
                        )

                    if routing_info.get("node_history"):
                        self.routing_latencies["node_flow"] = list(routing_info["node_history"])

                    # Log routing response
                    convert_to_request_log(
                        message=routing_data,
                        meta_info=meta_info,
                        model=routing_info.get("routing_model", ""),
                        component=LogComponent.GRAPH_ROUTING,
                        direction=LogDirection.RESPONSE,
                        run_id=self.run_id,
                        input_tokens=routing_usage.get("input_tokens"),
                        output_tokens=routing_usage.get("output_tokens"),
                        reasoning_tokens=routing_usage.get("reasoning_tokens"),
                        cached_tokens=routing_usage.get("cached_tokens"),
                    )
                    continue

                data = llm_message.data
                end_of_llm_stream = llm_message.end_of_stream
                latency = llm_message.latency
                trigger_function_call = llm_message.is_function_call
                function_tool = llm_message.function_name
                function_tool_message = llm_message.function_message

                # Capture actual token counts from any chunk that carries them
                if llm_message.input_tokens is not None:
                    actual_input_tokens = llm_message.input_tokens
                if llm_message.output_tokens is not None:
                    actual_output_tokens = llm_message.output_tokens
                if llm_message.reasoning_tokens is not None:
                    actual_reasoning_tokens = llm_message.reasoning_tokens
                if llm_message.cached_tokens is not None:
                    actual_cached_tokens = llm_message.cached_tokens
                if llm_message.reasoning_content is not None:
                    actual_reasoning_content = llm_message.reasoning_content

                if trigger_function_call:
                    logger.info(f"Triggering function call for {data}")
                    textual_response = data.textual_response if hasattr(data, "textual_response") else None
                    if textual_response:  # intentionally omitting tool_calls, which will be filled later if the tool_call flow completed (requirement from OpenAI)
                        self.__store_into_history(
                            meta_info,
                            messages,
                            textual_response,
                            should_trigger_function_call=should_trigger_function_call,
                            input_tokens=actual_input_tokens,
                            output_tokens=actual_output_tokens,
                            reasoning_tokens=actual_reasoning_tokens,
                            cached_tokens=actual_cached_tokens,
                            reasoning_content=actual_reasoning_content,
                        )
                    await self.__execute_function_call(next_step=next_step, **data.model_dump())
                    return

                if latency:
                    latency_dict = latency.model_dump()
                    previous_latency_item = (
                        self.llm_latencies.turn_latencies[-1] if self.llm_latencies.turn_latencies else None
                    )
                    if previous_latency_item and previous_latency_item.get("sequence_id") == latency_dict.get(
                        "sequence_id"
                    ):
                        self.llm_latencies.turn_latencies[-1] = latency_dict
                    else:
                        self.llm_latencies.turn_latencies.append(latency_dict)

                llm_response += " " + data

                logger.info(f"Got a response from LLM {llm_response}")
                if end_of_llm_stream:
                    meta_info["end_of_llm_stream"] = True

                if self.stream:
                    text_chunk = self.__process_stop_words(data, meta_info)

                    # A hack as during the 'await' part control passes to llm streaming function parameters
                    # So we have to make sure we've commited the filler message
                    filler_message = compute_function_pre_call_message(
                        self.language, function_tool, function_tool_message
                    )
                    # filler_message = PRE_FUNCTION_CALL_MESSAGE.get(self.language, PRE_FUNCTION_CALL_MESSAGE[DEFAULT_LANGUAGE_CODE])
                    if text_chunk == filler_message:
                        logger.info("Got a pre function call message")
                        messages.append({"role": "assistant", "content": filler_message})
                        self.conversation_history.append_assistant(filler_message)
                        self.conversation_history.sync_interim(messages)

                    await self._handle_llm_output(next_step, text_chunk, should_bypass_synth, meta_info)
        except BolnaComponentError:
            raise
        except Exception as e:
            raise LLMError(str(e), provider=self.llm_config.get("provider"), model=self.llm_config.get("model")) from e

        filler_message = compute_function_pre_call_message(self.language, function_tool, function_tool_message)
        if self.stream and llm_response != filler_message:
            self.__store_into_history(
                meta_info,
                messages,
                llm_response,
                should_trigger_function_call=should_trigger_function_call,
                input_tokens=actual_input_tokens,
                output_tokens=actual_output_tokens,
                reasoning_tokens=actual_reasoning_tokens,
                cached_tokens=actual_cached_tokens,
                reasoning_content=actual_reasoning_content,
            )
        elif not self.stream:
            llm_response = llm_response.strip()
            if self.turn_based_conversation:
                self.conversation_history.append_assistant(llm_response)
            await self._handle_llm_output(
                next_step, llm_response, should_bypass_synth, meta_info, is_function_call=should_trigger_function_call
            )
            convert_to_request_log(
                message=llm_response,
                meta_info=meta_info,
                component=LogComponent.LLM,
                direction=LogDirection.RESPONSE,
                model=self.llm_config["model"],
                run_id=self.run_id,
                input_tokens=actual_input_tokens,
                output_tokens=actual_output_tokens,
                reasoning_tokens=actual_reasoning_tokens,
                cached_tokens=actual_cached_tokens,
                reasoning_content=actual_reasoning_content,
            )

        # Collect RAG latency if present (from KnowledgeBaseAgent)
        if meta_info.get("rag_latency"):
            rag_latency = meta_info["rag_latency"]
            existing_seq_ids = [t.get("sequence_id") for t in self.rag_latencies["turn_latencies"]]
            if rag_latency.get("sequence_id") not in existing_seq_ids:
                self.rag_latencies["turn_latencies"].append(rag_latency)

    async def _process_conversation_task(self, message, sequence, meta_info):
        should_bypass_synth = "bypass_synth" in meta_info and meta_info["bypass_synth"] is True
        next_step = self._get_next_step(sequence, "llm")
        meta_info["llm_start_time"] = time.time()

        if self.turn_based_conversation:
            self.history.append({"role": "user", "content": message["data"]})
        messages = self.conversation_history.get_copy()

        # Request logs converted inside do_llm_generation for knowledgebase agent
        if not self.__is_knowledgebase_agent() and not self.__is_graph_agent():
            convert_to_request_log(
                message=format_messages(messages, use_system_prompt=True, include_tools=True),
                meta_info=meta_info,
                component=LogComponent.LLM,
                direction=LogDirection.REQUEST,
                model=self.llm_config["model"],
                run_id=self.run_id,
            )

        await self.__do_llm_generation(messages, meta_info, next_step, should_bypass_synth)
        # TODO : Write a better check for completion prompt

        # Hangup detection - now supported for all agent types including graph_agent
        if self.use_llm_to_determine_hangup and not self.turn_based_conversation:
            completion_res, metadata = await self.tools["llm_agent"].check_for_completion(
                messages, self.check_for_completion_prompt
            )

            should_hangup = (
                str(completion_res.get("hangup", "")).lower() == "yes" if isinstance(completion_res, dict) else False
            )

            # Track hangup check latency (latency returned by agent)
            self.llm_latencies.other_latencies.append(
                {
                    "type": "hangup_check",
                    "latency_ms": metadata.get("latency_ms", None),
                    "model": self.check_for_completion_llm,
                    "provider": "openai",  # TODO: Make dynamic based on provider used
                    "service_tier": metadata.get("service_tier", None),
                    "llm_host": metadata.get("llm_host", None),
                    "sequence_id": meta_info.get("sequence_id"),
                }
            )

            prompt = [
                {"role": "system", "content": self.check_for_completion_prompt},
                {"role": "user", "content": format_messages(self.history)},
            ]
            logger.info(f"##### Answer from the LLM {completion_res}")
            convert_to_request_log(
                message=format_messages(prompt, use_system_prompt=True),
                meta_info=meta_info,
                component=LogComponent.LLM_HANGUP,
                direction=LogDirection.REQUEST,
                model=self.check_for_completion_llm,
                run_id=self.run_id,
            )
            convert_to_request_log(
                message=completion_res,
                meta_info=meta_info,
                component=LogComponent.LLM_HANGUP,
                direction=LogDirection.RESPONSE,
                model=self.check_for_completion_llm,
                run_id=self.run_id,
                input_tokens=metadata.get("input_tokens"),
                output_tokens=metadata.get("output_tokens"),
                reasoning_tokens=metadata.get("reasoning_tokens"),
                cached_tokens=metadata.get("cached_tokens"),
            )

            if self.shadow_end_call_enabled and (should_hangup or self.shadow_end_call_events):
                logger.info(
                    f"hangup_after_LLMCall result: hangup={should_hangup}, seq={meta_info.get('sequence_id')}, "
                    f"shadow_end_call_events={json.dumps(self.shadow_end_call_events)}"
                )

            if should_hangup:
                if self.hangup_triggered or self.conversation_ended:
                    logger.info(f"Hangup already triggered or conversation ended, skipping duplicate hangup request")
                    return
                self.hangup_detail = HangupReason.LLM_PROMPTED_HANGUP
                await self.process_call_hangup()
                return

        self.llm_processed_request_ids.add(self.current_request_id)
        llm_response = ""

    async def process_call_hangup(self):
        # Guard: Prevent multiple concurrent hangup attempts
        if self.hangup_triggered or self.conversation_ended:
            logger.info(f"process_call_hangup: Hangup already in progress or conversation ended, skipping")
            return

        # Set immediately to prevent user interruptions from cancelling hangup
        self.hangup_triggered = True
        self.hangup_triggered_at = time.time()  # Track when hangup was triggered for timeout monitoring
        message = self.call_hangup_message if not self.voicemail_handler.detected else ""
        if not message or message.strip() == "":
            self.hangup_message_queued = False  # No hangup message to wait for
            await self.__process_end_of_conversation()
        else:
            self.hangup_message_queued = True  # Hangup message will be synthesized
            await self.wait_for_current_message()
            await self.__cleanup_downstream_tasks()
            meta_info = {
                "io": self.tools["output"].get_provider(),
                "request_id": str(uuid.uuid4()),
                "cached": False,
                "sequence_id": -1,
                "format": "pcm",
                "message_category": "agent_hangup",
                "end_of_llm_stream": True,
            }
            await self._synthesize(create_ws_data_packet(message, meta_info=meta_info))
        return

    async def _listen_llm_input_queue(self):
        logger.info(
            f"Starting listening to LLM queue as either Connected to dashboard = {self.turn_based_conversation} or  it's a textual chat agent {self.textual_chat_agent}"
        )
        while True:
            try:
                ws_data_packet = await self.queues["llm"].get()
                logger.info(f"ws_data_packet {ws_data_packet}")
                meta_info = self.__get_updated_meta_info(ws_data_packet["meta_info"])
                bos_packet = create_ws_data_packet("<beginning_of_stream>", meta_info)
                await self.tools["output"].handle(bos_packet)
                # self.interim_history = self.history.copy()
                # self.history.append({'role': 'user', 'content': ws_data_packet['data']})
                await self._run_llm_task(create_ws_data_packet(ws_data_packet["data"], meta_info))
                eos_packet = create_ws_data_packet("<end_of_stream>", meta_info)
                await self.tools["output"].handle(eos_packet)

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Something went wrong with LLM queue {e}")
                break

    async def _run_llm_task(self, message):
        sequence, meta_info = self._extract_sequence_and_meta(message)

        try:
            if self._is_extraction_task() or self._is_summarization_task():
                await self._process_followup_task(message)
            elif self._is_conversation_task():
                await self._process_conversation_task(message, sequence, meta_info)
            else:
                logger.error("unsupported task type: {}".format(self.task_config["task_type"]))
            self.llm_task = None
        except BolnaComponentError as e:
            self.response_in_pipeline = False
            await self._end_call_on_component_error(e, HangupReason.LLM_ERROR)
            raise
        except Exception as e:
            self.response_in_pipeline = False
            await self._end_call_on_component_error(
                LLMError(
                    str(e), provider=self.llm_config.get("provider", "unknown"), model=self.llm_config.get("model")
                ),
                HangupReason.LLM_ERROR,
            )

    #################################################################
    # Transcriber task
    #################################################################
    async def process_transcriber_request(self, meta_info):
        request_id = meta_info.get("request_id")
        if request_id and (not self.current_request_id or self.current_request_id != request_id):
            self.previous_request_id, self.current_request_id = self.current_request_id, request_id

        sequence = meta_info.get("sequence", 0)

        # check if previous request id is not in transmitted request id
        if self.previous_request_id is None:
            is_first_message = True
        elif self.previous_request_id not in self.llm_processed_request_ids:
            logger.info(f"Adding previous request id to LLM rejected request if")
            self.llm_rejected_request_ids.add(self.previous_request_id)
        else:
            skip_append_to_data = False
        return sequence

    def _trigger_voicemail_check(self, transcriber_message, meta_info, is_final=True):
        self.voicemail_handler.trigger_check(transcriber_message, meta_info, is_final)

    async def _handle_transcriber_output(self, next_task, transcriber_message, meta_info):
        if not self.tools["input"].welcome_message_played() and (
            self.discard_pre_welcome_utterance or len(self.conversation_history) > 2
        ):
            logger.info(f"Welcome message is playing while spoken: {transcriber_message}")
            return

        if self._speech_started_before_welcome:
            logger.info(
                f"Discarding transcript from speech that started before welcome finished: {transcriber_message}"
            )
            self._speech_started_before_welcome = False
            return

        if self.conversation_history.is_duplicate_user(transcriber_message):
            logger.info(f"Skipping duplicate transcript (same content): {transcriber_message}")
            return

        self._trigger_voicemail_check(transcriber_message, meta_info, is_final=True)

        if self.voicemail_handler.detected:
            logger.info("Voicemail already detected - skipping normal transcriber output processing")
            return

        await self.language_detector.collect_transcript(transcriber_message)

        if self.response_in_pipeline and next_task == "llm":
            self.conversation_history.pop_unheard_responses()
            self._invalidate_response_chain()
            original_message = transcriber_message
            transcriber_message = self.conversation_history.pop_and_merge_user(transcriber_message)
            if transcriber_message != original_message:
                logger.info(f"Merged transcript with unheard response: {transcriber_message}")
            await self.__cleanup_downstream_tasks()

        self.conversation_history.append_user(transcriber_message)

        convert_to_request_log(
            message=transcriber_message, meta_info=meta_info, model=self.transcriber_provider, run_id=self.run_id
        )
        if next_task == "llm":
            logger.info(f"Running llm Tasks")
            meta_info["origin"] = "transcriber"
            transcriber_package = create_ws_data_packet(transcriber_message, meta_info)

            # Cancel any existing LLM task to prevent orphaned concurrent responses
            if self.llm_task is not None and not self.llm_task.done():
                logger.info("Cancelling existing LLM task for new speech_final")
                self.llm_task.cancel()
                self.llm_task = None
                self.interruption_manager.invalidate_pending_responses()
                # Re-register the current sequence_id (already allocated by
                # __get_updated_meta_info) so the new response's audio is not blocked
                self.interruption_manager.revalidate_sequence_id(meta_info["sequence_id"])

            self.response_in_pipeline = True
            self.llm_task = asyncio.create_task(self._run_llm_task(transcriber_package))

        elif next_task == "synthesizer":
            self.synthesizer_tasks.append(
                asyncio.create_task(self._synthesize(create_ws_data_packet(transcriber_message, meta_info)))
            )
        else:
            logger.info(f"Need to separate out output task")

    async def _end_call_on_component_error(self, error, hangup_detail):
        """End the call gracefully when a critical pipeline component fails.

        Handles: CSV error logging, _component_error tracking, and triggering
        __process_end_of_conversation for immediate graceful shutdown.
        """
        if self._component_error is None:
            self._component_error = {
                "cls": type(error),
                "message": str(error),
                "provider": getattr(error, "provider", None),
                "model": getattr(error, "model", None),
            }

        # Log to CSV if not already done
        if self.run_id and not self._error_logged:
            if isinstance(error, BolnaComponentError):
                error_msg = format_error_message(error.component, error.provider or error.model or "-", str(error))
                model = error.model or error.provider or "-"
            else:
                error_msg = format_error_message("unknown", "-", str(error))
                model = "-"
            convert_to_request_log(
                error_msg,
                {"request_id": self.task_id, "sequence_id": None},
                model=model,
                component=LogComponent.ERROR,
                direction=LogDirection.ERROR,
                is_cached=False,
                run_id=self.run_id,
            )
            self._error_logged = True

        # Trigger graceful shutdown
        if not self.conversation_ended and not self._end_of_conversation_in_progress:
            self.hangup_detail = hangup_detail
            await self.__process_end_of_conversation()

    async def _log_transcriber_connection_error(self, connection_error):
        if connection_error:
            provider = self.task_config["tools_config"]["transcriber"].get("provider", "unknown")
            await self._end_call_on_component_error(
                TranscriberError(connection_error, provider=provider), HangupReason.TRANSCRIBER_CONNECTION_ERROR
            )

    async def _listen_transcriber(self):
        temp_transcriber_message = ""
        try:
            while True:
                message = await self.transcriber_output_queue.get()
                logger.info(f"Message from the transcriber class {message}")

                if self.hangup_triggered:
                    if message["data"] == "transcriber_connection_closed":
                        logger.info(f"Transcriber connection has been closed")
                        self.transcriber_duration += (
                            message.get("meta_info", {}).get("transcriber_duration", 0)
                            if message["meta_info"] is not None
                            else 0
                        )
                        await self._log_transcriber_connection_error(
                            (message.get("meta_info") or {}).get("connection_error")
                        )
                        break
                    continue

                if self.stream:
                    self._set_call_details(message)
                    meta_info = message["meta_info"]
                    sequence = await self.process_transcriber_request(meta_info)
                    next_task = self._get_next_step(sequence, "transcriber")
                    interim_transcript_len = 0

                    # Handling of transcriber events
                    if message["data"] == "speech_started":
                        if not self.tools["input"].welcome_message_played() and self.discard_pre_welcome_utterance:
                            self._speech_started_before_welcome = True
                        if self.tools["input"].welcome_message_played():
                            self._speech_started_before_welcome = False
                            logger.info(f"User has started speaking")
                            # self.callee_silent = False

                    # Whenever interim results would be received from Deepgram, this condition would get triggered
                    elif (
                        isinstance(message.get("data"), dict)
                        and message["data"].get("type", "") == "interim_transcript_received"
                    ):
                        self.time_since_last_spoken_human_word = time.time()
                        if temp_transcriber_message == message["data"].get("content"):
                            logger.info("Received the same transcript as the previous one we have hence continuing")
                            continue

                        temp_transcriber_message = message["data"].get("content")

                        if not self.tools["input"].welcome_message_played():
                            if self.discard_pre_welcome_utterance:
                                self._speech_started_before_welcome = True
                            continue

                        # Post-welcome interim → clear stale flag set by pre-welcome SpeechStarted (welcome-audio bleed).
                        self._speech_started_before_welcome = False

                        interim_transcript_len += len(message["data"].get("content").strip().split(" "))
                        transcript_content = message["data"].get("content", "")

                        if self.interruption_manager.should_trigger_interruption(
                            word_count=interim_transcript_len,
                            transcript=transcript_content,
                            is_audio_playing=self.tools["input"].is_audio_being_played_to_user()
                            or self.response_in_pipeline,
                            welcome_played=self.tools["input"].welcome_message_played(),
                        ):
                            logger.info(f"Condition for interruption hit")
                            self.interruption_manager.on_user_speech_started()
                            self.interruption_manager.on_interruption_triggered()
                            # Tag the ASR turn that was active at interrupt time so
                            # we can annotate turn_latencies with was_interrupted=True
                            _asr_turn_id = getattr(self.tools.get("transcriber"), "current_turn_id", None)
                            self.interruption_manager.record_interrupted_transcriber_turn(_asr_turn_id)
                            self.tools["input"].update_is_audio_being_played(False)
                            await self.__cleanup_downstream_tasks()
                        # User continuation detection: cancel pending response if user continues within grace period
                        elif (
                            not self.tools["input"].is_audio_being_played_to_user()
                            and self.tools["input"].welcome_message_played()
                        ):
                            self.interruption_manager.on_user_speech_started()
                            has_pending_response = self.interruption_manager.has_pending_responses()
                            time_since_utterance_end = self.interruption_manager.get_time_since_utterance_end()
                            within_grace_period = (
                                time_since_utterance_end != -1
                                and time_since_utterance_end < self.incremental_delay
                                and len(self.history) > 2
                            )

                            if (
                                has_pending_response
                                and within_grace_period
                                and interim_transcript_len > self.number_of_words_for_interruption
                            ):
                                logger.info(
                                    f"User continuation detected ({interim_transcript_len} words within {time_since_utterance_end:.0f}ms), canceling pending response"
                                )
                                # on_agent_interrupted_user MUST come before reset_utterance_end_time
                                # so it can still read the previous turn's utterance_end_time.
                                self.interruption_manager.on_agent_interrupted_user()
                                self.interruption_manager.reset_utterance_end_time()
                                await self.__cleanup_downstream_tasks()
                        elif (
                            (self.tools["input"].is_audio_being_played_to_user() or self.response_in_pipeline)
                            and self.tools["input"].welcome_message_played()
                            and self.number_of_words_for_interruption != 0
                        ):
                            # Not enough words for interruption - ignore (don't set callee_speaking)
                            logger.info(f"Ignoring transcript: {transcript_content.strip()}")
                            continue
                        else:
                            # Normal interim (no audio playing, no continuation) - mark user as speaking
                            self.interruption_manager.on_user_speech_started()

                        self.interruption_manager.update_required_delay(len(self.history))
                        self.interruption_manager.on_interim_transcript_received()

                        # Trigger background voicemail check on interim transcripts (non-blocking)
                        if self.voicemail_handler.enabled and not self.voicemail_handler.detected:
                            interim_content = message["data"].get("content", "")
                            self._trigger_voicemail_check(interim_content, meta_info, is_final=False)

                        self.llm_response_generated = False

                    # Whenever speech_final or UtteranceEnd is received from Deepgram, this condition would get triggered
                    elif isinstance(message.get("data"), dict) and message["data"].get("type", "") == "transcript":
                        logger.info(f"Received transcript, sending for further processing")
                        transcript_content = message["data"].get("content", "")
                        word_count = len(transcript_content.strip().split(" "))

                        if self.interruption_manager.is_false_interruption(
                            word_count=word_count,
                            transcript=transcript_content,
                            is_audio_playing=self.tools["input"].is_audio_being_played_to_user()
                            or self.response_in_pipeline,
                            welcome_played=self.tools["input"].welcome_message_played(),
                        ):
                            logger.info(
                                f"Continuing the loop and ignoring the transcript received ({transcript_content}) in speech final as it is false interruption"
                            )
                            self.interruption_manager.on_user_speech_ended(update_utterance_time=False)
                            self._speech_started_before_welcome = False
                            continue

                        _meta = message.get("meta_info") or {}
                        self.interruption_manager.on_user_speech_ended(
                            stop_offset_ms=_meta.get("user_stop_offset_ms", 0),
                            user_stop_ts_wall=_meta.get("user_stop_ts_wall"),
                        )
                        temp_transcriber_message = ""

                        if self.output_task is None:
                            logger.info(f"Output task was none and hence starting it")
                            self.output_task = asyncio.create_task(self.__process_output_loop())

                        self.interruption_manager.reset_delay_for_speech_final(len(self.history))

                        transcriber_message = message["data"].get("content")
                        meta_info = self.__get_updated_meta_info(meta_info)
                        await self._handle_transcriber_output(next_task, transcriber_message, meta_info)

                    # Handle speech_ended notification (UtteranceEnd with no new transcript)
                    elif isinstance(message.get("data"), dict) and message["data"].get("type", "") == "speech_ended":
                        logger.info(f"Received speech_ended notification, resetting callee_speaking state")
                        self.interruption_manager.on_user_speech_ended(update_utterance_time=False)
                        self._speech_started_before_welcome = False
                        temp_transcriber_message = ""

                    elif message["data"] == "transcriber_connection_closed":
                        self.transcriber_duration += (
                            message.get("meta_info", {}).get("transcriber_duration", 0)
                            if message["meta_info"] is not None
                            else 0
                        )
                        # In a pool, a standby transcriber closing is expected (e.g. Deepgram
                        # inactivity timeout). But if the active transcriber closed, the call
                        # is over (e.g. user hung up via telephony stop event).
                        if isinstance(self.tools.get("transcriber"), TranscriberPool):
                            if self.tools["transcriber"].is_active_transcriber_alive():
                                logger.info(f"TranscriberPool: standby transcriber closed, continuing")
                                continue
                            logger.info(f"TranscriberPool: active transcriber closed, ending call")
                        await self._log_transcriber_connection_error(
                            (message.get("meta_info") or {}).get("connection_error")
                        )
                        break

                else:
                    logger.info(f"Processing http transcription for message {message}")
                    if message["data"] == "transcriber_connection_closed":
                        self.transcriber_duration += (
                            message.get("meta_info", {}).get("transcriber_duration", 0)
                            if message["meta_info"] is not None
                            else 0
                        )
                        if isinstance(self.tools.get("transcriber"), TranscriberPool):
                            if self.tools["transcriber"].is_active_transcriber_alive():
                                logger.info(f"TranscriberPool: standby transcriber closed, continuing")
                                continue
                            logger.info(f"TranscriberPool: active transcriber closed, ending call")
                        await self._log_transcriber_connection_error(
                            (message.get("meta_info") or {}).get("connection_error")
                        )
                        break

                    await self.__process_http_transcription(message)

        except websockets.exceptions.ConnectionClosedOK:
            # Normal WebSocket closure (code 1000)
            pass
        except Exception as e:
            provider = self.task_config["tools_config"]["transcriber"].get("provider")
            await self._end_call_on_component_error(
                TranscriberError(str(e), provider=provider), HangupReason.TRANSCRIBER_ERROR
            )
            raise TranscriberError(str(e), provider=provider) from e

    async def __process_http_transcription(self, message):
        meta_info = self.__get_updated_meta_info(message["meta_info"])

        sequence = message["meta_info"].get("sequence", 0)
        next_task = self._get_next_step(sequence, "transcriber")
        self.transcriber_duration += (
            message["meta_info"]["transcriber_duration"] if "transcriber_duration" in message["meta_info"] else 0
        )

        await self._handle_transcriber_output(next_task, message["data"], meta_info)

    #################################################################
    # Synthesizer task
    #################################################################
    def __enqueue_chunk(self, chunk, i, number_of_chunks, meta_info):
        meta_info["chunk_id"] = i
        copied_meta_info = copy.deepcopy(meta_info)
        if i == 0 and "is_first_chunk" in meta_info and meta_info["is_first_chunk"]:
            logger.info("Sending first chunk")
            copied_meta_info["is_first_chunk_of_entire_response"] = True

        if i == number_of_chunks - 1 and (
            meta_info["sequence_id"] == -1 or meta_info.get("end_of_synthesizer_stream", False)
        ):
            logger.info(f"Sending first chunk")
            copied_meta_info["is_final_chunk_of_entire_response"] = True
            copied_meta_info.pop("is_first_chunk_of_entire_response", None)

        if copied_meta_info.get("message_category", None) == "agent_welcome_message":
            copied_meta_info["is_first_chunk_of_entire_response"] = True
            copied_meta_info["is_final_chunk_of_entire_response"] = True

        self.buffered_output_queue.put_nowait(create_ws_data_packet(chunk, copied_meta_info))

    def is_sequence_id_in_current_ids(self, sequence_id):
        """Check if sequence_id is valid. Delegates to InterruptionManager."""
        return self.interruption_manager.is_valid_sequence(sequence_id)

    def _get_voice_name_for_label(self, label):
        """Get agent name for a language label from configured agent_names."""
        return self.agent_names.get(label, "")

    async def switch_language(self, label, components=None):
        """Switch the active language for multilingual pools.

        Args:
            label: language label to switch to (e.g. "hi", "en").
            components: list of component names to switch. Defaults to both.
        """
        components = components or ["transcriber", "synthesizer"]
        if "transcriber" in components and isinstance(self.tools.get("transcriber"), TranscriberPool):
            await self.tools["transcriber"].switch(label)
        if "synthesizer" in components and isinstance(self.tools.get("synthesizer"), SynthesizerPool):
            await self.tools["synthesizer"].switch(label)

        # Update TaskManager state so silence detection, fillers, and LLM
        # language stay in sync with the active pools.
        self.language = label
        # Reset silence timers to prevent __check_for_completion from
        # interpreting the switch gap as inactivity and hanging up.
        self.last_transmitted_timestamp = time.time()
        self.time_since_last_spoken_human_word = time.time()
        self.asked_if_user_is_still_there = False
        logger.info(f"Language switched to '{label}'")

        if label in self.multilingual_prompts:
            new_prompt = self.multilingual_prompts[label]
            self.conversation_history.update_system_prompt(new_prompt)
            self.system_prompt["content"] = new_prompt
            logger.info(f"Switched system prompt to language '{label}'")

        active_transcriber_info = (
            self.tools.get("transcriber").get_active_transcriber_info()
            if isinstance(self.tools.get("transcriber"), TranscriberPool)
            else None
        )
        active_synthesizer_info = (
            self.tools.get("synthesizer").get_active_synthesizer_info()
            if isinstance(self.tools.get("synthesizer"), SynthesizerPool)
            else None
        )
        if active_transcriber_info:
            if "provider" in active_transcriber_info and active_transcriber_info["provider"]:
                self.transcriber_provider = active_transcriber_info["provider"]

        if active_synthesizer_info:
            if "provider" in active_synthesizer_info and active_synthesizer_info["provider"]:
                self.synthesizer_provider = active_synthesizer_info["provider"]
            if "voice" in active_synthesizer_info and active_synthesizer_info["voice"]:
                self.synthesizer_voice = active_synthesizer_info["voice"]

    async def __listen_synthesizer(self):
        all_text_to_be_synthesized = []
        try:
            while not self.conversation_ended:
                logger.info("Listening to synthesizer")
                try:
                    async for message in self.tools["synthesizer"].generate():
                        meta_info = message.get("meta_info", {})
                        current_text = meta_info.get("text", "")
                        write_to_log = False
                        if current_text not in all_text_to_be_synthesized:
                            all_text_to_be_synthesized.append(current_text)
                            write_to_log = True

                        is_first_message = meta_info.get("is_first_message", False)
                        sequence_id = meta_info.get("sequence_id", None)

                        # Check if the message is valid to process
                        if is_first_message or (
                            not self.conversation_ended and self.interruption_manager.is_valid_sequence(sequence_id)
                        ):
                            logger.info(f"Processing message with sequence_id: {sequence_id}")

                            if self.stream:
                                if meta_info.get("is_first_chunk", False):
                                    first_chunk_generation_timestamp = time.time()

                                if self.tools["output"].process_in_chunks(self.yield_chunks):
                                    number_of_chunks = math.ceil(len(message["data"]) / self.output_chunk_size)
                                    for chunk_idx, chunk in enumerate(
                                        yield_chunks_from_memory(message["data"], chunk_size=self.output_chunk_size)
                                    ):
                                        self.__enqueue_chunk(chunk, chunk_idx, number_of_chunks, meta_info)
                                else:
                                    self.buffered_output_queue.put_nowait(message)
                            else:
                                # Non-streaming output
                                logger.info("Stream not enabled, sending entire audio")
                                # TODO handle is audio playing over here
                                await self.tools["output"].handle(message)
                                if meta_info.get("end_of_synthesizer_stream", False):
                                    self._turn_audio_flushed.set()

                            if write_to_log:
                                logger.info(f"Writing response to log {meta_info.get('text')}")
                                convert_to_request_log(
                                    message=current_text,
                                    meta_info=meta_info,
                                    component=LogComponent.SYNTHESIZER,
                                    direction=LogDirection.RESPONSE,
                                    model=self.synthesizer_provider,
                                    is_cached=meta_info.get("is_cached", False),
                                    engine=self.tools["synthesizer"].get_engine(),
                                    run_id=self.run_id,
                                )
                        else:
                            logger.info(f"Skipping message with sequence_id: {sequence_id}")

                        # Give control to other tasks
                        sleep_time = self.tools["synthesizer"].get_sleep_time()
                        await asyncio.sleep(sleep_time)

                except asyncio.CancelledError:
                    logger.info("Synthesizer task was cancelled.")
                    # await self.handle_cancellation("Synthesizer task was cancelled.")
                    self._turn_audio_flushed.set()
                    break
                except Exception as e:
                    self._turn_audio_flushed.set()
                    await self._end_call_on_component_error(
                        SynthesizerError(str(e), provider=self.synthesizer_provider), HangupReason.SYNTHESIZER_ERROR
                    )
                    break

            logger.info("Exiting __listen_synthesizer gracefully.")

        except asyncio.CancelledError:
            logger.info("Synthesizer task cancelled outside loop.")
            # await self.handle_cancellation("Synthesizer task was cancelled outside loop.")
        except Exception as e:
            await self._end_call_on_component_error(
                SynthesizerError(str(e), provider=self.synthesizer_provider), HangupReason.SYNTHESIZER_ERROR
            )
            raise SynthesizerError(str(e), provider=self.synthesizer_provider) from e
        finally:
            await self.tools["synthesizer"].cleanup()

    async def __send_preprocessed_audio(self, meta_info, text):
        meta_info = copy.deepcopy(meta_info)
        yield_in_chunks = self.yield_chunks
        try:
            # TODO: Either load IVR audio into memory before call or user s3 iter_cunks
            # This will help with interruption in IVR
            audio_chunk = None
            if self.turn_based_conversation or self.task_config["tools_config"]["output"]["provider"] == "default":
                audio_chunk = await get_raw_audio_bytes(
                    text,
                    self.assistant_name,
                    self.task_config["tools_config"]["output"]["format"],
                    local=self.is_local,
                    assistant_id=self.assistant_id,
                )
                logger.info("Sending preprocessed audio")
                meta_info["format"] = self.task_config["tools_config"]["output"]["format"]
                meta_info["end_of_synthesizer_stream"] = True
                await self.tools["output"].handle(create_ws_data_packet(audio_chunk, meta_info))
            else:
                if meta_info.get("message_category", None) == "filler":
                    logger.info(f"Getting {text} filler from local fs")
                    audio = await get_raw_audio_bytes(
                        f"{self.filler_preset_directory}/{text}.wav", local=True, is_location=True
                    )
                    yield_in_chunks = False
                    if not self.turn_based_conversation and self.task_config["tools_config"]["output"] != "default":
                        logger.info(f"Got to convert it to pcm")
                        audio_chunk = wav_bytes_to_pcm(resample(audio, format="wav", target_sample_rate=8000))
                        meta_info["format"] = "pcm"
                else:
                    start_time = time.perf_counter()
                    audio_chunk = self.preloaded_welcome_audio if self.preloaded_welcome_audio else None
                    if meta_info["text"] == "":
                        audio_chunk = None
                    logger.info(f"Time to get response from S3 {time.perf_counter() - start_time}")
                    if not self.buffered_output_queue.empty():
                        logger.info(f"Output queue was not empty and hence emptying it")
                        self.buffered_output_queue = asyncio.Queue()
                    meta_info["format"] = "pcm"
                    if "message_category" in meta_info and meta_info["message_category"] == "agent_welcome_message":
                        if audio_chunk is None:
                            logger.info(f"File doesn't exist in S3. Hence we're synthesizing it from synthesizer")
                            meta_info["cached"] = False
                            await self._synthesize(create_ws_data_packet(meta_info["text"], meta_info=meta_info))
                            return
                        else:
                            meta_info["is_first_chunk"] = True
                meta_info["end_of_synthesizer_stream"] = True
                if yield_in_chunks and audio_chunk is not None:
                    i = 0
                    number_of_chunks = math.ceil(len(audio_chunk) / 100000000)
                    logger.info(f"Audio chunk size {len(audio_chunk)}, chunk size {100000000}")
                    for chunk in yield_chunks_from_memory(audio_chunk, chunk_size=100000000):
                        self.__enqueue_chunk(chunk, i, number_of_chunks, meta_info)
                        i += 1
                elif audio_chunk is not None:
                    meta_info["chunk_id"] = 1
                    meta_info["is_first_chunk_of_entire_response"] = True
                    meta_info["is_final_chunk_of_entire_response"] = True
                    message = create_ws_data_packet(audio_chunk, meta_info)
                    self.buffered_output_queue.put_nowait(message)

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Something went wrong {e}")

    async def _synthesize(self, message):
        meta_info = message["meta_info"]
        text = message["data"]
        meta_info["type"] = "audio"
        meta_info["synthesizer_start_time"] = time.time()
        try:
            if not self.conversation_ended and (
                "is_first_message" in meta_info
                and meta_info["is_first_message"]
                or self.interruption_manager.is_valid_sequence(message["meta_info"]["sequence_id"])
            ):
                if meta_info["is_md5_hash"]:
                    logger.info(
                        "sending preprocessed audio response to {}".format(
                            self.task_config["tools_config"]["output"]["provider"]
                        )
                    )
                    await self.__send_preprocessed_audio(meta_info, text)

                elif self.synthesizer_provider in SUPPORTED_SYNTHESIZER_MODELS.keys():
                    convert_to_request_log(
                        message=text,
                        meta_info=meta_info,
                        component=LogComponent.SYNTHESIZER,
                        direction=LogDirection.REQUEST,
                        model=self.synthesizer_provider,
                        engine=self.tools["synthesizer"].get_engine(),
                        run_id=self.run_id,
                    )
                    if "cached" in message["meta_info"] and meta_info["cached"] is True:
                        logger.info(f"Cached response and hence sending preprocessed text")
                        convert_to_request_log(
                            message=text,
                            meta_info=meta_info,
                            component=LogComponent.SYNTHESIZER,
                            direction=LogDirection.RESPONSE,
                            model=self.synthesizer_provider,
                            is_cached=True,
                            engine=self.tools["synthesizer"].get_engine(),
                            run_id=self.run_id,
                        )
                        await self.__send_preprocessed_audio(meta_info, get_md5_hash(text))
                    else:
                        self.synthesizer_characters += len(text)
                        await self.tools["synthesizer"].push(message)
                else:
                    logger.info("other synthesizer models not supported yet")
            else:
                logger.info(
                    f"{message['meta_info']['sequence_id']} is not a valid sequence id and hence not synthesizing this"
                )

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in synthesizer: {e}")
            self._turn_audio_flushed.set()

    ############################################################
    # Output handling
    ############################################################

    async def __send_first_message(self, message):
        meta_info = self.__get_updated_meta_info()
        sequence = meta_info.get("sequence", 0)
        next_task = self._get_next_step(sequence, "transcriber")
        await self._handle_transcriber_output(next_task, message, meta_info)
        self.interruption_manager.set_first_interim_for_immediate_response()

    """
    When the welcome message is playing we accumulate the transcript in the self.transcriber_message variable and once 
    the welcome message is completely played we send this transcript for further processing.
    """

    async def __handle_accumulated_message(self):
        logger.info("Setting up __handle_accumulated_message function")
        while True:
            if self.tools["input"].welcome_message_played():
                logger.info(f"Welcome message has been played")
                self.first_message_passing_time = time.time()
                if len(self.transcriber_message):
                    logger.info(f"Sending the accumulated transcribed message - {self.transcriber_message}")
                    await self.__send_first_message(self.transcriber_message)
                    self.transcriber_message = ""
                break

            await asyncio.sleep(0.1)
        self.handle_accumulated_message_task = None

    # Currently this loop only closes in case of interruption
    # but it shouldn't be the case.
    async def __process_output_loop(self):
        try:
            while True:
                if self.tools["input"].welcome_message_played():
                    should_delay, sleep_duration = self.interruption_manager.should_delay_output(
                        self.tools["input"].welcome_message_played()
                    )
                    if should_delay:
                        await asyncio.sleep(sleep_duration)
                        continue
                else:
                    logger.info(f"Started transmitting at {time.time()}")

                message = await self.buffered_output_queue.get()

                if "end_of_conversation" in message["meta_info"]:
                    await self.__process_end_of_conversation()

                sequence_id = message["meta_info"].get("sequence_id")

                # Centralized tri-state decision loop (handles race condition + grace period)
                should_continue_outer_loop = False
                while True:
                    status = self.interruption_manager.get_audio_send_status(sequence_id, len(self.history))

                    if status == "SEND":
                        # Audio approved - send it
                        self.tools["input"].update_is_audio_being_played(True)
                        self.response_in_pipeline = False
                        await self.tools["output"].handle(message)
                        # Track when agent audio first starts flowing for this sequence.
                        # Deduplicated inside on_agent_speech_started — safe to call per chunk.
                        if sequence_id is not None and sequence_id != -1:
                            self.interruption_manager.on_agent_speech_started(sequence_id)
                        try:
                            duration = calculate_audio_duration(
                                len(message["data"]), self.sampling_rate, format=message["meta_info"]["format"]
                            )
                            if self.should_record:
                                self.conversation_recording["output"].append(
                                    {"data": message["data"], "start_time": time.time(), "duration": duration}
                                )
                        except Exception as e:
                            duration = 0.256
                            logger.info("Exception in __process_output_loop: {}".format(str(e)))
                        break  # Exit inner loop, audio sent

                    elif status == "BLOCK":
                        # Audio blocked (user speaking or invalid sequence) - discard
                        logger.info(f"Audio blocked: discarding message (sequence_id={sequence_id})")
                        # Null byte (b'\x00') is the end-of-stream control signal, not audio.
                        # Always send it through handle() so the is_final_chunk post-mark is
                        # created and sent to Plivo. Without this, is_audio_being_played stays
                        # True forever because no final mark echo ever arrives.
                        if message["data"] == b"\x00":
                            await self.tools["output"].handle(message)
                        if message["meta_info"].get("end_of_llm_stream", False) or message["meta_info"].get(
                            "end_of_synthesizer_stream", False
                        ):
                            self._turn_audio_flushed.set()
                        should_continue_outer_loop = True
                        break  # Exit inner loop, skip to next message

                    elif status == "WAIT":
                        # Grace period active - hold and retry
                        await asyncio.sleep(0.05)
                        # Continue inner loop to re-check status

                if should_continue_outer_loop:
                    continue

                # Signal that the turn's audio has been fully flushed to the output.
                # This is reached only in the SEND path (BLOCK path breaks before here),
                # so it is a reliable signal that a complete response was delivered.
                if message["meta_info"].get("end_of_llm_stream", False) or message["meta_info"].get(
                    "end_of_synthesizer_stream", False
                ):
                    self._turn_audio_flushed.set()
                    if message["meta_info"].get("end_of_synthesizer_stream", False):
                        self.interruption_manager.on_successful_response_delivered()
                        self.interruption_manager.on_agent_speech_ended()
                    # Reset asked_if_user_is_still_there flag after any message except is_user_online_message
                    if message["meta_info"].get("message_category", "") != "is_user_online_message":
                        self.asked_if_user_is_still_there = False

                # # The below code is redundant in the case of telephony
                # if "is_final_chunk_of_entire_response" in message['meta_info'] and message['meta_info']['is_final_chunk_of_entire_response']:
                #     self.started_transmitting_audio = False
                #     logger.info("##### End of synthesizer stream")
                #
                #     if message['meta_info'].get('message_category', '') == 'agent_hangup':
                #         await self.__process_end_of_conversation()
                #         break
                #
                #     #If we're sending the message to check if user is still here, don't set asked_if_user_is_still_there to True
                #     if message['meta_info'].get('text', '') != self.check_user_online_message:
                #         self.asked_if_user_is_still_there = False
                #
                #     self.turn_id += 1
                #
                # # The below code is redundant in the case of telephony
                # if "is_first_chunk_of_entire_response" in message['meta_info'] and message['meta_info']['is_first_chunk_of_entire_response']:
                #     logger.info(f"First chunk stuff")
                #     self.started_transmitting_audio = True if "is_final_chunk_of_entire_response" not in message['meta_info'] else False
                #     self.consider_next_transcript_after = time.time() + self.duration_to_prevent_accidental_interruption
                #     self.__process_latency_data(message)
                # else:
                #     # Sleep until this particular audio frame is spoken only if the duration for the frame is atleast 500ms
                #     if duration > 0:
                #         logger.info(f"##### Sleeping for {duration} to maintain quueue on our side {self.sampling_rate}")
                #         await asyncio.sleep(duration - 0.030) #30 milliseconds less
                # if message['meta_info']['sequence_id'] != -1: #Making sure we only track the conversation's last transmitted timesatamp
                #     self.last_transmitted_timestamp = time.time()

                # try:
                #     logger.info(f"Updating Last transmitted timestamp to {str(self.last_transmitted_timestamp)}")
                # except Exception as e:
                #     logger.error(f'Error in printing Last transmitted timestamp: {str(e)}')

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in processing message output: {str(e)}")

    async def __check_for_completion(self):
        logger.info(f"Starting task to check for completion")
        while True:
            await asyncio.sleep(2)

            if self.is_web_based_call and time.time() - self.start_time >= int(
                self.task_config["task_config"]["call_terminate"]
            ):
                logger.info("Hanging up for web call as max time of call has been reached")
                await self.__process_end_of_conversation(web_call_timeout=True)
                self.hangup_detail = HangupReason.WEB_CALL_MAX_DURATION_REACHED
                break

            if self.last_transmitted_timestamp == 0:
                logger.info(f"Last transmitted timestamp is simply 0 and hence continuing")
                continue

            if self.hangup_triggered:
                if self.conversation_ended:
                    logger.info(f"Call hangup completed successfully")
                    break

                if self.hangup_triggered_at:
                    time_since_hangup = time.time() - self.hangup_triggered_at
                    if time_since_hangup > self.hangup_mark_event_timeout:
                        logger.warning(
                            f"Hangup mark event not received within {self.hangup_mark_event_timeout}s (waited {time_since_hangup:.1f}s), forcing conversation end"
                        )
                        # Set hangup_sent since mark event didn't arrive
                        if "output" in self.tools:
                            self.tools["output"].set_hangup_sent()
                        await self.__process_end_of_conversation()
                        break
                    else:
                        logger.info(
                            f"Waiting for hangup mark event ({time_since_hangup:.1f}s / {self.hangup_mark_event_timeout}s)"
                        )
                continue

            if self.tools["input"].is_audio_being_played_to_user() or self.response_in_pipeline:
                continue

            time_since_last_spoken_ai_word = time.time() - self.last_transmitted_timestamp
            time_since_user_last_spoke = (
                (time.time() - self.time_since_last_spoken_human_word)
                if self.time_since_last_spoken_human_word > 0
                else float("inf")
            )

            if (
                self.hang_conversation_after > 0
                and time_since_last_spoken_ai_word > self.hang_conversation_after
                and time_since_user_last_spoke > self.hang_conversation_after
            ):
                logger.info(
                    f"{time_since_last_spoken_ai_word} seconds since AI last spoke and {time_since_user_last_spoke} seconds since user last spoke, both exceed {self.hang_conversation_after}s timeout - hanging up"
                )
                self.hangup_detail = HangupReason.INACTIVITY_TIMEOUT
                await self.process_call_hangup()
                break

            elif (
                time_since_last_spoken_ai_word > self.trigger_user_online_message_after
                and not self.asked_if_user_is_still_there
                and time_since_user_last_spoke > self.trigger_user_online_message_after
            ):
                logger.info(
                    f"Asking if the user is still there (agent silent for {time_since_last_spoken_ai_word:.2f}s, user silent for {time_since_user_last_spoke:.2f}s)"
                )
                self.asked_if_user_is_still_there = True

                if self.check_if_user_online:
                    user_online_message = select_message_by_language(
                        self.check_user_online_message_config, self.language
                    )

                    if self.generate_precise_transcript:
                        self.tools["input"].reset_response_heard_by_user()

                    if self.should_record:
                        meta_info = {
                            "io": "default",
                            "request_id": str(uuid.uuid4()),
                            "cached": False,
                            "sequence_id": -1,
                            "format": "wav",
                            "message_category": "is_user_online_message",
                            "end_of_llm_stream": True,
                        }
                        await self._synthesize(create_ws_data_packet(user_online_message, meta_info=meta_info))
                    else:
                        meta_info = {
                            "io": self.tools["output"].get_provider(),
                            "request_id": str(uuid.uuid4()),
                            "cached": False,
                            "sequence_id": -1,
                            "format": "pcm",
                            "message_category": "is_user_online_message",
                            "end_of_llm_stream": True,
                        }
                        await self._synthesize(create_ws_data_packet(user_online_message, meta_info=meta_info))
                    self.conversation_history.append_assistant(user_online_message, exclude_from_llm=True)

                # Just in case we need to clear messages sent before
                await self.tools["output"].handle_interruption()
            else:
                logger.info(
                    f"Only {time_since_last_spoken_ai_word} seconds since last spoken time stamp and hence not cutting the phone call"
                )

    async def __check_for_backchanneling(self):
        while True:
            user_speaking_duration = self.interruption_manager.get_user_speaking_duration()
            if (
                self.interruption_manager.is_user_speaking()
                and user_speaking_duration > self.backchanneling_start_delay
            ):
                filename = random.choice(self.filenames)
                logger.info(f"Should send a random backchanneling words and sending them {filename}")
                audio = await get_raw_audio_bytes(
                    f"{self.backchanneling_audios}/{filename}", local=True, is_location=True
                )
                if not self.turn_based_conversation and self.task_config["tools_config"]["output"] != "default":
                    audio = resample(audio, target_sample_rate=8000, format="wav")
                    audio = wav_bytes_to_pcm(audio)
                await self.tools["output"].handle(create_ws_data_packet(audio, self.__get_updated_meta_info()))
            else:
                logger.info(
                    f"Callee isn't speaking and hence not sending or {user_speaking_duration} is not greater than {self.backchanneling_start_delay}"
                )
            await asyncio.sleep(self.backchanneling_message_gap)

    async def __first_message(self, timeout=10.0):
        logger.info(f"Executing the first message task")
        try:
            if self.is_web_based_call:
                logger.info("Sending agent welcome message for web based call")
                text = self.kwargs.get("agent_welcome_message", None)
                meta_info = {
                    "io": "default",
                    "message_category": "agent_welcome_message",
                    "stream_sid": self.stream_sid,
                    "request_id": str(uuid.uuid4()),
                    "cached": False,
                    "sequence_id": -1,
                    "format": self.task_config["tools_config"]["output"]["format"],
                    "text": text,
                    "end_of_llm_stream": True,
                }
                self.stream_sid_ts = time.time() * 1000
                if text and text.strip():
                    self.conversation_history.append_welcome_message(text)
                await self._synthesize(create_ws_data_packet(text, meta_info=meta_info))
                return

            start_time = asyncio.get_running_loop().time()
            while True:
                elapsed_time = asyncio.get_running_loop().time() - start_time
                if elapsed_time > timeout:
                    await self.__process_end_of_conversation()
                    logger.warning("Timeout reached while waiting for stream_sid")
                    break

                if not self.stream_sid and not self.default_io:
                    stream_sid = self.tools["input"].get_stream_sid()
                    if stream_sid is not None:
                        self.stream_sid_ts = time.time() * 1000
                        logger.info(f"Got stream sid and hence sending the first message {stream_sid}")
                        self.stream_sid = stream_sid
                        text = self.kwargs.get("agent_welcome_message", None)
                        meta_info = {
                            "io": self.tools["output"].get_provider(),
                            "message_category": "agent_welcome_message",
                            "stream_sid": stream_sid,
                            "request_id": str(uuid.uuid4()),
                            "cached": True,
                            "sequence_id": -1,
                            "format": self.task_config["tools_config"]["output"]["format"],
                            "text": text,
                            "end_of_llm_stream": True,
                        }
                        if text and text.strip():
                            self.conversation_history.append_welcome_message(text)
                        if self.turn_based_conversation:
                            meta_info["type"] = "text"
                            bos_packet = create_ws_data_packet("<beginning_of_stream>", meta_info)
                            await self.tools["output"].handle(bos_packet)
                            await self.tools["output"].handle(create_ws_data_packet(text, meta_info))
                            eos_packet = create_ws_data_packet("<end_of_stream>", meta_info)
                            await self.tools["output"].handle(eos_packet)
                        else:
                            await self._synthesize(create_ws_data_packet(text, meta_info=meta_info))
                        break
                    else:
                        logger.info(f"Stream id is still None, so not passing it")
                        await asyncio.sleep(0.01)  # Sleep for half a second to see if stream id goes past None
                elif self.default_io:
                    logger.info(f"Shouldn't record")
                    # meta_info={'io': 'default', 'is_first_message': True, "request_id": str(uuid.uuid4()), "cached": True, "sequence_id": -1, 'format': 'wav'}
                    # await self._synthesize(create_ws_data_packet(self.kwargs['agent_welcome_message'], meta_info= meta_info))
                    break

        except Exception as e:
            logger.error(f"Exception in __first_message {str(e)}")

    async def handle_init_event(self, init_meta_data):
        """
        This function is used to handle the init event which we get from the client side in the case of web calling.

        Args:
            init_meta_data: This consists of the metadata which has been sent via the client. It would consist of the
            context data which needs to be injected in the prompt.
        """
        try:
            logger.info(f"handle_init_event has been triggered with metadata = {init_meta_data}")
            self.context_data["recipient_data"].update(init_meta_data["context_data"])
            logger.info(f"Context data updated - {self.context_data}")

            self.prompts["system_prompt"] = update_prompt_with_context(self.prompts["system_prompt"], self.context_data)

            if self.system_prompt["content"]:
                system_prompt = self.system_prompt["content"]
                system_prompt = update_prompt_with_context(system_prompt, self.context_data)
                self.system_prompt["content"] = system_prompt
                self.conversation_history.update_system_prompt(system_prompt)

            if self.call_hangup_message_config and self.context_data:
                if isinstance(self.call_hangup_message_config, dict):
                    self.call_hangup_message_config = {
                        lang: update_prompt_with_context(msg, self.context_data)
                        for lang, msg in self.call_hangup_message_config.items()
                    }
                else:
                    self.call_hangup_message_config = update_prompt_with_context(
                        self.call_hangup_message_config, self.context_data
                    )

            agent_welcome_message = self.kwargs.get("agent_welcome_message", "")

            agent_welcome_message = update_prompt_with_context(agent_welcome_message, self.context_data)
            logger.info(f"Updated agent welcome message after context data replacement - {agent_welcome_message}")
            self.kwargs["agent_welcome_message"] = agent_welcome_message
            if len(self.conversation_history) == 2 and agent_welcome_message:
                self.conversation_history.update_welcome_message(agent_welcome_message)

            await self.tools["output"].send_init_acknowledgement()
            self.first_message_task = asyncio.create_task(self.__first_message())
        except Exception as e:
            logger.error(f"Error occurred in handling init event - {e}")

    async def run(self):
        self._component_error = None  # Reset for each run
        self._error_logged = False
        try:
            if self._is_conversation_task():
                logger.info("started running")
                # Create transcriber and synthesizer tasks
                tasks = []
                # tasks = [asyncio.create_task(self.tools['input'].handle())]

                # In the case of web call we would play the first message once we receive the init event
                if self.turn_based_conversation:
                    self.first_message_task = asyncio.create_task(self.__first_message())

                if not self.turn_based_conversation:
                    self.first_message_passing_time = None
                    self.handle_accumulated_message_task = asyncio.create_task(self.__handle_accumulated_message())
                if "transcriber" in self.tools:
                    tasks.append(asyncio.create_task(self._listen_transcriber()))
                    self.transcriber_task = asyncio.create_task(self.tools["transcriber"].run())

                if self.turn_based_conversation and self._is_conversation_task():
                    logger.info(
                        "Since it's connected through dashboard, I'll run listen_llm_tas too in case user wants to simply text"
                    )
                    self.llm_queue_task = asyncio.create_task(self._listen_llm_input_queue())

                if "synthesizer" in self.tools and self._is_conversation_task() and not self.turn_based_conversation:
                    try:
                        self.synthesizer_task = asyncio.create_task(self.__listen_synthesizer())
                    except asyncio.CancelledError as e:
                        logger.error(f"Synth task got cancelled {e}")
                        traceback.print_exc()

                self.output_task = asyncio.create_task(self.__process_output_loop())
                if not self.turn_based_conversation or self.enforce_streaming:
                    self.hangup_task = asyncio.create_task(self.__check_for_completion())

                    if self.should_backchannel:
                        self.backchanneling_task = asyncio.create_task(self.__check_for_backchanneling())

                try:
                    await asyncio.gather(*tasks)
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    if not isinstance(e, BolnaComponentError):
                        logger.error(f"Error: {e}")
                    if self.run_id and not self._error_logged:
                        if isinstance(e, BolnaComponentError):
                            error_msg = format_error_message(e.component, e.provider or e.model or "-", str(e))
                            model = e.model or "-"
                        else:
                            error_msg = format_error_message("unknown", "-", str(e))
                            model = "-"
                        convert_to_request_log(
                            error_msg,
                            {"request_id": self.task_id, "sequence_id": None},
                            model=model,
                            component=LogComponent.ERROR,
                            direction=LogDirection.ERROR,
                            is_cached=False,
                            run_id=self.run_id,
                        )
                    self._error_logged = True

                _stored_err = self._component_error
                if _stored_err is not None:
                    self._component_error = None
                    err_cls = _stored_err["cls"]
                    if issubclass(err_cls, BolnaComponentError):
                        raise err_cls(
                            _stored_err["message"], provider=_stored_err["provider"], model=_stored_err["model"]
                        )
                    else:
                        raise Exception(_stored_err["message"])
                for attr, cls, provider in [
                    ("synthesizer_task", SynthesizerError, getattr(self, "synthesizer_provider", None)),
                    (
                        "transcriber_task",
                        TranscriberError,
                        self.task_config.get("tools_config", {}).get("transcriber", {}).get("provider"),
                    ),
                ]:
                    task = getattr(self, attr, None)
                    if task and task.done() and not task.cancelled():
                        exc = task.exception()
                        if exc is not None:
                            raise cls(str(exc), provider=provider)

                if self.generate_precise_transcript:
                    has_pending_marks = len(self.mark_event_meta_data.mark_event_meta_data) > 0
                    has_response_heard = bool(self.tools["input"].response_heard_by_user)
                    if has_pending_marks or has_response_heard:
                        await self.sync_history(self.mark_event_meta_data.mark_event_meta_data.items(), time.time())
                    self.tools["input"].reset_response_heard_by_user()
                logger.info("Conversation completed")
                self.conversation_ended = True
            else:
                # Run agent followup tasks
                try:
                    if self.task_config["task_type"] == "webhook":
                        await self._process_followup_task()
                    else:
                        await self._run_llm_task(self.input_parameters)
                except BolnaComponentError:
                    raise
                except Exception as e:
                    raise

        except asyncio.CancelledError as e:
            traceback.print_exc()
            logger.info(f"Websocket got cancelled {self.task_id}")

        except Exception as e:
            # Cancel all tasks on error
            error_message = str(e)
            if not isinstance(e, BolnaComponentError):
                logger.error(f"Exception in task manager run: {error_message}")

            # Log call-breaking exception to CSV trace with component attribution (skip if already logged)
            if self.run_id and not self._error_logged:
                meta_info = {"request_id": self.task_id, "sequence_id": None}
                if isinstance(e, BolnaComponentError):
                    error_msg = format_error_message(e.component, e.provider or e.model or "-", error_message)
                    model = e.model or "-"
                else:
                    error_msg = format_error_message("unknown", "-", error_message)
                    model = "-"
                convert_to_request_log(
                    error_msg,
                    meta_info,
                    model=model,
                    component=LogComponent.ERROR,
                    direction=LogDirection.ERROR,
                    is_cached=False,
                    run_id=self.run_id,
                )

            logger.info(f"Exception occurred {e}")
            raise

        finally:
            self._component_error = None

            # Construct output
            tasks_to_cancel = []
            tasks_to_cancel.append(process_task_cancellation(self.first_message_task_new, "first_message_task_new"))
            tasks_to_cancel.append(process_task_cancellation(self.llm_task, "llm_task"))
            tasks_to_cancel.append(process_task_cancellation(self.llm_queue_task, "llm_queue_task"))
            tasks_to_cancel.append(
                process_task_cancellation(self.execute_function_call_task, "execute_function_call_task")
            )
            if "synthesizer" in self.tools and self.synthesizer_task is not None:
                tasks_to_cancel.append(process_task_cancellation(self.synthesizer_task, "synthesizer_task"))
                tasks_to_cancel.append(
                    process_task_cancellation(self.synthesizer_monitor_task, "synthesizer_monitor_task")
                )
                for task in self.synthesizer_tasks:
                    tasks_to_cancel.append(process_task_cancellation(task, "synthesizer_task_item"))
                self.synthesizer_tasks = []

            # Transcriber cleanup
            if "transcriber" in self.tools:
                tasks_to_cancel.append(self.tools["transcriber"].cleanup())
                if hasattr(self, "transcriber_task") and self.transcriber_task is not None:
                    tasks_to_cancel.append(process_task_cancellation(self.transcriber_task, "transcriber_task"))

            if self._is_conversation_task():
                self.transcriber_latencies.connection_latency_ms = self.tools["transcriber"].connection_time
                self.synthesizer_latencies.connection_latency_ms = self.tools["synthesizer"].connection_time

                self.transcriber_latencies.turn_latencies = self.tools["transcriber"].turn_latencies
                self.synthesizer_latencies.turn_latencies = self.tools["synthesizer"].turn_latencies

                # Annotate each transcriber turn with was_interrupted so callers
                # can see which ASR turns had a user barge-in without cross-referencing
                # the separate interruption_events list.
                _interrupted_ids = self.interruption_manager.interrupted_transcriber_turn_ids
                for _turn in self.transcriber_latencies.turn_latencies:
                    _tid = _turn.get("turn_id") or _turn.get("sequence_id")
                    _turn["was_interrupted"] = _tid in _interrupted_ids if _tid is not None else False

                # Collect language detection latency if available
                if hasattr(self, "language_detector") and self.language_detector.latency_data:
                    self.llm_latencies.other_latencies.append(self.language_detector.latency_data)

                welcome_message_sent_ts = self.tools["output"].get_welcome_message_sent_ts()

                _call_start_ms = self.conversation_start_init_ts
                _user_bot_latencies = [
                    {
                        "sequence_id": e["sequence_id"],
                        "user_end_ms": round(e["user_end_s"] * 1000 - _call_start_ms, 2),
                        "agent_start_ms": round(e["agent_start_s"] * 1000 - _call_start_ms, 2),
                        "latency_ms": e["latency_ms"],
                    }
                    for e in self.interruption_manager.user_bot_latencies
                ]

                output = {
                    "messages": self.history,
                    "conversation_time": time.time() - self.start_time,
                    "label_flow": self.label_flow,
                    "function_tool_api_call_details": copy.deepcopy(self.function_tool_api_call_details),
                    "call_sid": self.call_sid,
                    "stream_sid": self.stream_sid,
                    "transcriber_duration": self.transcriber_duration,
                    "synthesizer_characters": self.tools["synthesizer"].get_synthesized_characters(),
                    "ended_by_assistant": self.ended_by_assistant,
                    "latency_dict": {
                        "llm_latencies": self.llm_latencies.model_dump(),
                        "transcriber_latencies": self.transcriber_latencies.model_dump(),
                        "synthesizer_latencies": self.synthesizer_latencies.model_dump(),
                        "rag_latencies": self.rag_latencies,
                        "routing_latencies": self.routing_latencies,
                        "welcome_message_sent_ts": None,
                        "stream_sid_ts": None,
                        "interruption_stats": self.interruption_manager.get_interruption_stats(
                            self.conversation_start_init_ts
                        ),
                        "user_bot_latencies": _user_bot_latencies,
                        "mark_tracking": self.mark_event_meta_data.get_mark_tracking_summary(),
                    },
                    "hangup_detail": self.hangup_detail,
                    "has_transfer": self.has_transfer,
                }

                if self.shadow_end_call_enabled:
                    end_call_seq = self.shadow_end_call_events[0]["seq"] if self.shadow_end_call_events else None
                    hangup_detail_str = str(self.hangup_detail) if self.hangup_detail else None
                    logger.info(
                        f"Shadow end_call summary: "
                        f"end_call_count={len(self.shadow_end_call_events)}, "
                        f"first_end_call_seq={end_call_seq}, "
                        f"actual_hangup_reason={hangup_detail_str}, "
                        f"events={json.dumps(self.shadow_end_call_events)}"
                    )

                try:
                    if welcome_message_sent_ts:
                        output["latency_dict"]["welcome_message_sent_ts"] = (
                            welcome_message_sent_ts - self.conversation_start_init_ts
                        )
                    if self.stream_sid_ts:
                        output["latency_dict"]["stream_sid_ts"] = self.stream_sid_ts - self.conversation_start_init_ts
                except Exception as e:
                    logger.error(f"error in logging audio latency ts {str(e)}")

                tasks_to_cancel.append(process_task_cancellation(self.output_task, "output_task"))
                tasks_to_cancel.append(process_task_cancellation(self.hangup_task, "hangup_task"))
                tasks_to_cancel.append(process_task_cancellation(self.backchanneling_task, "backchanneling_task"))
                # tasks_to_cancel.append(process_task_cancellation(self.initial_silence_task, 'initial_silence_task'))
                tasks_to_cancel.append(process_task_cancellation(self.first_message_task, "first_message_task"))
                tasks_to_cancel.append(process_task_cancellation(self.dtmf_task, "dtmf_task"))
                tasks_to_cancel.append(
                    process_task_cancellation(self.voicemail_handler.check_task, "voicemail_check_task")
                )
                tasks_to_cancel.append(
                    process_task_cancellation(self.handle_accumulated_message_task, "handle_accumulated_message_task")
                )

                output["recording_url"] = None
                if self.should_record:
                    output["recording_url"] = await save_audio_file_to_s3(
                        self.conversation_recording, self.sampling_rate, self.assistant_id, self.run_id
                    )
            else:
                output = self.input_parameters
                if self.task_config["task_type"] == "extraction":
                    output = {
                        "extracted_data": self.extracted_data,
                        "task_type": "extraction",
                        "latency_dict": {"llm_latencies": self.llm_latencies.model_dump()},
                    }
                elif self.task_config["task_type"] == "summarization":
                    logger.info(f"self.summarized_data {self.summarized_data}")
                    output = {
                        "summary": self.summarized_data,
                        "task_type": "summarization",
                        "latency_dict": {"llm_latencies": self.llm_latencies.model_dump()},
                    }
                elif self.task_config["task_type"] == "webhook":
                    output = {"status": self.webhook_response, "task_type": "webhook"}

            try:
                await asyncio.gather(*tasks_to_cancel)
            except Exception as e:
                logger.error(f"Error during task cancellation: {e}")
            finally:
                llm_agents_to_close = set()
                llm_agent = self.tools.get("llm_agent")
                if llm_agent is not None:
                    llm_agents_to_close.add(llm_agent)
                for agent in getattr(self, "llm_agent_map", {}).values():
                    if agent is not None:
                        llm_agents_to_close.add(agent)
                for agent in llm_agents_to_close:
                    if hasattr(agent, "llm") and hasattr(agent.llm, "close"):
                        try:
                            await agent.llm.close()
                        except Exception as e:
                            logger.error(f"Error closing LLM: {e}")
                    for attr in ("conversation_completion_llm", "voicemail_llm"):
                        aux = getattr(agent, attr, None)
                        if aux and hasattr(aux, "close"):
                            try:
                                await aux.close()
                            except Exception as e:
                                logger.error(f"Error closing {attr}: {e}")
                lang_det = getattr(self, "language_detector", None)
                if lang_det:
                    aux_llm = getattr(lang_det, "_llm", None)
                    if aux_llm and hasattr(aux_llm, "close"):
                        try:
                            await aux_llm.close()
                        except Exception as e:
                            logger.error(f"Error closing language detector LLM: {e}")
                for obs in self.observable_variables.values():
                    obs._observers.clear()
                self.observable_variables.clear()
                for tool in self.tools.values():
                    if hasattr(tool, "task_manager_instance"):
                        tool.task_manager_instance = None
                self.tools.clear()
                self.kwargs.pop("task_manager_instance", None)
                self.conversation_recording = {"input": {"data": b""}, "output": [], "metadata": {}}
                self.conversation_history = None
                self.request_logs.clear()
                self.function_tool_api_call_details.clear()

            return output

    async def handle_cancellation(self, message):
        try:
            # Cancel all tasks on cancellation
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            logger.info(f"tasks {len(tasks)}")
            for task in tasks:
                await process_task_cancellation(task, task.get_name())
                logger.info(f"Cancelling task {task.get_name()}")
                task.cancel()
            logger.info(message)
        except Exception as e:
            traceback.print_exc()
            logger.info(e)
