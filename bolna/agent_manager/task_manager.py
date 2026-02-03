import asyncio
from collections import defaultdict
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

from bolna.constants import ACCIDENTAL_INTERRUPTION_PHRASES, DEFAULT_USER_ONLINE_MESSAGE, DEFAULT_USER_ONLINE_MESSAGE_TRIGGER_DURATION, FILLER_DICT, DEFAULT_LANGUAGE_CODE, DEFAULT_TIMEZONE, LANGUAGE_NAMES, LLM_DEFAULT_CONFIGS
from bolna.helpers.function_calling_helpers import trigger_api, computed_api_response
from bolna.memory.cache.vector_cache import VectorCache
from .base_manager import BaseManager
from .interruption_manager import InterruptionManager
from bolna.agent_types import *
from bolna.providers import *
from bolna.prompts import *
from bolna.helpers.language_detector import LanguageDetector
from bolna.helpers.utils import structure_system_prompt, compute_function_pre_call_message, select_message_by_language, get_date_time_from_timezone, get_route_info, calculate_audio_duration, create_ws_data_packet, get_file_names_in_directory, get_raw_audio_bytes, is_valid_md5, \
    get_required_input_types, format_messages, get_prompt_responses, resample, save_audio_file_to_s3, update_prompt_with_context, get_md5_hash, clean_json_string, wav_bytes_to_pcm, convert_to_request_log, yield_chunks_from_memory, process_task_cancellation
from bolna.helpers.logger_config import configure_logger
from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import FastEmbedEncoder

from ..helpers.mark_event_meta_data import MarkEventMetaData
from ..helpers.observable_variable import ObservableVariable

logger = configure_logger(__name__)


class TaskManager(BaseManager):
    def __init__(self, assistant_name, task_id, task, ws, input_parameters=None, context_data=None,
                 assistant_id=None, turn_based_conversation=False, cache=None,
                 input_queue=None, conversation_history=None, output_queue=None, yield_chunks=True, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.kwargs["task_manager_instance"] = self

        self.conversation_start_init_ts = time.time() * 1000
        self.llm_latencies = {
            'connection_latency_ms': None, 
            'turn_latencies': [],
            'other_latencies': []  # List of {type, latency_ms, timestamp}
        }
        self.transcriber_latencies = {'connection_latency_ms': None, 'turn_latencies': []}
        self.synthesizer_latencies = {'connection_latency_ms': None, 'turn_latencies': []}
        self.rag_latencies = {'turn_latencies': []}
        self.stream_sid_ts = None

        self.task_config = task

        self.timezone = pytz.timezone(DEFAULT_TIMEZONE)
        self.language = DEFAULT_LANGUAGE_CODE
        self.transfer_call_params = self.kwargs.get('transfer_call_params', None)

        if task['tools_config'].get('api_tools', None) is not None:
            self.kwargs['api_tools'] = task['tools_config']['api_tools']

        if task['tools_config']["llm_agent"] and task['tools_config']["llm_agent"]['llm_config'].get('assistant_id', None) is not None:
            self.kwargs['assistant_id'] = task['tools_config']["llm_agent"]['llm_config']['assistant_id']

        logger.info(f"doing task {task}")
        self.task_id = task_id
        self.assistant_name = assistant_name
        self.tools = {}
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
            "synthesizer": self.synthesizer_queue
        }
        self.pipelines = task['toolchain']['pipelines']
        self.textual_chat_agent = False
        if task['toolchain']['pipelines'][0] == "llm" and task["tools_config"]["llm_agent"]["agent_task"] == "conversation":
            self.textual_chat_agent = False

        # Assistant persistance stuff
        self.assistant_id = assistant_id
        self.run_id = kwargs.get("run_id")

        self.mark_event_meta_data = MarkEventMetaData()
        self.sampling_rate = 24000
        self.conversation_ended = False
        self.hangup_triggered = False
        self.hangup_triggered_at = None
        self.hangup_message_queued = False
        self._end_of_conversation_in_progress = False
        self.hangup_mark_event_timeout = 10

        # Prompts
        self.prompts, self.system_prompt = {}, {}
        self.input_parameters = input_parameters

        # Recording
        self.should_record = False
        self.conversation_recording = {
            "input": {
                'data': b'',
                'started': time.time()
            },
            "output": [],
            "metadata": {
                "started": 0
            }
        }

        self.welcome_message_audio = self.kwargs.pop('welcome_message_audio', None)
        # Pre-decode welcome audio for faster playback
        self.preloaded_welcome_audio = base64.b64decode(self.welcome_message_audio) if self.welcome_message_audio else None
        self.observable_variables = {}
        self.output_handler_set = False
        #IO HANDLERS
        if task_id == 0:
            if self.is_web_based_call:
                self.task_config["tools_config"]["input"]["provider"] = "default"
                self.task_config["tools_config"]["output"]["provider"] = "default"

            self.default_io = self.task_config["tools_config"]["output"]["provider"] == 'default'
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
                self.should_record = self.task_config["tools_config"]["output"]["provider"] == 'default' and self.enforce_streaming #In this case, this is a websocket connection and we should record

            self.__setup_input_handlers(turn_based_conversation, input_queue, self.should_record)
        self.__setup_output_handlers(turn_based_conversation, output_queue)

        self.first_message_task_new = asyncio.create_task(self.message_task_new())

        # Agent stuff
        # Need to maintain current conversation history and overall persona/history kinda thing.
        # Soon we will maintain a separate history for this
        self.history = [] if conversation_history is None else conversation_history
        self.interim_history = copy.deepcopy(self.history.copy())
        self.label_flow = []

        # Setup IO SERVICE, TRANSCRIBER, LLM, SYNTHESIZER
        self.llm_task = None
        self.execute_function_call_task = None
        self.synthesizer_tasks = []
        self.synthesizer_task = None
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

        # Language detection
        self.language_detector = LanguageDetector(
            self.task_config['task_config'],
            run_id=self.run_id
        )
        self.language_injection_mode = self.task_config['task_config'].get('language_injection_mode')
        self.language_instruction_template = self.task_config['task_config'].get('language_instruction_template')

        # Call conversations
        self.call_sid = None
        self.stream_sid = None

        # metering
        self.transcriber_duration = 0
        self.synthesizer_characters = 0
        self.ended_by_assistant = False
        self.start_time = time.time()

        #Tasks
        self.extracted_data = None
        self.summarized_data = None
        self.stream = (self.task_config["tools_config"]['synthesizer'] is not None and self.task_config["tools_config"]["synthesizer"]["stream"]) and (self.enforce_streaming or not self.turn_based_conversation)

        self.is_local = False
        self.llm_config = None
        self.agent_type = None

        self.llm_config_map = {}
        self.llm_agent_map = {}
        if self.__is_multiagent():
            for agent, config in self.task_config["tools_config"]["llm_agent"]['llm_config']['agent_map'].items():
                self.llm_config_map[agent] = config.copy()
                self.llm_config_map[agent]['buffer_size'] = self.task_config["tools_config"]["synthesizer"][
                    'buffer_size']
        else:
            if self.task_config["tools_config"]["llm_agent"] is not None:
                if self.__is_knowledgebase_agent():
                    self.llm_agent_config = self.task_config["tools_config"]["llm_agent"]
                    self.llm_config = {
                        "model": self.llm_agent_config['llm_config']['model'],
                        "max_tokens": self.llm_agent_config['llm_config']['max_tokens'],
                        "provider": self.llm_agent_config['llm_config']['provider'],
                        "buffer_size": self.task_config["tools_config"]["synthesizer"].get('buffer_size'),
                        "temperature": self.llm_agent_config['llm_config']['temperature']
                    }
                elif self.__is_graph_agent():
                    self.llm_agent_config = self.task_config["tools_config"]["llm_agent"]
                    self.llm_config = {
                        "model": self.llm_agent_config['llm_config']['model'],
                        "max_tokens": self.llm_agent_config['llm_config']['max_tokens'],
                        "provider": self.llm_agent_config['llm_config']['provider'],
                    }
                else:
                    agent_type = self.task_config["tools_config"]["llm_agent"].get("agent_type", None)
                    if not agent_type:
                        self.llm_agent_config = self.task_config["tools_config"]["llm_agent"]
                    else:
                        self.llm_agent_config = self.task_config["tools_config"]["llm_agent"]['llm_config']

                    self.llm_config = {
                        "model": self.llm_agent_config['model'],
                        "max_tokens": self.llm_agent_config['max_tokens'],
                        "provider": self.llm_agent_config['provider'],
                        "temperature": self.llm_agent_config['temperature']
                    }

                if 'reasoning_effort' in self.llm_agent_config:
                    self.llm_config['reasoning_effort'] = self.llm_agent_config['reasoning_effort']

        # Output stuff
        self.output_task = None
        self.buffered_output_queue = asyncio.Queue()

        # Memory
        self.cache = cache

        # Initialize InterruptionManager with defaults (will be reconfigured for task_id == 0)
        self.interruption_manager = InterruptionManager()

        #setup request logs
        self.request_logs = []
        self.hangup_task = None

        self.conversation_config = None

        if task_id == 0:
            provider_config = self.task_config["tools_config"]["synthesizer"].get("provider_config")
            self.synthesizer_voice = provider_config["voice"]
            self.hangup_detail = None

            self.handle_accumulated_message_task = None
            # self.initial_silence_task = None
            self.hangup_task = None
            self.transcriber_task = None
            self.output_chunk_size = 16384 if self.sampling_rate == 24000 else 4096 #0.5 second chunk size for calls
            # For nitro
            self.nitro = True
            self.conversation_config = task.get("task_config", {})
            logger.info(f"Conversation config {self.conversation_config}")
            self.generate_precise_transcript = self.conversation_config.get('generate_precise_transcript', False)

            # Enable DTMF flow
            dtmf_enabled = self.conversation_config.get('dtmf_enabled', False)
            if dtmf_enabled:
                self.tools['input'].is_dtmf_active = True
                self.dtmf_task = asyncio.create_task(self.inject_digits_to_conversation())

            self.trigger_user_online_message_after = self.conversation_config.get("trigger_user_online_message_after", DEFAULT_USER_ONLINE_MESSAGE_TRIGGER_DURATION)
            self.check_if_user_online = self.conversation_config.get("check_if_user_online", True)
            self.check_user_online_message_config = self.conversation_config.get("check_user_online_message", DEFAULT_USER_ONLINE_MESSAGE)
            if self.check_user_online_message_config and self.context_data:
                if isinstance(self.check_user_online_message_config, dict):
                    self.check_user_online_message_config = {
                        lang: update_prompt_with_context(msg, self.context_data)
                        for lang, msg in self.check_user_online_message_config.items()
                    }
                else:
                    self.check_user_online_message_config = update_prompt_with_context(self.check_user_online_message_config, self.context_data)

            self.kwargs["process_interim_results"] = "true" if self.conversation_config.get("optimize_latency", False) is True else "false"
            # Routes
            self.routes = task['tools_config']['llm_agent'].get("routes", None)
            self.route_layer = None

            if self.routes:
                start_time = time.time()
                if self.__is_multiagent():
                    routes_meta = self.kwargs.get('routes', None)
                    routes_meta = routes_meta['routes']
                else:
                    routes_meta = self.kwargs.get('routes', None)

                if routes_meta:
                    self.vector_caches = routes_meta["vector_caches"]
                    self.route_responses_dict = routes_meta["route_responses_dict"]
                    self.route_layer = routes_meta["route_layer"]
                    logger.info(f"Time to setup routes from warmed up cache {time.time() - start_time}")
                else:
                    self.__setup_routes(self.routes)
                    logger.info(f"Time to setup routes {time.time() - start_time}")

            if self.__is_multiagent():
                routes_meta = self.kwargs.pop('routes', None)
                self.agent_routing = routes_meta['agent_routing_config']['route_layer']
                self.default_agent = task['tools_config']['llm_agent']['llm_config']['default_agent']
                logger.info(f"Initialised with default agent {self.default_agent}, agent_routing {self.agent_routing}")

            # for long pauses and rushing
            if self.conversation_config is not None:
                # TODO need to get this for azure - for azure the subtraction would not happen
                self.minimum_wait_duration = self.task_config["tools_config"]["transcriber"]["endpointing"]
                self.last_spoken_timestamp = time.time() * 1000
                self.incremental_delay = self.conversation_config.get("incremental_delay", 100)

                #Cut conversation
                self.hang_conversation_after = self.conversation_config.get("hangup_after_silence", 10)
                self.last_transmitted_timestamp = 0

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
                        self.call_hangup_message_config = update_prompt_with_context(self.call_hangup_message_config, self.context_data)
                self.check_for_completion_llm = os.getenv("CHECK_FOR_COMPLETION_LLM")

                # Voicemail detection (time-based)
                self.voicemail_detection_enabled = self.conversation_config.get('voicemail', False)
                self.voicemail_llm = os.getenv('VOICEMAIL_DETECTION_LLM', "gpt-4.1-mini")

                if 'output' not in self.tools or (self.tools['output'] and not self.tools['output'].requires_custom_voicemail_detection()):
                    self.voicemail_detection_enabled = False

                self.voicemail_detection_duration = self.conversation_config.get('voicemail_detection_duration', 30.0)  # Time window in seconds
                self.voicemail_check_interval = self.conversation_config.get('voicemail_check_interval', 7.0)  # Min time between interim checks
                self.voicemail_min_transcript_length = self.conversation_config.get('voicemail_min_transcript_length', 7)  # Min words for interim check
                self.voicemail_detection_prompt = VOICEMAIL_DETECTION_PROMPT
                self.voicemail_detection_prompt += """
                    Respond only in this JSON format:
                        {{
                          "is_voicemail": "Yes" or "No"
                        }}
                """
                self.voicemail_detected = False
                self.voicemail_detection_start_time = None  # Will be set when detection window starts
                self.voicemail_last_check_time = None  # Last time we checked for voicemail
                self.voicemail_check_task = None  # Background task for voicemail detection

                self.time_since_last_spoken_human_word = 0

                #Handling accidental interruption
                self.number_of_words_for_interruption = self.conversation_config.get("number_of_words_for_interruption", 3)
                self.asked_if_user_is_still_there = False #Used to make sure that if user's phrase qualifies as acciedental interruption, we don't break the conversation loop
                self.started_transmitting_audio = False
                self.accidental_interruption_phrases = set(ACCIDENTAL_INTERRUPTION_PHRASES)
                #self.interruption_backoff_period = 1000 #conversation_config.get("interruption_backoff_period", 300) #this is the amount of time output loop will sleep before sending next audio
                self.allow_extra_sleep = False #It'll help us to back off as soon as we hear interruption for a while

                # Initialize InterruptionManager to centralize interruption logic
                self.interruption_manager = InterruptionManager(
                    number_of_words_for_interruption=self.number_of_words_for_interruption,
                    accidental_interruption_phrases=ACCIDENTAL_INTERRUPTION_PHRASES,
                    incremental_delay=self.incremental_delay,
                    minimum_wait_duration=self.minimum_wait_duration,
                )

                #Backchanneling
                self.should_backchannel = self.conversation_config.get("backchanneling", False)
                self.backchanneling_task = None
                self.backchanneling_start_delay = self.conversation_config.get("backchanneling_start_delay", 5)
                self.backchanneling_message_gap = self.conversation_config.get("backchanneling_message_gap", 2) #Amount of duration co routine will sleep
                if self.should_backchannel and not turn_based_conversation and task_id == 0:
                    logger.info(f"Should backchannel")
                    self.backchanneling_audios = f'{kwargs.get("backchanneling_audio_location", os.getenv("BACKCHANNELING_PRESETS_DIR"))}/{self.synthesizer_voice.lower()}'
                    #self.num_files = list_number_of_wav_files_in_directory(self.backchanneling_audios)
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
                    self.transcriber_message = ''

                # Ambient noise
                self.ambient_noise = self.conversation_config.get("ambient_noise", False)
                self.ambient_noise_task = None
                if self.ambient_noise:
                    logger.info(f"Ambient noise is True {self.ambient_noise}")
                    self.soundtrack = f"{self.conversation_config.get('ambient_noise_track', 'coffee-shop')}.wav"

            # Classifier for filler
            self.use_fillers = self.conversation_config.get("use_fillers", False)
            if self.use_fillers:
                self.filler_classifier = kwargs.get("classifier", None)
                if self.filler_classifier is None:
                    logger.info("Not using fillers to decrease latency")
                else:
                    self.filler_preset_directory = f"{os.getenv('FILLERS_PRESETS_DIR')}/{self.synthesizer_voice.lower()}"

        # setting transcriber and synthesizer in parallel
        self.__setup_transcriber()
        self.__setup_synthesizer(self.llm_config)
        if not self.turn_based_conversation and task_id == 0:
            self.synthesizer_monitor_task = asyncio.create_task(self.tools['synthesizer'].monitor_connection())

        # # setting llm
        # llm = self.__setup_llm(self.llm_config)
        # # Setup tasks
        # self.__setup_tasks(llm)

        # setting llm
        if self.llm_config is not None:
            llm = self.__setup_llm(self.llm_config, task_id)
            #Setup tasks
            agent_params = {
                'llm': llm,
                'agent_type': self.llm_agent_config.get("agent_type","simple_llm_agent")
            }
            self.__setup_tasks(**agent_params)

        elif self.__is_multiagent():
            # Setup task for multiagent conversation
            for agent in self.task_config["tools_config"]["llm_agent"]['llm_config']['agent_map']:
                if 'routes' in self.llm_config_map[agent]:
                    del self.llm_config_map[agent]['routes'] #Remove routes from here as it'll create conflict ahead
                llm = self.__setup_llm(self.llm_config_map[agent])
                agent_type = self.llm_config_map[agent].get("agent_type", "simple_llm_agent")
                logger.info(f"Getting response for {llm} and agent type {agent_type} and {agent}")
                agent_params = {
                    'llm': llm,
                    'agent_type': agent_type
                }
                llm_agent = self.__setup_tasks(**agent_params)
                self.llm_agent_map[agent] = llm_agent

        elif self.task_config["task_type"] == "webhook":
            if "webhookURL" in self.task_config["tools_config"]["api_tools"]:
                webhook_url = self.task_config["tools_config"]["api_tools"]["webhookURL"]
            else:
                webhook_url = self.task_config["tools_config"]["api_tools"]["tools_params"]["webhook"]["url"]
            logger.info(f"Webhook URL {webhook_url}")
            self.tools["webhook_agent"] = WebhookAgent(webhook_url=webhook_url)

    @property
    def call_hangup_message(self):
        """Get the hangup message based on detected language."""
        detected_lang = self.language_detector.dominant_language if hasattr(self, 'language_detector') else None
        return select_message_by_language(self.call_hangup_message_config, detected_lang)

    def __is_multiagent(self):
        if self.task_config["task_type"] == "webhook":
            return False
        agent_type = self.task_config['tools_config']["llm_agent"].get("agent_type", None)
        return agent_type == "multiagent"

    def __is_knowledgebase_agent(self):
        if self.task_config["task_type"] == "webhook":
            return False
        agent_type = self.task_config['tools_config']["llm_agent"].get("agent_type", None)
        return agent_type == "knowledgebase_agent"

    def __is_graph_agent(self):
        if self.task_config["task_type"] == "webhook":
            return False
        agent_type = self.task_config['tools_config']["llm_agent"].get("agent_type", None)
        return agent_type == "graph_agent"
    
    # def __is_knowledge_agent(self):
    #     if self.task_config["task_type"] == "webhook":
    #         return False
    #     agent_type = self.task_config['tools_config']["llm_agent"].get("agent_type", None)
    #     return agent_type == "knowledge_agent"

    def _inject_language_instruction(self, messages: list) -> list:
        """Inject language instruction into messages based on detected language."""
        lang = self.language_detector.dominant_language
        if not lang or not self.language_injection_mode or not self.language_instruction_template:
            return messages

        try:
            lang_name = LANGUAGE_NAMES.get(lang, lang)
            instruction = self.language_instruction_template.format(language=lang_name) + "\n\n"

            if self.language_injection_mode == 'system_only':
                for i, msg in enumerate(messages):
                    if msg.get('role') == 'system':
                        messages[i]['content'] = instruction + msg['content']
                        logger.info(f"[system_only] Injected: {lang_name} ({lang})")
                        break
            elif self.language_injection_mode == 'per_turn':
                for i, msg in enumerate(messages):
                    if msg.get('role') == 'user':
                        messages[i]['content'] = instruction + msg['content']
                logger.info(f"[per_turn] Injected to {sum(1 for m in messages if m.get('role') == 'user')} user messages: {lang_name} ({lang})")
        except Exception as e:
            logger.error(f"Language injection error: {e}")

        return messages

    def __setup_routes(self, routes):
        embedding_model = routes.get("embedding_model", os.getenv("ROUTE_EMBEDDING_MODEL"))
        route_encoder = FastEmbedEncoder(name=embedding_model)

        routes_list = []
        self.vector_caches = {}
        self.route_responses_dict = {}
        for route in routes['routes']:
            logger.info(f"Setting up route {route}")
            utterances = route['utterances']
            r = Route(
                name=route['route_name'],
                utterances=utterances,
                score_threshold=route['score_threshold']
            )
            utterance_response_dict = {}
            if type(route['response']) is list and len(route['response']) == len(route['utterances']):
                for i, utterance in enumerate(utterances):
                    utterance_response_dict[utterance] =  route['response'][i]
                self.route_responses_dict[route['route_name']] = utterance_response_dict
            elif type(route['response']) is str:
                self.route_responses_dict[route['route_name']] = route['response']
            else:
                raise Exception("Invalid number of responses for the responses array")

            routes_list.append(r)
            if type(route['response']) is list:
                logger.info(f"Setting up vector cache for {route} and embedding model {embedding_model}")
                vector_cache = VectorCache(embedding_model=embedding_model)
                vector_cache.set(utterances)
                self.vector_caches[route['route_name']] = vector_cache

        self.route_layer = RouteLayer(encoder=route_encoder, routes=routes_list)
        logger.info("Routes are set")

    def __setup_output_handlers(self, turn_based_conversation, output_queue):
        output_kwargs = {"websocket": self.websocket}

        if self.task_config["tools_config"]["output"] is None:
            logger.info("Not setting up any output handler as it is none")
        elif self.task_config["tools_config"]["output"]["provider"] in SUPPORTED_OUTPUT_HANDLERS.keys():
            #Explicitly use default for turn based conversation as we expect to use HTTP endpoints
            if turn_based_conversation:
                logger.info("Connected through dashboard and hence using default output handler")
                output_handler_class = SUPPORTED_OUTPUT_HANDLERS.get("default")
            else:
                output_handler_class = SUPPORTED_OUTPUT_HANDLERS.get(self.task_config["tools_config"]["output"]["provider"])

                if self.task_config["tools_config"]["output"]["provider"] in SUPPORTED_OUTPUT_TELEPHONY_HANDLERS.keys():
                    output_kwargs['mark_event_meta_data'] = self.mark_event_meta_data
                    logger.info(f"Making sure that the sampling rate for output handler is 8000")
                    self.task_config['tools_config']['synthesizer']['provider_config']['sampling_rate'] = 8000
                    self.task_config['tools_config']['synthesizer']['audio_format'] = 'pcm'
                else:
                    self.task_config['tools_config']['synthesizer']['provider_config']['sampling_rate'] = 24000
                    output_kwargs['queue'] = output_queue
                self.sampling_rate = self.task_config['tools_config']['synthesizer']['provider_config']['sampling_rate']

            if self.task_config["tools_config"]["output"]["provider"] == "default":
                output_kwargs["is_web_based_call"] = self.is_web_based_call
                output_kwargs['mark_event_meta_data'] = self.mark_event_meta_data

            self.tools["output"] = output_handler_class(**output_kwargs)
            self.output_handler_set = True
            logger.info("output handler set")
        else:
            raise "Other input handlers not supported yet"

    async def message_task_new(self):
        tasks = []
        if self._is_conversation_task():
            tasks.append(self.tools['input'].handle())

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
                "is_welcome_message_played": True if self.task_config["tools_config"]["output"]["provider"] == 'default' and not self.is_web_based_call else False
            }

            if should_record:
                input_kwargs['conversation_recording'] = self.conversation_recording

            if self.turn_based_conversation:
                input_kwargs['turn_based_conversation'] = True
                input_handler_class = SUPPORTED_INPUT_HANDLERS.get("default")
                input_kwargs['queue'] = input_queue
            else:
                input_handler_class = SUPPORTED_INPUT_HANDLERS.get(
                    self.task_config["tools_config"]["input"]["provider"])

                if self.task_config['tools_config']['input']['provider'] == 'default':
                    input_kwargs['queue'] = input_queue

                input_kwargs["observable_variables"] = self.observable_variables
            self.tools["input"] = input_handler_class(**input_kwargs)
        else:
            raise "Other input handlers not supported yet"

    async def __forced_first_message(self, timeout=10.0):
        logger.info(f"Executing the first message task")
        try:
            start_time = asyncio.get_running_loop().time()
            while True:
                elapsed_time = asyncio.get_running_loop().time() - start_time
                if elapsed_time > timeout:
                    await self.__process_end_of_conversation()
                    logger.warning("Timeout reached while waiting for stream_sid")
                    break

                text = self.kwargs.get('agent_welcome_message', None)
                meta_info = {'io': self.tools["output"].get_provider(), 'message_category': 'agent_welcome_message', "request_id": str(uuid.uuid4()), "cached": True, "sequence_id": -1, 'format': self.task_config["tools_config"]["output"]["format"], 'text': text, 'end_of_llm_stream': True}
                ws_data_packet = create_ws_data_packet(text, meta_info=meta_info)

                meta_info = ws_data_packet["meta_info"]
                text = ws_data_packet["data"]
                meta_info["type"] = "audio"
                meta_info["synthesizer_start_time"] = time.time()

                audio_chunk = self.preloaded_welcome_audio if self.preloaded_welcome_audio else None
                if meta_info['text'] == '':
                    audio_chunk = None

                meta_info["format"] = "pcm"
                meta_info['is_first_chunk'] = True
                meta_info["end_of_synthesizer_stream"] = True
                meta_info['chunk_id'] = 1
                meta_info["is_first_chunk_of_entire_response"] = True
                meta_info["is_final_chunk_of_entire_response"] = True
                message = create_ws_data_packet(audio_chunk, meta_info)

                stream_sid = self.tools["input"].get_stream_sid()
                if stream_sid is not None and self.output_handler_set:
                    self.stream_sid_ts = time.time() * 1000
                    logger.info(f"Got stream sid and hence sending the first message {stream_sid}")
                    self.stream_sid = stream_sid
                    await self.tools["output"].set_stream_sid(stream_sid)
                    self.tools["input"].update_is_audio_being_played(True)
                    convert_to_request_log(message=text, meta_info=meta_info, component="synthesizer", direction="response", model=self.synthesizer_provider, is_cached=meta_info.get("is_cached", False), engine=self.tools['synthesizer'].get_engine(), run_id=self.run_id)
                    await self.tools["output"].handle(message)
                    try:
                        data = message.get("data")
                        if data is not None:
                            duration = calculate_audio_duration(len(data), self.sampling_rate,
                                                                format=message['meta_info']['format'])
                            self.conversation_recording['output'].append(
                                {'data': data, "start_time": time.time(), "duration": duration})
                    except Exception as e:
                        duration = 0.256
                        logger.error("Exception in __forced_first_message for duration calculation: {}".format(str(e)))
                    break
                else:
                    logger.info(f"Stream id is still None ({stream_sid}) or output handler not set ({self.output_handler_set}), waiting...")
                    await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Exception in __forced_first_message {str(e)}")

        return

    def __setup_transcriber(self):
        try:
            if self.task_config["tools_config"]["transcriber"] is not None:
                self.language = self.task_config["tools_config"]["transcriber"].get('language', DEFAULT_LANGUAGE_CODE)
                if self.turn_based_conversation:
                    provider = "playground"
                elif self.is_web_based_call:
                    provider = "web_based_call"
                else:
                    provider = self.task_config["tools_config"]["input"]["provider"]

                self.task_config["tools_config"]["transcriber"]["input_queue"] = self.audio_queue
                self.task_config['tools_config']["transcriber"]["output_queue"] = self.transcriber_output_queue

                # Checking models for backwards compatibility
                if self.task_config["tools_config"]["transcriber"]["model"] in SUPPORTED_TRANSCRIBER_MODELS.keys() or self.task_config["tools_config"]["transcriber"]["provider"] in SUPPORTED_TRANSCRIBER_PROVIDERS.keys():
                    if self.turn_based_conversation:
                        self.task_config["tools_config"]["transcriber"]["stream"] = True if self.enforce_streaming else False
                        logger.info(f'self.task_config["tools_config"]["transcriber"]["stream"] {self.task_config["tools_config"]["transcriber"]["stream"]} self.enforce_streaming {self.enforce_streaming}')
                    if 'provider' in self.task_config["tools_config"]["transcriber"]:
                        transcriber_class = SUPPORTED_TRANSCRIBER_PROVIDERS.get(
                            self.task_config["tools_config"]["transcriber"]["provider"])
                    else:
                        transcriber_class = SUPPORTED_TRANSCRIBER_MODELS.get(
                            self.task_config["tools_config"]["transcriber"]["model"])
                    self.tools["transcriber"] = transcriber_class(provider, **self.task_config["tools_config"]["transcriber"], **self.kwargs)
        except Exception as e:
            logger.error(f"Something went wrong with starting transcriber {e}")

    def __setup_synthesizer(self, llm_config=None):
        if self._is_conversation_task():
            self.kwargs["use_turbo"] = self.task_config["tools_config"]["transcriber"]["language"] == DEFAULT_LANGUAGE_CODE
        if self.task_config["tools_config"]["synthesizer"] is not None:
            if "caching" in self.task_config['tools_config']['synthesizer']:
                caching = self.task_config["tools_config"]["synthesizer"].pop("caching")
            else:
                caching = True

            self.synthesizer_provider = self.task_config["tools_config"]["synthesizer"].pop("provider")
            synthesizer_class = SUPPORTED_SYNTHESIZER_MODELS.get(self.synthesizer_provider)
            provider_config = self.task_config["tools_config"]["synthesizer"].pop("provider_config")
            self.synthesizer_voice = provider_config["voice"]
            if self.turn_based_conversation:
                self.task_config["tools_config"]["synthesizer"]["audio_format"] = "mp3" # Hard code mp3 if we're connected through dashboard
                self.task_config["tools_config"]["synthesizer"]["stream"] = True if self.enforce_streaming else False #Hardcode stream to be False as we don't want to get blocked by a __listen_synthesizer co-routine

            self.tools["synthesizer"] = synthesizer_class(**self.task_config["tools_config"]["synthesizer"], **provider_config, **self.kwargs, caching=caching)
            # if not self.turn_based_conversation:
            #     self.synthesizer_monitor_task = asyncio.create_task(self.tools['synthesizer'].monitor_connection())
            if self.task_config["tools_config"]["llm_agent"] is not None and llm_config is not None:
                llm_config["buffer_size"] = self.task_config["tools_config"]["synthesizer"].get('buffer_size')

    def __setup_llm(self, llm_config, task_id=0):
        if self.task_config["tools_config"]["llm_agent"] is not None:
            if task_id and task_id > 0:
                self.kwargs.pop('llm_key', None)
                self.kwargs.pop('base_url', None)
                self.kwargs.pop('api_version', None)

                if self._is_summarization_task() or self._is_extraction_task():
                    llm_config['model'] = LLM_DEFAULT_CONFIGS["summarization"]["model"]
                    llm_config['provider'] = LLM_DEFAULT_CONFIGS["summarization"]["provider"]

            if llm_config["provider"] in SUPPORTED_LLM_PROVIDERS.keys():
                llm_class = SUPPORTED_LLM_PROVIDERS.get(llm_config["provider"])
                llm = llm_class(language=self.language, **llm_config, **self.kwargs)
                return llm
            else:
                raise Exception(f'LLM {llm_config["provider"]} not supported')

    def __get_agent_object(self, llm, agent_type, assistant_config=None):
        self.agent_type = agent_type
        if agent_type == "simple_llm_agent":
            llm_agent = StreamingContextualAgent(llm)
        elif agent_type == "graph_agent":
            logger.info("Setting up graph agent with rag-proxy-server support")
            llm_config = self.task_config["tools_config"]["llm_agent"].get("llm_config", {})
            rag_server_url = self.kwargs.get('rag_server_url', os.getenv('RAG_SERVER_URL', 'http://localhost:8000'))
            
            logger.info(f"Graph agent config: {llm_config}")
            logger.info(f"RAG server URL: {rag_server_url}")
            
            # Set RAG server URL in environment for GraphAgent to use
            os.environ['RAG_SERVER_URL'] = rag_server_url
            
            llm_agent = GraphAgent(llm_config)
            logger.info("Graph agent created with rag-proxy-server support")
        elif agent_type == "knowledgebase_agent":
            logger.info("Setting up knowledge agent with rag-proxy-server support")
            llm_config = self.task_config["tools_config"]["llm_agent"].get("llm_config", {})
            rag_server_url = self.kwargs.get('rag_server_url', os.getenv('RAG_SERVER_URL', 'http://localhost:8000'))
            
            logger.info(f"Knowledge agent config: {llm_config}")
            logger.info(f"RAG server URL: {rag_server_url}")
            
            # Set RAG server URL in environment for KnowledgeAgent to use
            os.environ['RAG_SERVER_URL'] = rag_server_url
            
            # Inject provider credentials and endpoints into KnowledgeAgent config
            injected_cfg = dict(llm_config)
            if 'llm_key' in self.kwargs:
                injected_cfg['llm_key'] = self.kwargs['llm_key']
            if 'base_url' in self.kwargs:
                injected_cfg['base_url'] = self.kwargs['base_url']
            if 'api_version' in self.kwargs:
                injected_cfg['api_version'] = self.kwargs['api_version']
            if 'api_tools' in self.kwargs:
                injected_cfg['api_tools'] = self.kwargs['api_tools']
            injected_cfg['buffer_size'] = self.task_config["tools_config"]["synthesizer"].get('buffer_size')
            injected_cfg['language'] = self.language

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
            if self.context_data and 'recipient_data' in self.context_data and self.context_data['recipient_data'] and self.context_data['recipient_data'].get('timezone', None):
                self.timezone = pytz.timezone(self.context_data['recipient_data']['timezone'])
        current_date, current_time = get_date_time_from_timezone(self.timezone)

        prompt_responses = kwargs.get('prompt_responses', None)
        if not prompt_responses:
            prompt_responses = await get_prompt_responses(assistant_id=self.assistant_id, local=self.is_local)

        current_task = "task_{}".format(task_id + 1)
        if self.__is_multiagent():
            logger.info(f"Getting {current_task} from prompt responses of type {type(prompt_responses)}, prompt responses key {prompt_responses.keys()}")
            prompts = prompt_responses.get(current_task, None)
            self.prompt_map = {}
            for agent in self.task_config["tools_config"]["llm_agent"]['llm_config']['agent_map']:
                prompt = prompts[agent]['system_prompt']
                prompt = self.__prefill_prompts(self.task_config, prompt, self.task_config['task_type'])
                prompt = self.__get_final_prompt(prompt, current_date, current_time, self.timezone)
                if agent == self.task_config["tools_config"]["llm_agent"]['llm_config']['default_agent']:
                    self.system_prompt = {
                        'role': 'system',
                        'content': prompt
                    }
                self.prompt_map[agent] = prompt
            logger.info(f"Initialised prompt dict {self.prompt_map}, Set default prompt {self.system_prompt}")
        else:
            self.prompts = self.__prefill_prompts(self.task_config, prompt_responses.get(current_task, None), self.task_config['task_type'])

        if "system_prompt" in self.prompts:
            # This isn't a graph based agent
            enriched_prompt = self.prompts["system_prompt"]
            if self.context_data and self.context_data.get('recipient_data', {}).get('call_sid'):
                self.call_sid = self.context_data['recipient_data']['call_sid']

            enriched_prompt = structure_system_prompt(self.prompts["system_prompt"], self.run_id, self.assistant_id, self.call_sid, self.context_data, self.timezone, self.is_web_based_call)

            notes = ""
            if self._is_conversation_task() and self.use_fillers:
                notes = "### Note:\n"
                notes += f"1.{FILLER_PROMPT}\n"

            final_prompt = f"\n## Agent Prompt:\n\n{enriched_prompt}\n{notes}\n\n## Transcript:\n"
            self.prompts["system_prompt"] = final_prompt

            self.system_prompt = {
                'role': "system",
                'content': final_prompt
            }
        else:
            self.system_prompt = {
                'role': "system",
                'content': ""
            }

        if len(self.system_prompt['content']) == 0:
            self.history = [] if len(self.history) == 0 else self.history
        else:
            self.history = [self.system_prompt] if len(self.history) == 0 else [self.system_prompt] + self.history

        # If using knowledge_agent, inject the prompt into agent config so agent can read it
        try:
            if self.__is_knowledgebase_agent() and 'llm_agent' in self.task_config['tools_config']:
                if 'llm_config' in self.task_config['tools_config']['llm_agent']:
                    self.task_config['tools_config']['llm_agent']['llm_config']['prompt'] = self.system_prompt['content']
        except Exception as e:
            logger.error(f"Failed to inject prompt into knowledge agent config: {e}")

        #If history is empty and agent welcome message is not empty add it to history
        if task_id == 0 and len(self.history) == 1 and len(self.kwargs['agent_welcome_message']) != 0:
            self.history.append({'role': 'assistant', 'content': self.kwargs['agent_welcome_message']})

        self.interim_history = copy.deepcopy(self.history)

    def __prefill_prompts(self, task, prompt, task_type):
        if self.context_data and 'recipient_data' in self.context_data and self.context_data[
            'recipient_data'] and self.context_data['recipient_data'].get('timezone', None):
            self.timezone = pytz.timezone(self.context_data['recipient_data']['timezone'])
        current_date, current_time = get_date_time_from_timezone(self.timezone)

        if not prompt and task_type in ('extraction', 'summarization'):
            if task_type == 'extraction':
                extraction_json = task.get("tools_config").get('llm_agent', {}).get('llm_config', {}).get('extraction_json')
                prompt = EXTRACTION_PROMPT.format(current_date, current_time, self.timezone, extraction_json)
                return {"system_prompt": prompt}
            elif task_type == 'summarization':
                return {"system_prompt": SUMMARIZATION_PROMPT}
        return prompt

    def __process_stop_words(self, text_chunk, meta_info):
         #THis is to remove stop words. Really helpful in smaller 7B models
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
            return original_stream[:index + len(heard_text)]

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

            if not response_heard:
                pending_marks = [{'mark_id': k, 'mark_data': v} for k, v in mark_events_data]
                pending_chunks = []
                for mark in pending_marks:
                    mark_data = mark.get('mark_data', {})
                    mark_type = mark_data.get('type', '')
                    text = mark_data.get('text_synthesized', '')
                    if mark_type in ['pre_mark_message', 'ambient_noise', 'backchanneling'] or not text:
                        continue
                    pending_chunks.append({
                        'text': text,
                        'duration': mark_data.get('duration', 0),
                        'sent_ts': mark_data.get('sent_ts', 0)
                    })

                if pending_chunks:
                    first_sent_ts = pending_chunks[0].get('sent_ts', 0)
                    plivo_latency = self.tools["input"].get_calculated_plivo_latency()
                    if first_sent_ts > 0:
                        time_since_first_send = interruption_processed_at - first_sent_ts
                        actual_play_time = max(0, time_since_first_send - plivo_latency)
                    else:
                        elapsed_time = interruption_processed_at - self.tools["input"].get_current_mark_started_time()
                        actual_play_time = max(0, elapsed_time - plivo_latency)

                    played_text = []
                    cumulative_duration = 0
                    for chunk in pending_chunks:
                        if cumulative_duration >= actual_play_time:
                            break
                        chunk_duration = chunk['duration']
                        if cumulative_duration + chunk_duration <= actual_play_time:
                            played_text.append(chunk['text'])
                        else:
                            remaining_time = actual_play_time - cumulative_duration
                            proportion = remaining_time / chunk_duration if chunk_duration > 0 else 0
                            char_count = int(len(chunk['text']) * proportion)
                            partial_text = chunk['text'][:char_count]
                            if partial_text and char_count < len(chunk['text']):
                                last_space = partial_text.rfind(' ')
                                if last_space > 0:
                                    partial_text = partial_text[:last_space]
                            if partial_text:
                                played_text.append(partial_text)
                        cumulative_duration += chunk_duration

                    if played_text:
                        response_heard = ''.join(played_text)
                else:
                    # No pending content marks - response likely completed normally
                    return

            # Update last assistant message in history
            for i in range(len(self.history) - 1, -1, -1):
                if self.history[i]['role'] == 'assistant':
                    original = self.history[i]['content']
                    if original is None:
                        continue
                    updated = self.update_transcript_for_interruption(original, response_heard)
                    if not updated or not updated.strip():
                        self.history.pop(i)
                    else:
                        self.history[i]['content'] = updated
                    break

            for i in range(len(self.interim_history) - 1, -1, -1):
                if self.interim_history[i]['role'] == 'assistant':
                    original = self.interim_history[i]['content']
                    if original is None:
                        continue
                    updated = self.update_transcript_for_interruption(original, response_heard)
                    if not updated or not updated.strip():
                        self.interim_history.pop(i)
                    else:
                        self.interim_history[i]['content'] = updated
                    break

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
        await self.tools["synthesizer"].flush_synthesizer_stream()

        #Stop the output loop first so that we do not transmit anything else
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

        # Cancel voicemail check task if running
        if self.voicemail_check_task is not None and not self.voicemail_check_task.done():
            logger.info("Cancelling voicemail check task")
            self.voicemail_check_task.cancel()
            self.voicemail_check_task = None

        # self.synthesizer_task.cancel()
        # self.synthesizer_task = asyncio.create_task(self.__listen_synthesizer())
        #for task in self.synthesizer_tasks:
        #    task.cancel()

        #self.synthesizer_tasks = []

        logger.info(f"Synth Task cancelled seconds")
        if not self.buffered_output_queue.empty():
            logger.info(f"Output queue was not empty and hence emptying it")
            self.buffered_output_queue = asyncio.Queue()

        #restart output task
        self.output_task = asyncio.create_task(self.__process_output_loop())
        self.started_transmitting_audio = False #Since we're interrupting we need to stop transmitting as well
        self.last_transmitted_timestamp = time.time()
        logger.info(f"Cleaning up downstream tasks. Time taken to send a clear message {time.time() - start_time}")

    def __get_updated_meta_info(self, meta_info = None):
        #This is used in case there's silence from callee's side
        if meta_info is None:
            meta_info = self.tools["transcriber"].get_meta_info()
            logger.info(f"Metainfo {meta_info}")
        meta_info_copy = meta_info.copy()

        new_sequence_id = self.interruption_manager.get_next_sequence_id()
        meta_info_copy["sequence_id"] = new_sequence_id
        meta_info_copy['turn_id'] = self.interruption_manager.get_turn_id()

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
            return next((self.pipelines[sequence][i + 1] for i in range(len(self.pipelines[sequence]) - 1) if
                         self.pipelines[sequence][i] == origin), "output")
        except Exception as e:
            logger.error(f"Error getting next step: {e}")

    def _set_call_details(self, message):
        if self.call_sid is not None and self.stream_sid is not None and "call_sid" not in message['meta_info'] and "stream_sid" not in message['meta_info']:
            return

        if "call_sid" in message.get("meta_info", {}):
            self.call_sid = message['meta_info']["call_sid"]
        if "stream_sid" in message.get("meta_info", {}):
            self.stream_sid = message['meta_info']["stream_sid"]

    async def _process_followup_task(self, message=None):
        if self.task_config["task_type"] == "webhook":
            logger.info(f"Input patrameters {self.input_parameters}")
            extraction_details = self.input_parameters.get('extraction_details', {})
            logger.info(f"DOING THE POST REQUEST TO WEBHOOK {extraction_details}")
            self.webhook_response = await self.tools["webhook_agent"].execute(extraction_details)
            logger.info(f"Response from the server {self.webhook_response}")
        else:
            message = format_messages(self.input_parameters["messages"], include_tools=True)  # Remove the initial system prompt
            self.history.append({
                'role': 'user',
                'content': message
            })

            start_time = time.time()
            json_data = await self.tools["llm_agent"].generate(self.history)
            latency_ms = (time.time() - start_time) * 1000
            
            if self.task_config["task_type"] == "summarization":
                self.summarized_data = json_data["summary"]
                self.llm_latencies['other_latencies'].append({
                    'type': 'summarization',
                    "latency_ms": latency_ms,
                    "model": LLM_DEFAULT_CONFIGS["summarization"]["model"],
                    "provider": LLM_DEFAULT_CONFIGS["summarization"]["provider"]
                })
            else:
                json_data = clean_json_string(json_data)
                if type(json_data) is not dict:
                    json_data = json.loads(json_data)
                self.extracted_data = json_data
                self.llm_latencies['other_latencies'].append({
                    "type": 'extraction',
                    "latency_ms": latency_ms,
                    "model": LLM_DEFAULT_CONFIGS["extraction"]["model"],
                    "provider": LLM_DEFAULT_CONFIGS["extraction"]["provider"]
                })

    # This observer works only for messages which have sequence_id != -1
    def final_chunk_played_observer(self, is_final_chunk_played):
        logger.info(f'Updating last_transmitted_timestamp')
        self.last_transmitted_timestamp = time.time()

    async def agent_hangup_observer(self, is_agent_hangup):
        logger.info(f"agent_hangup_observer triggered with is_agent_hangup = {is_agent_hangup}")
        if is_agent_hangup:
            self.tools["output"].set_hangup_sent()
            await self.__process_end_of_conversation()

    async def wait_for_current_message(self):
        while not self.conversation_ended:
            mark_events = self.mark_event_meta_data.mark_event_meta_data
            mark_items_list = [{'mark_id': k, 'mark_data': v} for k, v in mark_events.items()]
            logger.info(f"current_list: {mark_items_list}")

            if not mark_items_list:
                break

            first_item = mark_items_list[0]['mark_data']
            if len(mark_items_list) == 1 and first_item.get('type') == 'pre_mark_message':
                break

            # plivo mark_event bug
            if len(mark_items_list) == 2:
                second_item = mark_items_list[1]['mark_data']
                if first_item.get('type') == 'agent_hangup' and first_item.get('text_synthesized') == '' and second_item.get('type') == 'pre_mark_message':
                    break

            if first_item.get('text_synthesized') and first_item.get('is_final_chunk') is True:
                break

            await asyncio.sleep(0.5)
        return

    async def inject_digits_to_conversation(self) -> None:
        while True:
            try:
                dtmf_digits = await self.queues["dtmf"].get()
                logger.info(f"DTMF collected {dtmf_digits}")

                dtmf_message = "dtmf_number: " + dtmf_digits
                base_meta_info = {
                    'io': self.tools['input'].io_provider,
                    'type': 'text',
                    'sequence': 0,
                    'origin': 'dtmf'
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

        # Close output handler to prevent sends after websocket close
        if "output" in self.tools and self.tools["output"] is not None:
            self.tools["output"].close()

        await self.tools["input"].stop_handler()
        logger.info("Stopped input handler")
        if "transcriber" in self.tools and not self.turn_based_conversation:
            logger.info("Stopping transcriber")
            await self.tools["transcriber"].toggle_connection()
            await asyncio.sleep(2)  # Making sure whatever message was passed is over
            
        # Cancel voicemail check task if still running
        if self.voicemail_check_task is not None and not self.voicemail_check_task.done():
            logger.info("Cancelling voicemail check task during conversation end")
            self.voicemail_check_task.cancel()
            self.voicemail_check_task = None

    def __update_preprocessed_tree_node(self):
        logger.info(f"It's a preprocessed flow and hence updating current node")
        self.tools['llm_agent'].update_current_node()


    ##############################################################
    # LLM task
    ##############################################################
    async def _handle_llm_output(self, next_step, text_chunk, should_bypass_synth, meta_info, is_filler=False, is_function_call=False):
        if "request_id" not in meta_info:
            meta_info["request_id"] = str(uuid.uuid4())

        if not self.stream and not is_filler:
            first_buffer_latency = time.time() - meta_info["llm_start_time"]
            meta_info["llm_first_buffer_generation_latency"] = first_buffer_latency

        elif is_filler:
            logger.info(f"It's a filler message and hence adding required metadata")
            meta_info['origin'] = "classifier"
            meta_info['cached'] = True
            meta_info['local'] = True
            meta_info['message_category'] = 'filler'

        if next_step == "synthesizer" and not should_bypass_synth:
            task = asyncio.create_task(self._synthesize(create_ws_data_packet(text_chunk, meta_info)))
            self.synthesizer_tasks.append(asyncio.ensure_future(task))
        elif self.tools["output"] is not None:
            logger.info("Synthesizer not the next step and hence simply returning back")
            overall_time = time.time() - meta_info["llm_start_time"]
            #self.history = copy.deepcopy(self.interim_history)
            if is_function_call:
                bos_packet = create_ws_data_packet("<beginning_of_stream>", meta_info)
                await self.tools["output"].handle(bos_packet)
                await self.tools["output"].handle(create_ws_data_packet(text_chunk, meta_info))
                eos_packet = create_ws_data_packet("<end_of_stream>", meta_info)
                await self.tools["output"].handle(eos_packet)
            else:
                await self.tools["output"].handle(create_ws_data_packet(text_chunk, meta_info))

    async def _process_conversation_preprocessed_task(self, message, sequence, meta_info):
        if self.task_config["tools_config"]["llm_agent"]['agent_flow_type'] == "preprocessed":
            messages = copy.deepcopy(self.history)
            # TODO revisit this
            messages.append({'role': 'user', 'content': message['data']})
            logger.info(f"Starting LLM Agent {messages}")
            #Expose get current classification_response method from the agent class and use it for the response log
            convert_to_request_log(message=format_messages(messages, use_system_prompt= True), meta_info= meta_info, component="llm", direction="request", model=self.llm_agent_config["model"], is_cached= True, run_id= self.run_id)
            async for next_state in self.tools['llm_agent'].generate(messages, label_flow=self.label_flow):
                if next_state == "<end_of_conversation>":
                    meta_info["end_of_conversation"] = True
                    self.buffered_output_queue.put_nowait(create_ws_data_packet("<end_of_conversation>", meta_info))
                    return

                logger.info(f"Text chunk {next_state['text']}")
                # TODO revisit this
                messages.append({'role': 'assistant', 'content': next_state['text']})
                self.synthesizer_tasks.append(asyncio.create_task(
                        self._synthesize(create_ws_data_packet(next_state['audio'], meta_info, is_md5_hash=True))))
            logger.info(f"Interim history after the LLM task {messages}")
            self.llm_response_generated = True
            self.interim_history = copy.deepcopy(messages)
            # if self.callee_silent:
            #     logger.info("When we got utterance end, maybe LLM was still generating response. So, copying into history")
            #     self.history = copy.deepcopy(self.interim_history)

    async def _process_conversation_formulaic_task(self, message, sequence, meta_info):
        llm_response = ""
        logger.info("Agent flow is formulaic and hence moving smoothly")
        async for text_chunk in self.tools['llm_agent'].generate(self.history):
            if is_valid_md5(text_chunk):
                self.synthesizer_tasks.append(asyncio.create_task(
                    self._synthesize(create_ws_data_packet(text_chunk, meta_info, is_md5_hash=True))))
            else:
                # TODO Make it more modular
                llm_response += " " +text_chunk
                next_step = self._get_next_step(sequence, "llm")
                if next_step == "synthesizer":
                    self.synthesizer_tasks.append(asyncio.create_task(self._synthesize(create_ws_data_packet(text_chunk, meta_info))))
                else:
                    logger.info(f"Sending output text {sequence}")
                    await self.tools["output"].handle(create_ws_data_packet(text_chunk, meta_info))
                    self.synthesizer_tasks.append(asyncio.create_task(
                        self._synthesize(create_ws_data_packet(text_chunk, meta_info, is_md5_hash=False))))

    async def __filler_classification_task(self, message):
        logger.info(f"doing the classification task")
        sequence, meta_info = self._extract_sequence_and_meta(message)
        next_step = self._get_next_step(sequence, "llm")
        start_time = time.perf_counter()
        filler_class = self.filler_classifier.classify(message['data'])
        logger.info(f"doing the classification task in {time.perf_counter() - start_time}")
        new_meta_info = copy.deepcopy(meta_info)
        self.current_filler = filler_class
        should_bypass_synth = 'bypass_synth' in meta_info and meta_info['bypass_synth'] == True
        filler = random.choice((FILLER_DICT[filler_class]))
        await self._handle_llm_output(next_step, filler, should_bypass_synth, new_meta_info, is_filler=True)

    async def __execute_function_call(self, url, method, param, api_token, headers, model_args, meta_info, next_step, called_fun, **resp):
        self.check_if_user_online = False

        if called_fun.startswith("transfer_call"):
            await asyncio.sleep(2)

            try:
                from_number = self.context_data['recipient_data']['from_number']
            except Exception as e:
                from_number = None

            call_sid = None
            call_transfer_number = None
            payload = {
                'call_sid': call_sid,
                'provider': self.tools['input'].io_provider,
                'stream_sid': self.stream_sid,
                'from_number': from_number,
                'execution_id': self.run_id,
                **(self.transfer_call_params or {})
            }

            if self.tools['input'].io_provider != 'default':
                call_sid = self.tools["input"].get_call_sid()
                payload['call_sid'] = call_sid

            if url is None:
                url = os.getenv("CALL_TRANSFER_WEBHOOK_URL")

                try:
                    json_function_call_params = copy.deepcopy(param)
                    if isinstance(param, str):
                        json_function_call_params = json.loads(param)
                    call_transfer_number = json_function_call_params['call_transfer_number']
                    if call_transfer_number:
                        payload['call_transfer_number'] = call_transfer_number
                except Exception as e:
                    logger.error(f"Error in __execute_function_call {e}")

            if param is not None:
                logger.info(f"Gotten response {resp}")
                payload = {**payload, **resp}

            if self.tools['input'].io_provider == 'default':
                mock_response = f"This is a mocked response demonstrating a successful transfer of call to {call_transfer_number}"
                convert_to_request_log(str(payload), meta_info, None, "function_call", direction="request", run_id=self.run_id)
                convert_to_request_log(mock_response, meta_info, None, "function_call", direction="response", run_id=self.run_id)

                bos_packet = create_ws_data_packet("<beginning_of_stream>", meta_info)
                await self.tools["output"].handle(bos_packet)
                await self.tools["output"].handle(create_ws_data_packet(mock_response, meta_info))
                eos_packet = create_ws_data_packet("<end_of_stream>", meta_info)
                await self.tools["output"].handle(eos_packet)
                return

            async with aiohttp.ClientSession() as session:
                logger.info(f"Sending the payload to stop the conversation {payload} url {url}")
                while self.tools["input"].is_audio_being_played_to_user():
                    await asyncio.sleep(1)
                convert_to_request_log(str(payload), meta_info, None, "function_call", direction="request", is_cached=False,
                                       run_id=self.run_id)
                async with session.post(url, json = payload) as response:
                    response_text = await response.text()
                    logger.info(f"Response from the server after call transfer: {response_text}")
                    convert_to_request_log(str(response_text), meta_info, None, "function_call", direction="response", is_cached=False, run_id=self.run_id)
                    return

        response = await trigger_api(url=url, method=method.lower(), param=param, api_token=api_token, headers_data=headers, meta_info=meta_info, run_id=self.run_id, **resp)
        function_response = str(response)
        get_res_keys, get_res_values = await computed_api_response(function_response)
        if called_fun.startswith('check_availability_of_slots') and (not get_res_values or (len(get_res_values) == 1 and len(get_res_values[0]) == 0)):
            set_response_prompt = []
        elif called_fun.startswith('book_appointment') and 'id' not in get_res_keys:
            if get_res_values and get_res_values[0] == 'no_available_users_found_error':
                function_response = "Sorry, the host isn't available at this time. Are you available at any other time?"
            set_response_prompt = []
        else:
            set_response_prompt = function_response

        textual_response = resp.get("textual_response", None)
        self.history.append({"role": "assistant", "content": textual_response, "tool_calls": resp["model_response"]})
        self.history.append({"role": "tool", "tool_call_id": resp.get("tool_call_id", ""), "content": function_response})
        model_args["messages"].append({"role": "assistant", "content": textual_response, "tool_calls": resp["model_response"]})
        model_args["messages"].append({"role": "tool", "tool_call_id": resp.get("tool_call_id", ""), "content": function_response})

        logger.info(f"Logging function call parameters ")
        convert_to_request_log(function_response, meta_info , None, "function_call", direction = "response", is_cached= False, run_id = self.run_id)

        convert_to_request_log(format_messages(model_args['messages'], True), meta_info, self.llm_config['model'], "llm", direction = "request", is_cached= False, run_id = self.run_id)
        self.check_if_user_online = self.conversation_config.get("check_if_user_online", True)

        if not called_fun.startswith("transfer_call"):
            should_bypass_synth = meta_info.get('bypass_synth', False)
            await self.__do_llm_generation(model_args["messages"], meta_info, next_step, should_bypass_synth=should_bypass_synth, should_trigger_function_call=True)

        self.execute_function_call_task = None

    def __store_into_history(self, meta_info, messages, llm_response, should_trigger_function_call = False):
        # TODO revisit this
        # if self.current_request_id in self.llm_rejected_request_ids:
        if False:
            logger.info("##### User spoke while LLM was generating response")
        else:
            self.llm_response_generated = True
            convert_to_request_log(message=llm_response, meta_info= meta_info, component="llm", direction="response", model=self.llm_config["model"], run_id= self.run_id)
            if should_trigger_function_call:
                logger.info(f"There was a function call and need to make that work")
                self.history.append({"role": "assistant", "content": llm_response})
                #Assuming that callee was silent
                # self.history = copy.deepcopy(self.interim_history)
            else:
                messages.append({"role": "assistant", "content": llm_response})
                self.history.append({"role": "assistant", "content": llm_response})
                self.interim_history = copy.deepcopy(messages)
                # if self.callee_silent:
                #     logger.info("##### When we got utterance end, maybe LLM was still generating response. So, copying into history")
                #     self.history = copy.deepcopy(self.interim_history)
                #self.__update_transcripts()

    async def __do_llm_generation(self, messages, meta_info, next_step, should_bypass_synth=False, should_trigger_function_call=False):
        # Reset response tracking for new turn
        if self.generate_precise_transcript:
            self.tools["input"].reset_response_heard_by_user()

        llm_response, function_tool, function_tool_message = '', '', ''
        synthesize = True
        if should_bypass_synth:
            synthesize = False

        # Inject language instruction if detection complete
        messages = self._inject_language_instruction(messages)

        # Pass detected language to LLM for pre_call_message selection
        detected_lang = self.language_detector.dominant_language
        if detected_lang:
            meta_info['detected_language'] = detected_lang

        async for llm_message in self.tools['llm_agent'].generate(messages, synthesize=synthesize, meta_info=meta_info):
            if isinstance(llm_message, dict) and 'messages' in llm_message: # custom list of messages before the llm call
                convert_to_request_log(format_messages(llm_message['messages'], True), meta_info, self.llm_config['model'], "llm", direction="request", is_cached=False, run_id=self.run_id)
                continue

            data, end_of_llm_stream, latency, trigger_function_call, function_tool, function_tool_message = llm_message

            if trigger_function_call:
                logger.info(f"Triggering function call for {data}")
                self.llm_task = asyncio.create_task(self.__execute_function_call(next_step = next_step, **data))
                return

            if latency:
                previous_latency_item = self.llm_latencies['turn_latencies'][-1] if self.llm_latencies['turn_latencies'] else None
                if previous_latency_item and previous_latency_item.get('sequence_id') == latency.get('sequence_id'):
                    self.llm_latencies['turn_latencies'][-1] = latency
                else:
                    self.llm_latencies['turn_latencies'].append(latency)

            llm_response += " " + data

            logger.info(f"Got a response from LLM {llm_response}")
            if end_of_llm_stream:
                meta_info["end_of_llm_stream"] = True

            if self.stream:
                text_chunk = self.__process_stop_words(data, meta_info)

                # A hack as during the 'await' part control passes to llm streaming function parameters
                # So we have to make sure we've commited the filler message
                detected_lang = self.language_detector.dominant_language or self.language
                filler_message = compute_function_pre_call_message(detected_lang, function_tool, function_tool_message)
                #filler_message = PRE_FUNCTION_CALL_MESSAGE.get(self.language, PRE_FUNCTION_CALL_MESSAGE[DEFAULT_LANGUAGE_CODE])
                if text_chunk == filler_message:
                    logger.info("Got a pre function call message")
                    messages.append({'role':'assistant', 'content': filler_message})
                    self.history.append({'role': 'assistant', 'content': filler_message})
                    self.interim_history = copy.deepcopy(messages)

                await self._handle_llm_output(next_step, text_chunk, should_bypass_synth, meta_info)

        detected_lang = self.language_detector.dominant_language or self.language
        filler_message = compute_function_pre_call_message(detected_lang, function_tool, function_tool_message)
        if self.stream and llm_response != filler_message:
            self.__store_into_history(meta_info, messages, llm_response, should_trigger_function_call= should_trigger_function_call)
        elif not self.stream:
            llm_response = llm_response.strip()
            if self.turn_based_conversation:
                self.history.append({"role": "assistant", "content": llm_response})
            await self._handle_llm_output(next_step, llm_response, should_bypass_synth, meta_info, is_function_call=should_trigger_function_call)
            convert_to_request_log(message=llm_response, meta_info=meta_info, component="llm", direction="response", model=self.llm_config["model"], run_id=self.run_id)

        # Collect RAG latency if present (from KnowledgeBaseAgent)
        if meta_info.get('rag_latency'):
            rag_latency = meta_info['rag_latency']
            existing_seq_ids = [t.get('sequence_id') for t in self.rag_latencies['turn_latencies']]
            if rag_latency.get('sequence_id') not in existing_seq_ids:
                self.rag_latencies['turn_latencies'].append(rag_latency)

    async def _process_conversation_task(self, message, sequence, meta_info):
        should_bypass_synth = 'bypass_synth' in meta_info and meta_info['bypass_synth'] is True
        next_step = self._get_next_step(sequence, "llm")
        meta_info['llm_start_time'] = time.time()
        route = None

        if self.__is_multiagent():
            tasks = [(lambda: get_route_info(message['data'], self.agent_routing))]
            if self.route_layer is not None:
                tasks.append(lambda: get_route_info(message['data'], self.route_layer))
            tasks_op = await asyncio.gather(*tasks)
            current_agent = tasks_op[0]
            if self.route_layer is not None:
                route = tasks_op[1]

            logger.info(f"Current agent {current_agent}")
            self.tools['llm_agent'] = self.llm_agent_map[current_agent]
        elif self.route_layer is not None:
            route_layer_data = self.route_layer(message['data'])
            if route_layer_data:
                route = route_layer_data.name
            logger.info(f"Got route name {route}")

        if route is not None:
            logger.info(f"It was a route hit and we've got to respond from cache hence simply returning and the route is {route}")
            # Check if for the particular route if there's a vector store
            # If not send the response else get the response from the vector store
            logger.info(f"Vector caches {self.vector_caches}")
            if route in self.vector_caches:
                logger.info(f"Route {route} has a vector cache")
                relevant_utterance = self.vector_caches[route].get(message['data'])
                cache_response = self.route_responses_dict[route][relevant_utterance]
                convert_to_request_log(message=message['data'], meta_info=meta_info, component="llm", direction="request", model=self.llm_config["model"], run_id=self.run_id)
                convert_to_request_log(message=message['data'], meta_info=meta_info, component="llm", direction="response", model=self.llm_config["model"], is_cached=True, run_id= self.run_id)
                messages = copy.deepcopy(self.history)
                # TODO revisit this
                messages += [{'role': 'user', 'content': message['data']},{'role': 'assistant', 'content': cache_response}]
                self.interim_history = copy.deepcopy(messages)
                self.llm_response_generated = True
                # if self.callee_silent:
                #     logger.info("##### When we got utterance end, maybe LLM was still generating response. So, copying into history")
                #     self.history = copy.deepcopy(self.interim_history)

            else:
                logger.info(f"Route doesn't have a vector cache, and hence simply returning back a given response")
                cache_response = self.route_responses_dict[route]

            logger.info(f"Cached response {cache_response}")
            meta_info['cached'] = False
            meta_info["end_of_llm_stream"] = True

            await self._handle_llm_output(next_step, cache_response, should_bypass_synth, meta_info)
            self.llm_processed_request_ids.add(self.current_request_id)
        else:
            if self.turn_based_conversation:
                self.history.append({"role": "user", "content": message['data']})
            messages = copy.deepcopy(self.history)
            # messages.append({'role': 'user', 'content': message['data']})
            ### TODO CHECK IF THIS IS EVEN REQUIRED

            # Request logs converted inside do_llm_generation for knowledgebase agent
            if not self.__is_knowledgebase_agent():
                convert_to_request_log(message=format_messages(messages, use_system_prompt=True), meta_info=meta_info, component="llm", direction="request", model=self.llm_config["model"], run_id= self.run_id)

            await self.__do_llm_generation(messages, meta_info, next_step, should_bypass_synth)
            # TODO : Write a better check for completion prompt

            if self.agent_type not in ["graph_agent"]:
                if self.use_llm_to_determine_hangup and not self.turn_based_conversation:
                    completion_res, metadata = await self.tools["llm_agent"].check_for_completion(messages, self.check_for_completion_prompt)

                    should_hangup = (
                        str(completion_res.get("hangup", "")).lower() == "yes"
                        if isinstance(completion_res, dict)
                        else False
                    )

                    # Track hangup check latency (latency returned by agent)
                    self.llm_latencies['other_latencies'].append({
                        "type": 'hangup_check',
                        "latency_ms": metadata.get("latency_ms", None),
                        "model": self.check_for_completion_llm,
                        "provider": "openai",  # TODO: Make dynamic based on provider used
                        "service_tier": metadata.get("service_tier", None),
                        "llm_host": metadata.get("llm_host", None),
                        "sequence_id": meta_info.get("sequence_id")
                    })

                    prompt = [
                            {'role': 'system', 'content': self.check_for_completion_prompt},
                            {'role': 'user', 'content': format_messages(self.history, use_system_prompt= True)}]
                    logger.info(f"##### Answer from the LLM {completion_res}")
                    convert_to_request_log(message=format_messages(prompt, use_system_prompt= True), meta_info= meta_info, component="llm_hangup", direction="request", model=self.check_for_completion_llm, run_id= self.run_id)
                    convert_to_request_log(message=completion_res, meta_info= meta_info, component="llm_hangup", direction="response", model=self.check_for_completion_llm, run_id= self.run_id)

                    if should_hangup:
                        if self.hangup_triggered or self.conversation_ended:
                            logger.info(f"Hangup already triggered or conversation ended, skipping duplicate hangup request")
                            return
                        self.hangup_detail = "llm_prompted_hangup"
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
        message = self.call_hangup_message if not self.voicemail_detected else ""
        if not message or message.strip() == "":
            self.hangup_message_queued = False  # No hangup message to wait for
            await self.__process_end_of_conversation()
        else:
            self.hangup_message_queued = True  # Hangup message will be synthesized
            await self.wait_for_current_message()
            await self.__cleanup_downstream_tasks()
            meta_info = {'io': self.tools["output"].get_provider(), "request_id": str(uuid.uuid4()),
                             "cached": False, "sequence_id": -1, 'format': 'pcm', 'message_category': 'agent_hangup',
                         'end_of_llm_stream': True}
            await self._synthesize(create_ws_data_packet(message, meta_info=meta_info))
        return

    async def _listen_llm_input_queue(self):
        logger.info(
            f"Starting listening to LLM queue as either Connected to dashboard = {self.turn_based_conversation} or  it's a textual chat agent {self.textual_chat_agent}")
        while True:
            try:
                ws_data_packet = await self.queues["llm"].get()
                logger.info(f"ws_data_packet {ws_data_packet}")
                meta_info = self.__get_updated_meta_info(ws_data_packet['meta_info'])
                bos_packet = create_ws_data_packet("<beginning_of_stream>", meta_info)
                await self.tools["output"].handle(bos_packet)
                # self.interim_history = self.history.copy()
                # self.history.append({'role': 'user', 'content': ws_data_packet['data']})
                await self._run_llm_task(
                    create_ws_data_packet(ws_data_packet['data'], meta_info))
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
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Something went wrong in llm: {e}")


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

    def _should_check_voicemail(self, transcriber_message, is_final=True):
        """
        Determine if voicemail check should be triggered based on timing and conditions.
        This is a non-blocking check that returns quickly.

        Args:
            transcriber_message (str): The transcribed message from the user
            is_final (bool): Whether this is a final transcript or interim

        Returns:
            bool: True if voicemail check should be triggered, False otherwise
        """
        # Skip if voicemail detection is disabled
        if not self.voicemail_detection_enabled:
            return False

        # Skip if voicemail already detected (call is ending)
        if self.voicemail_detected:
            return False

        # Skip if a voicemail check is already in progress
        if self.voicemail_check_task is not None and not self.voicemail_check_task.done():
            logger.info("Voicemail check already in progress, skipping")
            return False

        current_time = time.time()

        # Initialize detection start time on first check
        if self.voicemail_detection_start_time is None:
            self.voicemail_detection_start_time = current_time
            logger.info(f"Voicemail detection window started at {self.voicemail_detection_start_time}")

        # Skip if we've exceeded the detection time window
        time_elapsed = current_time - self.voicemail_detection_start_time
        if time_elapsed > self.voicemail_detection_duration:
            logger.info(f"Voicemail detection window expired ({time_elapsed:.2f}s > {self.voicemail_detection_duration}s)")
            return False

        # For interim transcripts, check additional conditions
        if not is_final:
            # Check if enough time has passed since the last check
            time_since_last_check = (current_time - self.voicemail_last_check_time) if self.voicemail_last_check_time else float('inf')
            if time_since_last_check < self.voicemail_check_interval:
                logger.info(f"Skipping interim voicemail check - only {time_since_last_check:.2f}s since last check (need {self.voicemail_check_interval}s)")
                return False

            # Check if transcript length meets the minimum threshold
            word_count = len(transcriber_message.strip().split())
            if word_count < self.voicemail_min_transcript_length:
                logger.info(f"Skipping interim voicemail check - transcript too short ({word_count} words < {self.voicemail_min_transcript_length})")
                return False

        return True

    def _trigger_voicemail_check(self, transcriber_message, meta_info, is_final=True):
        """
        Trigger a background voicemail check if conditions are met.
        This is non-blocking and returns immediately.

        Args:
            transcriber_message (str): The transcribed message from the user
            meta_info (dict): Metadata from transcriber
            is_final (bool): Whether this is a final transcript or interim
        """
        if not self._should_check_voicemail(transcriber_message, is_final):
            return

        # Update last check time before starting the background task
        self.voicemail_last_check_time = time.time()
        time_elapsed = self.voicemail_last_check_time - self.voicemail_detection_start_time
        logger.info(f"Triggering background voicemail check at {time_elapsed:.2f}s into detection window (is_final={is_final}): {transcriber_message}")

        # Create background task for voicemail detection
        try:
            self.voicemail_check_task = asyncio.create_task(
                self._voicemail_check_background_task(transcriber_message, meta_info, is_final)
            )
        except Exception as e:
            logger.error(f"Error starting voicemail check background task: {e}")

    async def _voicemail_check_background_task(self, transcriber_message, meta_info, is_final):
        """
        Background task that performs the actual voicemail detection LLM call.
        If voicemail is detected, it will trigger the hangup process.

        Args:
            transcriber_message (str): The transcribed message from the user
            meta_info (dict): Metadata from transcriber
            is_final (bool): Whether this is a final transcript or interim
        """
        try:
            # Use the LLM agent to check for voicemail
            if "llm_agent" in self.tools and hasattr(self.tools["llm_agent"], 'check_for_voicemail'):
                prompt = [
                    {"role": "system", "content": self.voicemail_detection_prompt},
                    {"role": "user", "content": f"User message: {transcriber_message}"}
                ]

                convert_to_request_log(
                    message=format_messages(prompt, use_system_prompt=True),
                    meta_info=meta_info,
                    component="llm_voicemail",
                    direction="request",
                    model=self.voicemail_llm,
                    run_id=self.run_id
                )

                voicemail_result, metadata = await self.tools["llm_agent"].check_for_voicemail(
                    transcriber_message,
                    self.voicemail_detection_prompt
                )

                is_voicemail = (
                    str(voicemail_result.get("is_voicemail", "")).lower() == "yes"
                    if isinstance(voicemail_result, dict)
                    else False
                )

                # Track voicemail check latency (latency returned by agent)
                self.llm_latencies['other_latencies'].append({
                    "type": 'voicemail_check',
                    "latency_ms": metadata.get("latency_ms", None),
                    "model": self.voicemail_llm,
                    "provider": "openai",  # TODO: Make dynamic based on provider used
                    "service_tier": metadata.get("service_tier", None),
                    "llm_host": metadata.get("llm_host", None),
                    "sequence_id": meta_info.get("sequence_id")  # May be None for interim checks
                })

                convert_to_request_log(
                    message=voicemail_result,
                    meta_info=meta_info,
                    component="llm_voicemail",
                    direction="response",
                    model=self.voicemail_llm,
                    run_id=self.run_id
                )

                if is_voicemail:
                    logger.info(f"Voicemail detected in background task! Message: {transcriber_message}")
                    self.voicemail_detected = True
                    self.hangup_detail = "voicemail_detected"

                    # Trigger voicemail handling and call hangup
                    await self._handle_voicemail_detected()
            else:
                logger.warning("Voicemail detection enabled but llm_agent doesn't support check_for_voicemail")
        except Exception as e:
            logger.error(f"Error during background voicemail detection: {e}")

    async def _handle_voicemail_detected(self):
        """
        Handle the case when voicemail is detected - say the message and end the call.
        """
        logger.info(f"Handling voicemail detection - ending call")

        await self.process_call_hangup()

    async def _handle_transcriber_output(self, next_task, transcriber_message, meta_info):
        if not self.tools["input"].welcome_message_played() and len(self.history) > 2:
            logger.info(f"Welcome message is playing while spoken: {transcriber_message}")
            return

        # Skip only if exact same content as last user message (true duplicate)
        if (len(self.history) > 0
            and self.history[-1].get("role") == "user"
            and self.history[-1].get("content", "").strip() == transcriber_message.strip()):
            logger.info(f"Skipping duplicate transcript (same content): {transcriber_message}")
            return

        # Trigger background voicemail check (non-blocking) for final transcripts
        self._trigger_voicemail_check(transcriber_message, meta_info, is_final=True)

        # Skip processing if voicemail was already detected
        if self.voicemail_detected:
            logger.info("Voicemail already detected - skipping normal transcriber output processing")
            return

        # Collect transcript for language detection
        await self.language_detector.collect_transcript(transcriber_message)

        self.history.append({"role": "user", "content": transcriber_message})

        convert_to_request_log(message=transcriber_message, meta_info=meta_info, model=self.task_config["tools_config"]["transcriber"]["provider"], run_id= self.run_id)
        if next_task == "llm":
            logger.info(f"Running llm Tasks")
            meta_info["origin"] = "transcriber"
            transcriber_package = create_ws_data_packet(transcriber_message, meta_info)
            self.llm_task = asyncio.create_task(
                self._run_llm_task(transcriber_package))
            if self.use_fillers:
                self.filler_task = asyncio.create_task(self.__filler_classification_task(transcriber_package))

        elif next_task == "synthesizer":
            self.synthesizer_tasks.append(asyncio.create_task(
                self._synthesize(create_ws_data_packet(transcriber_message, meta_info))))
        else:
            logger.info(f"Need to separate out output task")

    async def _listen_transcriber(self):
        temp_transcriber_message = ""
        try:
            while True:
                message = await self.transcriber_output_queue.get()
                logger.info(f"Message from the transcriber class {message}")

                if self.hangup_triggered:
                    if message["data"] == "transcriber_connection_closed":
                        logger.info(f"Transcriber connection has been closed")
                        self.transcriber_duration += message.get("meta_info", {}).get("transcriber_duration", 0) if message['meta_info'] is not None else 0
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
                        if self.tools["input"].welcome_message_played():
                            logger.info(f"User has started speaking")
                            # self.callee_silent = False

                    # Whenever interim results would be received from Deepgram, this condition would get triggered
                    elif isinstance(message.get("data"), dict) and message["data"].get("type", "") == "interim_transcript_received":
                        self.time_since_last_spoken_human_word = time.time()
                        if temp_transcriber_message == message["data"].get("content"):
                            logger.info("Received the same transcript as the previous one we have hence continuing")
                            continue

                        temp_transcriber_message = message["data"].get("content")

                        if not self.tools["input"].welcome_message_played():
                            continue

                        self.interruption_manager.on_user_speech_started()

                        interim_transcript_len += len(message["data"].get("content").strip().split(" "))
                        transcript_content = message["data"].get("content", "")

                        if self.interruption_manager.should_trigger_interruption(
                            word_count=interim_transcript_len,
                            transcript=transcript_content,
                            is_audio_playing=self.tools["input"].is_audio_being_played_to_user(),
                            welcome_played=self.tools["input"].welcome_message_played()
                        ):
                            logger.info(f"Condition for interruption hit")
                            self.interruption_manager.on_interruption_triggered()
                            self.tools["input"].update_is_audio_being_played(False)
                            await self.__cleanup_downstream_tasks()
                        # User continuation detection: cancel pending response if user continues within grace period
                        elif not self.tools["input"].is_audio_being_played_to_user() and self.tools["input"].welcome_message_played():
                            has_pending_response = self.interruption_manager.has_pending_responses()
                            time_since_utterance_end = self.interruption_manager.get_time_since_utterance_end()
                            within_grace_period = (
                                time_since_utterance_end != -1 and
                                time_since_utterance_end < self.incremental_delay and
                                len(self.history) > 2
                            )

                            if has_pending_response and within_grace_period and interim_transcript_len > self.number_of_words_for_interruption:
                                logger.info(f"User continuation detected ({interim_transcript_len} words within {time_since_utterance_end:.0f}ms), canceling pending response")
                                self.interruption_manager.reset_utterance_end_time()
                                await self.__cleanup_downstream_tasks()
                        elif self.tools["input"].is_audio_being_played_to_user() and self.tools["input"].welcome_message_played() and self.number_of_words_for_interruption != 0:
                            # Not enough words for interruption - ignore
                            logger.info(f"Ignoring transcript: {transcript_content.strip()}")
                            continue

                        self.interruption_manager.update_required_delay(len(self.history))
                        self.interruption_manager.on_interim_transcript_received()

                        # Trigger background voicemail check on interim transcripts (non-blocking)
                        if self.voicemail_detection_enabled and not self.voicemail_detected:
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
                            is_audio_playing=self.tools["input"].is_audio_being_played_to_user(),
                            welcome_played=self.tools["input"].welcome_message_played()
                        ):
                            logger.info(f"Continuing the loop and ignoring the transcript received ({transcript_content}) in speech final as it is false interruption")
                            continue

                        self.interruption_manager.on_user_speech_ended()
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
                        self.interruption_manager.on_user_speech_ended()
                        temp_transcriber_message = ""

                    elif message["data"] == "transcriber_connection_closed":
                        logger.info(f"Transcriber connection has been closed")
                        self.transcriber_duration += message.get("meta_info", {}).get("transcriber_duration", 0) if message["meta_info"] is not None else 0
                        break

                else:
                    logger.info(f"Processing http transcription for message {message}")
                    if message["data"] == "transcriber_connection_closed":
                        logger.info(f"Transcriber connection has been closed")
                        self.transcriber_duration += message.get("meta_info", {}).get("transcriber_duration", 0) if message["meta_info"] is not None else 0
                        break

                    await self.__process_http_transcription(message)

        except websockets.exceptions.ConnectionClosedOK:
            # Normal WebSocket closure (code 1000)
            pass
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in transcriber {e}")

    async def __process_http_transcription(self, message):
        meta_info = self.__get_updated_meta_info(message["meta_info"])

        sequence = message["meta_info"].get('sequence', meta_info['sequence_id'])
        next_task = self._get_next_step(sequence, "transcriber")
        self.transcriber_duration += message["meta_info"]["transcriber_duration"] if "transcriber_duration" in message["meta_info"] else 0

        await self._handle_transcriber_output(next_task, message['data'], meta_info)


    #################################################################
    # Synthesizer task
    #################################################################
    def __enqueue_chunk(self, chunk, i, number_of_chunks, meta_info):
        meta_info['chunk_id'] = i
        copied_meta_info = copy.deepcopy(meta_info)
        if i == 0 and "is_first_chunk" in meta_info and meta_info["is_first_chunk"]:
            logger.info("Sending first chunk")
            copied_meta_info["is_first_chunk_of_entire_response"] = True

        if i == number_of_chunks - 1 and (meta_info['sequence_id'] == -1 or meta_info.get('end_of_synthesizer_stream', False)):
            logger.info(f"Sending first chunk")
            copied_meta_info["is_final_chunk_of_entire_response"] = True
            copied_meta_info.pop("is_first_chunk_of_entire_response", None)

        if copied_meta_info.get('message_category', None) == 'agent_welcome_message':
            copied_meta_info["is_first_chunk_of_entire_response"] = True
            copied_meta_info["is_final_chunk_of_entire_response"] = True

        self.buffered_output_queue.put_nowait(create_ws_data_packet(chunk, copied_meta_info))

    def is_sequence_id_in_current_ids(self, sequence_id):
        """Check if sequence_id is valid. Delegates to InterruptionManager."""
        return self.interruption_manager.is_valid_sequence(sequence_id)

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
                        if is_first_message or (not self.conversation_ended and self.interruption_manager.is_valid_sequence(sequence_id)):
                            logger.info(f"Processing message with sequence_id: {sequence_id}")

                            if self.stream:
                                if meta_info.get("is_first_chunk", False):
                                    first_chunk_generation_timestamp = time.time()

                                if self.tools["output"].process_in_chunks(self.yield_chunks):
                                    number_of_chunks = math.ceil(len(message['data']) / self.output_chunk_size)
                                    for chunk_idx, chunk in enumerate(
                                            yield_chunks_from_memory(message['data'], chunk_size=self.output_chunk_size)
                                    ):
                                        self.__enqueue_chunk(chunk, chunk_idx, number_of_chunks, meta_info)
                                else:
                                    self.buffered_output_queue.put_nowait(message)
                            else:
                                # Non-streaming output
                                logger.info("Stream not enabled, sending entire audio")
                                # TODO handle is audio playing over here
                                await self.tools["output"].handle(message)

                            logger.info(f"writing response to log {meta_info.get('text')}")
                            if write_to_log:
                                convert_to_request_log(
                                    message=current_text,
                                    meta_info=meta_info,
                                    component="synthesizer",
                                    direction="response",
                                    model=self.synthesizer_provider,
                                    is_cached=meta_info.get("is_cached", False),
                                    engine=self.tools['synthesizer'].get_engine(),
                                    run_id=self.run_id
                                )
                        else:
                            logger.info(f"Skipping message with sequence_id: {sequence_id}")

                        # Give control to other tasks
                        sleep_time = self.tools["synthesizer"].get_sleep_time()
                        await asyncio.sleep(sleep_time)

                except asyncio.CancelledError:
                    logger.info("Synthesizer task was cancelled.")
                    #await self.handle_cancellation("Synthesizer task was cancelled.")
                    break
                except Exception as e:
                    logger.error(f"Error in synthesizer: {e}", exc_info=True)
                    break

            logger.info("Exiting __listen_synthesizer gracefully.")

        except asyncio.CancelledError:
            logger.info("Synthesizer task cancelled outside loop.")
            #await self.handle_cancellation("Synthesizer task was cancelled outside loop.")
        except Exception as e:
            logger.error(f"Unexpected error in __listen_synthesizer: {e}", exc_info=True)
        finally:
            await self.tools["synthesizer"].cleanup()

    async def __send_preprocessed_audio(self, meta_info, text):
        meta_info = copy.deepcopy(meta_info)
        yield_in_chunks = self.yield_chunks
        try:
            #TODO: Either load IVR audio into memory before call or user s3 iter_cunks
            # This will help with interruption in IVR
            audio_chunk = None
            if self.turn_based_conversation or self.task_config['tools_config']['output']['provider'] == "default":
                audio_chunk = await get_raw_audio_bytes(text, self.assistant_name,
                                                                self.task_config["tools_config"]["output"][
                                                                    "format"], local=self.is_local,
                                                                assistant_id=self.assistant_id)
                logger.info("Sending preprocessed audio")
                meta_info["format"] = self.task_config["tools_config"]["output"]["format"]
                meta_info["end_of_synthesizer_stream"] = True
                await self.tools["output"].handle(create_ws_data_packet(audio_chunk, meta_info))
            else:
                if meta_info.get('message_category', None ) == 'filler':
                    logger.info(f"Getting {text} filler from local fs")
                    audio = await get_raw_audio_bytes(f'{self.filler_preset_directory}/{text}.wav', local= True, is_location=True)
                    yield_in_chunks = False
                    if not self.turn_based_conversation and self.task_config['tools_config']['output'] != "default":
                        logger.info(f"Got to convert it to pcm")
                        audio_chunk = wav_bytes_to_pcm(resample(audio, format = "wav", target_sample_rate = 8000 ))
                        meta_info["format"] = "pcm"
                else:
                    start_time = time.perf_counter()
                    audio_chunk = self.preloaded_welcome_audio if self.preloaded_welcome_audio else None
                    if meta_info['text'] == '':
                        audio_chunk = None
                    logger.info(f"Time to get response from S3 {time.perf_counter() - start_time }")
                    if not self.buffered_output_queue.empty():
                        logger.info(f"Output queue was not empty and hence emptying it")
                        self.buffered_output_queue = asyncio.Queue()
                    meta_info["format"] = "pcm"
                    if 'message_category' in meta_info and meta_info['message_category'] == "agent_welcome_message":
                        if audio_chunk is None:
                            logger.info(f"File doesn't exist in S3. Hence we're synthesizing it from synthesizer")
                            meta_info['cached'] = False
                            await self._synthesize(create_ws_data_packet(meta_info['text'], meta_info= meta_info))
                            return
                        else:
                            meta_info['is_first_chunk'] = True
                meta_info["end_of_synthesizer_stream"] = True
                if yield_in_chunks and audio_chunk is not None:
                    i = 0
                    number_of_chunks = math.ceil(len(audio_chunk) / 100000000)
                    logger.info(f"Audio chunk size {len(audio_chunk)}, chunk size {100000000}")
                    for chunk in yield_chunks_from_memory(audio_chunk, chunk_size=100000000):
                        self.__enqueue_chunk(chunk, i, number_of_chunks, meta_info)
                        i += 1
                elif audio_chunk is not None:
                    meta_info['chunk_id'] = 1
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
            if not self.conversation_ended and ('is_first_message' in meta_info and meta_info['is_first_message'] or self.interruption_manager.is_valid_sequence(message["meta_info"]["sequence_id"])):
                if meta_info["is_md5_hash"]:
                    logger.info('sending preprocessed audio response to {}'.format(self.task_config["tools_config"]["output"]["provider"]))
                    await self.__send_preprocessed_audio(meta_info, text)

                elif self.synthesizer_provider in SUPPORTED_SYNTHESIZER_MODELS.keys():
                    convert_to_request_log(message = text, meta_info= meta_info, component="synthesizer", direction="request", model = self.synthesizer_provider, engine=self.tools['synthesizer'].get_engine(), run_id= self.run_id)
                    if 'cached' in message['meta_info'] and meta_info['cached'] is True:
                        logger.info(f"Cached response and hence sending preprocessed text")
                        convert_to_request_log(message = text, meta_info= meta_info, component="synthesizer", direction="response", model = self.synthesizer_provider, is_cached= True, engine=self.tools['synthesizer'].get_engine(), run_id= self.run_id)
                        await self.__send_preprocessed_audio(meta_info, get_md5_hash(text))
                    else:
                        self.synthesizer_characters += len(text)
                        await self.tools["synthesizer"].push(message)
                else:
                    logger.info("other synthesizer models not supported yet")
            else:
                logger.info(f"{message['meta_info']['sequence_id']} is not a valid sequence id and hence not synthesizing this")

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in synthesizer: {e}")

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

    #Currently this loop only closes in case of interruption
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

                if "end_of_conversation" in message['meta_info']:
                    await self.__process_end_of_conversation()

                sequence_id = message['meta_info'].get('sequence_id')

                # Centralized tri-state decision loop (handles race condition + grace period)
                should_continue_outer_loop = False
                while True:
                    status = self.interruption_manager.get_audio_send_status(
                        sequence_id,
                        len(self.history)
                    )

                    if status == "SEND":
                        # Audio approved - send it
                        self.tools["input"].update_is_audio_being_played(True)
                        await self.tools["output"].handle(message)
                        try:
                            duration = calculate_audio_duration(len(message["data"]), self.sampling_rate, format=message['meta_info']['format'])
                            self.conversation_recording['output'].append({'data': message['data'], "start_time": time.time(), "duration": duration})
                        except Exception as e:
                            duration = 0.256
                            logger.info("Exception in __process_output_loop: {}".format(str(e)))
                        break  # Exit inner loop, audio sent

                    elif status == "BLOCK":
                        # Audio blocked (user speaking or invalid sequence) - discard
                        logger.info(f'Audio blocked: discarding message (sequence_id={sequence_id})')
                        should_continue_outer_loop = True
                        break  # Exit inner loop, skip to next message

                    elif status == "WAIT":
                        # Grace period active - hold and retry
                        await asyncio.sleep(0.05)
                        # Continue inner loop to re-check status

                if should_continue_outer_loop:
                    continue

                # Reset asked_if_user_is_still_there flag after any message except is_user_online_message
                if (message['meta_info'].get("end_of_llm_stream", False) or message['meta_info'].get("end_of_synthesizer_stream", False)) and \
                        message['meta_info'].get('message_category', '') != 'is_user_online_message':
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
            logger.error(f'Error in processing message output: {str(e)}')

    async def __check_for_completion(self):
        logger.info(f"Starting task to check for completion")
        while True:
            await asyncio.sleep(2)

            if self.is_web_based_call and time.time() - self.start_time >= int(
                    self.task_config["task_config"]["call_terminate"]):
                logger.info("Hanging up for web call as max time of call has been reached")
                await self.__process_end_of_conversation(web_call_timeout=True)
                self.hangup_detail = "web_call_max_duration_reached"
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
                        logger.warning(f"Hangup mark event not received within {self.hangup_mark_event_timeout}s (waited {time_since_hangup:.1f}s), forcing conversation end")
                        # Set hangup_sent since mark event didn't arrive
                        if "output" in self.tools:
                            self.tools["output"].set_hangup_sent()
                        await self.__process_end_of_conversation()
                        break
                    else:
                        logger.info(f"Waiting for hangup mark event ({time_since_hangup:.1f}s / {self.hangup_mark_event_timeout}s)")
                continue

            if self.tools["input"].is_audio_being_played_to_user():
                continue

            time_since_last_spoken_ai_word = (time.time() - self.last_transmitted_timestamp)
            time_since_user_last_spoke = (time.time() - self.time_since_last_spoken_human_word) if self.time_since_last_spoken_human_word > 0 else float('inf')

            if self.hang_conversation_after > 0 and time_since_last_spoken_ai_word > self.hang_conversation_after and time_since_user_last_spoke > self.hang_conversation_after:
                logger.info(f"{time_since_last_spoken_ai_word} seconds since AI last spoke and {time_since_user_last_spoke} seconds since user last spoke, both exceed {self.hang_conversation_after}s timeout - hanging up")
                self.hangup_detail = "inactivity_timeout"
                await self.process_call_hangup()
                break

            elif (time_since_last_spoken_ai_word > self.trigger_user_online_message_after and
                  not self.asked_if_user_is_still_there and
                  time_since_user_last_spoke > self.trigger_user_online_message_after):
                logger.info(f"Asking if the user is still there (agent silent for {time_since_last_spoken_ai_word:.2f}s, user silent for {time_since_user_last_spoke:.2f}s)")
                self.asked_if_user_is_still_there = True

                if self.check_if_user_online:
                    detected_lang = self.language_detector.dominant_language
                    user_online_message = select_message_by_language(self.check_user_online_message_config, detected_lang)

                    if self.should_record:
                        meta_info={'io': 'default', "request_id": str(uuid.uuid4()), "cached": False, "sequence_id": -1, 'format': 'wav', "message_category": "is_user_online_message", 'end_of_llm_stream': True}
                        await self._synthesize(create_ws_data_packet(user_online_message, meta_info= meta_info))
                    else:
                        meta_info={'io': self.tools["output"].get_provider(), "request_id": str(uuid.uuid4()), "cached": False, "sequence_id": -1, 'format': 'pcm', "message_category": "is_user_online_message", 'end_of_llm_stream': True}
                        await self._synthesize(create_ws_data_packet(user_online_message, meta_info= meta_info))
                    self.history.append({'role': 'assistant', 'content': user_online_message})

                # Just in case we need to clear messages sent before
                await self.tools["output"].handle_interruption()
            else:
                logger.info(f"Only {time_since_last_spoken_ai_word} seconds since last spoken time stamp and hence not cutting the phone call")

    async def __check_for_backchanneling(self):
        while True:
            user_speaking_duration = self.interruption_manager.get_user_speaking_duration()
            if self.interruption_manager.is_user_speaking() and user_speaking_duration > self.backchanneling_start_delay:
                filename = random.choice(self.filenames)
                logger.info(f"Should send a random backchanneling words and sending them {filename}")
                audio = await get_raw_audio_bytes(f"{self.backchanneling_audios}/{filename}", local= True, is_location=True)
                if not self.turn_based_conversation and self.task_config['tools_config']['output'] != "default":
                    audio = resample(audio, target_sample_rate= 8000, format="wav")
                    audio = wav_bytes_to_pcm(audio)
                await self.tools["output"].handle(create_ws_data_packet(audio, self.__get_updated_meta_info()))
            else:
                logger.info(f"Callee isn't speaking and hence not sending or {user_speaking_duration} is not greater than {self.backchanneling_start_delay}")
            await asyncio.sleep(self.backchanneling_message_gap)

    async def __first_message(self, timeout=10.0):
        logger.info(f"Executing the first message task")
        try:
            if self.is_web_based_call:
                logger.info("Sending agent welcome message for web based call")
                text = self.kwargs.get('agent_welcome_message', None)
                meta_info = {'io': 'default', 'message_category': 'agent_welcome_message',
                             'stream_sid': self.stream_sid, "request_id": str(uuid.uuid4()), "cached": False,
                             "sequence_id": -1, 'format': self.task_config["tools_config"]["output"]["format"],
                             'text': text, 'end_of_llm_stream': True}
                self.stream_sid_ts = time.time() * 1000
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
                        text = self.kwargs.get('agent_welcome_message', None)
                        meta_info = {'io': self.tools["output"].get_provider(), 'message_category': 'agent_welcome_message', 'stream_sid': stream_sid, "request_id": str(uuid.uuid4()), "cached": True, "sequence_id": -1, 'format': self.task_config["tools_config"]["output"]["format"], 'text': text, 'end_of_llm_stream': True}
                        if self.turn_based_conversation:
                            meta_info['type'] = 'text'
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
                        await asyncio.sleep(0.01) #Sleep for half a second to see if stream id goes past None
                elif self.default_io:
                    logger.info(f"Shouldn't record")
                    # meta_info={'io': 'default', 'is_first_message': True, "request_id": str(uuid.uuid4()), "cached": True, "sequence_id": -1, 'format': 'wav'}
                    # await self._synthesize(create_ws_data_packet(self.kwargs['agent_welcome_message'], meta_info= meta_info))
                    break

        except Exception as e:
            logger.error(f"Exception in __first_message {str(e)}")

    async def __start_transmitting_ambient_noise(self):
        try:
            audio = await get_raw_audio_bytes(f'{os.getenv("AMBIENT_NOISE_PRESETS_DIR")}/{self.soundtrack}', local=True, is_location=True)
            audio = resample(audio, self.sampling_rate, format = "wav")
            if self.task_config["tools_config"]["output"]["provider"] in SUPPORTED_OUTPUT_TELEPHONY_HANDLERS.keys():
                audio = wav_bytes_to_pcm(audio)
            logger.info(f"Length of audio {len(audio)} {self.sampling_rate}")
            # TODO whenever this feature is redone ensure to have a look at the metadata of other messages which have the sequence_id of -1. Fields such as end_of_synthesizer_stream and end_of_llm_stream would need to be added here
            if self.should_record:
                meta_info={'io': 'default', 'message_category': 'ambient_noise', "request_id": str(uuid.uuid4()), "sequence_id": -1, "type":'audio', 'format': 'wav'}
            else:

                meta_info={'io': self.tools["output"].get_provider(), 'message_category': 'ambient_noise', 'stream_sid': self.stream_sid , "request_id": str(uuid.uuid4()), "cached": True, "type":'audio', "sequence_id": -1, 'format': 'pcm'}
            while True:
                logger.info(f"Before yielding ambient noise")
                for chunk in yield_chunks_from_memory(audio, self.output_chunk_size*2):
                    # Only play ambient noise if no content is being transmitted
                    # Check if any real content audio (sequence_id > 0) is being played
                    is_content_playing = self.tools["input"].is_audio_being_played_to_user()
                    
                    if not is_content_playing:
                        logger.info(f"Transmitting ambient noise {len(chunk)}")
                        await self.tools["output"].handle(create_ws_data_packet(chunk, meta_info=meta_info))
                    logger.info("Sleeping for 800 ms")
                    await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Something went wrong while transmitting noise {e}")

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

            if self.system_prompt['content']:
                system_prompt = self.system_prompt['content']
                system_prompt = update_prompt_with_context(system_prompt, self.context_data)
                self.system_prompt['content'] = system_prompt
                self.history[0]['content'] = system_prompt

            if self.call_hangup_message_config and self.context_data:
                if isinstance(self.call_hangup_message_config, dict):
                    self.call_hangup_message_config = {
                        lang: update_prompt_with_context(msg, self.context_data)
                        for lang, msg in self.call_hangup_message_config.items()
                    }
                else:
                    self.call_hangup_message_config = update_prompt_with_context(self.call_hangup_message_config, self.context_data)

            agent_welcome_message = self.kwargs.get("agent_welcome_message", "")
            agent_welcome_message = update_prompt_with_context(agent_welcome_message, self.context_data)
            logger.info(f"Updated agent welcome message after context data replacement - {agent_welcome_message}")
            self.kwargs["agent_welcome_message"] = agent_welcome_message
            if len(self.history) == 2 and agent_welcome_message and self.history[1]["role"] == "assistant":
                self.history[1]["content"] = agent_welcome_message

            await self.tools["output"].send_init_acknowledgement()
            self.first_message_task = asyncio.create_task(self.__first_message())
        except Exception as e:
            logger.error(f"Error occurred in handling init event - {e}")

    async def run(self):
        try:
            if self._is_conversation_task():
                logger.info("started running")
                # Create transcriber and synthesizer tasks
                tasks = []
                #tasks = [asyncio.create_task(self.tools['input'].handle())]

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
                        "Since it's connected through dashboard, I'll run listen_llm_tas too in case user wants to simply text")
                    self.llm_queue_task = asyncio.create_task(self._listen_llm_input_queue())

                if "synthesizer" in self.tools and self._is_conversation_task() and not self.turn_based_conversation:
                    try:
                        self.synthesizer_task = asyncio.create_task(self.__listen_synthesizer())
                    except asyncio.CancelledError as e:
                        logger.error(f'Synth task got cancelled {e}')
                        traceback.print_exc()

                self.output_task = asyncio.create_task(self.__process_output_loop())
                if not self.turn_based_conversation or self.enforce_streaming:
                    self.hangup_task = asyncio.create_task(self.__check_for_completion())

                    if self.should_backchannel:
                        self.backchanneling_task = asyncio.create_task(self.__check_for_backchanneling())
                    if self.ambient_noise:
                        self.ambient_noise_task = asyncio.create_task(self.__start_transmitting_ambient_noise())
                try:
                    await asyncio.gather(*tasks)
                except asyncio.CancelledError as e:
                    logger.error(f'task got cancelled {e}')
                    traceback.print_exc()
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Error: {e}")

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
                except Exception as e:
                    logger.error(f"Could not do llm call: {e}")
                    raise Exception(e)

        except asyncio.CancelledError as e:
            # Cancel all tasks on cancel
            traceback.print_exc()
            self.transcriber_task.cancel()
            await self.handle_cancellation(f"Websocket got cancelled {self.task_id}")

        except Exception as e:
            # Cancel all tasks on error
            error_message = str(e)
            logger.error(f"Exception in task manager run: {error_message}")
            traceback.print_exc()

            # Log call-breaking exception to CSV trace
            if self.run_id:
                meta_info = {
                    'request_id': self.task_id,
                    'sequence_id': None
                }
                convert_to_request_log(
                    f"Call Breaking Error: {error_message}",
                    meta_info,
                    model="-",
                    component="error",
                    direction="error",
                    is_cached=False,
                    run_id=self.run_id
                )

            await self.handle_cancellation(f"Exception occurred {e}")
            raise Exception(e)

        finally:
            # Construct output
            tasks_to_cancel = []
            if "synthesizer" in self.tools and self.synthesizer_task is not None:
                tasks_to_cancel.append(self.tools["synthesizer"].cleanup())
                tasks_to_cancel.append(process_task_cancellation(self.synthesizer_task, 'synthesizer_task'))
                tasks_to_cancel.append(process_task_cancellation(self.synthesizer_monitor_task, 'synthesizer_monitor_task'))

            # Transcriber cleanup
            if "transcriber" in self.tools:
                tasks_to_cancel.append(self.tools["transcriber"].cleanup())
                if hasattr(self, 'transcriber_task') and self.transcriber_task is not None:
                    tasks_to_cancel.append(process_task_cancellation(self.transcriber_task, 'transcriber_task'))

            if self._is_conversation_task():
                self.transcriber_latencies['connection_latency_ms'] = self.tools["transcriber"].connection_time
                self.synthesizer_latencies['connection_latency_ms'] = self.tools["synthesizer"].connection_time

                self.transcriber_latencies['turn_latencies'] = self.tools["transcriber"].turn_latencies
                self.synthesizer_latencies['turn_latencies'] = self.tools["synthesizer"].turn_latencies

                # Collect language detection latency if available
                if hasattr(self, 'language_detector') and self.language_detector.latency_data:
                    self.llm_latencies['other_latencies'].append(self.language_detector.latency_data)

                welcome_message_sent_ts = self.tools["output"].get_welcome_message_sent_ts()

                output = {
                    "messages": self.history,
                    "conversation_time": time.time() - self.start_time,
                    "label_flow": self.label_flow,
                    "call_sid": self.call_sid,
                    "stream_sid": self.stream_sid,
                    "transcriber_duration": self.transcriber_duration,
                    "synthesizer_characters": self.tools['synthesizer'].get_synthesized_characters(), "ended_by_assistant": self.ended_by_assistant,
                    "latency_dict": {
                        "llm_latencies": self.llm_latencies,
                        "transcriber_latencies": self.transcriber_latencies,
                        "synthesizer_latencies": self.synthesizer_latencies,
                        "rag_latencies": self.rag_latencies,
                        "welcome_message_sent_ts": None,
                        "stream_sid_ts": None
                    },
                    "hangup_detail": self.hangup_detail
                }

                try:
                    if welcome_message_sent_ts:
                        output["latency_dict"]["welcome_message_sent_ts"] = welcome_message_sent_ts - self.conversation_start_init_ts
                    if self.stream_sid_ts:
                        output["latency_dict"]["stream_sid_ts"] = self.stream_sid_ts - self.conversation_start_init_ts
                except Exception as e:
                    logger.error(f"error in logging audio latency ts {str(e)}")

                tasks_to_cancel.append(process_task_cancellation(self.output_task,'output_task'))
                tasks_to_cancel.append(process_task_cancellation(self.hangup_task,'hangup_task'))
                tasks_to_cancel.append(process_task_cancellation(self.backchanneling_task, 'backchanneling_task'))
                tasks_to_cancel.append(process_task_cancellation(self.ambient_noise_task, 'ambient_noise_task'))
                # tasks_to_cancel.append(process_task_cancellation(self.initial_silence_task, 'initial_silence_task'))
                tasks_to_cancel.append(process_task_cancellation(self.first_message_task, 'first_message_task'))
                tasks_to_cancel.append(process_task_cancellation(self.dtmf_task, 'dtmf_task'))
                tasks_to_cancel.append(process_task_cancellation(self.voicemail_check_task, 'voicemail_check_task'))
                tasks_to_cancel.append(
                    process_task_cancellation(self.handle_accumulated_message_task, "handle_accumulated_message_task"))

                output['recording_url'] = ""
                if self.should_record:
                    output['recording_url'] = await save_audio_file_to_s3(self.conversation_recording, self.sampling_rate, self.assistant_id, self.run_id)
            else:
                output = self.input_parameters
                if self.task_config["task_type"] == "extraction":
                    output = { "extracted_data" : self.extracted_data,
                              "task_type": "extraction",
                              "latency_dict": {
                                "llm_latencies": self.llm_latencies
                            }}
                elif self.task_config["task_type"] == "summarization":
                    logger.info(f"self.summarized_data {self.summarized_data}")
                    output = {"summary" : self.summarized_data,
                              "task_type": "summarization",
                              "latency_dict": {
                                "llm_latencies": self.llm_latencies
                            }}
                elif self.task_config["task_type"] == "webhook":
                    output = {"status": self.webhook_response, "task_type": "webhook"}
                    

            await asyncio.gather(*tasks_to_cancel)
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
