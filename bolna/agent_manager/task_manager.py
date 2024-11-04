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
from datetime import datetime
import pytz

import aiohttp

from bolna.constants import ACCIDENTAL_INTERRUPTION_PHRASES, DEFAULT_USER_ONLINE_MESSAGE, DEFAULT_USER_ONLINE_MESSAGE_TRIGGER_DURATION, FILLER_DICT, PRE_FUNCTION_CALL_MESSAGE, DEFAULT_LANGUAGE_CODE
from bolna.helpers.function_calling_helpers import trigger_api
from bolna.memory.cache.vector_cache import VectorCache
from .base_manager import BaseManager
from bolna.agent_types import *
from bolna.providers import *
from bolna.prompts import *
from bolna.helpers.utils import get_route_info, calculate_audio_duration, create_ws_data_packet, get_file_names_in_directory, get_raw_audio_bytes, is_valid_md5, \
    get_required_input_types, format_messages, get_prompt_responses, resample, save_audio_file_to_s3, update_prompt_with_context, get_md5_hash, clean_json_string, wav_bytes_to_pcm, convert_to_request_log, yield_chunks_from_memory, process_task_cancellation
from bolna.helpers.logger_config import configure_logger
from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import FastEmbedEncoder

asyncio.get_event_loop().set_debug(True)
logger = configure_logger(__name__)


class TaskManager(BaseManager):
    def __init__(self, assistant_name, task_id, task, ws, input_parameters=None, context_data=None,
                 assistant_id=None, turn_based_conversation=False, cache=None,
                 input_queue=None, conversation_history=None, output_queue=None, yield_chunks=True, **kwargs):
        super().__init__()
        # Latency and logging 
        self.latency_dict = defaultdict(dict)
        self.kwargs = kwargs
        #Setup Latency part
        self.llm_latencies = []
        self.synthesizer_latencies = []
        self.transcriber_latencies = []
        self.average_llm_latency = 0.0
        self.average_synthesizer_latency = 0.0
        self.average_transcriber_latency = 0.0
        self.task_config = task

        self.timezone = pytz.timezone('America/Los_Angeles')
        self.language = DEFAULT_LANGUAGE_CODE

        logger.info(f"API TOOLS IN TOOLS CONFIG {task['tools_config'].get('api_tools')}")
        if task['tools_config'].get('api_tools', None) is not None:
            logger.info(f"API TOOLS is present {task['tools_config']['api_tools']}")
            self.kwargs['api_tools'] = task['tools_config']['api_tools']

        if task['tools_config']["llm_agent"] and task['tools_config']["llm_agent"]['llm_config'].get('assistant_id', None) is not None:
            self.kwargs['assistant_id'] = task['tools_config']["llm_agent"]['llm_config']['assistant_id']
            logger.info(f"Assistant id for agent is {self.kwargs['assistant_id']}")

        if self.__is_openai_assistant():
            self.kwargs['assistant_id'] = task['tools_config']["llm_agent"]['llm_config']['assistant_id']
            logger.info(f"Assistant id for agent is {self.kwargs['assistant_id']}")

        logger.info(f"doing task {task}")
        self.task_id = task_id
        self.assistant_name = assistant_name
        self.tools = {}
        self.websocket = ws
        self.context_data = context_data
        logger.info(f"turn_based_conversation {turn_based_conversation}")
        self.turn_based_conversation = turn_based_conversation
        self.enforce_streaming = kwargs.get("enforce_streaming", False)
        self.room_url = kwargs.get("room_url", None)
        self.callee_silent = True
        self.yield_chunks = yield_chunks
        # Set up communication queues between processes
        self.audio_queue = asyncio.Queue()
        self.llm_queue = asyncio.Queue()
        self.synthesizer_queue = asyncio.Queue()
        self.transcriber_output_queue = asyncio.Queue()
        self.queues = {
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
        self.run_id = kwargs.get("run_id", "1234#0")
        self.agent_id, self.run_id_ts = self.run_id.split('#')

        self.mark_set = set()
        self.sampling_rate = 24000
        self.conversation_ended = False

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
        #IO HANDLERS
        if task_id == 0:
            self.default_io = self.task_config["tools_config"]["output"]["provider"] == 'default'
            logger.info(f"Connected via websocket")
            self.should_record = self.task_config["tools_config"]["output"]["provider"] == 'default' and self.enforce_streaming #In this case, this is a websocket connection and we should record 
            self.__setup_input_handlers(turn_based_conversation, input_queue, self.should_record)
        self.__setup_output_handlers(turn_based_conversation, output_queue)

        # Agent stuff
        # Need to maintain current conversation history and overall persona/history kinda thing. 
        # Soon we will maintain a separate history for this 
        self.history = [] if conversation_history is None else conversation_history 
        self.interim_history = copy.deepcopy(self.history.copy())
        logger.info(f'History {self.history}')
        self.label_flow = []

        # Setup IO SERVICE, TRANSCRIBER, LLM, SYNTHESIZER
        self.llm_task = None
        self.execute_function_call_task = None
        self.synthesizer_tasks = []
        self.synthesizer_task = None
        self.synthesizer_monitor_task = None

        # state of conversation
        self.current_request_id = None
        self.previous_request_id = None
        self.llm_rejected_request_ids = set()
        self.llm_processed_request_ids = set()
        self.was_long_pause = False
        self.buffers = []
        self.should_respond = False
        self.last_response_time = time.time()
        self.is_an_ivr_call = self._is_conversation_task() and self._is_preprocessed_flow() and not self.turn_based_conversation
        self.consider_next_transcript_after = time.time()
        self.duration_to_prevent_accidental_interruption = 3 if self.is_an_ivr_call else 0
        self.callee_speaking = False
        self.callee_speaking_start_time = -1
        self.llm_response_generated = False
        self.turn_id = 0

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
        logger.info(f"TASK CONFIG {self.task_config['tools_config'] }")
        self.stream = (self.task_config["tools_config"]['synthesizer'] is not None and self.task_config["tools_config"]["synthesizer"]["stream"]) and (self.enforce_streaming or not self.turn_based_conversation)
        #self.stream = not turn_based_conversation #Currently we are allowing only realtime conversation based usecases. Hence it'll always be true unless connected through dashboard
        self.is_local = False
        self.llm_config = None

        self.llm_config_map = {}
        self.llm_agent_map = {}
        if self.__is_multiagent():
            for agent, config in self.task_config["tools_config"]["llm_agent"]['llm_config']['agent_map'].items():
                self.llm_config_map[agent] = config.copy()
                self.llm_config_map[agent]['buffer_size'] = self.task_config["tools_config"]["synthesizer"][
                    'buffer_size']
                if 'assistant_id' in config:
                    self.llm_config_map[agent]['agent_type'] = "openai_assistant"
        elif not self.__is_openai_assistant():
            if self.task_config["tools_config"]["llm_agent"] is not None:
                if self.__is_knowledgebase_agent():
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
                    }

        # if self.task_config["tools_config"]["llm_agent"] is not None:
        #     self.llm_config = {
        #         "model": self.llm_agent_config["model"],
        #         "max_tokens": self.task_config["tools_config"]["llm_agent"]["max_tokens"],
        #         "provider": self.task_config["tools_config"]["llm_agent"]["provider"]
        #     }

        # Output stuff
        self.output_task = None
        self.buffered_output_queue = asyncio.Queue()

        # Memory
        self.cache = cache
        logger.info("task initialization completed")

        # Sequence id for interruption
        self.curr_sequence_id = 0
        self.sequence_ids = {-1} #-1 is used for data that needs to be passed and is developed by task manager like backchannleing etc.

        #setup request logs
        self.request_logs = []
        self.hangup_task = None
        
        self.conversation_config = None

        if task_id == 0:
            provider_config = self.task_config["tools_config"]["synthesizer"].get("provider_config")
            self.synthesizer_voice = provider_config["voice"]

            self.background_check_task = None
            self.hangup_task = None
            self.output_chunk_size = 16384 if self.sampling_rate == 24000 else 4096 #0.5 second chunk size for calls
            # For nitro
            self.nitro = True 
            self.conversation_config = task.get("task_config", {})
            logger.info(f"Conversation config {self.conversation_config}")

            self.trigger_user_online_message_after = self.conversation_config.get("trigger_user_online_message_after", DEFAULT_USER_ONLINE_MESSAGE_TRIGGER_DURATION)
            self.check_if_user_online = self.conversation_config.get("check_if_user_online", True)
            self.check_user_online_message = self.conversation_config.get("check_user_online_message", DEFAULT_USER_ONLINE_MESSAGE)

            self.kwargs["process_interim_results"] = "true" if self.conversation_config.get("optimize_latency", False) is True else "false"
            logger.info(f"Processing interim results {self.kwargs['process_interim_results'] }")
            # Routes
            self.routes = task['tools_config']['llm_agent'].get("routes", None)
            self.route_layer = None

            if self.routes:
                start_time = time.time()
                routes_meta = self.kwargs.get('routes', None)
                if self.__is_multiagent():
                    routes_meta = self.kwargs.get('routes', None)
                    routes_meta = routes_meta['routes']
                else:
                    routes_meta = self.kwargs.get('routes', None)

                if self.kwargs['routes']:
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
                self.minimum_wait_duration = self.task_config["tools_config"]["transcriber"]["endpointing"]
                logger.info(f"minimum wait duration {self.minimum_wait_duration}")
                self.last_spoken_timestamp = time.time() * 1000
                self.incremental_delay = self.conversation_config.get("incremental_delay", 100)
                logger.info(f"incremental_delay - {self.incremental_delay}")
                self.required_delay_before_speaking = max(self.minimum_wait_duration - self.incremental_delay, 0)  #Everytime we get a message we increase it by 100 miliseconds 
                self.time_since_first_interim_result = -1

                #Cut conversation
                self.hang_conversation_after = self.conversation_config.get("hangup_after_silence", 10)
                logger.info(f"hangup_after_silence {self.hang_conversation_after}")
                self.last_transmitted_timestamp = 0
                self.let_remaining_audio_pass_through = False #Will be used to let remaining audio pass through in case of utterenceEnd event and there's still audio left to be sent
                self.use_llm_to_determine_hangup = self.conversation_config.get("hangup_after_LLMCall", False)

                self.check_for_completion_prompt = self.conversation_config.get("call_cancellation_prompt", None)
                if self.check_for_completion_prompt is not None:
                    completion_json_format = {"answer": "A simple Yes or No based on if you should cut the phone or not"}
                    self.check_for_completion_prompt = f"{self.check_for_completion_prompt}\nYour response should be in the following json format\n{completion_json_format}"
                self.check_for_completion_llm = os.getenv("CHECK_FOR_COMPLETION_LLM")
                self.time_since_last_spoken_human_word = 0 

                #Handling accidental interruption
                self.number_of_words_for_interruption = self.conversation_config.get("number_of_words_for_interruption", 3)
                self.asked_if_user_is_still_there = False #Used to make sure that if user's phrase qualifies as acciedental interruption, we don't break the conversation loop
                self.first_message_passed = True if self.task_config["tools_config"]["output"]["provider"] == 'default' else False
                self.started_transmitting_audio = False
                self.accidental_interruption_phrases = set(ACCIDENTAL_INTERRUPTION_PHRASES)
                #self.interruption_backoff_period = 1000 #conversation_config.get("interruption_backoff_period", 300) #this is the amount of time output loop will sleep before sending next audio
                self.allow_extra_sleep = False #It'll help us to back off as soon as we hear interruption for a while

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
                    logger.info(f"Not setting up backchanneling")
                    self.backchanneling_audio_map = []
                # Agent welcome message
                if "agent_welcome_message" in self.kwargs:
                    logger.info(f"Agent welcome message present {self.kwargs['agent_welcome_message']}")
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

        # setting transcriber
        self.__setup_transcriber()
        # setting synthesizer
        self.__setup_synthesizer(self.llm_config)
        # # setting llm
        # llm = self.__setup_llm(self.llm_config)
        # # Setup tasks
        # self.__setup_tasks(llm)

        # setting llm
        if self.llm_config is not None:
            logger.info(f"LLM CONFIG IS NOT NONE {self.task_config['task_type']} llm agent config {self.llm_agent_config}")
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
                agent_type = self.llm_config_map[agent].get("agent_type","simple_llm_agent")
                logger.info(f"Getting response for {llm} and agent type {agent_type} and {agent}")
                agent_params = {
                    'llm': llm,
                    'agent_type': agent_type
                }
                if agent_type == "openai_assistant":
                    agent_params['assistant_config'] = self.llm_config_map[agent]
                llm_agent = self.__setup_tasks(**agent_params)
                self.llm_agent_map[agent] = llm_agent

        elif self.__is_openai_assistant():
            # if self.task_config['tools_config']["llm_agent"].get("agent_type", None) is None:
            #     assistant_config = {"assistant_id": self.task_config['tools_config']["llm_agent"]['assistant_id']}
            self.__setup_tasks(agent_type="openai_assistant", assistant_config=task['tools_config']["llm_agent"]['llm_config'])

        elif self.task_config["task_type"] == "webhook":
            if "webhookURL" in self.task_config["tools_config"]["api_tools"]:
                webhook_url = self.task_config["tools_config"]["api_tools"]["webhookURL"]
            else:
                webhook_url = self.task_config["tools_config"]["api_tools"]["tools_params"]["webhook"]["url"]
            logger.info(f"Webhook URL {webhook_url}")
            self.tools["webhook_agent"] = WebhookAgent(webhook_url=webhook_url)

    def __is_openai_assistant(self):
        if self.task_config["task_type"] == "webhook":
            return False
        agent_type = self.task_config['tools_config']["llm_agent"].get("agent_type", self.task_config['tools_config']["llm_agent"].get("agent_flow_type"))
        return agent_type == "openai_assistant"

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

                if self.task_config["tools_config"]["output"]["provider"] == "daily":
                    output_kwargs['room_url'] = self.room_url

                if self.task_config["tools_config"]["output"]["provider"] in SUPPORTED_OUTPUT_TELEPHONY_HANDLERS.keys():
                    output_kwargs['mark_set'] = self.mark_set
                    logger.info(f"Making sure that the sampling rate for output handler is 8000")
                    self.task_config['tools_config']['synthesizer']['provider_config']['sampling_rate'] = 8000
                    self.task_config['tools_config']['synthesizer']['audio_format'] = 'pcm'
                else:
                    self.task_config['tools_config']['synthesizer']['provider_config']['sampling_rate'] = 24000
                    output_kwargs['queue'] = output_queue
                self.sampling_rate = self.task_config['tools_config']['synthesizer']['provider_config']['sampling_rate']

            self.tools["output"] = output_handler_class(**output_kwargs)
        else:
            raise "Other input handlers not supported yet"

    def __setup_input_handlers(self, turn_based_conversation, input_queue, should_record):
        if self.task_config["tools_config"]["input"]["provider"] in SUPPORTED_INPUT_HANDLERS.keys():
            logger.info(f"Connected through dashboard {turn_based_conversation}")
            input_kwargs = {"queues": self.queues,
                            "websocket": self.websocket,
                            "input_types": get_required_input_types(self.task_config),
                            "mark_set": self.mark_set}
            
            if self.task_config["tools_config"]["input"]["provider"] == "daily":
                input_kwargs['room_url'] = self.room_url

            if should_record:
                input_kwargs['conversation_recording'] = self.conversation_recording

            if self.turn_based_conversation:
                logger.info("Connected through dashboard and hence using default input handler")
                input_kwargs['turn_based_conversation'] = True
                input_handler_class = SUPPORTED_INPUT_HANDLERS.get("default")
                input_kwargs['queue'] = input_queue
            else:
                input_handler_class = SUPPORTED_INPUT_HANDLERS.get(
                    self.task_config["tools_config"]["input"]["provider"])
                
                if self.task_config['tools_config']['input']['provider'] == 'default':
                    input_kwargs['queue'] = input_queue
                    
            self.tools["input"] = input_handler_class(**input_kwargs)
        else:
            raise "Other input handlers not supported yet"


    def __setup_transcriber(self):
        try:
            if self.task_config["tools_config"]["transcriber"] is not None:
                logger.info("Setting up transcriber")
                self.language = self.task_config["tools_config"]["transcriber"].get('language', DEFAULT_LANGUAGE_CODE)
                provider = "playground" if self.turn_based_conversation else self.task_config["tools_config"]["input"][
                    "provider"]
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
                    self.tools["transcriber"] = transcriber_class( provider, **self.task_config["tools_config"]["transcriber"], **self.kwargs)
        except Exception as e:
            logger.error(f"Something went wrong with starting transcriber {e}")

    def __setup_synthesizer(self, llm_config=None):
        logger.info(f"Synthesizer config: {self.task_config['tools_config']['synthesizer']}")
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
            self.synthesizer_monitor_task = asyncio.create_task(self.tools['synthesizer'].monitor_connection())
            if self.task_config["tools_config"]["llm_agent"] is not None and llm_config is not None:
                llm_config["buffer_size"] = self.task_config["tools_config"]["synthesizer"].get('buffer_size')

    def __setup_llm(self, llm_config, task_id=0):
        if self.task_config["tools_config"]["llm_agent"] is not None:
            logger.info(f'### PROVIDER {llm_config["provider"] }')
            if llm_config["provider"] in SUPPORTED_LLM_PROVIDERS.keys():
                llm_class = SUPPORTED_LLM_PROVIDERS.get(llm_config["provider"])
                logger.info(f"LLM CONFIG {llm_config}")

                if task_id and task_id > 0:
                    self.kwargs.pop('llm_key', None)
                    self.kwargs.pop('base_url', None)
                    self.kwargs.pop('api_version', None)
                llm = llm_class(language=self.language, **llm_config, **self.kwargs)
                return llm
            else:
                raise Exception(f'LLM {llm_config["provider"]} not supported')

    def __get_agent_object(self, llm, agent_type, assistant_config=None):
        if agent_type == "simple_llm_agent":
            logger.info(f"Simple llm agent")
            llm_agent = StreamingContextualAgent(llm)
        elif agent_type == "openai_assistant":
            logger.info(f"setting up backend as openai_assistants {assistant_config}")
            llm_agent = OpenAIAssistantAgent(**assistant_config)
        elif agent_type == "knowledgebase_agent":
            logger.info("#### Setting up knowledgebase_agent agent ####")
            llm_config = self.task_config["tools_config"]["llm_agent"].get("llm_config", {})
            vector_store_config = llm_config.get("vector_store", {})
            llm_agent = RAGAgent(
                temperature=llm_config.get("temperature", 0.1),
                model=llm_config.get("model", "gpt-3.5-turbo-16k"),
                buffer=self.task_config["tools_config"]["synthesizer"].get('buffer_size'),
                max_tokens=self.llm_agent_config['llm_config']['max_tokens'],
                provider_config=vector_store_config
            )
            logger.info("Knowledge Base Agent created")
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
        logger.info("prompt and config setup started")
        if self.task_config["task_type"] == "webhook":
            return

        agent_type = self.task_config["tools_config"]["llm_agent"].get("agent_type", "simple_llm_agent")
        if agent_type in ["openai_assistant", "knowledgebase_agent"]:
            return

        self.is_local = local
        if task_id == 0:
            if self.context_data and 'recipient_data' in self.context_data and self.context_data['recipient_data'] and self.context_data['recipient_data'].get('timezone', None):
                self.timezone = pytz.timezone(self.context_data['recipient_data']['timezone'])

        today = datetime.now(self.timezone).strftime("%A, %B %d, %Y")
        current_time = datetime.now(self.timezone).strftime("%I:%M:%S %p")

        prompt_responses = kwargs.get('prompt_responses', None)
        if not prompt_responses:
            prompt_responses = await get_prompt_responses(assistant_id=self.assistant_id, local=self.is_local)

        logger.info(f"GOT prompt responses {prompt_responses}")
        current_task = "task_{}".format(task_id + 1)
        if self.__is_multiagent():
            logger.info(f"Getting {current_task} from prompt responses of type {type(prompt_responses)}, prompt responses key {prompt_responses.keys()}")
            prompts = prompt_responses.get(current_task, None)
            self.prompt_map = {}
            for agent in self.task_config["tools_config"]["llm_agent"]['llm_config']['agent_map']:
                prompt = prompts[agent]['system_prompt']
                prompt = self.__prefill_prompts(self.task_config, prompt, self.task_config['task_type'])
                prompt = self.__get_final_prompt(prompt, today, current_time, self.timezone)
                if agent == self.task_config["tools_config"]["llm_agent"]['llm_config']['default_agent']:
                    self.system_prompt = {
                        'role': 'system',
                        'content': prompt
                    }
                self.prompt_map[agent] = prompt
            logger.info(f"Initialised prompt dict {self.prompt_map}, Set default prompt {self.system_prompt}")
        else:
            self.prompts = self.__prefill_prompts(self.task_config, prompt_responses.get(current_task, None), self.task_config['task_type'])
            if self.task_config["tools_config"]["llm_agent"]['agent_flow_type'] == "preprocessed":
                self.tools["llm_agent"].load_prompts_and_create_graph(self.prompts)

        if "system_prompt" in self.prompts:
            # This isn't a graph based agent
            enriched_prompt = self.prompts["system_prompt"]
            logger.info("There's a system prompt")
            if self.context_data is not None:
                enriched_prompt = update_prompt_with_context(self.prompts["system_prompt"], self.context_data)

                if 'recipient_data' in self.context_data and self.context_data['recipient_data'] and self.context_data['recipient_data'].get('call_sid', None):
                    self.call_sid = self.context_data['recipient_data']['call_sid']
                    enriched_prompt = f'{enriched_prompt}\nPhone call_sid is "{self.call_sid}"\n'

                enriched_prompt = f'{enriched_prompt}\nagent_id is "{self.agent_id}"\nexecution_id is "{self.run_id_ts}"\n'
                self.prompts["system_prompt"] = enriched_prompt

            notes = "### Note:\n"
            
            if self._is_conversation_task() and self.use_fillers:
                notes += f"1.{FILLER_PROMPT}\n"
            
            self.system_prompt = {
                'role': "system",
                'content': f"{enriched_prompt}\n{notes}\n{DATE_PROMPT.format(today, current_time, self.timezone)}"
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

        #If history is empty and agent welcome message is not empty add it to history
        if len(self.history) == 1 and len(self.kwargs['agent_welcome_message']) != 0:
            self.history.append({'role': 'assistant', 'content': self.kwargs['agent_welcome_message']})

        self.interim_history = copy.deepcopy(self.history)

    def __prefill_prompts(self, task, prompt, task_type):
        if not prompt and task_type in ('extraction', 'summarization'):
            if task_type == 'extraction':
                extraction_json = task.get("tools_config").get('llm_agent', {}).get('llm_config', {}).get('extraction_json')
                prompt = EXTRACTION_PROMPT.format(extraction_json)
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

    async def __cleanup_downstream_tasks(self):
        logger.info(f"Cleaning up downstream task")
        start_time = time.time()
        await self.tools["output"].handle_interruption()
        self.sequence_ids = {-1} 
        
        #Stop the output loop first so that we do not transmit anything else
        if self.output_task is not None:
            logger.info(f"Cancelling output task")
            self.output_task.cancel()

        if self.llm_task is not None:
            logger.info(f"Cancelling LLM Task")
            self.llm_task.cancel()
            self.llm_task = None

        if self.first_message_task is not None:
            logger.info("Cancelling first message task")
            self.first_message_task.cancel()
            self.first_message_task = None

        # self.synthesizer_task.cancel()
        # self.synthesizer_task = asyncio.create_task(self.__listen_synthesizer())
        for task in self.synthesizer_tasks:
            task.cancel()
        
        self.synthesizer_tasks = []

        logger.info(f"Synth Task cancelled seconds")
        if not self.buffered_output_queue.empty():
            logger.info(f"Output queue was not empty and hence emptying it")
            self.buffered_output_queue = asyncio.Queue()
        
        #restart output task
        self.output_task = asyncio.create_task(self.__process_output_loop())
        self.started_transmitting_audio = False #Since we're interrupting we need to stop transmitting as well
        logger.info(f"Cleaning up downstream tasks sequenxce ids {self.sequence_ids}. Time taken to send a clear message {time.time() - start_time}")

    def __get_updated_meta_info(self, meta_info = None):
        #This is used in case there's silence from callee's side
        if meta_info is None:
            meta_info = self.tools["transcriber"].get_meta_info()
            logger.info(f"Metainfo {meta_info}")
        meta_info_copy = meta_info.copy()
        self.curr_sequence_id +=1
        meta_info_copy["sequence_id"] = self.curr_sequence_id
        meta_info_copy['turn_id'] = self.turn_id
        self.sequence_ids.add(meta_info_copy["sequence_id"])
        return meta_info_copy
    
    def _extract_sequence_and_meta(self, message):
        sequence, meta_info = None, None
        if isinstance(message, dict) and "meta_info" in message:
            self._set_call_details(message)
            sequence = message["meta_info"]["sequence"]
            meta_info = message["meta_info"]
        return sequence, meta_info

    def _is_extraction_task(self):
        return self.task_config["task_type"] == "extraction"

    def _is_summarization_task(self):
        return self.task_config["task_type"] == "summarization"

    def _is_conversation_task(self):
        return self.task_config["task_type"] == "conversation"

    def __is_openai_assistant_agent(self):
        return self.task_config["tools_config"]["llm_agent"].get("agent_type", None) == "openai_assistant"

    def _is_preprocessed_flow(self):
        return "agent_flow_type" in self.task_config["tools_config"]["llm_agent"] and self.task_config["tools_config"]["llm_agent"]['agent_flow_type'] == "preprocessed"

    def _get_next_step(self, sequence, origin):
        try:
            return next((self.pipelines[sequence][i + 1] for i in range(len(self.pipelines[sequence]) - 1) if
                         self.pipelines[sequence][i] == origin), "output")
        except Exception as e:
            logger.error(f"Error getting next step: {e}")

    def _set_call_details(self, message):
        if self.call_sid is not None and self.stream_sid is not None and "call_sid" not in message['meta_info'] and "stream_sid" not in message['meta_info']:
            return

        if "call_sid" in message['meta_info']:
            self.call_sid = message['meta_info']["call_sid"]
        if "stream_sid" in message:
            self.stream_sid = message['meta_info']["stream_sid"]

    async def _process_followup_task(self, message=None):
        logger.info(f" TASK CONFIG  {self.task_config['task_type']}")
        if self.task_config["task_type"] == "webhook":
            logger.info(f"Input patrameters {self.input_parameters}")
            extraction_details = self.input_parameters.get('extraction_details', {})
            logger.info(f"DOING THE POST REQUEST TO WEBHOOK {extraction_details}")
            self.webhook_response = await self.tools["webhook_agent"].execute(extraction_details)
            logger.info(f"Response from the server {self.webhook_response}")
        
        else:
            message = format_messages(self.input_parameters["messages"])  # Remove the initial system prompt
            self.history.append({
                'role': 'user',
                'content': message
            })

            today = datetime.now().strftime("%A, %B %d, %Y")
            self.history[0]['content'] += f"\n Today's Date is {today}"

            json_data = await self.tools["llm_agent"].generate(self.history)
            if self.task_config["task_type"] == "summarization":
                logger.info(f'Summary {json_data["summary"]}')
                self.summarized_data = json_data["summary"]
                logger.info(f"self.summarize {self.summarized_data}")
            else:
                logger.info(f"Extraction task output {json_data}")
                json_data = clean_json_string(json_data)
                logger.info(f"After replacing {json_data}")
                if type(json_data) is not dict:
                    json_data = json.loads(json_data)
                self.extracted_data = json_data
        logger.info("Done")

    async def __process_end_of_conversation(self):
        logger.info("Got end of conversation. I'm stopping now")
        self.conversation_ended = True
        self.ended_by_assistant = True
        await self.tools["input"].stop_handler()
        logger.info("Stopped input handler")
        if "transcriber" in self.tools and not self.turn_based_conversation:
            logger.info("Stopping transcriber")
            await self.tools["transcriber"].toggle_connection()
            await asyncio.sleep(5)  # Making sure whatever message was passed is over

    def __update_preprocessed_tree_node(self):
        logger.info(f"It's a preprocessed flow and hence updating current node")
        self.tools['llm_agent'].update_current_node()


    ##############################################################
    # LLM task
    ##############################################################
    async def _handle_llm_output(self, next_step, text_chunk, should_bypass_synth, meta_info, is_filler = False):

        logger.info("received text from LLM for output processing: {} which belongs to sequence id {}".format(text_chunk, meta_info['sequence_id']))
        if "request_id" not in meta_info:
            meta_info["request_id"] = str(uuid.uuid4())
        
        if not self.stream and not is_filler:
            first_buffer_latency = time.time() - meta_info["llm_start_time"]
            meta_info["llm_first_buffer_generation_latency"] = first_buffer_latency
            latency_metrics = {
                "transcriber": {
                    "latency": meta_info.get('transcriber_latency', 0),
                    "average_latency": self.average_transcriber_latency,
                },
                "llm": {
                    "first_llm_buffer_latency": meta_info.get('llm_latency', 0),
                    "average_latency": self.average_llm_latency,
                },
            }
            self.latency_dict[meta_info["request_id"]] = latency_metrics
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
            self.latency_dict[meta_info["request_id"]]['overall_first_byte_latency'] = overall_time
            #self.history = copy.deepcopy(self.interim_history)
            await self.tools["output"].handle(create_ws_data_packet(text_chunk, meta_info))

    async def _process_conversation_preprocessed_task(self, message, sequence, meta_info):
        if self.task_config["tools_config"]["llm_agent"]['agent_flow_type'] == "preprocessed":
            messages = copy.deepcopy(self.history)
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
                messages.append({'role': 'assistant', 'content': next_state['text']})
                self.synthesizer_tasks.append(asyncio.create_task(
                        self._synthesize(create_ws_data_packet(next_state['audio'], meta_info, is_md5_hash=True))))
            logger.info(f"Interim history after the LLM task {messages}")
            self.llm_response_generated = True
            self.interim_history = copy.deepcopy(messages)
            if self.callee_silent:
                logger.info("When we got utterance end, maybe LLM was still generating response. So, copying into history")
                self.history = copy.deepcopy(self.interim_history)

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
        await self._handle_llm_output(next_step, filler, should_bypass_synth, new_meta_info, is_filler = True)
    
    async def __execute_function_call(self, url, method, param, api_token, model_args, meta_info, next_step, called_fun, **resp):
        self.check_if_user_online = False 

        if called_fun.startswith("transfer_call"):
            logger.info(f"Transfer call function called param {param}. First sleeping for 2 seconds to make sure we're done speaking the filler")
            await asyncio.sleep(2) #Sleep for 1 second to ensure that the filler is spoken before transfering call
            call_sid = self.tools["input"].get_call_sid()

            try:
                from_number = self.context_data['recipient_data']['from_number']
            except Exception as e:
                from_number = None

            user_id, agent_id = self.assistant_id.split("/")
            self.history = copy.deepcopy(model_args["messages"])

            if url is None:
                url = os.getenv("CALL_TRANSFER_WEBHOOK_URL")
                payload = {
                    'call_sid': call_sid,
                    'agent_id': agent_id,
                    'user_id': user_id,
                    'provider': self.tools['input'].io_provider,
                    'stream_sid': self.stream_sid,
                    'from_number': from_number,
                    'execution_id': self.run_id_ts
                }

                try:
                    json_function_call_params = json.loads(param)
                    call_transfer_number = json_function_call_params['call_transfer_number']
                    if call_transfer_number:
                        payload['call_transfer_number'] = call_transfer_number
                except Exception as e:
                    logger.error(f"Error in __execute_function_call {e}")
            else:
                payload = {'call_sid': call_sid, "agent_id": agent_id}

            if param is not None:
                logger.info(f"Gotten response {resp}")
                payload = {**payload, **resp}

            async with aiohttp.ClientSession() as session:
                logger.info(f"Sending the payload to stop the conversation {payload} url {url}")
                convert_to_request_log(str(payload), meta_info, None, "function_call", direction="request", is_cached=False,
                                       run_id=self.run_id)
                async with session.post(url, json = payload) as response:
                    response_text = await response.text()
                    logger.info(f"Response from the server after call transfer: {response_text}")
                    convert_to_request_log(str(response_text), meta_info, None, "function_call", direction="response", is_cached=False, run_id=self.run_id)
                    return
                
        response = await trigger_api(url= url, method=method.lower(), param= param, api_token= api_token, meta_info = meta_info, run_id = self.run_id, **resp)
        content = FUNCTION_CALL_PROMPT.format(called_fun, method, str(response))
        model_args["messages"].append({"role":"system","content":content})
        logger.info(f"Logging function call parameters ")
        convert_to_request_log(str(response), meta_info , None, "function_call", direction = "response", is_cached= False, run_id = self.run_id)

        convert_to_request_log(format_messages(model_args['messages'], True), meta_info, self.llm_config['model'], "llm", direction = "request", is_cached= False, run_id = self.run_id)
        self.check_if_user_online = self.conversation_config.get("check_if_user_online", True)

        if not called_fun.startswith("transfer_call"):
            await self.__do_llm_generation(model_args["messages"], meta_info, next_step, should_trigger_function_call = True)

        self.execute_function_call_task = None
            
    def __store_into_history(self, meta_info, messages, llm_response, should_trigger_function_call = False):
        if self.current_request_id in self.llm_rejected_request_ids:
            logger.info("##### User spoke while LLM was generating response")
        else:
            self.llm_response_generated = True
            convert_to_request_log(message=llm_response, meta_info= meta_info, component="llm", direction="response", model=self.llm_config["model"], run_id= self.run_id)
            if should_trigger_function_call:
                #Now, we need to consider 2 things here
                #1. There was silence between function call and now
                #2. There was a conversation till now
                logger.info(f"There was a function call and need to make that work")
                
                if self.interim_history[-1]['role'] == 'assistant' and self.interim_history[-1]['content'] == PRE_FUNCTION_CALL_MESSAGE:
                    logger.info(f"There was a no conversation between function call")
                    #Nothing was spoken
                    self.interim_history[-1]['content'] += llm_response
                else:
                    
                    logger.info(f"There was a conversation between function call and this and changing relevant history point")
                    #There was a conversation
                    messages = copy.deepcopy(self.interim_history)
                    for entry in reversed(messages):
                        if entry['content'] == PRE_FUNCTION_CALL_MESSAGE:
                            entry['content'] += llm_response
                            break
                    
                    self.interim_history = copy.deepcopy(messages)
                #Assuming that callee was silent
                self.history = copy.deepcopy(self.interim_history)
            else:
                logger.info(f"There was no function call {messages}")
                messages.append({"role": "assistant", "content": llm_response})
                self.interim_history = copy.deepcopy(messages)
                if self.callee_silent:
                    logger.info("##### When we got utterance end, maybe LLM was still generating response. So, copying into history")
                    self.history = copy.deepcopy(self.interim_history)
                #self.__update_transcripts()
                        
    async def __do_llm_generation(self, messages, meta_info, next_step, should_bypass_synth = False, should_trigger_function_call = False):
        llm_response = ""
        logger.info(f"MEssages before generation {messages}")
        synthesize = True
        if should_bypass_synth:
            synthesize = False

        async for llm_message in self.tools['llm_agent'].generate(messages, synthesize=synthesize, meta_info=meta_info):
            logger.info(f"llm_message {llm_message}")
            data, end_of_llm_stream, latency, trigger_function_call = llm_message

            if trigger_function_call:
                logger.info(f"Triggering function call for {data}")
                self.llm_task = asyncio.create_task(self.__execute_function_call(next_step = next_step, **data))
                return
            
            if latency and (len(self.llm_latencies) == 0 or self.llm_latencies[-1] != latency):
                meta_info["llm_latency"] = latency
                self.llm_latencies.append(latency)
                self.average_llm_latency = sum(self.llm_latencies) / len(self.llm_latencies)
                logger.info(f"Got llm latencies {self.llm_latencies}")

            llm_response += " " + data

            # Checking the size of the llm_response as it should be less than 20 words
            # if len(llm_response.strip()) > 20:
            #     # Let's make a gpt3.5 turbo openai async call to summarize the text and make it shorter
            #     summary_prompt = f"Summarize the following text in less than 100 characters and the answer should be precise:\n\n{llm_response}"
            #     summary_response = ""
            #     async for summarized_response in self.tools['llm_agent'].generate([{'role': 'user', 'content': summary_prompt}]):
            #         logger.info(f"Summarized response : {summary_response}")
                
            #     llm_response = summarized_response

            logger.info(f"Got a response from LLM {llm_response}")
            if end_of_llm_stream:
                meta_info["end_of_llm_stream"] = True

            if self.stream:
                text_chunk = self.__process_stop_words(data, meta_info)
                logger.info(f"##### O/P from LLM {text_chunk} {llm_response}")

                # A hack as during the 'await' part control passes to llm streaming function parameters
                # So we have to make sure we've commited the filler message
                if text_chunk == PRE_FUNCTION_CALL_MESSAGE:
                    logger.info("Got a pre function call message")
                    messages.append({'role':'assistant', 'content': PRE_FUNCTION_CALL_MESSAGE})
                    self.interim_history = copy.deepcopy(messages)

                await self._handle_llm_output(next_step, text_chunk, should_bypass_synth, meta_info)
            else:
                messages.append({"role": "assistant", "content": llm_response})
                self.history = copy.deepcopy(messages)
                await self._handle_llm_output(next_step, llm_response, should_bypass_synth, meta_info)
                convert_to_request_log(message=llm_response, meta_info=meta_info, component="llm", direction="response", model=self.llm_config["model"], run_id= self.run_id)
        
        if self.stream and llm_response != PRE_FUNCTION_CALL_MESSAGE:
            logger.info(f"Storing {llm_response} into history should_trigger_function_call {should_trigger_function_call}")
            self.__store_into_history(meta_info, messages, llm_response, should_trigger_function_call= should_trigger_function_call)

    async def _process_conversation_task(self, message, sequence, meta_info):
        next_step = None
        logger.info("agent flow is not preprocessed")

        start_time = time.time()
        should_bypass_synth = 'bypass_synth' in meta_info and meta_info['bypass_synth'] == True
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
                messages += [{'role': 'user', 'content': message['data']},{'role': 'assistant', 'content': cache_response}]
                self.interim_history = copy.deepcopy(messages)
                self.llm_response_generated = True
                if self.callee_silent:
                    logger.info("##### When we got utterance end, maybe LLM was still generating response. So, copying into history")
                    self.history = copy.deepcopy(self.interim_history)

            else:
                logger.info(f"Route doesn't have a vector cache, and hence simply returning back a given response")
                cache_response = self.route_responses_dict[route]

            logger.info(f"Cached response {cache_response}")
            meta_info['cached'] = True
            meta_info["end_of_llm_stream"] = True

            await self._handle_llm_output(next_step, cache_response, should_bypass_synth, meta_info)
            self.llm_processed_request_ids.add(self.current_request_id)
        else:
            messages = copy.deepcopy(self.history)
            logger.info(f"Message {messages} history {self.history}")
            messages.append({'role': 'user', 'content': message['data']})
            ### TODO CHECK IF THIS IS EVEN REQUIRED
            convert_to_request_log(message=format_messages(messages, use_system_prompt=True), meta_info=meta_info, component="llm", direction="request", model=self.llm_config["model"], run_id= self.run_id)

            await self.__do_llm_generation(messages, meta_info, next_step, should_bypass_synth)
            # TODO : Write a better check for completion prompt
            if self.use_llm_to_determine_hangup and not self.turn_based_conversation:
                answer = await self.tools["llm_agent"].check_for_completion(self.history, self.check_for_completion_prompt)
                should_hangup = answer['answer'].lower() == "yes"
                prompt = [
                        {'role': 'system', 'content': self.check_for_completion_prompt},
                        {'role': 'user', 'content': format_messages(self.history, use_system_prompt= True)}]
                logger.info(f"##### Answer from the LLM {answer}")
                convert_to_request_log(message=format_messages(prompt, use_system_prompt= True), meta_info= meta_info, component="llm", direction="request", model=self.llm_config["model"], run_id= self.run_id)
                convert_to_request_log(message=answer, meta_info= meta_info, component="llm", direction="response", model= self.check_for_completion_llm, run_id= self.run_id)

                if should_hangup:
                    await self.__process_end_of_conversation()
                    return

            self.llm_processed_request_ids.add(self.current_request_id)
            llm_response = ""
    
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
                if self._is_preprocessed_flow():
                    self.__update_preprocessed_tree_node()
                eos_packet = create_ws_data_packet("<end_of_stream>", meta_info)
                await self.tools["output"].handle(eos_packet)

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Something went wrong with LLM queue {e}")
                break

    async def _run_llm_task(self, message):
        sequence, meta_info = self._extract_sequence_and_meta(message)
        logger.info(f"After adding {self.curr_sequence_id} into sequence id {self.sequence_ids} for message {message}")

        try:
            if self._is_extraction_task() or self._is_summarization_task():
                await self._process_followup_task(message)
            elif self._is_conversation_task():
                if self._is_preprocessed_flow():
                    if time.time() < self.consider_next_transcript_after:
                        logger.info("Not considering transcript as we're still in cool down period")
                        await asyncio.sleep(self.consider_next_transcript_after - time.time())
                    logger.info(f"Running preprocessedf task")
                    await self._process_conversation_preprocessed_task(message, sequence, meta_info)

                else:
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
        if not self.current_request_id or self.current_request_id != meta_info["request_id"]:
            self.previous_request_id, self.current_request_id = self.current_request_id, meta_info["request_id"]

        sequence = meta_info["sequence"]

        # check if previous request id is not in transmitted request id
        if self.previous_request_id is None:
            is_first_message = True
        elif self.previous_request_id not in self.llm_processed_request_ids:
            logger.info(f"Adding previous request id to LLM rejected request if")
            self.llm_rejected_request_ids.add(self.previous_request_id)
        else:
            skip_append_to_data = False
        return sequence

    async def _handle_transcriber_output(self, next_task, transcriber_message, meta_info):
        convert_to_request_log(message=transcriber_message, meta_info= meta_info, model = "deepgram", run_id= self.run_id)
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
        transcriber_message = ""
        logger.info(f"Starting transcriber task")
        response_started = False
        num_words = 0
        try:
            while True:
                message = await self.transcriber_output_queue.get()
                logger.info(f"##### Message from the transcriber class {message}")
                if message['meta_info'] is not None and message['meta_info'].get('transcriber_latency', False):
                    self.transcriber_latencies.append(message['meta_info']['transcriber_latency'])
                    self.average_transcriber_latency = sum(self.transcriber_latencies) / len(self.transcriber_latencies)
                if message["data"].strip() == "":
                    continue
                if message['data'] == "transcriber_connection_closed":
                    self.transcriber_duration += message['meta_info']["transcriber_duration"] if message['meta_info'] is not None else 0
                    logger.info("transcriber connection closed")
                    break

                if self.stream:
                    self._set_call_details(message)
                    meta_info = message["meta_info"]
                    sequence = await self.process_transcriber_request(meta_info)
                    next_task = self._get_next_step(sequence, "transcriber")
                    num_words = 0
                    if message['data'] == "TRANSCRIBER_BEGIN":
                        response_started = False #This signifies if we've gotten the first bit of interim text for the given response or not
                        self.callee_silent = False

                    elif "speech_final" in meta_info and meta_info['speech_final'] and message['data'] != "":
                        logger.info(f"Starting the TRANSCRIBER_END TASK")
                        self.callee_speaking = False
                        self.callee_silent = True

                        if self.output_task is None:
                            logger.info(f"Output task was none and hence starting it")
                            self.output_task = asyncio.create_task(self.__process_output_loop())

                        if self._is_preprocessed_flow():
                            self.__update_preprocessed_tree_node()
                        
                        logger.info(f"INTERIM TRANSCRIPT WHEN EVERYTING IS OVER {self.interim_history}")
                        if self.llm_response_generated:
                            logger.info(f"LLM RESPONSE WAS GENERATED AND HENCE MOVING INTERIM HISTORY TO HISTORY")
                            self.history = copy.deepcopy(self.interim_history)
                        meta_info = message['meta_info']
                        transcriber_message = ""
                        self.let_remaining_audio_pass_through = True 

                        if self.nitro:
                            self.time_since_first_interim_result = -1
                            self.required_delay_before_speaking = max(self.minimum_wait_duration - self.incremental_delay, 0)
                            logger.info(f"#### Resetting time since first interim result and resetting required delay {self.required_delay_before_speaking}")
                        
                    else:
                        self.time_since_last_spoken_human_word = time.time()
                        logger.info(f'invoking next_task {next_task} with transcriber_message: {message["data"]}')
                        if transcriber_message.strip() == message['data'].strip():
                            logger.info(f"###### Transcriber message and message data are same and hence not changing anything else. Probably just an is_final thingy. {message}")
                            continue

                        elif len(message['data'].strip()) != 0:
                            #Currently simply cancel the next task

                            if not self.first_message_passed:
                                logger.info(f"Adding to transcrber message")
                                self.transcriber_message += message['data']
                                continue

                            num_words += len(message['data'].split(" "))
                            if self.callee_speaking is False:
                                self.callee_speaking_start_time = time.time()
                                self.callee_speaking = True
                            
                            # This means we are generating response from an interim transcript 
                            # Hence we transmit quickly 
                            if not self.started_transmitting_audio:
                                logger.info("##### Haven't started transmitting audio and hence cleaning up downstream tasks")
                                await self.__cleanup_downstream_tasks()
                            else:
                                logger.info(f"Started transmitting and hence moving fursther")
                            
                            # If we've started transmitting audio this is probably an interruption, so calculate number of words
                            if self.started_transmitting_audio and self.number_of_words_for_interruption != 0 and self.first_message_passed:
                                if num_words > self.number_of_words_for_interruption or message['data'].strip() in self.accidental_interruption_phrases:
                                    #Process interruption only if number of words is higher than the threshold 
                                    logger.info(f"###### Number of words {num_words} is higher than the required number of words for interruption, hence, definitely interrupting. Interruption and hence changing the turn id")
                                    self.turn_id += 1
                                    await self.__cleanup_downstream_tasks()
                                else:
                                    logger.info(f"Not starting a cleanup because {num_words} number of words are lesser {self.number_of_words_for_interruption} and hence continuing,")
                                    continue
                            elif self.number_of_words_for_interruption == 0:
                                logger.info(f"Not interrupting")
                                    
                            self.last_response_time = time.time()
                            transcriber_message = message['data']
                            
                            # Use last spoken timestamp to give out endpointing in nitro
                            logger.info(f"###### last spoken timestamp before changing {self.last_spoken_timestamp}")
                            self.last_spoken_timestamp = time.time() * 1000

                            if not response_started:
                                response_started = True
                            
                            if self.nitro:
                                self.let_remaining_audio_pass_through = False
                                self.required_delay_before_speaking += self.incremental_delay
                                logger.info(f"Increase the incremental delay time {self.required_delay_before_speaking}")
                                if self.time_since_first_interim_result == -1:
                                    self.time_since_first_interim_result = time.time() * 1000
                                    logger.info(f"###### Updating Time since first interim result {self.time_since_first_interim_result}")
                                #In this case user has already started speaking
                                # Hence check the previous message if it's user or assistant
                                # If it's user, simply change user's message
                                # If it's assistant remover assistant message and append user
                                self.callee_silent = False
                            self.llm_response_generated = False
                            logger.info("###### Current transcript: {} Predicting next few tokens and changing last spoken timestampt to {}".format(transcriber_message, self.last_spoken_timestamp))
                            meta_info = self.__get_updated_meta_info(meta_info)
                            await self._handle_transcriber_output(next_task, transcriber_message, meta_info)

                        else:
                            logger.info(f"Got a null message")
                else:
                    logger.info(f"Processing http transcription for message {message}")
                    await self.__process_http_transcription(message)
        
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in transcriber {e}")

    async def __process_http_transcription(self, message):
        meta_info = self.__get_updated_meta_info(message["meta_info"])
        include_latency = meta_info.get("include_latency", False)
        if include_latency:
            self.latency_dict[meta_info['request_id']]["transcriber"] = {"total_latency":  meta_info["transcriber_latency"], "audio_duration": meta_info["audio_duration"], "last_vocal_frame_timestamp": meta_info["last_vocal_frame_timestamp"] }

        sequence = message["meta_info"]["sequence"]
        next_task = self._get_next_step(sequence, "transcriber")
        self.transcriber_duration += message["meta_info"]["transcriber_duration"] if "transcriber_duration" in message["meta_info"] else 0
        #self.history.append({'role': 'user', 'content': message['data']})
        if self._is_preprocessed_flow():
            self.__update_preprocessed_tree_node()

        await self._handle_transcriber_output(next_task, message['data'], meta_info)


    #################################################################
    # Synthesizer task
    #################################################################
    def __enqueue_chunk(self, chunk, i, number_of_chunks, meta_info):
        logger.info(f"Meta_info of chunk {meta_info} {i} {number_of_chunks}")
        meta_info['chunk_id'] = i
        copied_meta_info = copy.deepcopy(meta_info)
        if i == 0 and "is_first_chunk" in meta_info and meta_info["is_first_chunk"]:
            logger.info(f"##### Sending first chunk")
            copied_meta_info["is_first_chunk_of_entire_response"] = True
        
        if i == number_of_chunks - 1 and (meta_info['sequence_id'] == -1 or ("end_of_synthesizer_stream" in meta_info and meta_info['end_of_synthesizer_stream'])):
            logger.info(f"##### Truly a final chunk")
            copied_meta_info["is_final_chunk_of_entire_response"] = True
            
        self.buffered_output_queue.put_nowait(create_ws_data_packet(chunk, copied_meta_info))

    async def __listen_synthesizer(self):
        try:
            # if self.stream and self.tools['synthesizer'].supports_websocket() and not self.is_an_ivr_call:
            #     logger.info("Opening websocket connection to synthesizer")
            #     await self.tools["synthesizer"].open_connection()

            while True:
                logger.info("Listening to synthesizer")
                try:
                    async for message in self.tools["synthesizer"].generate():
                        meta_info = message["meta_info"]
                        is_first_message = 'is_first_message' in meta_info and meta_info['is_first_message']
                        if is_first_message or (not self.conversation_ended and message["meta_info"]["sequence_id"] in self.sequence_ids):
                            logger.info(f"{message['meta_info']['sequence_id'] } is in sequence ids  {self.sequence_ids} and hence letting it pass by")
                            first_chunk_generation_timestamp = time.time()
                            meta_info["synthesizer_latency"] = first_chunk_generation_timestamp - message['meta_info']['synthesizer_start_time']
                            self.synthesizer_latencies.append(meta_info["synthesizer_latency"])
                            self.average_synthesizer_latency = sum(self.synthesizer_latencies) / len(self.synthesizer_latencies)

                            if self.stream:
                                message['data'] = await self.process_audio_data_for_output(meta_info, message)
                                if "is_first_chunk" in message['meta_info'] and message['meta_info']['is_first_chunk']:
                                    first_chunk_generation_timestamp = time.time()
                                    meta_info["synthesizer_first_chunk_latency"] = first_chunk_generation_timestamp - message['meta_info']['synthesizer_start_time']
                                logger.info(f"Simply Storing in buffered output queue for now")

                                #self.buffered_output_queue.put_nowait(message)
                                if self.tools["output"].process_in_chunks(self.yield_chunks):
                                    number_of_chunks = math.ceil(len(message['data'])/self.output_chunk_size)
                                    chunk_idx = 0
                                    logger.info(f"Audio chunk size {len(message['data'])}, chunk size {self.output_chunk_size}, number_of_chunks: {number_of_chunks}")
                                    for chunk in yield_chunks_from_memory(message['data'], chunk_size=self.output_chunk_size):
                                        self.__enqueue_chunk(chunk, chunk_idx, number_of_chunks, meta_info)
                                        chunk_idx += 1
                                else:
                                    #await self.tools["output"].handle(message)
                                    self.buffered_output_queue.put_nowait(message)
                            else:
                                logger.info("Stream is not enabled and hence sending entire audio")
                                self.latency_dict[meta_info["request_id"]]["synthesizer"] = {
                                    "synthesizer_first_chunk_latency": meta_info.get("synthesizer_latency", 0),
                                    "average_latency": self.average_synthesizer_latency
                                }
                                overall_time = time.time() - meta_info["start_time"]
                                await self.tools["output"].handle(message)
                        else:
                            logger.info(f"{message['meta_info']['sequence_id']} is not in sequence ids  {self.sequence_ids} and hence not sending to output")

                        convert_to_request_log(message = meta_info['text'], meta_info= meta_info, component="synthesizer", direction="response", model = self.synthesizer_provider, is_cached= 'is_cached' in meta_info and meta_info['is_cached'], engine=self.tools['synthesizer'].get_engine(), run_id= self.run_id)
                        await asyncio.sleep(0.1) #Sleeping for 100ms after receiving every chunk so other tasks can execute
                except asyncio.CancelledError:
                    logger.info("Synthesizer task was cancelled.")
                    raise  # Make sure to re-raise the exception to stop the loop
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Error in synthesizer {e}")
                    break

        except asyncio.CancelledError:
            logger.info("Synthesizer task cancelled outside loop.")
            raise  # Allow cancellation outside the loop

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in synthesizer outside loop.{e}")

    async def process_audio_data_for_output(self, meta_info, message):
        return message['data']
        if self.task_config["tools_config"]["output"]["format"] == "pcm" and meta_info.get('format', '') != 'mulaw':
            message['data'] = wav_bytes_to_pcm(message['data'])

        # TODO remove this hard-coded condition
        elif self.synthesizer_provider == 'elevenlabs' and self.tools["output"].get_provider() == 'plivo':
            message['data'] = wav_bytes_to_pcm(message['data'])

        return message['data']

    async def __send_preprocessed_audio(self, meta_info, text):
        meta_info = copy.deepcopy(meta_info)

        # if self.first_message_passed is True else False. chunking->false to make it interrupted
        yield_in_chunks = self.yield_chunks
        try:
            #TODO: Either load IVR audio into memory before call or user s3 iter_cunks
            # This will help with interruption in IVR
            audio_chunk = None
            if self.turn_based_conversation or self.task_config['tools_config']['output'] == "default":
                audio_chunk = await get_raw_audio_bytes(text, self.assistant_name,
                                                                self.task_config["tools_config"]["output"][
                                                                    "format"], local=self.is_local,
                                                                assistant_id=self.assistant_id)
                logger.info("Sending preprocessed audio")
                meta_info["format"] = self.task_config["tools_config"]["output"]["format"]
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
                    audio_chunk = await get_raw_audio_bytes(text, self.assistant_name,
                                                                'pcm', local=self.is_local,
                                                                assistant_id=self.assistant_id)
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
                            logger.info(f"Sending the agent welcome message")
                            meta_info['is_first_chunk'] = True
                if yield_in_chunks and audio_chunk is not None:
                    i = 0
                    number_of_chunks = math.ceil(len(audio_chunk)/self.output_chunk_size)
                    logger.info(f"Audio chunk size {len(audio_chunk)}, chunk size {self.output_chunk_size}")
                    for chunk in yield_chunks_from_memory(audio_chunk, chunk_size=self.output_chunk_size):
                        self.__enqueue_chunk(chunk, i, number_of_chunks, meta_info)
                        i +=1
                elif audio_chunk is not None:
                    meta_info['chunk_id'] = 1
                    meta_info["is_first_chunk_of_entire_response"] = True
                    meta_info["is_final_chunk_of_entire_response"] = True
                    message = create_ws_data_packet(audio_chunk, meta_info)
                    logger.info(f"Yield in chunks is false and hence sending a full")
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
            if not self.conversation_ended and ('is_first_message' in meta_info and meta_info['is_first_message'] or message["meta_info"]["sequence_id"] in self.sequence_ids):
                if meta_info["is_md5_hash"]:
                    logger.info('sending preprocessed audio response to {}'.format(self.task_config["tools_config"]["output"]["provider"]))
                    await self.__send_preprocessed_audio(meta_info, text)
                    
                elif self.synthesizer_provider in SUPPORTED_SYNTHESIZER_MODELS.keys():
                    # self.sequence_ids.add(meta_info["sequence_id"])
                    # logger.info(f"After adding into sequence id {self.sequence_ids}")
                    convert_to_request_log(message = text, meta_info= meta_info, component="synthesizer", direction="request", model = self.synthesizer_provider, engine=self.tools['synthesizer'].get_engine(), run_id= self.run_id)
                    logger.info('##### sending text to {} for generation: {} '.format(self.synthesizer_provider, text))
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
                logger.info(f"{message['meta_info']['sequence_id']} is not in sequence ids  {self.sequence_ids} and hence not synthesizing this")                

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in synthesizer: {e}")

    ############################################################
    # Output handling
    ############################################################
    
    async def __send_first_message(self, message):
        meta_info = self.__get_updated_meta_info()
        sequence = meta_info["sequence"]
        next_task = self._get_next_step(sequence, "transcriber")
        await self._handle_transcriber_output(next_task, message, meta_info)
        self.time_since_first_interim_result = (time.time() * 1000) - 1000

    async def __handle_initial_silence(self, duration=5):
        while True:
            logger.info(f"Checking for initial silence {duration}")
            #logger.info(f"Woke up from my slumber {self.callee_silent}, {self.history}, {self.interim_history}")
            if self.first_message_passed and self.callee_silent and len(self.history) == 1 and len(self.interim_history) == 1 and time.time() - self.first_message_passing_time > duration:
                logger.info(f"Callee was silent and hence speaking Hello on callee's behalf")
                await self.__send_first_message("Hello")
                break
            elif len(self.history) > 1:
                break
            await asyncio.sleep(3)
        self.background_check_task = None

    def __process_latency_data(self, message):
        utterance_end = message['meta_info'].get("utterance_end", None)
        overall_first_byte_latency = time.time() - message['meta_info']['utterance_end'] if utterance_end is not None else 0
        transcriber_latency = message["meta_info"].get("transcriber_latency", 0) if utterance_end is not None else 0
        first_llm_buffer_latency = message["meta_info"].get("llm_latency", 0) if utterance_end is not None else 0
        synthesizer_first_chunk_latency = message["meta_info"].get("synthesizer_latency", 0) if utterance_end is not None else 0

        if utterance_end is None:
            logger.info(f"First chunk is none")

        latency_metrics = {
            "transcriber": {
                "utterance_end": utterance_end,
                "latency": transcriber_latency,
                "average_latency": self.average_transcriber_latency
                },
            "llm": {
                "first_llm_buffer_latency" : first_llm_buffer_latency,
                "average_latency": self.average_llm_latency,
                },
            "synthesizer": {
                "synthesizer_first_chunk_latency": synthesizer_first_chunk_latency,
                "average_latency": self.average_synthesizer_latency
                },
            "overall_first_byte_latency": overall_first_byte_latency,
            
            }

        if message['meta_info']["request_id"] not in self.latency_dict:
            self.latency_dict[message['meta_info']["request_id"]] = latency_metrics
            logger.info("LATENCY METRICS FOR {} are {}".format(message['meta_info']["request_id"], latency_metrics))
            
    #Currently this loop only closes in case of interruption 
    # but it shouldn't be the case. 
    async def __process_output_loop(self):
        try:
            num_chunks = 0
            while True:
                if (not self.let_remaining_audio_pass_through) and self.first_message_passed:
                    time_since_first_interim_result = (time.time() * 1000) - self.time_since_first_interim_result if self.time_since_first_interim_result != -1 else -1
                    logger.info(f"##### It's been {time_since_first_interim_result} ms since first  interim result and required time to wait for it is {self.required_delay_before_speaking}. Hence sleeping for 100ms. self.time_since_first_interim_result {self.time_since_first_interim_result}")
                    if time_since_first_interim_result != -1 and time_since_first_interim_result < self.required_delay_before_speaking:
                        await asyncio.sleep(0.1) #sleep for 100ms and continue 
                        continue
                    else:
                        logger.info(f"First interim result hasn't been gotten yet and hence sleeping ")
                        await asyncio.sleep(0.1)
                    
                    logger.info(f"##### Got to wait {self.required_delay_before_speaking} ms before speaking and alreasy waited {time_since_first_interim_result} since the first interim result")
                
                elif self.let_remaining_audio_pass_through:
                    time_since_first_interim_result = (time.time() *1000)- self.time_since_first_interim_result if self.time_since_first_interim_result != -1 else -1
                    logger.info(f"##### In elif been {time_since_first_interim_result} ms since first  interim result and required time to wait for it is {self.required_delay_before_speaking}. Hence sleeping for 100ms. self.time_since_first_interim_result {self.time_since_first_interim_result}")
                    if time_since_first_interim_result != -1 and time_since_first_interim_result < self.required_delay_before_speaking:
                        await asyncio.sleep(0.1) #sleep for 100ms and continue 
                        continue
                else:
                    logger.info(f"Started transmitting at {time.time()}")

                message = await self.buffered_output_queue.get()
                chunk_id = message['meta_info']['chunk_id']

                logger.info("##### Start response is True for {} and hence starting to speak {} Current sequence ids {}".format(chunk_id, message['meta_info'], self.sequence_ids))
                if "end_of_conversation" in message['meta_info']:
                    await self.__process_end_of_conversation()
                
                if 'sequence_id' in message['meta_info'] and message["meta_info"]["sequence_id"] in self.sequence_ids:
                    num_chunks += 1
                    await self.tools["output"].handle(message)                    
                    duration = calculate_audio_duration(len(message["data"]), self.sampling_rate, format = message['meta_info']['format'])
                    logger.info(f"Duration of the byte {duration}")
                    self.conversation_recording['output'].append({'data': message['data'], "start_time": time.time(), "duration": duration})
                else:
                    logger.info(f'{message["meta_info"]["sequence_id"]} is not in {self.sequence_ids} and hence not speaking')
                    continue
                
                if "is_final_chunk_of_entire_response" in message['meta_info'] and message['meta_info']['is_final_chunk_of_entire_response']:
                    self.started_transmitting_audio = False
                    logger.info("##### End of synthesizer stream and ")

                    #If we're sending the message to check if user is still here, don't set asked_if_user_is_still_there to True
                    if message['meta_info'].get('text', '') != self.check_user_online_message:
                        self.asked_if_user_is_still_there = False

                    num_chunks = 0
                    self.turn_id += 1
                    if not self.first_message_passed:
                        self.first_message_passed = True
                        logger.info(f"Making first message passed as True")
                        self.first_message_passing_time = time.time()
                        if len(self.transcriber_message) != 0:
                            logger.info(f"Sending the first message as the first message is still not passed and we got a response")
                            await self.__send_first_message(self.transcriber_message)
                            self.transcriber_message = ''

                if "is_first_chunk_of_entire_response" in message['meta_info'] and message['meta_info']['is_first_chunk_of_entire_response']:
                    logger.info(f"First chunk stuff")
                    self.started_transmitting_audio = True if "is_final_chunk_of_entire_response" not in message['meta_info'] else False
                    self.consider_next_transcript_after = time.time() + self.duration_to_prevent_accidental_interruption
                    self.__process_latency_data(message) 
                else:
                    # Sleep until this particular audio frame is spoken only if the duration for the frame is atleast 500ms
                    if duration > 0:
                        logger.info(f"##### Sleeping for {duration} to maintain quueue on our side {self.sampling_rate}")
                        await asyncio.sleep(duration - 0.030) #30 milliseconds less
                if message['meta_info']['sequence_id'] != -1: #Making sure we only track the conversation's last transmitted timesatamp
                    self.last_transmitted_timestamp = time.time()
                logger.info(f"##### Updating Last transmitted timestamp to {self.last_transmitted_timestamp}")
                
        except Exception as e:
            traceback.print_exc()
            logger.error(f'Error in processing message output')

    async def __check_for_completion(self):
        logger.info(f"Starting task to check for completion")
        while True:
            await asyncio.sleep(2)
            if self.last_transmitted_timestamp == 0:
                logger.info(f"Last transmitted timestamp is simply 0 and hence continuing")
                continue

            time_since_last_spoken_ai_word = (time.time() - self.last_transmitted_timestamp)
            if time_since_last_spoken_ai_word > self.hang_conversation_after and self.time_since_last_spoken_human_word < self.last_transmitted_timestamp:
                logger.info(f"{time_since_last_spoken_ai_word} seconds since last spoken time stamp and hence cutting the phone call and last transmitted timestampt ws {self.last_transmitted_timestamp} and time since last spoken human word {self.time_since_last_spoken_human_word}")
                await self.__process_end_of_conversation()
                break

            elif time_since_last_spoken_ai_word > self.trigger_user_online_message_after and not self.asked_if_user_is_still_there and self.time_since_last_spoken_human_word < self.last_transmitted_timestamp:
                logger.info(f"Asking if the user is still there")
                self.asked_if_user_is_still_there = True

                if self.check_if_user_online:
                    if self.should_record:
                        meta_info={'io': 'default', "request_id": str(uuid.uuid4()), "cached": False, "sequence_id": -1, 'format': 'wav'}
                        await self._synthesize(create_ws_data_packet(self.check_user_online_message, meta_info= meta_info))
                    else:
                        meta_info={'io': self.tools["output"].get_provider(), "request_id": str(uuid.uuid4()), "cached": False, "sequence_id": -1, 'format': 'pcm'}
                        await self._synthesize(create_ws_data_packet(self.check_user_online_message, meta_info= meta_info))

                # Just in case we need to clear messages sent before
                await self.tools["output"].handle_interruption()
            else:
                logger.info(f"Only {time_since_last_spoken_ai_word} seconds since last spoken time stamp and hence not cutting the phone call")
    
    async def __check_for_backchanneling(self):
        while True:
            if self.callee_speaking and time.time() - self.callee_speaking_start_time > self.backchanneling_start_delay:
                filename = random.choice(self.filenames)
                logger.info(f"Should send a random backchanneling words and sending them {filename}")
                audio = await get_raw_audio_bytes(f"{self.backchanneling_audios}/{filename}", local= True, is_location=True)
                if not self.turn_based_conversation and self.task_config['tools_config']['output'] != "default":
                    audio = resample(audio, target_sample_rate= 8000, format="wav")
                    audio = wav_bytes_to_pcm(audio)
                await self.tools["output"].handle(create_ws_data_packet(audio, self.__get_updated_meta_info())) 
            else:
                logger.info(f"Callee isn't speaking and hence not sending or {time.time() - self.callee_speaking_start_time} is not greater than {self.backchanneling_start_delay}") 
            await asyncio.sleep(self.backchanneling_message_gap) 
    
    async def __first_message(self):
        logger.info(f"Executing the first message task")
        try:
            while True:
                if not self.stream_sid and not self.default_io:
                    stream_sid = self.tools["input"].get_stream_sid()
                    if stream_sid is not None:
                        logger.info(f"Got stream sid and hence sending the first message {stream_sid}")
                        self.stream_sid = stream_sid
                        text = self.kwargs.get('agent_welcome_message', None)
                        logger.info(f"Generating {text}")
                        meta_info = {'io': self.tools["output"].get_provider(), 'message_category': 'agent_welcome_message', 'stream_sid': stream_sid, "request_id": str(uuid.uuid4()), "cached": True, "sequence_id": -1, 'format': self.task_config["tools_config"]["output"]["format"], 'text': text}
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
            logger.error(f"Error happeneed {e}")

    async def __start_transmitting_ambient_noise(self):
        try:
            audio = await get_raw_audio_bytes(f'{os.getenv("AMBIENT_NOISE_PRESETS_DIR")}/{self.soundtrack}', local= True, is_location=True)
            audio = resample(audio, self.sampling_rate, format = "wav")
            if self.task_config["tools_config"]["output"]["provider"] in SUPPORTED_OUTPUT_TELEPHONY_HANDLERS.keys():
                audio = wav_bytes_to_pcm(audio)
            logger.info(f"Length of audio {len(audio)} {self.sampling_rate}")
            if self.should_record:
                meta_info={'io': 'default', 'message_category': 'ambient_noise', "request_id": str(uuid.uuid4()), "sequence_id": -1, "type":'audio', 'format': 'wav'}
            else:

                meta_info={'io': self.tools["output"].get_provider(), 'message_category': 'ambient_noise', 'stream_sid': self.stream_sid , "request_id": str(uuid.uuid4()), "cached": True, "type":'audio', "sequence_id": -1, 'format': 'pcm'}
            while True:
                logger.info(f"Before yielding ambient noise")
                for chunk in yield_chunks_from_memory(audio, self.output_chunk_size*2 ):
                    if not self.started_transmitting_audio:
                        logger.info(f"Transmitting ambient noise {len(chunk)}")
                        await self.tools["output"].handle(create_ws_data_packet(chunk, meta_info=meta_info))
                    logger.info("Sleeping for 800 ms")
                    await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Something went wrong while transmitting noise {e}")

    async def run(self):
        try:
            if self._is_conversation_task():
                # Create transcriber and synthesizer tasks
                logger.info("starting task_id {}".format(self.task_id))                
                tasks = [asyncio.create_task(self.tools['input'].handle())]
                if not self.turn_based_conversation:
                    self.background_check_task = asyncio.create_task(self.__handle_initial_silence(duration=15))
                if "transcriber" in self.tools:
                    tasks.append(asyncio.create_task(self._listen_transcriber()))
                    self.transcriber_task = asyncio.create_task(self.tools["transcriber"].run())

                if self.turn_based_conversation and self._is_conversation_task():
                    logger.info(
                        "Since it's connected through dashboard, I'll run listen_llm_tas too in case user wants to simply text")
                    self.llm_queue_task = asyncio.create_task(self._listen_llm_input_queue())
                
                if "synthesizer" in self.tools and self._is_conversation_task():
                    logger.info("Starting synthesizer task")
                    try:
                        self.synthesizer_task = asyncio.create_task(self.__listen_synthesizer())
                    except asyncio.CancelledError as e:
                        logger.error(f'Synth task got cancelled {e}')
                        traceback.print_exc()
                    
                logger.info(f"Starting the first message task {self.enforce_streaming}")
                self.output_task = asyncio.create_task(self.__process_output_loop())
                if not self.turn_based_conversation or self.enforce_streaming:
                    logger.info(f"Setting up other servers")
                    self.first_message_task = asyncio.create_task(self.__first_message())
                    #if not self.use_llm_to_determine_hangup :
                    # By default we will hang up after x amount of silence
                    # We still need to
                    self.hangup_task = asyncio.create_task(self.__check_for_completion())
                    
                    if self.should_backchannel:
                        self.backchanneling_task = asyncio.create_task(self.__check_for_backchanneling())
                    if self.ambient_noise:
                        logger.info(f"Transmitting ambient noise")
                        self.ambient_noise_task = asyncio.create_task(self.__start_transmitting_ambient_noise())
                try:
                    await asyncio.gather(*tasks)
                except asyncio.CancelledError as e:
                    logger.error(f'task got cancelled {e}')
                    traceback.print_exc()
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Error: {e}")

                logger.info("Conversation completed")
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
            await self.handle_cancellation(f"Exception occurred {e}")
            raise Exception(e)

        finally:
            # Construct output
            tasks_to_cancel = []
            if "synthesizer" in self.tools and self.synthesizer_task is not None:   
                tasks_to_cancel.append(process_task_cancellation(self.synthesizer_task, 'synthesizer_task'))
                tasks_to_cancel.append(process_task_cancellation(self.synthesizer_monitor_task, 'synthesizer_monitor_task'))

            if self._is_conversation_task():
                output = {"messages": self.history, "conversation_time": time.time() - self.start_time,
                    "label_flow": self.label_flow, "call_sid": self.call_sid, "stream_sid": self.stream_sid,
                    "transcriber_duration": self.transcriber_duration,
                    "synthesizer_characters": self.tools['synthesizer'].get_synthesized_characters(), "ended_by_assistant": self.ended_by_assistant,
                    "latency_dict": self.latency_dict}

                tasks_to_cancel.append(process_task_cancellation(self.output_task,'output_task'))
                #self.output_task.cancel()
                tasks_to_cancel.append(process_task_cancellation(self.hangup_task,'hangup_task'))
                tasks_to_cancel.append(process_task_cancellation(self.backchanneling_task, 'backchanneling_task'))
                tasks_to_cancel.append(process_task_cancellation(self.ambient_noise_task, 'ambient_noise_task'))
                tasks_to_cancel.append(process_task_cancellation(self.background_check_task, 'background_check_task'))
                tasks_to_cancel.append(process_task_cancellation(self.first_message_task, 'first_message_task'))

                # if self.hangup_task is not None:
                #     self.hangup_task.cancel()
                #     try:
                #         await self.hangup_task
                #     except asyncio.CancelledError:
                #         logger.info("hangup_task has been successfully cancelled")

                # if self.backchanneling_task is not None:
                #     self.backchanneling_task.cancel()
                #     try:
                #         await self.backchanneling_task
                #     except asyncio.CancelledError:
                #         logger.info("backchanneling_task has been successfully cancelled")

                # if self.ambient_noise_task is not None:
                #     self.ambient_noise_task.cancel()
                  
                # if self.background_check_task is not None:
                #     self.background_check_task.cancel()

                # if self.first_message_task is not None:
                #     self.first_message_task.cancel()
                #     try:
                #         await self.first_message_task
                #     except asyncio.CancelledError:
                #         logger.info("first_message_task has been successfully cancelled")

                if self.should_record:
                    output['recording_url'] = await save_audio_file_to_s3(self.conversation_recording, self.sampling_rate, self.assistant_id, self.run_id)

                if self.task_config['tools_config']['output']['provider'] == "daily":
                    logger.info("calling release function")
                    await self.tools['output'].release_call()

            else:
                output = self.input_parameters
                if self.task_config["task_type"] == "extraction":
                    output = { "extracted_data" : self.extracted_data, "task_type": "extraction"}
                elif self.task_config["task_type"] == "summarization":
                    logger.info(f"self.summarized_data {self.summarized_data}")
                    output = {"summary" : self.summarized_data, "task_type": "summarization"}
                elif self.task_config["task_type"] == "webhook":
                    output = {"status": self.webhook_response, "task_type": "webhook"}

            await asyncio.gather(*tasks_to_cancel)
            return output

    async def handle_cancellation(self, message):
        try:
            # Cancel all tasks on cancellation
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            #if "synthesizer" in self.tools and self.synthesizer_task:
            #    self.synthesizer_task.cancel()
            logger.info(f"tasks {len(tasks)}")
            for task in tasks:
                await process_task_cancellation(task, task.get_name())
                logger.info(f"Cancelling task {task.get_name()}")
                task.cancel()
            logger.info(message)
        except Exception as e:
            traceback.print_exc()
            logger.info(e)
