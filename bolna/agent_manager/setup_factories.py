from __future__ import annotations

import os
from typing import Any, Optional, TYPE_CHECKING

from bolna.agent_types import *
from bolna.constants import DEFAULT_LANGUAGE_CODE, LLM_DEFAULT_CONFIGS
from bolna.enums import TaskType, TelephonyProvider
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.utils import get_required_input_types
from bolna.providers import (
    SUPPORTED_INPUT_HANDLERS,
    SUPPORTED_LLM_PROVIDERS,
    SUPPORTED_OUTPUT_HANDLERS,
    SUPPORTED_OUTPUT_TELEPHONY_HANDLERS,
    SUPPORTED_SYNTHESIZER_MODELS,
    SUPPORTED_TRANSCRIBER_MODELS,
    SUPPORTED_TRANSCRIBER_PROVIDERS,
)

if TYPE_CHECKING:
    from .task_manager import TaskManager

logger = configure_logger(__name__)

# Keys forwarded from tm.kwargs into injected agent configs
_AGENT_CONFIG_KEYS = (
    'llm_key', 'base_url', 'api_version', 'api_tools',
    'reasoning_effort', 'service_tier',
)


def _inject_kwargs_into_config(cfg: dict, kwargs: dict, extra_keys: tuple = ()) -> None:
    for key in _AGENT_CONFIG_KEYS + extra_keys:
        if key in kwargs:
            cfg[key] = kwargs[key]


def setup_input_handlers(
    tm: "TaskManager",
    turn_based_conversation: bool,
    input_queue: Any,
    should_record: bool,
) -> None:
    if tm.task_config["tools_config"]["input"]["provider"] in SUPPORTED_INPUT_HANDLERS.keys():
        input_kwargs = {
            "queues": tm.queues,
            "websocket": tm.websocket,
            "input_types": get_required_input_types(tm.task_config),
            "mark_event_meta_data": tm.mark_event_meta_data,
            "is_welcome_message_played": True if tm.task_config["tools_config"]["output"]["provider"] == 'default' and not tm.is_web_based_call else False
        }

        if should_record:
            input_kwargs['conversation_recording'] = tm.conversation_recording

        if turn_based_conversation:
            input_kwargs['turn_based_conversation'] = True
            input_handler_class = SUPPORTED_INPUT_HANDLERS.get("default")
            input_kwargs['queue'] = input_queue
        else:
            input_handler_class = SUPPORTED_INPUT_HANDLERS.get(
                tm.task_config["tools_config"]["input"]["provider"])

            if tm.task_config['tools_config']['input']['provider'] == 'default':
                input_kwargs['queue'] = input_queue

            input_kwargs["observable_variables"] = tm.observable_variables

            # Asterisk (sip-trunk): pass context data for pre-parsed MEDIA_START
            if tm.task_config["tools_config"]["input"]["provider"] == TelephonyProvider.SIP_TRUNK.value and tm.context_data:
                input_kwargs["ws_context_data"] = tm.context_data
                input_kwargs["agent_config"] = {"tasks": [tm.task_config]}
        tm.tools["input"] = input_handler_class(**input_kwargs)
    else:
        raise "Other input handlers not supported yet"


def setup_output_handlers(
    tm: "TaskManager",
    turn_based_conversation: bool,
    output_queue: Any,
) -> None:
    output_kwargs = {"websocket": tm.websocket}

    if tm.task_config["tools_config"]["output"] is None:
        logger.info("Not setting up any output handler as it is none")
    elif tm.task_config["tools_config"]["output"]["provider"] in SUPPORTED_OUTPUT_HANDLERS.keys():
        # Explicitly use default for turn based conversation as we expect to use HTTP endpoints
        if turn_based_conversation:
            logger.info("Connected through dashboard and hence using default output handler")
            output_handler_class = SUPPORTED_OUTPUT_HANDLERS.get("default")
        else:
            output_handler_class = SUPPORTED_OUTPUT_HANDLERS.get(tm.task_config["tools_config"]["output"]["provider"])

            if tm.task_config["tools_config"]["output"]["provider"] in SUPPORTED_OUTPUT_TELEPHONY_HANDLERS.keys():
                output_kwargs['mark_event_meta_data'] = tm.mark_event_meta_data
                logger.info(f"Making sure that the sampling rate for output handler is 8000")
                tm.task_config['tools_config']['synthesizer']['provider_config']['sampling_rate'] = 8000
                # sip-trunk (Asterisk) uses ulaw; other telephony use pcm (handler converts to mulaw)
                if tm.task_config["tools_config"]["output"]["provider"] == TelephonyProvider.SIP_TRUNK.value:
                    tm.task_config['tools_config']['synthesizer']['audio_format'] = 'ulaw'
                    logger.info(f"Setting synthesizer audio format to ulaw for Asterisk sip-trunk")
                    # Pass input handler to output handler so it can simulate mark events
                    input_handler = tm.tools.get("input")
                    output_kwargs['input_handler'] = input_handler
                    output_kwargs['asterisk_media_start'] = (tm.context_data or {}).get("media_start_data")
                    output_kwargs['agent_config'] = {"tasks": [tm.task_config]}
                    logger.info(f"Passing input_handler to sip-trunk output handler for mark event simulation: {input_handler is not None}")
                else:
                    tm.task_config['tools_config']['synthesizer']['audio_format'] = 'pcm'
            else:
                tm.task_config['tools_config']['synthesizer']['provider_config']['sampling_rate'] = 24000
                output_kwargs['queue'] = output_queue
            tm.sampling_rate = tm.task_config['tools_config']['synthesizer']['provider_config']['sampling_rate']

        if tm.task_config["tools_config"]["output"]["provider"] == "default":
            output_kwargs["is_web_based_call"] = tm.is_web_based_call
            output_kwargs['mark_event_meta_data'] = tm.mark_event_meta_data

        tm.tools["output"] = output_handler_class(**output_kwargs)
        tm.output_handler_set = True
        logger.info("output handler set")
    else:
        raise "Other input handlers not supported yet"


def setup_transcriber(tm: "TaskManager") -> None:
    try:
        if tm.task_config["tools_config"]["transcriber"] is not None:
            tm.language = tm.task_config["tools_config"]["transcriber"].get('language', DEFAULT_LANGUAGE_CODE)
            if tm.turn_based_conversation:
                provider = "playground"
            elif tm.is_web_based_call:
                provider = "web_based_call"
            else:
                provider = tm.task_config["tools_config"]["input"]["provider"]

            tm.task_config["tools_config"]["transcriber"]["input_queue"] = tm.audio_queue
            tm.task_config['tools_config']["transcriber"]["output_queue"] = tm.transcriber_output_queue

            # Configure encoding for Asterisk/sip-trunk (uses ulaw like Twilio)
            if provider == TelephonyProvider.SIP_TRUNK.value:
                tm.task_config["tools_config"]["transcriber"]["encoding"] = "mulaw"
                tm.task_config["tools_config"]["transcriber"]["sampling_rate"] = 8000
                logger.info(f"Configured transcriber for Asterisk sip-trunk with mulaw encoding @ 8kHz")

            # Checking models for backwards compatibility
            if tm.task_config["tools_config"]["transcriber"]["model"] in SUPPORTED_TRANSCRIBER_MODELS.keys() or tm.task_config["tools_config"]["transcriber"]["provider"] in SUPPORTED_TRANSCRIBER_PROVIDERS.keys():
                if tm.turn_based_conversation:
                    tm.task_config["tools_config"]["transcriber"]["stream"] = True if tm.enforce_streaming else False
                    logger.info(f'tm.task_config["tools_config"]["transcriber"]["stream"] {tm.task_config["tools_config"]["transcriber"]["stream"]} tm.enforce_streaming {tm.enforce_streaming}')
                if 'provider' in tm.task_config["tools_config"]["transcriber"]:
                    transcriber_class = SUPPORTED_TRANSCRIBER_PROVIDERS.get(
                        tm.task_config["tools_config"]["transcriber"]["provider"])
                else:
                    transcriber_class = SUPPORTED_TRANSCRIBER_MODELS.get(
                        tm.task_config["tools_config"]["transcriber"]["model"])
                tm.tools["transcriber"] = transcriber_class(provider, **tm.task_config["tools_config"]["transcriber"], **tm.kwargs)
    except Exception as e:
        logger.error(f"Something went wrong with starting transcriber {e}")


def setup_synthesizer(tm: "TaskManager", llm_config: Optional[dict] = None) -> None:
    if tm._is_conversation_task():
        tm.kwargs["use_turbo"] = tm.task_config["tools_config"]["transcriber"]["language"] == DEFAULT_LANGUAGE_CODE
    if tm.task_config["tools_config"]["synthesizer"] is not None:
        if "caching" in tm.task_config['tools_config']['synthesizer']:
            caching = tm.task_config["tools_config"]["synthesizer"].pop("caching")
        else:
            caching = True

        tm.synthesizer_provider = tm.task_config["tools_config"]["synthesizer"].pop("provider")
        synthesizer_class = SUPPORTED_SYNTHESIZER_MODELS.get(tm.synthesizer_provider)
        provider_config = tm.task_config["tools_config"]["synthesizer"].pop("provider_config")
        tm.synthesizer_voice = provider_config["voice"]
        if tm.turn_based_conversation:
            tm.task_config["tools_config"]["synthesizer"]["audio_format"] = "mp3"
            tm.task_config["tools_config"]["synthesizer"]["stream"] = True if tm.enforce_streaming else False

        # Configure use_mulaw for Asterisk/sip-trunk to ensure synthesizer outputs ulaw
        synthesizer_kwargs = tm.kwargs.copy()
        if tm.task_config["tools_config"]["output"]["provider"] == TelephonyProvider.SIP_TRUNK.value:
            synthesizer_kwargs['use_mulaw'] = True
            logger.info(f"[SIP-TRUNK] Configuring synthesizer with use_mulaw=True for Asterisk sip-trunk")

        tm.tools["synthesizer"] = synthesizer_class(**tm.task_config["tools_config"]["synthesizer"], **provider_config, **synthesizer_kwargs, caching=caching)
        if tm.task_config["tools_config"]["llm_agent"] is not None and llm_config is not None:
            llm_config["buffer_size"] = tm.task_config["tools_config"]["synthesizer"].get('buffer_size')


def setup_llm(tm: "TaskManager", llm_config: dict, task_id: int = 0) -> Any:
    if tm.task_config["tools_config"]["llm_agent"] is not None:
        if task_id and task_id > 0:
            tm.kwargs.pop('llm_key', None)
            tm.kwargs.pop('base_url', None)
            tm.kwargs.pop('api_version', None)

            if tm._is_summarization_task() or tm._is_extraction_task():
                llm_config['model'] = LLM_DEFAULT_CONFIGS["summarization"]["model"]
                llm_config['provider'] = LLM_DEFAULT_CONFIGS["summarization"]["provider"]

        if llm_config["provider"] in SUPPORTED_LLM_PROVIDERS.keys():
            llm_class = SUPPORTED_LLM_PROVIDERS.get(llm_config["provider"])
            llm = llm_class(language=tm.language, **llm_config, **tm.kwargs)
            return llm
        else:
            raise Exception(f'LLM {llm_config["provider"]} not supported')


def get_agent_object(
    tm: "TaskManager",
    llm: Any,
    agent_type: str,
    assistant_config: Any = None,
) -> Any:
    tm.agent_type = agent_type
    if agent_type == "simple_llm_agent":
        return StreamingContextualAgent(llm)

    if agent_type == "graph_agent":
        logger.info("Setting up graph agent with rag-proxy-server support")
        llm_config = tm.task_config["tools_config"]["llm_agent"].get("llm_config", {})
        rag_server_url = tm.kwargs.get('rag_server_url', os.getenv('RAG_SERVER_URL', 'http://localhost:8000'))
        os.environ['RAG_SERVER_URL'] = rag_server_url

        injected_cfg = dict(llm_config)
        _inject_kwargs_into_config(injected_cfg, tm.kwargs, extra_keys=('routing_reasoning_effort', 'routing_max_tokens'))
        if tm.context_data:
            injected_cfg['context_data'] = tm.context_data
        if tm.llm_config.get('use_responses_api'):
            injected_cfg['use_responses_api'] = True
        injected_cfg['buffer_size'] = tm.task_config["tools_config"]["synthesizer"].get('buffer_size')
        injected_cfg['language'] = tm.language

        llm_agent = GraphAgent(injected_cfg)
        logger.info("Graph agent created with rag-proxy-server support")
        return llm_agent

    if agent_type == "knowledgebase_agent":
        logger.info("Setting up knowledge agent with rag-proxy-server support")
        llm_config = tm.task_config["tools_config"]["llm_agent"].get("llm_config", {})
        rag_server_url = tm.kwargs.get('rag_server_url', os.getenv('RAG_SERVER_URL', 'http://localhost:8000'))
        os.environ['RAG_SERVER_URL'] = rag_server_url

        injected_cfg = dict(llm_config)
        _inject_kwargs_into_config(injected_cfg, tm.kwargs)
        if tm.llm_config.get('use_responses_api'):
            injected_cfg['use_responses_api'] = True
        injected_cfg['buffer_size'] = tm.task_config["tools_config"]["synthesizer"].get('buffer_size')
        injected_cfg['language'] = tm.language

        llm_agent = KnowledgeBaseAgent(injected_cfg)
        logger.info("Knowledge agent created with rag-proxy-server support")
        return llm_agent

    raise f"{agent_type} Agent type is not created yet"


def setup_tasks(
    tm: "TaskManager",
    llm: Any = None,
    agent_type: Optional[str] = None,
    assistant_config: Any = None,
) -> Optional[Any]:
    if tm.task_config["task_type"] == TaskType.CONVERSATION and not tm._is_multiagent():
        tm.tools["llm_agent"] = get_agent_object(tm, llm, agent_type, assistant_config)
    elif tm._is_multiagent():
        return get_agent_object(tm, llm, agent_type, assistant_config)
    elif tm.task_config["task_type"] == TaskType.EXTRACTION:
        logger.info("Setting up extraction agent")
        tm.tools["llm_agent"] = ExtractionContextualAgent(llm, prompt=tm.system_prompt)
        tm.extracted_data = None
    elif tm.task_config["task_type"] == TaskType.SUMMARIZATION:
        logger.info("Setting up summarization agent")
        tm.tools["llm_agent"] = SummarizationContextualAgent(llm, prompt=tm.system_prompt)
        tm.summarized_data = None
    logger.info("prompt and config setup completed")
