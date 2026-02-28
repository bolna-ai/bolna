from __future__ import annotations

import asyncio
import copy
import json
import os
import time
import traceback
import uuid
from typing import TYPE_CHECKING

import aiohttp

from bolna.helpers.function_calling_helpers import trigger_api, computed_api_response
from bolna.helpers.utils import (
    compute_function_pre_call_message,
    convert_to_request_log,
    create_ws_data_packet,
    format_messages,
    is_valid_md5,
)
from bolna.helpers.logger_config import configure_logger

if TYPE_CHECKING:
    from .task_manager import TaskManager

logger = configure_logger(__name__)


class LLMPipeline:
    def __init__(self, tm: "TaskManager"):
        self.tm = tm

    async def handle_output(self, next_step: str, text_chunk: str, should_bypass_synth: bool,
                            meta_info: dict, is_filler: bool = False, is_function_call: bool = False) -> None:
        if "request_id" not in meta_info:
            meta_info["request_id"] = str(uuid.uuid4())

        if not self.tm.stream and not is_filler:
            first_buffer_latency = time.time() - meta_info["llm_start_time"]
            meta_info["llm_first_buffer_generation_latency"] = first_buffer_latency

        elif is_filler:
            logger.info(f"It's a filler message and hence adding required metadata")
            meta_info['origin'] = "classifier"
            meta_info['cached'] = True
            meta_info['local'] = True
            meta_info['message_category'] = 'filler'

        if next_step == "synthesizer" and not should_bypass_synth:
            self.tm._turn_audio_flushed.clear()
            task = asyncio.create_task(self.tm._synthesize(create_ws_data_packet(text_chunk, meta_info)))
            self.tm.synthesizer_tasks.append(asyncio.ensure_future(task))
        elif self.tm.tools["output"] is not None:
            logger.info("Synthesizer not the next step and hence simply returning back")
            overall_time = time.time() - meta_info["llm_start_time"]
            if is_function_call:
                bos_packet = create_ws_data_packet("<beginning_of_stream>", meta_info)
                await self.tm.tools["output"].handle(bos_packet)
                await self.tm.tools["output"].handle(create_ws_data_packet(text_chunk, meta_info))
                eos_packet = create_ws_data_packet("<end_of_stream>", meta_info)
                await self.tm.tools["output"].handle(eos_packet)
            else:
                await self.tm.tools["output"].handle(create_ws_data_packet(text_chunk, meta_info))

    async def process_preprocessed(self, message: dict, sequence: int, meta_info: dict) -> None:
        if self.tm.task_config["tools_config"]["llm_agent"]['agent_flow_type'] == "preprocessed":
            messages = self.tm.conversation_history.get_copy()
            messages.append({'role': 'user', 'content': message['data']})
            logger.info(f"Starting LLM Agent {messages}")
            convert_to_request_log(message=format_messages(messages, use_system_prompt= True), meta_info= meta_info, component="llm", direction="request", model=self.tm.llm_agent_config["model"], is_cached= True, run_id= self.tm.run_id)
            async for next_state in self.tm.tools['llm_agent'].generate(messages, label_flow=self.tm.label_flow):
                if next_state == "<end_of_conversation>":
                    meta_info["end_of_conversation"] = True
                    self.tm.buffered_output_queue.put_nowait(create_ws_data_packet("<end_of_conversation>", meta_info))
                    return

                logger.info(f"Text chunk {next_state['text']}")
                messages.append({'role': 'assistant', 'content': next_state['text']})
                self.tm.synthesizer_tasks.append(asyncio.create_task(
                        self.tm._synthesize(create_ws_data_packet(next_state['audio'], meta_info, is_md5_hash=True))))
            logger.info(f"Interim history after the LLM task {messages}")
            self.tm.llm_response_generated = True
            self.tm.conversation_history.sync_interim(messages)

    async def process_formulaic(self, message: dict, sequence: int, meta_info: dict) -> None:
        llm_response = ""
        logger.info("Agent flow is formulaic and hence moving smoothly")
        async for text_chunk in self.tm.tools['llm_agent'].generate(self.tm.history):
            if is_valid_md5(text_chunk):
                self.tm.synthesizer_tasks.append(asyncio.create_task(
                    self.tm._synthesize(create_ws_data_packet(text_chunk, meta_info, is_md5_hash=True))))
            else:
                llm_response += " " +text_chunk
                next_step = self.tm._get_next_step(sequence, "llm")
                if next_step == "synthesizer":
                    self.tm.synthesizer_tasks.append(asyncio.create_task(self.tm._synthesize(create_ws_data_packet(text_chunk, meta_info))))
                else:
                    logger.info(f"Sending output text {sequence}")
                    await self.tm.tools["output"].handle(create_ws_data_packet(text_chunk, meta_info))
                    self.tm.synthesizer_tasks.append(asyncio.create_task(
                        self.tm._synthesize(create_ws_data_packet(text_chunk, meta_info, is_md5_hash=False))))

    async def execute_function_call(self, url, method, param, api_token, headers, model_args, meta_info, next_step, called_fun, **resp) -> None:
        self.tm.check_if_user_online = False

        if called_fun.startswith("transfer_call"):
            await asyncio.sleep(2)
            try:
                from_number = self.tm.context_data['recipient_data']['from_number']
            except Exception as e:
                from_number = None

            call_sid = None
            call_transfer_number = None
            payload = {
                'call_sid': call_sid,
                'provider': self.tm.tools['input'].io_provider,
                'stream_sid': self.tm.stream_sid,
                'from_number': from_number,
                'execution_id': self.tm.run_id,
                **(self.tm.transfer_call_params or {})
            }

            if self.tm.tools['input'].io_provider != 'default':
                call_sid = self.tm.tools["input"].get_call_sid()
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

            if self.tm.tools['input'].io_provider != 'default':
                payload['call_sid'] = self.tm.tools["input"].get_call_sid()

            if self.tm.tools['input'].io_provider == 'default':
                mock_response = f"This is a mocked response demonstrating a successful transfer of call to {call_transfer_number}"
                convert_to_request_log(str(payload), meta_info, None, "function_call", direction="request", run_id=self.tm.run_id)
                convert_to_request_log(mock_response, meta_info, None, "function_call", direction="response", run_id=self.tm.run_id)

                bos_packet = create_ws_data_packet("<beginning_of_stream>", meta_info)
                await self.tm.tools["output"].handle(bos_packet)
                await self.tm.tools["output"].handle(create_ws_data_packet(mock_response, meta_info))
                eos_packet = create_ws_data_packet("<end_of_stream>", meta_info)
                await self.tm.tools["output"].handle(eos_packet)
                return

            async with aiohttp.ClientSession() as session:
                logger.info(f"Sending the payload to stop the conversation {payload} url {url}")
                while self.tm.tools["input"].is_audio_being_played_to_user():
                    await asyncio.sleep(1)
                convert_to_request_log(str(payload), meta_info, None, "function_call", direction="request", is_cached=False,
                                       run_id=self.tm.run_id)
                async with session.post(url, json = payload) as response:
                    response_text = await response.text()
                    logger.info(f"Response from the server after call transfer: {response_text}")
                    convert_to_request_log(str(response_text), meta_info, None, "function_call", direction="response", is_cached=False, run_id=self.tm.run_id)
                    return

        await self.tm.wait_for_current_message()
        response = await trigger_api(url=url, method=method.lower(), param=param, api_token=api_token, headers_data=headers, meta_info=meta_info, run_id=self.tm.run_id, **resp)
        function_response = str(response)
        get_res_keys, get_res_values = await computed_api_response(function_response)

        # Merge API response data into context_data for routing decisions
        if self.tm._is_graph_agent():
            try:
                response_data = json.loads(function_response) if isinstance(function_response, str) else function_response
                if isinstance(response_data, dict):
                    if self.tm.context_data is None:
                        self.tm.context_data = {}
                    self.tm.context_data.update(response_data)
                    if hasattr(self.tm.tools.get('llm_agent'), 'context_data'):
                        self.tm.tools['llm_agent'].context_data.update(response_data)
                    logger.info(f"Merged API response into context_data: {list(response_data.keys())}")
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug(f"Could not parse API response as JSON for context merge: {e}")
        if called_fun.startswith('check_availability_of_slots') and (not get_res_values or (len(get_res_values) == 1 and len(get_res_values[0]) == 0)):
            set_response_prompt = []
        elif called_fun.startswith('book_appointment') and 'id' not in get_res_keys:
            if get_res_values and get_res_values[0] == 'no_available_users_found_error':
                function_response = "Sorry, the host isn't available at this time. Are you available at any other time?"
            set_response_prompt = []
        else:
            set_response_prompt = function_response

        textual_response = resp.get("textual_response", None)
        self.tm.conversation_history.append_assistant(textual_response, tool_calls=resp["model_response"])
        self.tm.conversation_history.append_tool_result(resp.get("tool_call_id", ""), function_response)

        logger.info(f"Logging function call parameters ")
        convert_to_request_log(function_response, meta_info , None, "function_call", direction = "response", is_cached= False, run_id = self.tm.run_id)

        messages = self.tm.conversation_history.get_copy()
        convert_to_request_log(format_messages(messages, True), meta_info, self.tm.llm_config['model'], "llm", direction = "request", is_cached= False, run_id = self.tm.run_id)
        self.tm.check_if_user_online = self.tm.conversation_config.get("check_if_user_online", True)

        if not called_fun.startswith("transfer_call"):
            should_bypass_synth = meta_info.get('bypass_synth', False)
            await self.do_generation(messages, meta_info, next_step, should_bypass_synth=should_bypass_synth, should_trigger_function_call=True)

        self.tm.execute_function_call_task = None

    def store_into_history(self, meta_info: dict, messages: list, llm_response: str,
                           should_trigger_function_call: bool = False) -> None:
        self.tm.llm_response_generated = True
        convert_to_request_log(message=llm_response, meta_info= meta_info, component="llm", direction="response", model=self.tm.llm_config["model"], run_id= self.tm.run_id)
        if should_trigger_function_call:
            logger.info(f"There was a function call and need to make that work")
            self.tm.conversation_history.append_assistant(llm_response)
        else:
            messages.append({"role": "assistant", "content": llm_response})
            self.tm.conversation_history.append_assistant(llm_response)
            self.tm.conversation_history.sync_interim(messages)

    async def do_generation(self, messages: list, meta_info: dict, next_step: str,
                            should_bypass_synth: bool = False, should_trigger_function_call: bool = False) -> None:
        # Reset response tracking for new turn
        if self.tm.generate_precise_transcript:
            self.tm.tools["input"].reset_response_heard_by_user()

        llm_response, function_tool, function_tool_message = '', '', ''
        synthesize = True
        if should_bypass_synth:
            synthesize = False

        # Inject language instruction if detection complete
        messages = self.tm._inject_language_instruction(messages)

        # Pass detected language to LLM for pre_call_message selection
        detected_lang = self.tm.language_detector.dominant_language
        if detected_lang:
            meta_info['detected_language'] = detected_lang

        async for llm_message in self.tm.tools['llm_agent'].generate(messages, synthesize=synthesize, meta_info=meta_info):
            if isinstance(llm_message, dict) and 'messages' in llm_message:
                convert_to_request_log(format_messages(llm_message['messages'], True), meta_info, self.tm.llm_config['model'], "llm", direction="request", is_cached=False, run_id=self.tm.run_id)
                continue

            # Handle graph agent routing info
            if isinstance(llm_message, dict) and 'routing_info' in llm_message:
                routing_info = llm_message['routing_info']

                routing_messages = routing_info.get('routing_messages')
                routing_tools = routing_info.get('routing_tools', [])
                if routing_messages:
                    tools_summary = ""
                    if routing_tools:
                        tool_lines = []
                        for t in routing_tools:
                            if 'function' in t:
                                name = t['function']['name']
                                desc = t['function'].get('description', '')
                                tool_lines.append(f"  - {name}: {desc}")
                        if tool_lines:
                            tools_summary = "\n\nAvailable transitions:\n" + "\n".join(tool_lines)

                    convert_to_request_log(
                        message=format_messages(routing_messages, use_system_prompt=True) + tools_summary,
                        meta_info=meta_info,
                        model=routing_info.get('routing_model', ''),
                        component="graph_routing",
                        direction="request",
                        run_id=self.tm.run_id
                    )

                if routing_info.get('transitioned'):
                    routing_data = f"Node: {routing_info.get('previous_node', '?')} → {routing_info['current_node']}"
                else:
                    routing_data = f"Node: {routing_info['current_node']} (no transition)"
                if routing_info.get('extracted_params'):
                    routing_data += f" | Params: {json.dumps(routing_info['extracted_params'])}"
                if routing_info.get('node_history'):
                    routing_data += f" | Flow: {' → '.join(routing_info['node_history'])}"

                meta_info['llm_metadata'] = meta_info.get('llm_metadata') or {}
                meta_info['llm_metadata']['graph_routing_info'] = routing_info

                if routing_info.get('routing_latency_ms') is not None:
                    self.tm.routing_latencies['turn_latencies'].append({
                        'latency_ms': routing_info['routing_latency_ms'],
                        'routing_model': routing_info.get('routing_model'),
                        'routing_provider': routing_info.get('routing_provider'),
                        'previous_node': routing_info.get('previous_node'),
                        'current_node': routing_info.get('current_node'),
                        'transitioned': routing_info.get('transitioned', False),
                        'sequence_id': meta_info.get('sequence_id')
                    })

                if routing_info.get('node_history'):
                    self.tm.routing_latencies['node_flow'] = list(routing_info['node_history'])

                convert_to_request_log(
                    message=routing_data,
                    meta_info=meta_info,
                    model=routing_info.get('routing_model', ''),
                    component="graph_routing",
                    direction="response",
                    run_id=self.tm.run_id
                )
                continue

            data = llm_message.data
            end_of_llm_stream = llm_message.end_of_stream
            latency = llm_message.latency
            trigger_function_call = llm_message.is_function_call
            function_tool = llm_message.function_name
            function_tool_message = llm_message.function_message

            if trigger_function_call:
                logger.info(f"Triggering function call for {data}")
                self.tm.llm_task = asyncio.create_task(self.execute_function_call(next_step = next_step, **data.model_dump()))
                return

            if latency:
                latency_dict = latency.model_dump()
                previous_latency_item = self.tm.llm_latencies['turn_latencies'][-1] if self.tm.llm_latencies['turn_latencies'] else None
                if previous_latency_item and previous_latency_item.get('sequence_id') == latency_dict.get('sequence_id'):
                    self.tm.llm_latencies['turn_latencies'][-1] = latency_dict
                else:
                    self.tm.llm_latencies['turn_latencies'].append(latency_dict)

            llm_response += " " + data

            logger.info(f"Got a response from LLM {llm_response}")
            if end_of_llm_stream:
                meta_info["end_of_llm_stream"] = True

            if self.tm.stream:
                text_chunk = self.tm._TaskManager__process_stop_words(data, meta_info)

                detected_lang = self.tm.language_detector.dominant_language or self.tm.language
                filler_message = compute_function_pre_call_message(detected_lang, function_tool, function_tool_message)
                if text_chunk == filler_message:
                    logger.info("Got a pre function call message")
                    messages.append({'role':'assistant', 'content': filler_message})
                    self.tm.conversation_history.append_assistant(filler_message)
                    self.tm.conversation_history.sync_interim(messages)

                await self.handle_output(next_step, text_chunk, should_bypass_synth, meta_info)

        detected_lang = self.tm.language_detector.dominant_language or self.tm.language
        filler_message = compute_function_pre_call_message(detected_lang, function_tool, function_tool_message)
        if self.tm.stream and llm_response != filler_message:
            self.store_into_history(meta_info, messages, llm_response, should_trigger_function_call= should_trigger_function_call)
        elif not self.tm.stream:
            llm_response = llm_response.strip()
            if self.tm.turn_based_conversation:
                self.tm.conversation_history.append_assistant(llm_response)
            await self.handle_output(next_step, llm_response, should_bypass_synth, meta_info, is_function_call=should_trigger_function_call)
            convert_to_request_log(message=llm_response, meta_info=meta_info, component="llm", direction="response", model=self.tm.llm_config["model"], run_id=self.tm.run_id)

        # Collect RAG latency if present (from KnowledgeBaseAgent)
        if meta_info.get('rag_latency'):
            rag_latency = meta_info['rag_latency']
            existing_seq_ids = [t.get('sequence_id') for t in self.tm.rag_latencies['turn_latencies']]
            if rag_latency.get('sequence_id') not in existing_seq_ids:
                self.tm.rag_latencies['turn_latencies'].append(rag_latency)

    async def process_conversation(self, message: dict, sequence: int, meta_info: dict) -> None:
        should_bypass_synth = 'bypass_synth' in meta_info and meta_info['bypass_synth'] is True
        next_step = self.tm._get_next_step(sequence, "llm")
        meta_info['llm_start_time'] = time.time()

        if self.tm.turn_based_conversation:
            self.tm.history.append({"role": "user", "content": message['data']})
        messages = copy.deepcopy(self.tm.history)

        # Request logs converted inside do_generation for knowledgebase agent
        if not self.tm._is_knowledgebase_agent() and not self.tm._is_graph_agent():
            convert_to_request_log(message=format_messages(messages, use_system_prompt=True), meta_info=meta_info, component="llm", direction="request", model=self.tm.llm_config["model"], run_id= self.tm.run_id)

        await self.do_generation(messages, meta_info, next_step, should_bypass_synth)

        # Hangup detection - now supported for all agent types including graph_agent
        if self.tm.use_llm_to_determine_hangup and not self.tm.turn_based_conversation:
            completion_res, metadata = await self.tm.tools["llm_agent"].check_for_completion(messages, self.tm.check_for_completion_prompt)

            should_hangup = (
                str(completion_res.get("hangup", "")).lower() == "yes"
                if isinstance(completion_res, dict)
                else False
            )

            # Track hangup check latency (latency returned by agent)
            self.tm.llm_latencies['other_latencies'].append({
                "type": 'hangup_check',
                "latency_ms": metadata.get("latency_ms", None),
                "model": self.tm.check_for_completion_llm,
                "provider": "openai",  # TODO: Make dynamic based on provider used
                "service_tier": metadata.get("service_tier", None),
                "llm_host": metadata.get("llm_host", None),
                "sequence_id": meta_info.get("sequence_id")
            })

            prompt = [
                {'role': 'system', 'content': self.tm.check_for_completion_prompt},
                {'role': 'user', 'content': format_messages(self.tm.history)}
            ]
            logger.info(f"##### Answer from the LLM {completion_res}")
            convert_to_request_log(message=format_messages(prompt, use_system_prompt=True), meta_info=meta_info, component="llm_hangup", direction="request", model=self.tm.check_for_completion_llm, run_id=self.tm.run_id)
            convert_to_request_log(message=completion_res, meta_info=meta_info, component="llm_hangup", direction="response", model=self.tm.check_for_completion_llm, run_id=self.tm.run_id)

            if should_hangup:
                if self.tm.hangup_triggered or self.tm.conversation_ended:
                    logger.info(f"Hangup already triggered or conversation ended, skipping duplicate hangup request")
                    return
                self.tm.hangup_detail = "llm_prompted_hangup"
                await self.tm.process_call_hangup()
                return

        self.tm.llm_processed_request_ids.add(self.tm.current_request_id)
        llm_response = ""

    async def run_task(self, message: dict) -> None:
        sequence, meta_info = self.tm._extract_sequence_and_meta(message)

        try:
            if self.tm._is_extraction_task() or self.tm._is_summarization_task():
                await self.tm._process_followup_task(message)
            elif self.tm._is_conversation_task():
                await self.process_conversation(message, sequence, meta_info)
            else:
                logger.error("unsupported task type: {}".format(self.tm.task_config["task_type"]))
            self.tm.llm_task = None
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Something went wrong in llm: {e}")
            self.tm.response_in_pipeline = False

    async def listen_input_queue(self) -> None:
        logger.info(
            f"Starting listening to LLM queue as either Connected to dashboard = {self.tm.turn_based_conversation} or  it's a textual chat agent {self.tm.textual_chat_agent}")
        while True:
            try:
                ws_data_packet = await self.tm.queues["llm"].get()
                logger.info(f"ws_data_packet {ws_data_packet}")
                meta_info = self.tm._TaskManager__get_updated_meta_info(ws_data_packet['meta_info'])
                bos_packet = create_ws_data_packet("<beginning_of_stream>", meta_info)
                await self.tm.tools["output"].handle(bos_packet)
                await self.run_task(
                    create_ws_data_packet(ws_data_packet['data'], meta_info))
                eos_packet = create_ws_data_packet("<end_of_stream>", meta_info)
                await self.tm.tools["output"].handle(eos_packet)

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Something went wrong with LLM queue {e}")
                break
