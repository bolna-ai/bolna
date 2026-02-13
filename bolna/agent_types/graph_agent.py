import asyncio
from collections import defaultdict
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
import json

from bolna.models import *
from bolna.agent_types.base_agent import BaseAgent
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.rag_service_client import RAGServiceClientSingleton
from bolna.helpers.utils import now_ms, format_messages, update_prompt_with_context, DictWithMissing
from bolna.llms import OpenAiLLM
from bolna.providers import SUPPORTED_LLM_PROVIDERS
from bolna.prompts import VOICEMAIL_DETECTION_PROMPT

from typing import List, Tuple, AsyncGenerator, Optional, Dict, Any

# Optional Groq support for fast routing
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

load_dotenv()
logger = configure_logger(__name__)

class GraphAgent(BaseAgent):
    def __init__(self, config: GraphAgentConfig):
        super().__init__()
        self.config = config
        self.agent_information = self.config.get('agent_information')
        self.current_node_id = self.config.get('current_node_id')
        self.context_data = self.config.get('context_data') or {}
        self.llm_model = self.config.get('model')

        # Get credentials from config (injected by task_manager) or fall back to env vars
        self.llm_key = self.config.get('llm_key') or os.getenv('OPENAI_API_KEY')
        self.base_url = self.config.get('base_url')

        # Initialize OpenAI client with credentials (supports EU routing)
        if self.base_url:
            self.openai = OpenAI(api_key=self.llm_key, base_url=self.base_url)
            logger.info(f"OpenAI client initialized with custom base_url: {self.base_url}")
        else:
            self.openai = OpenAI(api_key=self.llm_key)

        self.node_history = [self.current_node_id]
        self.current_node_entry_index = 0  # Track message index when we entered current node
        self.rag_configs = self.initialize_rag_configs()
        self.rag_server_url = os.getenv('RAG_SERVER_URL', 'http://localhost:8000')

        # Cache transition tools per node for faster routing (bounded to prevent unbounded growth)
        self._transition_tools_cache: Dict[str, List[dict]] = {}
        self._transition_tools_cache_max_size = 100

        # Initialize routing client (Groq for speed, or fallback to OpenAI)
        self.routing_provider = self.config.get('routing_provider')
        self.routing_model = self.config.get('routing_model')
        self.routing_instructions = self.config.get('routing_instructions')  # Custom routing instructions
        self.routing_reasoning_effort = self.config.get('routing_reasoning_effort')
        self.routing_max_tokens = self.config.get('routing_max_tokens')
        logger.info(f"GraphAgent routing_instructions loaded: {bool(self.routing_instructions)} (length: {len(self.routing_instructions) if self.routing_instructions else 0})")
        self._init_routing_client()

        # Initialize main LLM for response generation (supports api_tools/function calling + real streaming)
        self.llm = self._initialize_llm()

        # Initialize LLMs for hangup and voicemail detection
        llm_kwargs = {}
        if self.llm_key:
            llm_kwargs['llm_key'] = self.llm_key
        if self.base_url:
            llm_kwargs['base_url'] = self.base_url
        self.conversation_completion_llm = OpenAiLLM(model=os.getenv('CHECK_FOR_COMPLETION_LLM', self.llm_model or 'gpt-4o-mini'), **llm_kwargs)
        self.voicemail_llm = OpenAiLLM(model=os.getenv('VOICEMAIL_DETECTION_LLM', 'gpt-4.1-mini'), **llm_kwargs)

    def _initialize_llm(self):
        """Initialize LLM with api_tools support (same pattern as KnowledgeBaseAgent)."""
        try:
            provider = self.config.get('provider') or self.config.get('llm_provider', 'openai')
            if provider not in SUPPORTED_LLM_PROVIDERS:
                logger.warning(f"Unknown provider: {provider}, using openai")
                provider = 'openai'

            llm_kwargs = {
                'model': self.llm_model,
                'temperature': self.config.get('temperature', 0.7),
                'max_tokens': self.config.get('max_tokens', 150),
                'provider': provider,
            }

            for key in ['llm_key', 'base_url', 'api_version', 'language', 'api_tools', 'buffer_size', 'reasoning_effort', 'service_tier']:
                if self.config.get(key, None):
                    llm_kwargs[key] = self.config[key]

            llm_class = SUPPORTED_LLM_PROVIDERS[provider]
            return llm_class(**llm_kwargs)
        except Exception as e:
            logger.error(f"Failed to create LLM: {e}, falling back to default OpenAiLLM")
            return OpenAiLLM(model=self.llm_model or 'gpt-4o-mini', llm_key=self.llm_key or os.getenv('OPENAI_API_KEY'))

    def initialize_rag_configs(self) -> Dict[str, Dict]:
        """Initialize RAG configurations for each node."""
        rag_configs = {}
        for node in self.config.get('nodes', []):
            rag_config = node.get('rag_config')
            if rag_config:
                # Extract collection/vector IDs
                collections = []
                # Legacy: direct vector_id on rag_config
                if 'vector_id' in rag_config:
                    collections.append(rag_config['vector_id'])
                # Legacy: rag_config.provider_config.vector_id
                elif 'provider_config' in rag_config and isinstance(rag_config.get('provider_config'), dict) and 'vector_id' in rag_config['provider_config']:
                    collections.append(rag_config['provider_config']['vector_id'])
                # New: rag_config.vector_store.provider_config.vector_id
                else:
                    try:
                        vector_store = rag_config.get('vector_store') or {}
                        provider_config = vector_store.get('provider_config') or {}
                        vs_vector_id = provider_config.get('vector_id')
                        if vs_vector_id:
                            collections.append(vs_vector_id)
                    except Exception:
                        pass
                
                rag_configs[node['id']] = {
                    'collections': collections,
                    'similarity_top_k': rag_config.get('similarity_top_k', 10),
                    'temperature': rag_config.get('temperature', 0.7),
                    'model': rag_config.get('model', 'gpt-4o'),
                    'max_tokens': rag_config.get('max_tokens', 150)
                }
                
                logger.info(f"Initialized RAG config for node {node['id']} with collections: {collections}")

        return rag_configs

    def _init_routing_client(self):
        """Initialize routing client. Uses Groq if available, else OpenAI."""
        groq_available = GROQ_AVAILABLE and os.getenv('GROQ_API_KEY')

        # Auto-detect provider if not specified
        if not self.routing_provider:
            self.routing_provider = 'groq' if groq_available else 'openai'

        if self.routing_provider == 'groq':
            if groq_available:
                self.routing_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
                # Default to llama-3.3-70b-versatile (best for multilingual routing)
                if not self.routing_model:
                    self.routing_model = os.getenv('DEFAULT_ROUTING_MODEL_GROQ', 'llama-3.3-70b-versatile')
                logger.info(f"Routing initialized with Groq ({self.routing_model}) - fast mode ~200ms")
            else:
                logger.warning("Groq requested but GROQ_API_KEY not set or groq package not installed, falling back to OpenAI")
                self.routing_client = self.openai
                self.routing_provider = 'openai'
                self.routing_model = os.getenv('DEFAULT_ROUTING_MODEL_OPENAI', 'gpt-4.1-mini')
        else:
            self.routing_client = self.openai
            if not self.routing_model:
                self.routing_model = os.getenv('DEFAULT_ROUTING_MODEL_OPENAI', 'gpt-4.1-mini')
            logger.info(f"Routing initialized with OpenAI ({self.routing_model})")

    async def check_for_completion(self, messages, check_for_completion_prompt):
        """Check if the conversation should end. Returns (hangup_dict, metadata)."""
        try:
            prompt = [
                {'role': 'system', 'content': check_for_completion_prompt},
                {'role': 'user', 'content': format_messages(messages)}
            ]

            start_time = time.time()
            response, metadata = await self.conversation_completion_llm.generate(prompt, request_json=True, ret_metadata=True)
            latency_ms = (time.time() - start_time) * 1000

            hangup = json.loads(response)
            metadata['latency_ms'] = latency_ms

            return hangup, metadata
        except Exception as e:
            logger.error(f'check_for_completion exception: {str(e)}')
            return {'hangup': 'No'}, {}

    async def check_for_voicemail(self, user_message, voicemail_detection_prompt=None):
        """Check if message indicates a voicemail system. Returns (result_dict, metadata)."""
        try:
            detection_prompt = voicemail_detection_prompt or VOICEMAIL_DETECTION_PROMPT
            prompt = [
                {'role': 'system', 'content': detection_prompt + """
                    Respond only in this JSON format:
                    {
                      "is_voicemail": "Yes" or "No"
                    }
                """},
                {'role': 'user', 'content': f"User message: {user_message}"}
            ]

            start_time = time.time()
            response, metadata = await self.voicemail_llm.generate(prompt, request_json=True, ret_metadata=True)
            latency_ms = (time.time() - start_time) * 1000

            result = json.loads(response)
            metadata['latency_ms'] = latency_ms
            return result, metadata
        except Exception as e:
            logger.error(f'check_for_voicemail exception: {str(e)}')
            return {'is_voicemail': 'No'}, {}

    def _build_transition_tools(self, node: dict) -> List[dict]:
        """Build and cache function/tool definitions for node edges."""
        node_id = node.get('id')
        if node_id and node_id in self._transition_tools_cache:
            return self._transition_tools_cache[node_id]

        tools = []
        for edge in node.get('edges', []):
            to_node_id = edge.get('to_node_id')
            func_name = edge.get('function_name') or f"transition_to_{to_node_id}"
            func_description = edge.get('function_description') or f"Call this function when: {edge.get('condition', '')}"

            parameters = {"type": "object", "properties": {}, "required": []}
            if edge.get('parameters'):
                for param_name, param_type in edge['parameters'].items():
                    parameters["properties"][param_name] = {"type": param_type, "description": f"The {param_name} provided by the user"}
                    parameters["required"].append(param_name)

            tools.append({
                "type": "function",
                "function": {"name": func_name, "description": func_description, "parameters": parameters}
            })

        tools.append({
            "type": "function",
            "function": {"name": "stay_on_current_node", "description": "No transition matches. Need more info or clarification.", "parameters": {"type": "object", "properties": {}, "required": []}}
        })

        if node_id:
            if len(self._transition_tools_cache) >= self._transition_tools_cache_max_size:
                # Evict oldest entry
                oldest_key = next(iter(self._transition_tools_cache))
                del self._transition_tools_cache[oldest_key]
            self._transition_tools_cache[node_id] = tools
        return tools

    def _get_edge_by_function_name(self, node: dict, function_name: str) -> Optional[dict]:
        """Find the edge that corresponds to a function name."""
        for edge in node.get('edges', []):
            expected_name = edge.get('function_name') or f"transition_to_{edge['to_node_id']}"
            if expected_name == function_name:
                return edge
        return None

    async def decide_next_node_with_functions(self, history: List[dict]) -> Tuple[Optional[str], Optional[Dict[str, Any]], float, Optional[List[dict]], Optional[List[dict]]]:
        """Decide next node using LLM function calling.

        Returns (next_node_id, extracted_params, latency_ms, routing_messages, routing_tools).
        """
        start_time = time.perf_counter()

        current_node = self.get_node_by_id(self.current_node_id)
        if not current_node:
            logger.error(f"Current node '{self.current_node_id}' not found")
            return None, None, 0, None, None

        edges = current_node.get('edges', [])
        if not edges:
            logger.debug(f"Node '{self.current_node_id}' has no edges, staying on current node")
            return None, None, 0, None, None

        tools = self._build_transition_tools(current_node)

        # Build compact context for routing
        context_section = ""
        if self.context_data:
            context_items = [f"{k}={v}" for k, v in self.context_data.items()
                          if v is not None and not isinstance(v, dict) and k != 'detected_language']
            if context_items:
                context_section = f"\nContext: {', '.join(context_items)}"

        default_instructions = "Call the transition function matching user intent, or stay_on_current_node if unclear."
        instructions = self.routing_instructions or default_instructions

        if self.context_data and instructions:
            try:
                substitution_data = dict(self.context_data)
                if 'recipient_data' in self.context_data and isinstance(self.context_data['recipient_data'], dict):
                    substitution_data.update(self.context_data['recipient_data'])
                instructions = instructions.format_map(defaultdict(lambda: 'NULL', substitution_data))
            except Exception as e:
                logger.debug(f"Variable substitution in routing_instructions failed: {e}")

        node_objective = current_node.get('prompt', '')
        system_prompt = f"""Routing Guidelines: \n {instructions}\n Current Node: {current_node['id']}{context_section} \n Node Objective: {node_objective}\n\n Node Conversation History:\n"""

        logger.debug(f"Routing system prompt:\n{system_prompt}")
        messages = [{"role": "system", "content": system_prompt}]
        node_history = history[self.current_node_entry_index:] if self.current_node_entry_index < len(history) else history
        has_tool_context = any(msg.get("role") == "assistant" and msg.get("tool_calls") for msg in node_history)

        if has_tool_context:
            for msg in node_history:
                role = msg.get("role")
                if role == "assistant":
                    if msg.get("tool_calls"):
                        messages.append({"role": "assistant", "content": None, "tool_calls": msg["tool_calls"]})
                    elif msg.get("content"):
                        messages.append({"role": "assistant", "content": msg["content"]})
                elif role == "tool":
                    content = msg.get("content", "")
                    messages.append({"role": "tool", "tool_call_id": msg.get("tool_call_id", ""), "content": content})
                elif role == "user" and msg.get("content"):
                    messages.append({"role": "user", "content": msg["content"]})
        else:
            for msg in node_history:
                role = msg.get("role")
                content = msg.get("content")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        if len(messages) == 1:
            user_message = history[-1].get("content", "") if history else ""
            if user_message:
                messages.append({"role": "user", "content": user_message})

        try:
            routing_kwargs = {
                "model": self.routing_model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "required",
                "parallel_tool_calls": False,
            }

            if self.routing_model and self.routing_model.startswith("gpt-5"):
                routing_kwargs["max_completion_tokens"] = self.routing_max_tokens or 150
                routing_kwargs["reasoning_effort"] = self.routing_reasoning_effort or os.getenv('GPT5_ROUTING_REASONING_EFFORT', 'minimal')
            else:
                routing_kwargs["max_tokens"] = self.routing_max_tokens or 50
                routing_kwargs["temperature"] = 0.0

            response = await asyncio.to_thread(self.routing_client.chat.completions.create, **routing_kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract the function call
            message = response.choices[0].message
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

                logger.info(f"Routing decision: {function_name} (latency: {latency_ms:.1f}ms)")

                if function_name == "stay_on_current_node":
                    return None, None, latency_ms, messages, tools

                # Find the edge for this function
                edge = self._get_edge_by_function_name(current_node, function_name)
                if edge:
                    return edge['to_node_id'], function_args, latency_ms, messages, tools
                else:
                    logger.warning(f"Function {function_name} not found in edges")
                    return None, None, latency_ms, messages, tools
            else:
                logger.warning("No tool call in response")
                return None, None, latency_ms, messages, tools

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Routing error: {e} (latency: {latency_ms:.1f}ms)")
            return None, None, latency_ms, messages, tools

    def get_node_by_id(self, node_id: str) -> Optional[dict]:
        return next((node for node in self.config.get('nodes', []) if node['id'] == node_id), None)

    def _get_prompt_with_example(self, node: dict, detected_lang: str) -> str:
        """Get node prompt with language-specific example appended."""
        prompt = node.get('prompt', '')
        examples = node.get('examples', {})

        if not examples:
            return prompt

        if detected_lang and detected_lang in examples:
            return f"{prompt}\n\nLANGUAGE GUIDELINES\n\nPlease make sure to generate replies in the {detected_lang} language only. You can refer to the example given below to generate a reply in the given language. Example response: \"{examples[detected_lang]}\""

        # Language not yet detected — include all examples
        example_lines = [f"  {lang.upper()}: \"{text}\"" for lang, text in examples.items()]
        return f"{prompt}\n\nExample responses:\n" + "\n".join(example_lines)

    def _get_tool_choice_for_node(self):
        """Check if current node should force a tool call.

        Returns:
        - {"type": "function", "function": {"name": "..."}} if force_function_call is a function name string
        - "required" if force_function_call is True (LLM must call some function)
        - None otherwise (defaults to "auto")
        """
        if not self.llm or not getattr(self.llm, 'trigger_function_call', False):
            return None

        current_node = self.get_node_by_id(self.current_node_id)
        if not current_node:
            return None

        fn = current_node.get('function_call')
        if fn:
            logger.info(f"Node '{self.current_node_id}' forcing specific function: {fn}")
            return {"type": "function", "function": {"name": fn}}

        return None

    async def _build_messages(self, history: List[dict]) -> List[dict]:
        """Build messages array: system prompt (+ optional RAG) + conversation history."""
        current_node = self.get_node_by_id(self.current_node_id)
        if not current_node:
            raise ValueError("Current node not found.")

        detected_lang = self.context_data.get('detected_language')  # None if not yet detected
        node_prompt = self._get_prompt_with_example(current_node, detected_lang)

        if self.context_data:
            node_prompt = update_prompt_with_context(node_prompt, self.context_data)

        if self.agent_information:
            agent_info = self.agent_information
            if self.context_data:
                agent_info = update_prompt_with_context(agent_info, self.context_data)
            prompt = f"{agent_info}\n\n{node_prompt}"
        else:
            prompt = node_prompt

        # Try RAG if configured for this node
        rag_config = self.rag_configs.get(self.current_node_id)
        if rag_config and rag_config.get('collections'):
            try:
                client = await RAGServiceClientSingleton.get_client(self.rag_server_url)
                latest_message = history[-1]["content"] if history else ""
                rag_response = await client.query_for_conversation(
                    query=latest_message,
                    collections=rag_config['collections'],
                    max_results=rag_config.get('similarity_top_k', 10),
                    similarity_threshold=0.0
                )
                if rag_response.contexts:
                    rag_context = await client.format_context_for_prompt(rag_response.contexts)
                    prompt = f"{prompt}\n\nKnowledge base:\n{rag_context}\n\nUse this information naturally."
            except Exception as e:
                logger.error(f"RAG error for node {self.current_node_id}: {e}")

        max_history = 50
        history_subset = history[-max_history:] if len(history) > max_history else history

        # Pass conversation history as-is to preserve tool_calls/tool_call_id fields
        conversation = [msg for msg in history_subset if msg.get("role") != "system"]
        return [{"role": "system", "content": prompt}] + conversation

    async def generate(self, message: List[dict], **kwargs) -> AsyncGenerator:
        meta_info = kwargs.get('meta_info', {})
        synthesize = kwargs.get('synthesize', True)
        start_time = now_ms()

        detected_language = meta_info.get('detected_language')  # None if not yet detected
        if detected_language:
            self.context_data['detected_language'] = detected_language

        try:
            previous_node = self.current_node_id
            next_node_id, extracted_params, routing_latency_ms, routing_messages, routing_tools = await self.decide_next_node_with_functions(message)

            if next_node_id:
                logger.info(f"Transitioning: {self.current_node_id} -> {next_node_id} (params: {extracted_params})")
                self.current_node_id = next_node_id
                self.current_node_entry_index = len(message)
                if extracted_params:
                    self.context_data.update(extracted_params)

            if next_node_id and (not self.node_history or self.node_history[-1] != self.current_node_id):
                self.node_history.append(self.current_node_id)

            yield {
                'routing_info': {
                    'previous_node': previous_node,
                    'current_node': self.current_node_id,
                    'transitioned': next_node_id is not None,
                    'routing_model': self.routing_model,
                    'routing_provider': getattr(self, 'routing_provider', None),
                    'routing_latency_ms': round(routing_latency_ms, 1),
                    'extracted_params': extracted_params or {},
                    'node_history': list(self.node_history),
                    'routing_messages': routing_messages,
                    'routing_tools': routing_tools,
                }
            }

            messages = await self._build_messages(message)
            yield {'messages': messages}
            tool_choice = self._get_tool_choice_for_node()
            async for chunk in self.llm.generate_stream(messages, synthesize=synthesize, meta_info=meta_info, tool_choice=tool_choice):
                yield chunk

        except Exception as e:
            logger.error(f"Error in generate: {e}")
            latency_data = {
                "sequence_id": meta_info.get("sequence_id") if meta_info else None,
                "first_token_latency_ms": 0,
                "total_stream_duration_ms": now_ms() - start_time
            }
            yield f"An error occurred: {str(e)}", True, latency_data, False, None, None
