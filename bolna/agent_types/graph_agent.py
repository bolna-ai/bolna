import asyncio
from collections import defaultdict
import os
import re
import time
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv
import json

from bolna.models import *
from bolna.agent_types.base_agent import BaseAgent
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.rag_service_client import RAGServiceClientSingleton
from bolna.helpers.utils import (
    now_ms,
    format_messages,
    update_prompt_with_context,
    enrich_context_with_time_variables,
    DictWithMissing,
    get_md5_hash,
)
from bolna.helpers.expression_evaluator import evaluate_edge_expression, describe_edge_expression
from bolna.enums import EdgeConditionType, NodeType, ToolScope
from bolna.llms.types import LLMStreamChunk, LatencyData
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

_DETERMINISTIC_REASONING_PREFIX = "deterministic:"
_ROUTER_REASONING_PREFIX = f"{_DETERMINISTIC_REASONING_PREFIX}router:"
_PROMPT_VAR_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

# Time variables frozen per call for the conversation prompt; see _prompt_context.
_TIME_VAR_KEYS = (
    "current_date",
    "current_time",
    "current_hour",
    "current_minute",
    "current_weekday",
    "current_day",
    "current_month",
    "current_year",
)


class GraphAgent(BaseAgent):
    def __init__(self, config: GraphAgentConfig):
        super().__init__()
        self.config = config
        self.agent_information = self.config.get("agent_information")
        self.current_node_id = self.config.get("current_node_id")
        self.context_data = self.config.get("context_data") or {}
        self.variable_types = self.config.get("variable_types") or {}
        self.llm_model = self.config.get("model")

        # Get credentials from config (injected by task_manager) or fall back to env vars
        self.llm_key = self.config.get("llm_key") or os.getenv("OPENAI_API_KEY")
        self.base_url = self.config.get("base_url")

        # Initialize OpenAI client with credentials (supports EU routing)
        if self.base_url:
            self.openai = OpenAI(api_key=self.llm_key, base_url=self.base_url)
            logger.info(f"OpenAI client initialized with custom base_url: {self.base_url}")
        else:
            self.openai = OpenAI(api_key=self.llm_key)

        self.node_history = [self.current_node_id]
        self.current_node_entry_index = 0
        self._silence_repeats = 0
        self._event_triggered_generation = False
        self._last_deterministic_eval = None
        self._frozen_time_vars: Optional[Dict[str, Any]] = None
        self.rag_configs = self.initialize_rag_configs()
        self.global_rag_config = self._initialize_global_rag_config()
        self.rag_server_url = os.getenv("RAG_SERVER_URL", "http://localhost:8000")

        # Cache transition tools per node for faster routing (bounded to prevent unbounded growth)
        self._transition_tools_cache: Dict[str, List[dict]] = {}
        self._transition_tools_cache_max_size = 100

        # Initialize routing client (Groq for speed, or fallback to OpenAI)
        self.routing_provider = self.config.get("routing_provider")
        self.routing_model = self.config.get("routing_model")
        self.routing_instructions = self.config.get("routing_instructions")  # Custom routing instructions
        self.routing_reasoning_effort = self.config.get("routing_reasoning_effort")
        self.routing_max_tokens = self.config.get("routing_max_tokens")
        self.service_tier = self.config.get("service_tier")
        logger.info(
            f"GraphAgent routing_instructions loaded: {bool(self.routing_instructions)} (length: {len(self.routing_instructions) if self.routing_instructions else 0})"
        )
        self._init_routing_client()

        # Initialize main LLM for response generation (supports api_tools/function calling + real streaming)
        self.llm = self._initialize_llm()

        # Initialize LLMs for hangup and voicemail detection
        llm_kwargs = {}
        if self.llm_key:
            llm_kwargs["llm_key"] = self.llm_key
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        self.conversation_completion_llm = OpenAiLLM(
            model=os.getenv("CHECK_FOR_COMPLETION_LLM", self.llm_model or "gpt-4o-mini"), **llm_kwargs
        )
        self.voicemail_llm = OpenAiLLM(model=os.getenv("VOICEMAIL_DETECTION_LLM", "gpt-4.1-mini"), **llm_kwargs)

    def _initialize_llm(self):
        """Initialize LLM with api_tools support (same pattern as KnowledgeBaseAgent)."""
        try:
            provider = self.config.get("provider") or self.config.get("llm_provider", "openai")
            if provider not in SUPPORTED_LLM_PROVIDERS:
                logger.warning(f"Unknown provider: {provider}, using openai")
                provider = "openai"

            llm_kwargs = {
                "model": self.llm_model,
                "temperature": self.config.get("temperature", 0.7),
                "max_tokens": self.config.get("max_tokens", 150),
                "provider": provider,
            }

            for key in [
                "llm_key",
                "base_url",
                "api_version",
                "language",
                "api_tools",
                "buffer_size",
                "reasoning_effort",
                "service_tier",
                "use_responses_api",
                "compact_threshold",
            ]:
                if self.config.get(key, None):
                    llm_kwargs[key] = self.config[key]

            llm_class = SUPPORTED_LLM_PROVIDERS[provider]
            return llm_class(**llm_kwargs)
        except Exception as e:
            logger.error(f"Failed to create LLM: {e}, falling back to default OpenAiLLM")
            return OpenAiLLM(model=self.llm_model or "gpt-4o-mini", llm_key=self.llm_key or os.getenv("OPENAI_API_KEY"))

    @staticmethod
    def _extract_rag_collections(rag_config: Dict) -> List[str]:
        """Extract collection/vector IDs from a rag_config dict, supporting all known formats."""
        collections = []
        legacy_pc = rag_config.get("provider_config") if isinstance(rag_config.get("provider_config"), dict) else {}
        provider_config = (rag_config.get("vector_store") or {}).get("provider_config") or {}
        # Prefer the first location that actually carries a vector id; a present-but-empty
        # field (e.g. top-level "vector_ids": []) must not shadow a valid vector_store.
        if isinstance(rag_config.get("vector_ids"), list) and rag_config["vector_ids"]:
            collections.extend(rag_config["vector_ids"])
        elif rag_config.get("vector_id"):
            collections.append(rag_config["vector_id"])
        elif legacy_pc.get("vector_id"):
            collections.append(legacy_pc["vector_id"])
        elif isinstance(provider_config.get("vector_ids"), list) and provider_config["vector_ids"]:
            collections.extend(provider_config["vector_ids"])
        elif provider_config.get("vector_id"):
            collections.append(provider_config["vector_id"])
        return [c for c in collections if c]

    @staticmethod
    def _extract_similarity_top_k(rag_config: Dict, default: int = 10) -> int:
        provider_config = (rag_config.get("vector_store") or {}).get("provider_config") or {}
        return rag_config.get("similarity_top_k") or provider_config.get("similarity_top_k") or default

    def initialize_rag_configs(self) -> Dict[str, Dict]:
        """Initialize RAG configurations for each node."""
        rag_configs = {}
        for node in self.config.get("nodes", []):
            rag_config = node.get("rag_config")
            if not rag_config:
                continue

            collections = self._extract_rag_collections(rag_config)
            # Nodes without resolvable collections fall back to the global rag_config
            if not collections:
                continue

            rag_configs[node["id"]] = {
                "collections": collections,
                "similarity_top_k": self._extract_similarity_top_k(rag_config),
            }

            logger.info(f"Initialized RAG config for node {node['id']} with collections: {collections}")

        return rag_configs

    def _initialize_global_rag_config(self) -> Dict:
        """Initialize the agent-level RAG config (same shape as KnowledgeBaseAgent's rag_config)."""
        rag_config = self.config.get("rag_config")
        if not rag_config:
            return {}

        collections = []
        used_sources = rag_config.get("used_sources")
        if used_sources:
            collections = [
                source["vector_id"] for source in used_sources if isinstance(source, dict) and source.get("vector_id")
            ]
        if not collections:
            collections = self._extract_rag_collections(rag_config)

        if not collections:
            logger.warning("Global rag_config present but no collections resolved")
            return {}

        logger.info(f"Initialized global RAG config with collections: {collections}")
        return {
            "collections": collections,
            "similarity_top_k": self._extract_similarity_top_k(rag_config),
            "used_sources": used_sources or [],
        }

    def _init_routing_client(self):
        """Initialize routing client. Uses Groq if available, else OpenAI."""
        groq_available = GROQ_AVAILABLE and os.getenv("GROQ_API_KEY")

        # Auto-detect provider if not specified
        if not self.routing_provider:
            self.routing_provider = "groq" if groq_available else "openai"

        if self.routing_provider == "groq":
            if groq_available:
                self.routing_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                # Default to llama-3.3-70b-versatile (best for multilingual routing)
                if not self.routing_model:
                    self.routing_model = os.getenv("DEFAULT_ROUTING_MODEL_GROQ", "llama-3.3-70b-versatile")
                logger.info(f"Routing initialized with Groq ({self.routing_model}) - fast mode ~200ms")
            else:
                logger.warning(
                    "Groq requested but GROQ_API_KEY not set or groq package not installed, falling back to OpenAI"
                )
                self.routing_client = self.openai
                self.routing_provider = "openai"
                self.routing_model = os.getenv("DEFAULT_ROUTING_MODEL_OPENAI", "gpt-4.1-mini")
        elif self.routing_provider == "azure":
            azure_endpoint = self.base_url or os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = self.config.get("api_version") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            self.routing_client = AzureOpenAI(
                azure_endpoint=azure_endpoint, api_key=self.llm_key, api_version=api_version
            )
            if self.routing_model:
                self.routing_model = self.routing_model.split("/", 1)[-1]
            else:
                self.routing_model = os.getenv("DEFAULT_ROUTING_MODEL_AZURE", "gpt-4.1-mini")
            logger.info(f"Routing initialized with Azure ({self.routing_model})")
        else:
            self.routing_client = self.openai
            if not self.routing_model:
                self.routing_model = os.getenv("DEFAULT_ROUTING_MODEL_OPENAI", "gpt-4.1-mini")
            logger.info(f"Routing initialized with OpenAI ({self.routing_model})")

    async def check_for_completion(self, messages, check_for_completion_prompt, meta_info=None):
        """Check if the conversation should end. Returns (hangup_dict, metadata)."""
        try:
            prompt = [
                {"role": "system", "content": check_for_completion_prompt},
                {"role": "user", "content": format_messages(messages)},
            ]

            start_time = time.time()
            response, metadata = await self.conversation_completion_llm.generate(
                prompt, request_json=True, ret_metadata=True, meta_info=meta_info
            )
            latency_ms = (time.time() - start_time) * 1000

            hangup = json.loads(response)
            metadata["latency_ms"] = latency_ms

            return hangup, metadata
        except Exception as e:
            logger.error(f"check_for_completion exception: {str(e)}")
            return {"hangup": "No"}, {}

    async def check_for_voicemail(self, user_message, voicemail_detection_prompt=None):
        """Check if message indicates a voicemail system. Returns (result_dict, metadata)."""
        try:
            detection_prompt = voicemail_detection_prompt or VOICEMAIL_DETECTION_PROMPT
            prompt = [
                {
                    "role": "system",
                    "content": detection_prompt
                    + """
                    Respond only in this JSON format:
                    {
                      "is_voicemail": "Yes" or "No"
                    }
                """,
                },
                {"role": "user", "content": f"User message: {user_message}"},
            ]

            start_time = time.time()
            response, metadata = await self.voicemail_llm.generate(prompt, request_json=True, ret_metadata=True)
            latency_ms = (time.time() - start_time) * 1000

            result = json.loads(response)
            metadata["latency_ms"] = latency_ms
            return result, metadata
        except Exception as e:
            logger.error(f"check_for_voicemail exception: {str(e)}")
            return {"is_voicemail": "No"}, {}

    @staticmethod
    def _edge_function_name(edge: dict) -> str:
        return edge.get("function_name") or f"transition_to_{edge['to_node_id']}"

    def _build_transition_tools_for_edges(self, edges: list) -> list:
        """Build function/tool definitions for a list of edges."""
        tools = []
        for edge in edges:
            func_name = self._edge_function_name(edge)
            func_description = (
                edge.get("function_description") or f"Call this function when: {edge.get('condition', '')}"
            )

            parameters = {"type": "object", "properties": {}, "required": []}
            if edge.get("parameters"):
                for param_name, param_type in edge["parameters"].items():
                    parameters["properties"][param_name] = {
                        "type": param_type,
                        "description": f"The {param_name} provided by the user",
                    }
                    parameters["required"].append(param_name)

            parameters["properties"]["reasoning"] = {
                "type": "string",
                "description": "Brief explanation of why this routing decision was made",
            }
            parameters["properties"]["confidence"] = {
                "type": "number",
                "description": "Confidence score from 0.0 to 1.0 for this routing decision",
            }
            parameters["required"].extend(["reasoning", "confidence"])

            tools.append(
                {
                    "type": "function",
                    "function": {"name": func_name, "description": func_description, "parameters": parameters},
                }
            )

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "stay_on_current_node",
                    "description": "No transition matches. Need more info or clarification.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of why this routing decision was made",
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence score from 0.0 to 1.0 for this routing decision",
                            },
                        },
                        "required": ["reasoning", "confidence"],
                    },
                },
            }
        )
        return tools

    def _build_transition_tools(self, node: dict) -> List[dict]:
        """Build and cache function/tool definitions for all node edges."""
        node_id = node.get("id")
        if node_id and node_id in self._transition_tools_cache:
            return self._transition_tools_cache[node_id]

        tools = self._build_transition_tools_for_edges(node.get("edges", []))

        if node_id:
            if len(self._transition_tools_cache) >= self._transition_tools_cache_max_size:
                oldest_key = next(iter(self._transition_tools_cache))
                del self._transition_tools_cache[oldest_key]
            self._transition_tools_cache[node_id] = tools
        return tools

    def _get_edge_by_function_name_from_edges(self, edges: list, function_name: str) -> Optional[dict]:
        for edge in edges:
            if self._edge_function_name(edge) == function_name:
                return edge
        return None

    def _get_edge_by_function_name(self, node: dict, function_name: str) -> Optional[dict]:
        return self._get_edge_by_function_name_from_edges(node.get("edges", []), function_name)

    def _classify_edges(self, edges: list) -> tuple:
        """Split edges into (deterministic_edges, llm_edges), sorted by priority.
        Event edges are excluded — they only fire via process_event()."""
        deterministic = []
        llm = []
        for edge in edges:
            ct = edge.get("condition_type")
            if ct == EdgeConditionType.EVENT:
                continue  # event edges only fire via process_event()
            elif ct in (EdgeConditionType.EXPRESSION, EdgeConditionType.UNCONDITIONAL):
                deterministic.append(edge)
            else:
                llm.append(edge)

        deterministic.sort(key=lambda e: e["priority"] if e.get("priority") is not None else 0)
        llm.sort(key=lambda e: e["priority"] if e.get("priority") is not None else 100)
        return deterministic, llm

    def _evaluate_deterministic_edges(self, edges: list) -> Tuple[Optional[dict], List[str]]:
        """Return (first matching deterministic edge or None, per-edge evaluation traces)."""
        evaluations = []
        for edge in edges:
            is_match = evaluate_edge_expression(edge, self.context_data, self.variable_types)
            evaluations.append(
                f"-> {edge.get('to_node_id')}: {describe_edge_expression(edge, self.context_data, self.variable_types)} | matched={is_match}"
            )
            if is_match:
                return edge, evaluations
        return None, evaluations

    def process_event(self, event: dict) -> dict:
        """Process an external event. Merges properties into context_data
        and checks current node's event edges for a matching transition.

        Returns dict with: matched, event, new_node_id, node_type, etc.
        """
        parsed = CallEvent(**event) if not isinstance(event, CallEvent) else event
        event_name = parsed.event
        properties = parsed.properties or {}

        # Always merge properties into context_data
        if properties:
            self.context_data.update(properties)
        self.context_data["_last_event"] = event_name

        current_node = self.get_node_by_id(self.current_node_id)
        if not current_node:
            logger.warning(f"process_event: current node '{self.current_node_id}' not found")
            return {"matched": False, "event": event_name}

        # Collect event edges, sorted by priority
        event_edges = [e for e in current_node.get("edges", []) if e.get("condition_type") == EdgeConditionType.EVENT]
        event_edges.sort(key=lambda e: e.get("priority") or 0)

        for edge in event_edges:
            if edge.get("event_name") == event_name:
                previous_node = self.current_node_id
                self.current_node_id = edge["to_node_id"]
                self.current_node_entry_index = 0  # caller should set to len(history)
                self._silence_repeats = 0

                if self.current_node_id not in self.node_history or self.node_history[-1] != self.current_node_id:
                    self.node_history.append(self.current_node_id)

                target_node = self.get_node_by_id(edge["to_node_id"])
                node_type = self._node_type_of(target_node)

                logger.info(f"Event '{event_name}' matched edge: {previous_node} -> {self.current_node_id}")
                return {
                    "matched": True,
                    "event": event_name,
                    "previous_node": previous_node,
                    "new_node_id": edge["to_node_id"],
                    "node_type": node_type,
                    "target_node": target_node,
                }

        logger.info(
            f"Event '{event_name}' did not match any event edge on node '{self.current_node_id}' — context updated silently"
        )
        return {"matched": False, "event": event_name}

    def _compute_turn_counts(self, history: list) -> tuple:
        """Count (node_turns, total_turns) from history. A node just entered this
        turn (entry_index == len(history), as during a router hop) has 0 node turns."""
        total_turns = sum(1 for msg in history if msg.get("role") == "user")
        node_history = history[self.current_node_entry_index :]
        node_turns = sum(1 for msg in node_history if msg.get("role") == "user")
        return node_turns, total_turns

    def _enrich_routing_context(self, history: list) -> None:
        """Refresh time variables and turn counts in context_data for expression evaluation."""
        recipient_data = self.context_data.get("recipient_data")
        timezone_str = recipient_data.get("timezone") if isinstance(recipient_data, dict) else None
        if timezone_str:
            enrich_context_with_time_variables(self.context_data, timezone_str)

        node_turns, total_turns = self._compute_turn_counts(history)
        self.context_data["_node_turns"] = node_turns
        self.context_data["_total_turns"] = total_turns
        self.context_data["_silence_repeats"] = self._silence_repeats

    @staticmethod
    def _node_type_of(node: Optional[dict]) -> str:
        return node.get("node_type", NodeType.LLM) if node else NodeType.LLM

    def _match_expression_edge(self, node: dict) -> Tuple[Optional[dict], str]:
        """First matching expression edge in priority order, with the evaluation trace."""
        deterministic_edges, _ = self._classify_edges(node.get("edges", []))
        expression_edges = [e for e in deterministic_edges if e.get("condition_type") == EdgeConditionType.EXPRESSION]
        matched_edge, evaluations = self._evaluate_deterministic_edges(expression_edges)
        return matched_edge, "; ".join(evaluations) or "no expression edge matched"

    @staticmethod
    def _catch_all_edge(node: dict) -> Optional[dict]:
        """Lowest-priority unconditional edge, matching the priority precedence used everywhere else."""
        unconditional = [e for e in node.get("edges", []) if e.get("condition_type") == EdgeConditionType.UNCONDITIONAL]
        if not unconditional:
            return None
        return min(unconditional, key=lambda e: e["priority"] if e.get("priority") is not None else 0)

    def _router_hop_info(
        self,
        previous_node: str,
        *,
        routing_type: str,
        latency_ms: float,
        reasoning: Optional[str],
        confidence: Optional[float],
        is_silence_trigger: bool = False,
        extracted_params: Optional[dict] = None,
        routing_messages: Optional[List[dict]] = None,
        routing_tools: Optional[List[dict]] = None,
        routing_expression: Optional[str] = None,
        routing_usage: Optional[dict] = None,
    ) -> dict:
        # A routing-LLM call happened whenever it produced messages, even on a hop that
        # then fell back to the catch-all — so its model/usage stay attributed and counted.
        made_llm_call = routing_messages is not None
        return {
            "previous_node": previous_node,
            "current_node": self.current_node_id,
            "transitioned": True,
            "routing_type": routing_type,
            "routing_model": self.routing_model if made_llm_call else None,
            "routing_provider": self.routing_provider if made_llm_call else None,
            "routing_latency_ms": round(latency_ms, 1),
            "extracted_params": extracted_params or {},
            "node_history": list(self.node_history),
            "routing_messages": routing_messages,
            "routing_tools": routing_tools,
            "reasoning": reasoning,
            "routing_expression": routing_expression,
            "confidence": confidence,
            "routing_usage": routing_usage,
            "node_type": self._node_type_of(self.get_node_by_id(self.current_node_id)),
            "is_silence_trigger": is_silence_trigger,
        }

    async def _resolve_router_chain(self, history: list) -> List[dict]:
        """Hop silently through router nodes until a speaking node is reached, one
        routing_info per hop. Each hop: expression edges first (priority order), then
        intent edges via one routing-LLM call, then the unconditional catch-all. The
        LLM is called at most once per chain so a chain never stacks routing latency;
        the visited-set bounds the hops, so the chain always terminates."""
        hops = []
        visited = set()
        intent_call_spent = False
        is_silence_trigger = bool(history and history[-1].get("content", "").startswith("[silence]"))

        while self._node_type_of(self.get_node_by_id(self.current_node_id)) == NodeType.ROUTER:
            if self.current_node_id in visited:
                logger.error(
                    f"Router cycle detected at '{self.current_node_id}', stopping chain. "
                    f"Flow: {' -> '.join(self.node_history)}"
                )
                break
            visited.add(self.current_node_id)

            hop_start = time.perf_counter()
            self._enrich_routing_context(history)
            router_node = self.get_node_by_id(self.current_node_id)
            previous_node = self.current_node_id

            edge, eval_trace = self._match_expression_edge(router_node)

            # Telemetry from an intent call that returned no match, carried onto the
            # catch-all hop so its tokens are still counted and the call is still logged.
            spent_messages = spent_tools = spent_usage = None

            if edge is None and not intent_call_spent:
                _, intent_edges = self._classify_edges(router_node.get("edges", []))
                if intent_edges:
                    intent_call_spent = True
                    (
                        next_node_id,
                        extracted_params,
                        latency_ms,
                        routing_messages,
                        routing_tools,
                        reasoning,
                        confidence,
                        routing_usage,
                    ) = await self._decide_next_node_llm(router_node, intent_edges, history, hop_start)
                    if next_node_id:
                        self._advance_to_node(next_node_id, entry_index=len(history))
                        if extracted_params:
                            self.context_data.update(extracted_params)
                        logger.info(
                            f"Router dispatch (intent) on node '{previous_node}': -> {self.current_node_id} "
                            f"| {reasoning} (latency: {latency_ms:.1f}ms)"
                        )
                        hops.append(
                            self._router_hop_info(
                                previous_node,
                                routing_type="llm",
                                latency_ms=latency_ms,
                                reasoning=reasoning,
                                confidence=confidence,
                                is_silence_trigger=is_silence_trigger,
                                extracted_params=extracted_params,
                                routing_messages=routing_messages,
                                routing_tools=routing_tools,
                                routing_usage=routing_usage,
                            )
                        )
                        continue
                    spent_messages, spent_tools, spent_usage = routing_messages, routing_tools, routing_usage
                    eval_trace = f"{eval_trace}; intent: no match"

            if edge is None:
                edge = self._catch_all_edge(router_node)
            if edge is None:
                logger.error(
                    f"Router node '{self.current_node_id}' has no matching edge and no catch-all, "
                    f"stopping chain. Evaluations: {eval_trace}"
                )
                break

            self._advance_to_node(edge["to_node_id"], entry_index=len(history))
            latency_ms = (time.perf_counter() - hop_start) * 1000
            condition = edge.get("condition") or edge.get("condition_type", "router")

            logger.info(
                f"Router dispatch on node '{previous_node}': -> {self.current_node_id} "
                f"| {eval_trace} (latency: {latency_ms:.1f}ms)"
            )
            hops.append(
                self._router_hop_info(
                    previous_node,
                    routing_type="deterministic",
                    latency_ms=latency_ms,
                    reasoning=f"{_ROUTER_REASONING_PREFIX}{condition}",
                    confidence=1.0,
                    is_silence_trigger=is_silence_trigger,
                    routing_expression=eval_trace,
                    routing_messages=spent_messages,
                    routing_tools=spent_tools,
                    routing_usage=spent_usage,
                )
            )

        return hops

    def _advance_to_node(self, node_id: str, entry_index: int) -> None:
        self.current_node_id = node_id
        self.current_node_entry_index = entry_index
        self._silence_repeats = 0
        if not self.node_history or self.node_history[-1] != self.current_node_id:
            self.node_history.append(self.current_node_id)

    def _end_turn_chunk(self, meta_info: Optional[dict], start_time: float) -> LLMStreamChunk:
        """Terminal empty end-of-stream chunk for a turn that produces no speech (a router
        that could not resolve — only reachable via an invalid config that skipped
        validation). Closes the stream on the end_of_llm_stream contract; the turn stays
        silent and the next user turn resumes normally."""
        return LLMStreamChunk(
            data="",
            end_of_stream=True,
            latency=LatencyData(
                sequence_id=meta_info.get("sequence_id") if meta_info else None,
                first_token_latency_ms=0,
                total_stream_duration_ms=now_ms() - start_time,
            ),
        )

    async def _decide_next_node_llm(
        self, node: dict, llm_edges: list, history: List[dict], start_time: float
    ) -> Tuple[
        Optional[str],
        Optional[Dict[str, Any]],
        float,
        Optional[List[dict]],
        Optional[List[dict]],
        Optional[str],
        Optional[float],
    ]:
        """LLM-based routing. Only called when no deterministic edge matched."""
        tools = self._build_transition_tools_for_edges(llm_edges)

        # Build compact context for routing
        context_section = ""
        if self.context_data:
            context_items = [
                f"{k}={v}"
                for k, v in self.context_data.items()
                if v is not None and not isinstance(v, dict) and k != "detected_language"
            ]
            if context_items:
                context_section = f"\nContext: {', '.join(context_items)}"

        default_instructions = "Call the transition function matching user intent, or stay_on_current_node if unclear."
        instructions = self.routing_instructions or default_instructions

        if self.context_data and instructions:
            try:
                substitution_data = dict(self.context_data)
                if "recipient_data" in self.context_data and isinstance(self.context_data["recipient_data"], dict):
                    substitution_data.update(self.context_data["recipient_data"])
                instructions = instructions.format_map(defaultdict(lambda: "NULL", substitution_data))
            except Exception as e:
                logger.debug(f"Variable substitution in routing_instructions failed: {e}")

        node_objective = node.get("prompt") or node.get("description") or ""
        system_prompt = f"""Routing Guidelines: \n {instructions}\n Current Node: {node["id"]}{context_section} \n Node Objective: {node_objective}\n\n Node Conversation History:\n"""

        logger.debug(f"Routing system prompt:\n{system_prompt}")
        messages = [{"role": "system", "content": system_prompt}]
        node_history = (
            history[self.current_node_entry_index :] if self.current_node_entry_index < len(history) else history
        )
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
                routing_kwargs["reasoning_effort"] = self.routing_reasoning_effort or os.getenv(
                    "GPT5_ROUTING_REASONING_EFFORT", "minimal"
                )
            else:
                routing_kwargs["max_tokens"] = self.routing_max_tokens or 250
                routing_kwargs["temperature"] = 0.0

            if self.routing_provider in ("openai", "azure") and self.service_tier:
                routing_kwargs["service_tier"] = self.service_tier

            self._routing_reasoning_effort_used = routing_kwargs.get("reasoning_effort")

            response = await asyncio.to_thread(self.routing_client.chat.completions.create, **routing_kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract token usage from routing LLM call
            usage_info = None
            if response.usage:
                usage_info = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "reasoning_tokens": response.usage.completion_tokens_details.reasoning_tokens
                    if response.usage.completion_tokens_details
                    else None,
                    "cached_tokens": response.usage.prompt_tokens_details.cached_tokens
                    if response.usage.prompt_tokens_details
                    else None,
                    "service_tier": getattr(response, "service_tier", None),
                }

            # Extract the function call
            message = response.choices[0].message
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

                # Pop reasoning and confidence before they pollute extracted_params/context_data
                reasoning = function_args.pop("reasoning", None)
                confidence = function_args.pop("confidence", None)

                logger.info(
                    f"Routing decision (LLM): {function_name} | confidence: {confidence} | reasoning: {reasoning} (latency: {latency_ms:.1f}ms)"
                )

                if function_name == "stay_on_current_node":
                    return None, None, latency_ms, messages, tools, reasoning, confidence, usage_info

                # Find the edge for this function
                edge = self._get_edge_by_function_name_from_edges(llm_edges, function_name)
                if edge:
                    return (
                        edge["to_node_id"],
                        function_args,
                        latency_ms,
                        messages,
                        tools,
                        reasoning,
                        confidence,
                        usage_info,
                    )
                else:
                    logger.warning(f"Function {function_name} not found in edges")
                    return None, None, latency_ms, messages, tools, reasoning, confidence, usage_info
            else:
                logger.warning("No tool call in response")
                return None, None, latency_ms, messages, tools, None, None, usage_info

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Routing error: {e} (latency: {latency_ms:.1f}ms)")
            return None, None, latency_ms, messages, tools, None, None, None

    async def decide_next_node_with_functions(
        self, history: List[dict]
    ) -> Tuple[
        Optional[str],
        Optional[Dict[str, Any]],
        float,
        Optional[List[dict]],
        Optional[List[dict]],
        Optional[str],
        Optional[float],
        Optional[dict],
    ]:
        """Two-phase routing: deterministic expressions first, then LLM fallback."""
        start_time = time.perf_counter()
        self._last_deterministic_eval = None

        current_node = self.get_node_by_id(self.current_node_id)
        if not current_node:
            logger.error(f"Current node '{self.current_node_id}' not found")
            return None, None, 0, None, None, None, None, None

        edges = current_node.get("edges", [])
        if not edges:
            logger.debug(f"Node '{self.current_node_id}' has no edges, staying on current node")
            return None, None, 0, None, None, None, None, None

        # Inject time variables and turn counts for expression evaluation
        self._enrich_routing_context(history)

        deterministic_edges, llm_edges = self._classify_edges(edges)

        # Phase 1: deterministic (0ms)
        if deterministic_edges:
            matched_edge, det_evaluations = self._evaluate_deterministic_edges(deterministic_edges)
            self._last_deterministic_eval = "; ".join(det_evaluations)
            if matched_edge:
                latency_ms = (time.perf_counter() - start_time) * 1000
                ct = matched_edge.get("condition_type", EdgeConditionType.EXPRESSION)
                reasoning = f"{_DETERMINISTIC_REASONING_PREFIX}{ct}:{matched_edge.get('condition', ct)}"
                logger.info(
                    f"Routing decision (deterministic) on node '{self.current_node_id}': "
                    f"-> {matched_edge['to_node_id']} | {self._last_deterministic_eval} (latency: {latency_ms:.1f}ms)"
                )
                return matched_edge["to_node_id"], None, latency_ms, None, None, reasoning, 1.0, None

            logger.info(
                f"No deterministic edge matched on node '{self.current_node_id}' "
                f"({'falling back to LLM routing' if llm_edges else 'staying on node'}): "
                f"{self._last_deterministic_eval}"
            )

        # Phase 2: LLM
        if llm_edges:
            return await self._decide_next_node_llm(current_node, llm_edges, history, start_time)

        return None, None, 0, None, None, None, None, None

    def get_node_by_id(self, node_id: str) -> Optional[dict]:
        return next((node for node in self.config.get("nodes", []) if node["id"] == node_id), None)

    def _get_prompt_with_example(self, node: dict, detected_lang: str) -> str:
        """Get node prompt with language-specific example appended."""
        prompt = node.get("prompt", "")
        examples = node.get("examples", {})

        if not examples:
            return prompt

        if detected_lang and detected_lang in examples:
            return f'{prompt}\n\nLANGUAGE GUIDELINES\n\nPlease make sure to generate replies in the {detected_lang} language only. You can refer to the example given below to generate a reply in the given language. Example response: "{examples[detected_lang]}"'

        # Language not yet detected — include all examples
        example_lines = [f'  {lang.upper()}: "{text}"' for lang, text in examples.items()]
        return f"{prompt}\n\nExample responses:\n" + "\n".join(example_lines)

    def _get_tool_choice_for_node(self, history: Optional[List[dict]] = None):
        """Return forced tool_choice for the current node, or None if not forced.

        Drops the force when required prompt vars aren't in recipient_data,
        or when the tool has already been called this node visit.
        """
        if not self.llm or not getattr(self.llm, "trigger_function_call", False):
            return None

        current_node = self.get_node_by_id(self.current_node_id)
        if not current_node:
            return None

        fn = current_node.get("function_call")
        if not fn:
            return None

        missing = self._missing_forced_function_vars(current_node, fn)
        if missing:
            logger.warning(
                f"Dropping forced function call '{fn}' on node '{self.current_node_id}': "
                f"recipient_data missing required vars referenced in prompt: {missing}"
            )
            return None

        if history is not None and self._forced_function_already_called(fn, history):
            logger.info(
                f"Dropping forced function call '{fn}' on node '{self.current_node_id}': "
                f"tool already invoked this node visit, letting LLM speak from the result"
            )
            return None

        logger.info(f"Node '{self.current_node_id}' forcing specific function: {fn}")
        return {"type": "function", "function": {"name": fn}}

    def _tools_for_node(self, node: Optional[dict], forced_name: Optional[str] = None) -> Optional[List[dict]]:
        """Tools visible on this node (global + node-scoped + forced), or None to use the full set.

        forced_name is the tool the resolved tool_choice forces this turn (or None); a forced tool
        must stay visible for the tool_choice to be valid, but only when the force actually survives.
        """
        if not self.llm or not getattr(self.llm, "trigger_function_call", False):
            return None
        raw = getattr(self.llm, "tools", None)
        if not raw:
            return None
        full = json.loads(raw) if isinstance(raw, str) else raw
        if not full:
            return None

        api_params = getattr(self.llm, "api_params", {}) or {}
        node_id = node.get("id") if node else None
        forced = forced_name

        subset = []
        for tool in full:
            name = tool.get("function", {}).get("name")
            params = api_params.get(name, {}) or {}
            if name == forced:
                subset.append(tool)  # must stay visible so the forced tool_choice is valid
            elif params.get("scope") == ToolScope.NODE.value:
                if node_id is not None and node_id in (params.get("nodes") or []):
                    subset.append(tool)
            else:
                subset.append(tool)

        if len(subset) == len(full):
            return None  # nothing filtered -> let generate_stream use self.tools
        return subset

    def _missing_forced_function_vars(self, node: dict, fn: str) -> List[str]:
        tools = getattr(self.llm, "tools", None) or []
        if isinstance(tools, str):
            tools = json.loads(tools)
        spec = next(
            (t["function"] for t in tools if t.get("function", {}).get("name") == fn),
            None,
        )
        required = set((spec or {}).get("parameters", {}).get("required") or [])
        if not required:
            return []

        prompt_vars = set(_PROMPT_VAR_PATTERN.findall(node.get("prompt", "") or ""))
        referenced = prompt_vars & required
        recipient_data = self.context_data.get("recipient_data") or {}
        return sorted(v for v in referenced if not recipient_data.get(v))

    def _forced_function_already_called(self, fn: str, history: List[dict]) -> bool:
        node_history = history[self.current_node_entry_index :] if self.current_node_entry_index < len(history) else []
        call_ids_for_fn = {
            tc.get("id")
            for msg in node_history
            if msg.get("role") == "assistant" and msg.get("tool_calls")
            for tc in msg["tool_calls"]
            if tc.get("function", {}).get("name") == fn
        }
        if not call_ids_for_fn:
            return False
        return any(msg.get("role") == "tool" and msg.get("tool_call_id") in call_ids_for_fn for msg in node_history)

    def _prompt_context(self) -> Optional[dict]:
        """Context for prompt substitution with time frozen at call start.

        Routing re-enriches time live each turn for expression edges; the prompt
        freezes it so its text stays identical across turns and the prompt cache
        keeps hitting, matching normal agents which render time once at setup.
        """
        if not self.context_data or not isinstance(self.context_data.get("recipient_data"), dict):
            return self.context_data
        recipient = self.context_data["recipient_data"]
        if self._frozen_time_vars is None:
            timezone_str = recipient.get("timezone")
            if timezone_str:
                enrich_context_with_time_variables(self.context_data, timezone_str)
            self._frozen_time_vars = {k: recipient[k] for k in _TIME_VAR_KEYS if k in recipient}
        if not self._frozen_time_vars:
            return self.context_data
        return {**self.context_data, "recipient_data": {**recipient, **self._frozen_time_vars}}

    async def _build_messages(self, history: List[dict], meta_info: Optional[dict] = None) -> List[dict]:
        """Build messages array: system prompt + conversation history (+ optional trailing RAG)."""
        current_node = self.get_node_by_id(self.current_node_id)
        if not current_node:
            raise ValueError("Current node not found.")

        detected_lang = self.context_data.get("detected_language")  # None if not yet detected
        node_prompt = self._get_prompt_with_example(current_node, detected_lang)

        prompt_context = self._prompt_context()
        if prompt_context:
            node_prompt = update_prompt_with_context(node_prompt, prompt_context)

        if self.agent_information:
            agent_info = self.agent_information
            if prompt_context:
                agent_info = update_prompt_with_context(agent_info, prompt_context)
            prompt = f"{agent_info}\n\n{node_prompt}"
        else:
            prompt = node_prompt

        # RAG depends on the latest message, so it goes in a trailing message rather
        # than the system prompt, keeping [system + history] a cacheable prefix.
        rag_message = None
        rag_config = self.rag_configs.get(self.current_node_id) or self.global_rag_config
        if rag_config and rag_config.get("collections"):
            try:
                client = await RAGServiceClientSingleton.get_client(self.rag_server_url)
                latest_message = history[-1]["content"] if history else ""
                rag_response = await client.query_for_conversation(
                    query=latest_message,
                    collections=rag_config["collections"],
                    max_results=rag_config.get("similarity_top_k", 10),
                    similarity_threshold=0.0,
                )
                if meta_info is not None:
                    meta_info["rag_latency"] = {
                        "sequence_id": meta_info.get("sequence_id"),
                        "total_query_time_ms": rag_response.total_query_time_ms,
                        "server_processing_time_ms": rag_response.server_processing_time_ms,
                        "collections_count": len(rag_config["collections"]),
                        "results_count": rag_response.total_results,
                    }
                if rag_response.contexts:
                    rag_context = await client.format_context_for_prompt(rag_response.contexts)
                    rag_message = {
                        "role": "system",
                        "content": f"Knowledge base for the latest user message:\n{rag_context}\n\nUse this information naturally.",
                    }
            except Exception as e:
                logger.error(f"RAG error for node {self.current_node_id}: {e}")

        max_history = 50
        history_subset = history[-max_history:] if len(history) > max_history else history

        # Pass conversation history as-is to preserve tool_calls/tool_call_id fields
        conversation = [msg for msg in history_subset if msg.get("role") != "system"]
        messages = [{"role": "system", "content": prompt}] + conversation
        if rag_message:
            messages.append(rag_message)
        return messages

    async def generate(self, message: List[dict], **kwargs) -> AsyncGenerator:
        meta_info = kwargs.get("meta_info", {})
        synthesize = kwargs.get("synthesize", True)
        start_time = now_ms()

        detected_language = meta_info.get("detected_language")  # None if not yet detected
        if detected_language:
            self.context_data["detected_language"] = detected_language

        try:
            # Event-triggered generation: process_event() already handled routing
            is_event = self._event_triggered_generation
            if is_event:
                self._event_triggered_generation = False
                current_node = self.get_node_by_id(self.current_node_id)
                node_type = self._node_type_of(current_node)

                yield {
                    "routing_info": {
                        "previous_node": self.context_data.get("_event_previous_node", self.current_node_id),
                        "current_node": self.current_node_id,
                        "transitioned": True,
                        "routing_type": "event",
                        "routing_model": None,
                        "routing_provider": None,
                        "routing_latency_ms": 0,
                        "extracted_params": {},
                        "node_history": list(self.node_history),
                        "routing_messages": None,
                        "routing_tools": None,
                        "reasoning": f"event:{self.context_data.get('_last_event', '')}",
                        "confidence": 1.0,
                        "node_type": node_type,
                        "is_silence_trigger": False,
                        "event_triggered": True,
                    }
                }

                if node_type == NodeType.ROUTER:
                    for hop in await self._resolve_router_chain(message):
                        yield {"routing_info": hop}
                    current_node = self.get_node_by_id(self.current_node_id)
                    node_type = self._node_type_of(current_node)
                    if node_type == NodeType.ROUTER:
                        logger.error(f"Router '{self.current_node_id}' did not resolve to a speaking node")
                        yield self._end_turn_chunk(meta_info, start_time)
                        return

                if node_type == NodeType.STATIC:
                    static_text = current_node.get("static_message", "") if current_node else ""
                    if static_text:
                        if self.context_data:
                            static_text = update_prompt_with_context(static_text, self.context_data)
                        yield {
                            "static_message": static_text,
                            "static_audio_hash": get_md5_hash(static_text),
                        }
                    return

                messages = await self._build_messages(message, meta_info=meta_info)
                # Inject ephemeral event hint (NOT persisted in conversation_history)
                event_name = self.context_data.get("_last_event", "")
                messages.append(
                    {
                        "role": "system",
                        "content": f"[Event: {event_name}. Respond proactively — speak first, do not wait for the user.]",
                    }
                )
                yield {"messages": messages}
                tool_choice = self._get_tool_choice_for_node(history=message)
                forced_name = tool_choice["function"]["name"] if tool_choice else None
                node_tools = self._tools_for_node(current_node, forced_name)
                async for chunk in self.llm.generate_stream(
                    messages, synthesize=synthesize, meta_info=meta_info, tool_choice=tool_choice, tools=node_tools
                ):
                    yield chunk
                return

            is_silence_trigger = bool(message and message[-1].get("content", "").startswith("[silence]"))
            if is_silence_trigger:
                self._silence_repeats += 1

            # Entry-node dispatch: the turn begins on a router start node. Resolve it
            # and let the resolved node speak this turn — do NOT re-route it through
            # decide_next (that would stack another routing call and could transition it
            # away before it speaks, unlike a router reached mid-turn by a transition).
            if self._node_type_of(self.get_node_by_id(self.current_node_id)) == NodeType.ROUTER:
                for hop in await self._resolve_router_chain(message):
                    yield {"routing_info": hop}
            else:
                previous_node = self.current_node_id
                (
                    next_node_id,
                    extracted_params,
                    routing_latency_ms,
                    routing_messages,
                    routing_tools,
                    reasoning,
                    confidence,
                    routing_usage,
                ) = await self.decide_next_node_with_functions(message)

                if next_node_id:
                    logger.info(f"Transitioning: {self.current_node_id} -> {next_node_id} (params: {extracted_params})")
                    self._advance_to_node(next_node_id, entry_index=len(message))
                    if extracted_params:
                        self.context_data.update(extracted_params)

                routing_type = (
                    "deterministic" if (reasoning and reasoning.startswith(_DETERMINISTIC_REASONING_PREFIX)) else "llm"
                )
                node_type = self._node_type_of(self.get_node_by_id(self.current_node_id))

                yield {
                    "routing_info": {
                        "previous_node": previous_node,
                        "current_node": self.current_node_id,
                        "transitioned": next_node_id is not None,
                        "routing_type": routing_type,
                        "routing_model": self.routing_model,
                        "routing_provider": getattr(self, "routing_provider", None),
                        "routing_latency_ms": round(routing_latency_ms, 1),
                        "routing_reasoning_effort": getattr(self, "_routing_reasoning_effort_used", None),
                        "extracted_params": extracted_params or {},
                        "node_history": list(self.node_history),
                        "routing_messages": routing_messages,
                        "routing_tools": routing_tools,
                        "reasoning": reasoning,
                        "routing_expression": self._last_deterministic_eval,
                        "confidence": confidence,
                        "routing_usage": routing_usage,
                        "node_type": node_type,
                        "is_silence_trigger": is_silence_trigger,
                    }
                }

                # Silent deterministic dispatch: if the transition landed on a router,
                # resolve the chain in this turn until a speaking node is reached.
                if node_type == NodeType.ROUTER:
                    for hop in await self._resolve_router_chain(message):
                        yield {"routing_info": hop}

            # A router that could not resolve (invalid config that bypassed validation)
            # ends the turn cleanly rather than speaking from an empty node prompt.
            current_node = self.get_node_by_id(self.current_node_id)
            node_type = self._node_type_of(current_node)
            if node_type == NodeType.ROUTER:
                logger.error(f"Router '{self.current_node_id}' did not resolve to a speaking node")
                yield self._end_turn_chunk(meta_info, start_time)
                return

            if node_type == NodeType.STATIC:
                static_text = current_node.get("static_message", "") if current_node else ""
                if static_text:
                    if self.context_data:
                        static_text = update_prompt_with_context(static_text, self.context_data)
                    yield {
                        "static_message": static_text,
                        "static_audio_hash": get_md5_hash(static_text),
                    }
                return

            messages = await self._build_messages(message, meta_info=meta_info)
            yield {"messages": messages}
            tool_choice = self._get_tool_choice_for_node(history=message)
            forced_name = tool_choice["function"]["name"] if tool_choice else None
            node_tools = self._tools_for_node(current_node, forced_name)
            async for chunk in self.llm.generate_stream(
                messages, synthesize=synthesize, meta_info=meta_info, tool_choice=tool_choice, tools=node_tools
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Error in generate: {e}")
            latency_data = LatencyData(
                sequence_id=meta_info.get("sequence_id") if meta_info else None,
                first_token_latency_ms=0,
                total_stream_duration_ms=now_ms() - start_time,
            )
            yield LLMStreamChunk(data=f"An error occurred: {str(e)}", end_of_stream=True, latency=latency_data)
