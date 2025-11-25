import os
import time
import asyncio
from typing import List, Tuple, AsyncGenerator, Optional, Dict

from bolna.models import *
from bolna.agent_types.base_agent import BaseAgent
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.rag_service_client import RAGServiceClient, RAGServiceClientSingleton
from bolna.helpers.utils import now_ms
from bolna.providers import SUPPORTED_LLM_PROVIDERS

logger = configure_logger(__name__)

class KnowledgeBaseAgent(BaseAgent):
    """
    Simplified Knowledge-based agent that replicates GraphAgent RAG flow
    without the complexity of multi-node graph management.
    
    This agent is designed for single-node conversations with document knowledge,
    providing the same RAG functionality as GraphAgent but with simpler configuration.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.agent_information = self.config.get('agent_information', 'Knowledge-based AI assistant')
        self.context_data = self.config.get('context_data', {})
        self.llm_model = self.config.get('model', 'gpt-4o')
        
        # Initialize provider LLM (same provider system simple agents use)
        self.llm = self._initialize_llm()
        
        # Initialize RAG configurations - simplified from GraphAgent's approach
        self.rag_config = self._initialize_rag_config()
        self.rag_server_url = os.getenv('RAG_SERVER_URL', 'http://localhost:8000')
        
        logger.info(f"KnowledgeAgent initialized with RAG collections: {self.rag_config.get('collections', [])}")

    def _initialize_llm(self):
        """Initialize provider LLM using the standard provider system (OpenAiLLM/LiteLLM)."""
        try:
            # Prefer "provider" like simple agents; fallback to legacy "llm_provider"
            provider_name = self.config.get('provider') or self.config.get('llm_provider', 'openai')

            if provider_name in SUPPORTED_LLM_PROVIDERS:
                # Prepare kwargs as expected by provider LLM classes
                llm_kwargs = {
                    'model': self.llm_model,
                    'temperature': self.config.get('temperature', 0.7),
                    'max_tokens': self.config.get('max_tokens', 150),
                    'provider': provider_name,
                }

                # Credentials / endpoints (injected by backend via TaskManager)
                if 'llm_key' in self.config:
                    llm_kwargs['llm_key'] = self.config['llm_key']
                if 'base_url' in self.config:
                    llm_kwargs['base_url'] = self.config['base_url']
                if 'api_version' in self.config:
                    llm_kwargs['api_version'] = self.config['api_version']
                if 'language' in self.config:
                    llm_kwargs['language'] = self.config['language']

                llm_class = SUPPORTED_LLM_PROVIDERS[provider_name]
                return llm_class(**llm_kwargs)

            logger.warning(f"Unknown LLM provider: {provider_name}, defaulting to openai")
            llm_class = SUPPORTED_LLM_PROVIDERS['openai']
            return llm_class(model=self.llm_model, llm_key=os.getenv('OPENAI_API_KEY'))

        except Exception as e:
            logger.error(f"Error initializing provider LLM: {e}")
            import importlib
            openai_mod = importlib.import_module('openai')
            OpenAI = getattr(openai_mod, 'OpenAI')
            return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def _initialize_rag_config(self) -> Dict:
        """Initialize RAG configuration using the current format."""
        rag_config = self.config.get('rag_config', {})

        if not rag_config:
            logger.warning("No RAG config provided")
            return {}

        # Support both single vector_id and multiple vector_ids
        collections = []

        if 'vector_store' in rag_config:
            vector_store = rag_config['vector_store']
            provider_config = vector_store.get('provider_config', {})

            # Try multiple vector_ids first (new format)
            vector_ids = provider_config.get('vector_ids')
            if vector_ids and isinstance(vector_ids, list):
                collections.extend(vector_ids)
                logger.info(f"Found {len(vector_ids)} collections in vector_ids")
            # Fallback to single vector_id (backward compatibility)
            elif vector_id := provider_config.get('vector_id'):
                collections.append(vector_id)
                logger.info(f"Found single collection in vector_id")
            else:
                logger.error("Neither vector_ids nor vector_id found in rag_config.vector_store.provider_config")
        else:
            logger.error("vector_store not found in rag_config")

        processed_config = {
            'collections': collections,
            'similarity_top_k': rag_config.get('similarity_top_k', 10),
            'temperature': rag_config.get('temperature', 0.7),
            'model': rag_config.get('model', 'gpt-4o'),
            'max_tokens': rag_config.get('max_tokens', 150)
        }

        logger.info(f"Initialized RAG config with {len(collections)} collection(s): {collections}")
        return processed_config

    async def generate_response(self, history: List[dict]) -> dict:
        """Generate response with natural conversation flow and optional RAG augmentation."""

        if not self.rag_config.get('collections'):
            return await self._generate_standard_response(history)

        try:
            client = await RAGServiceClientSingleton.get_client(self.rag_server_url)
            latest_message = history[-1]["content"]

            rag_response = await client.query_for_conversation(
                query=latest_message,
                collections=self.rag_config['collections'],
                max_results=self.rag_config.get('similarity_top_k', 10),
                similarity_threshold=0.0
            )

            if not rag_response.contexts or len(rag_response.contexts) == 0:
                return await self._generate_standard_response(history)

            top_score = rag_response.contexts[0].score if rag_response.contexts else 0.0
            logger.info(f"RAG: Retrieved {rag_response.total_results} contexts, top score: {top_score:.3f}")

            return await self._generate_response_with_knowledge(history, rag_response.contexts)

        except asyncio.TimeoutError:
            logger.error("RAG service timeout, using standard conversation")
            return await self._generate_standard_response(history)

        except Exception as e:
            logger.error(f"RAG service error: {e}")
            return await self._generate_standard_response(history)

    async def _generate_standard_response(self, history: List[dict]) -> dict:
        """Generate response using conversation history without RAG augmentation."""
        max_history = 50
        messages = history[-max_history:] if len(history) > max_history else history

        try:
            if hasattr(self.llm, 'generate'):
                response_text = await self.llm.generate(messages)
            else:
                response = self.llm.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    max_tokens=self.config.get('max_tokens', 150),
                    temperature=self.config.get('temperature', 0.7),
                )
                response_text = response.choices[0].message.content

        except Exception as e:
            logger.error(f"Standard response error: {e}")
            import importlib
            openai_mod = importlib.import_module('openai')
            OpenAI = getattr(openai_mod, 'OpenAI')
            fallback_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = fallback_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_tokens=self.config.get('max_tokens', 150),
                temperature=self.config.get('temperature', 0.7),
            )
            response_text = response.choices[0].message.content

        return {"role": "assistant", "content": response_text}

    async def _generate_response_with_knowledge(self, history: List[dict], rag_contexts: List) -> dict:
        """Generate response augmented with RAG knowledge base context."""
        client = await RAGServiceClientSingleton.get_client(self.rag_server_url)
        rag_context = await client.format_context_for_prompt(rag_contexts)

        if history and history[0].get('role') == 'system':
            original_system_prompt = history[0]['content']
            rest_of_history = history[1:]
        else:
            original_system_prompt = self.config.get('prompt', f"You are {self.agent_information}.")
            rest_of_history = history

        augmented_system_prompt = f"""{original_system_prompt}

You have access to relevant information from the knowledge base:

{rag_context}

Use this information naturally when it helps answer the user's questions. Don't force references if not relevant to the conversation."""

        augmented_history = [{"role": "system", "content": augmented_system_prompt}] + rest_of_history

        max_history = 50
        if len(augmented_history) > max_history:
            logger.warning(f"History truncated from {len(augmented_history)} to {max_history}")
            augmented_history = [augmented_history[0]] + augmented_history[-(max_history-1):]

        try:
            if hasattr(self.llm, 'generate'):
                response_text = await self.llm.generate(augmented_history)
            else:
                response = self.llm.chat.completions.create(
                    model=self.llm_model,
                    messages=augmented_history,
                    max_tokens=self.config.get('max_tokens', 150),
                    temperature=self.config.get('temperature', 0.7),
                )
                response_text = response.choices[0].message.content

        except Exception as e:
            logger.error(f"Knowledge-augmented response error: {e}")
            import importlib
            openai_mod = importlib.import_module('openai')
            OpenAI = getattr(openai_mod, 'OpenAI')
            fallback_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = fallback_client.chat.completions.create(
                model=self.llm_model,
                messages=augmented_history,
                max_tokens=self.config.get('max_tokens', 150),
                temperature=self.config.get('temperature', 0.7),
            )
            response_text = response.choices[0].message.content

        return {"role": "assistant", "content": response_text}
    
    async def generate(self, message: List[dict], **kwargs) -> AsyncGenerator[Tuple[str, bool, Optional[Dict], bool, None, None], None]:
        """Main generate method that streams response."""
        meta_info = kwargs.get('meta_info')
        start_time = now_ms()
        first_token_time = None
        buffer = ""
        buffer_size = 20
        latency_data = {
            "sequence_id": meta_info.get("sequence_id") if meta_info else None,
            "first_token_latency_ms": None,
            "total_stream_duration_ms": None
        }

        try:
            response = await self.generate_response(message)
            response_text = response["content"]

            words = response_text.split()
            for i, word in enumerate(words):
                if first_token_time is None:
                    first_token_time = now_ms()
                    latency_data["first_token_latency_ms"] = first_token_time - start_time

                buffer += word + " "

                if len(buffer.split()) >= buffer_size or i == len(words) - 1:
                    is_final = (i == len(words) - 1)
                    if is_final and latency_data:
                        latency_data["total_stream_duration_ms"] = now_ms() - start_time
                    yield buffer.strip(), is_final, latency_data, False, None, None
                    buffer = ""

            if buffer:
                if latency_data:
                    latency_data["total_stream_duration_ms"] = now_ms() - start_time
                yield buffer.strip(), True, latency_data, False, None, None

        except Exception as e:
            logger.error(f"Error in generate function: {e}")
            latency_data["first_token_latency_ms"] = latency_data.get("first_token_latency_ms") or 0
            latency_data["total_stream_duration_ms"] = now_ms() - start_time
            yield f"An error occurred: {str(e)}", True, latency_data, False, None, None