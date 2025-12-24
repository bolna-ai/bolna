import os
import asyncio
import json
from typing import List, Tuple, AsyncGenerator, Optional, Dict

from bolna.models import *
from bolna.agent_types.base_agent import BaseAgent
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.rag_service_client import RAGServiceClientSingleton
from bolna.helpers.utils import now_ms, format_messages
from bolna.providers import SUPPORTED_LLM_PROVIDERS
from bolna.llms import OpenAiLLM
from bolna.prompts import VOICEMAIL_DETECTION_PROMPT

logger = configure_logger(__name__)


class KnowledgeBaseAgent(BaseAgent):
    """
    Knowledge-based conversational agent with RAG (Retrieval Augmented Generation) support.

    This agent:
    - Fetches relevant context from a knowledge base before responding
    - Supports function calling (API tools) just like the simple LLM agent
    - Supports hangup detection via check_for_completion()
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.agent_information = self.config.get('agent_information', 'Knowledge-based AI assistant')
        self.context_data = self.config.get('context_data', {})
        self.llm_model = self.config.get('model', 'gpt-4o')

        # Main LLM for conversation
        self.llm = self._initialize_llm()

        # Separate LLM for checking if call should end
        self.conversation_completion_llm = OpenAiLLM(model=os.getenv('CHECK_FOR_COMPLETION_LLM', self.llm_model))

        # RAG configuration
        self.rag_config = self._initialize_rag_config()
        self.rag_server_url = os.getenv('RAG_SERVER_URL', 'http://localhost:8000')

        logger.info(f"KnowledgeBaseAgent initialized with RAG collections: {self.rag_config.get('collections', [])}")

    def _initialize_llm(self):
        """Initialize the LLM instance with all necessary config (including api_tools for function calling)."""
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

            # Pass through credentials
            for key in ['llm_key', 'base_url', 'api_version', 'language', 'api_tools', 'buffer_size']:
                if key in self.config:
                    llm_kwargs[key] = self.config[key]

            llm_class = SUPPORTED_LLM_PROVIDERS[provider]
            return llm_class(**llm_kwargs)

        except Exception as e:
            logger.error(f"Failed to create LLM: {e}, falling back to basic OpenAI")
            from openai import OpenAI
            return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def _initialize_rag_config(self) -> Dict:
        """Initialize RAG configuration from the provided config."""
        rag_config = self.config.get('rag_config', {})

        if not rag_config:
            logger.warning("No RAG config provided")
            return {}

        collections = []
        used_sources = rag_config.get('used_sources', None)

        if 'vector_store' in rag_config:
            provider_config = rag_config['vector_store'].get('provider_config', {})

            if used_sources:
                for source in used_sources:
                    vector_id = source.get('vector_id')
                    if vector_id:
                        collections.append(vector_id)

            else:
                # Support both formats: vector_ids (list) and vector_id (single)
                vector_ids = provider_config.get('vector_ids')
                if vector_ids and isinstance(vector_ids, list):
                    collections.extend(vector_ids)
                elif vector_id := provider_config.get('vector_id'):
                    collections.append(vector_id)
                else:
                    logger.error("No vector_id or vector_ids found in rag_config")
        else:
            logger.error("No vector_store in rag_config")

        return {
            'collections': collections,
            'similarity_top_k': rag_config.get('similarity_top_k', 10),
            'used_sources': used_sources
        }

    async def check_for_completion(self, messages, check_for_completion_prompt):
        """Check if the conversation should end (used for auto-hangup feature)."""
        try:
            prompt = [
                {'role': 'system', 'content': check_for_completion_prompt},
                {'role': 'user', 'content': format_messages(messages)}
            ]
            response = await self.conversation_completion_llm.generate(prompt, request_json=True)
            return json.loads(response)
        except Exception as e:
            logger.error(f'check_for_completion error: {e}')
            return {'hangup': 'No'}
        
    async def check_for_voicemail(self, user_message, voicemail_detection_prompt=None):
        """
        Check if the user message indicates a voicemail system.
        
        Args:
            user_message: The transcribed message from the user
            voicemail_detection_prompt: Custom prompt for voicemail detection (optional)
        
        Returns:
            dict with 'is_voicemail': 'Yes' or 'No'
        """
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

            response = await self.conversation_completion_llm.generate(prompt, request_json=True)
            result = json.loads(response)
            logger.info(f"Voicemail detection result: {result}")
            return result
        except Exception as e:
            logger.error('check_for_voicemail exception: {}'.format(str(e)))
            return {'is_voicemail': 'No'}

    async def _add_rag_context(self, messages: List[dict]) -> Tuple[List[dict], dict]:
        """
        Add relevant knowledge base context to the messages.
        Returns the original messages if RAG is not configured or fails.
        """
        if not self.rag_config.get('collections'):
            return messages, {'status': 'error', 'message': 'No knowledgebases configured'}

        try:
            client = await RAGServiceClientSingleton.get_client(self.rag_server_url)

            latest_message = messages[-1]["content"] if messages else ""

            rag_response = await client.query_for_conversation(
                query=latest_message,
                collections=self.rag_config['collections'],
                max_results=self.rag_config.get('similarity_top_k', 10),
                similarity_threshold=0.0
            )

            if not rag_response.contexts:
                return messages, {'status': 'error', 'message': 'No knowledgebase contexts found'}
            
            used_vector_ids = set()
            for context in rag_response.contexts:
                metadata = context.metadata
                vector_id = metadata.get('collection_id', None)
                if vector_id:
                    used_vector_ids.add(vector_id)

            retrieved_sources = []
            for source in self.rag_config.get('used_sources', []):
                vector_id = source.get('vector_id')
                if vector_id in used_vector_ids:
                    retrieved_sources.append(source)

            logger.info(f"RAG: Found {rag_response.total_results} contexts, top score: {rag_response.contexts[0].score:.3f}")

            rag_context = await client.format_context_for_prompt(rag_response.contexts)

            if messages and messages[0].get('role') == 'system':
                system_prompt = messages[0]['content']
                other_messages = messages[1:]
            else:
                system_prompt = self.config.get('prompt', f"You are {self.agent_information}.")
                other_messages = messages

            # Add RAG context to system prompt
            enhanced_system_prompt = f"""{system_prompt}

You have access to relevant information from the knowledge base:

{rag_context}

Use this information naturally when it helps answer the user's questions. Don't force references if not relevant to the conversation."""

            # Build final messages
            final_messages = [{"role": "system", "content": enhanced_system_prompt}] + other_messages

            # Limit history size
            max_messages = 50
            if len(final_messages) > max_messages:
                final_messages = [final_messages[0]] + final_messages[-(max_messages-1):]

            return final_messages, {'status': 'success', 'retrieved_sources': retrieved_sources}

        except asyncio.TimeoutError:
            logger.error("RAG service timeout")
            return messages, {'status': 'error', 'message': 'Internal Service Error'}
        except Exception as e:
            logger.error(f"RAG error: {e}")
            return messages, {'status': 'error', 'message': 'Internal Service Error'}

    async def generate(self, message: List[dict], **kwargs) -> AsyncGenerator[Tuple[str, bool, Optional[Dict], bool, None, None], None]:
        """
        Generate a streaming response with RAG context
        """
        meta_info = kwargs.get('meta_info')
        synthesize = kwargs.get('synthesize', True)
        start_time = now_ms()

        meta_info['llm_metadata'] = meta_info.get('llm_metadata', {})
        meta_info['llm_metadata']['rag_info'] = {}
        meta_info['llm_metadata']['rag_info']['all_sources'] = self.rag_config.get('used_sources', [])
        try:
            messages_with_context, metadata = await self._add_rag_context(message)

            meta_info['llm_metadata']['rag_info']['context_retrieval'] = metadata

            async for chunk in self.llm.generate_stream(messages_with_context, synthesize=synthesize, meta_info=meta_info):
                yield chunk

        except Exception as e:
            logger.error(f"generate() error: {e}")
            latency_data = {
                "sequence_id": meta_info.get("sequence_id") if meta_info else None,
                "first_token_latency_ms": 0,
                "total_stream_duration_ms": now_ms() - start_time
            }
            yield f"An error occurred: {str(e)}", True, latency_data, False, None, None
