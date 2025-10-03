import os
import time
import asyncio
from typing import List, Tuple, AsyncGenerator, Optional, Dict

from bolna.models import *
from bolna.agent_types.base_agent import BaseAgent
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.rag_service_client import RAGServiceClient, RAGServiceClientSingleton
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
        self.rag_service_url = os.getenv('RAG_SERVICE_URL', 'http://localhost:8000')
        
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
            # Final fallback to OpenAI client (import lazily to avoid lints when not installed)
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
        
        # Current format: rag_config should have vector_store.provider_config.vector_id
        collections = []
        
        if 'vector_store' in rag_config:
            vector_store = rag_config['vector_store']
            provider_config = vector_store.get('provider_config', {})
            vector_id = provider_config.get('vector_id')
            
            if vector_id:
                collections.append(vector_id)
            else:
                logger.error("vector_id not found in rag_config.vector_store.provider_config")
        else:
            logger.error("vector_store not found in rag_config")
        
        processed_config = {
            'collections': collections,
            'similarity_top_k': rag_config.get('similarity_top_k', 10),
            'temperature': rag_config.get('temperature', 0.7), 
            'model': rag_config.get('model', 'gpt-4o'),
            'max_tokens': rag_config.get('max_tokens', 150)
        }
        
        logger.info(f"Initialized RAG config with collections: {collections}")
        return processed_config

    async def generate_response(self, history: List[dict]) -> dict:
        """Generate response with RAG context - simplified from GraphAgent's approach."""
        
        if not self.rag_config.get('collections'):
            logger.info("No RAG collections configured, generating fallback response")
            return await self._generate_fallback_response(history)

        logger.info(f"Using RAG service for knowledge-based response")

        try:
            # Get RAG service client (same as GraphAgent)
            client = await RAGServiceClientSingleton.get_client(self.rag_service_url)
            
            # Get user query
            latest_message = history[-1]["content"]
            
            # Query RAG service (same parameters as GraphAgent)
            rag_response = await client.query_for_conversation(
                query=latest_message,
                collections=self.rag_config['collections'],
                max_results=self.rag_config.get('similarity_top_k', 10),
                similarity_threshold=0.1
            )

            if not rag_response.contexts:
                logger.warning(f"No relevant context retrieved from RAG")
                return await self._generate_fallback_response(history)

            # Format context (same as GraphAgent)
            rag_context = await client.format_context_for_prompt(rag_response.contexts)
            
            logger.info(f"Retrieved {rag_response.total_results} contexts in {rag_response.processing_time:.3f}s")

            # Combine prompt with RAG context (simplified from GraphAgent)
            system_prompt = self.config.get('prompt', f"You are {self.agent_information}. Answer questions based on the provided context.")
            combined_context = f"{system_prompt}\n\nRelevant Information:\n{rag_context}\n\nPlease respond based on the latest user message: '{latest_message}'."

            return await self._generate_response_with_llm(combined_context, history)

        except asyncio.TimeoutError:
            logger.error(f"Timeout occurred while fetching data from RAG service")
            return await self._generate_fallback_response(history)

        except Exception as e:
            logger.error(f"An error occurred while processing the RAG service: {e}")
            return await self._generate_fallback_response(history)

    async def _generate_response_with_llm(self, context: str, history: List[dict]) -> dict:
        """Generate response using LLM with the provided context (same logic as GraphAgent)."""
        latest_message = history[-1]["content"]
        system_message = f"{context}\n\nPlease respond based on the latest user message: '{latest_message}'."
        
        messages = [{"role": "system", "content": system_message}] + [
            {"role": item["role"], "content": item["content"]} for item in history[-5:]
        ]
        
        try:
            # Use provider LLM instance like simple agents
            if hasattr(self.llm, 'generate'):
                response_text = await self.llm.generate(messages)
            else:
                # Very rare case: directly use OpenAI-compatible client
                response = self.llm.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    max_tokens=self.config.get('max_tokens', 150),
                    temperature=self.config.get('temperature', 0.7),
                )
                response_text = response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response with LLM provider: {e}")
            # Emergency fallback - same as GraphAgent
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

    async def _generate_fallback_response(self, history: List[dict]) -> dict:
        """Generate a fallback response when RAG is not available (same as GraphAgent)."""
        latest_message = history[-1]["content"]
        system_prompt = self.config.get('prompt', f"You are {self.agent_information}. Respond helpfully to user questions.")
        system_message = f"{system_prompt}\n\nPlease respond based on the latest user message: '{latest_message}'."

        messages = [{"role": "system", "content": system_message}] + [
            {"role": item["role"], "content": item["content"]} for item in history[-5:]
        ]
        
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
            logger.error(f"Error in fallback response generation: {e}")
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
    
    async def generate(self, message: List[dict], **kwargs) -> AsyncGenerator[Tuple[str, bool, float, bool, None, None], None]:
        """
        Main generate method that streams response - same pattern as GraphAgent but simplified.
        """
        start_time = time.time()
        first_token_time = None
        buffer = ""
        buffer_size = 20  # Default buffer size of 20 words
        
        try:
            # Generate response using RAG
            response = await self.generate_response(message)
            response_text = response["content"]

            words = response_text.split()
            for i, word in enumerate(words):
                if first_token_time is None:
                    first_token_time = time.time()
                    latency = first_token_time - start_time
                
                buffer += word + " "
                
                if len(buffer.split()) >= buffer_size or i == len(words) - 1:
                    is_final = (i == len(words) - 1)
                    yield buffer.strip(), is_final, latency, False, None, None
                    buffer = ""
            
            if buffer:
                yield buffer.strip(), True, latency, False, None, None

        except Exception as e:
            logger.error(f"Error in generate function: {e}")
            yield f"An error occurred: {str(e)}", True, time.time() - start_time, False, None, None