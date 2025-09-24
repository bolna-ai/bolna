import os
import time
import asyncio
import logging
from typing import List, Tuple, AsyncGenerator

from .base_agent import BaseAgent
from bolna.helpers.logger_config import configure_logger
from bolna.helpers.rag_service_client import RAGServiceClient, RAGServiceClientSingleton

logger = configure_logger(__name__)


class RAGEnhancedAgent(BaseAgent):
    """
    RAG-Enhanced Agent that uses rag-proxy-server for retrieval.
    
    This replaces the old knowledgebase_agent.py and provides unified RAG functionality
    for all agent types that need knowledge base access.
    """

    def __init__(self, 
                 provider_config: dict, 
                 temperature: float, 
                 model: str, 
                 buffer: int = 20, 
                 max_tokens: int = 100,
                 rag_service_url: str = None):
        """
        Initialize the RAG Enhanced Agent.

        Args:
            provider_config (dict): Configuration containing RAG settings
            temperature (float): Temperature setting for the language model
            model (str): The name of the model to use
            buffer (int): Size of the token buffer for streaming responses
            max_tokens (int): Maximum number of tokens for output
            rag_service_url (str): URL of the rag-proxy-server
        """
        super().__init__()
        self.model = model
        self.buffer = buffer
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.provider_config = provider_config
        
        # RAG Service Configuration
        self.rag_service_url = rag_service_url or os.getenv('RAG_SERVICE_URL', 'http://localhost:8000')
        self.collections = self._extract_collections_from_config()
        self.similarity_top_k = provider_config.get('similarity_top_k', 15)
        
        logger.info(f"RAGEnhancedAgent initialized with collections: {self.collections}")
        logger.info(f"RAG Service URL: {self.rag_service_url}")

    def _extract_collections_from_config(self) -> List[str]:
        """Extract collection IDs from the provider config."""
        collections = []
        
        # Handle different config structures
        if isinstance(self.provider_config, dict):
            # Direct vector_id
            if 'vector_id' in self.provider_config:
                collections.append(self.provider_config['vector_id'])
            
            # Nested provider_config
            elif 'provider_config' in self.provider_config:
                nested_config = self.provider_config['provider_config']
                if 'vector_id' in nested_config:
                    collections.append(nested_config['vector_id'])
            
            # Multiple collections
            elif 'collections' in self.provider_config:
                collections.extend(self.provider_config['collections'])
        
        return collections

    async def _get_rag_client(self) -> RAGServiceClient:
        """Get or create a RAG service client."""
        return await RAGServiceClientSingleton.get_client(self.rag_service_url)

    async def _health_check_collections(self) -> List[str]:
        """
        Check which collections are healthy and accessible.
        
        Returns:
            List of healthy collection IDs
        """
        if not self.collections:
            return []
        
        healthy_collections = []
        client = await self._get_rag_client()
        
        # Check each collection
        for collection_id in self.collections:
            try:
                health_status = await client.check_collection_health(collection_id)
                if health_status.get('accessible', False):
                    healthy_collections.append(collection_id)
                    logger.info(f"Collection {collection_id} is healthy")
                else:
                    logger.warning(f"Collection {collection_id} is not accessible: {health_status.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Failed to check health for collection {collection_id}: {e}")
        
        return healthy_collections

    async def _retrieve_context(self, query: str) -> str:
        """
        Retrieve relevant context from RAG service.
        
        Args:
            query (str): The user's query
            
        Returns:
            str: Formatted context string
        """
        if not self.collections:
            logger.info("No collections configured for RAG")
            return ""
        
        try:
            client = await self._get_rag_client()
            
            # Check collection health first
            healthy_collections = await self._health_check_collections()
            if not healthy_collections:
                logger.warning("No healthy collections available")
                return ""
            
            # Query for context
            rag_response = await client.query_for_conversation(
                query=query,
                collections=healthy_collections,
                max_results=self.similarity_top_k,
                similarity_threshold=0.1  # Minimum relevance threshold
            )
            
            if not rag_response.contexts:
                logger.info("No relevant context found")
                return ""
            
            # Format context for LLM
            context_text = await client.format_context_for_prompt(rag_response.contexts)
            
            logger.info(f"Retrieved {rag_response.total_results} contexts in {rag_response.processing_time:.3f}s")
            return context_text
            
        except Exception as e:
            logger.error(f"Error retrieving RAG context: {e}")
            return ""

    async def generate(self, message: List[dict], **kwargs) -> AsyncGenerator[Tuple[str, bool, float, bool, None, None], None]:
        """
        Generate a response with RAG-enhanced context.

        Args:
            message (List[dict]): A list of dictionaries containing the message data and chat history.
            **kwargs: Additional keyword arguments.

        Yields:
            Tuple[str, bool, float, bool, None, None]: Generated text chunks with metadata.
        """
        start_time = time.time()
        buffer = ""
        latency = -1
        
        try:
            # Get the latest user message
            if not message:
                yield "I didn't receive any message to respond to.", True, 0.0, False, None, None
                return
            
            latest_message = message[-1]
            user_query = latest_message.get('content', '')
            
            if not user_query.strip():
                yield "I received an empty message.", True, 0.0, False, None, None
                return
            
            # Retrieve relevant context from RAG service
            logger.info(f"Processing query: {user_query[:100]}...")
            context = await self._retrieve_context(user_query)
            
            # Prepare response based on context availability
            if context:
                # Format response with context
                response_parts = [
                    f"Based on the information I have access to:\n\n",
                    f"Context: {context}\n\n",
                    f"Regarding your question: {user_query}\n\n",
                    "Let me provide you with a relevant response based on this information."
                ]
                response_text = "".join(response_parts)
            else:
                # Fallback response when no context is available
                response_text = f"I understand you're asking about: {user_query}. However, I don't have specific context available to provide a detailed response. Could you provide more information or rephrase your question?"
            
            # Stream the response in chunks
            words = response_text.split()
            for i, word in enumerate(words):
                if latency < 0:
                    latency = time.time() - start_time
                
                buffer += word + " "
                
                # Yield buffer when it's full or at sentence boundaries
                if (len(buffer.split()) >= self.buffer or 
                    buffer.rstrip().endswith(('.', '!', '?')) or 
                    i == len(words) - 1):
                    
                    is_final = (i == len(words) - 1)
                    yield buffer.strip(), is_final, latency, False, None, None
                    
                    logger.info(f"RAG Agent yielding: {buffer.strip()[:100]}...")
                    buffer = ""
            
            # Yield any remaining buffer
            if buffer.strip():
                yield buffer.strip(), True, latency, False, None, None
                
        except Exception as e:
            logger.error(f"Error in RAG agent generation: {e}")
            error_message = f"I encountered an error while processing your request: {str(e)}"
            yield error_message, True, time.time() - start_time, False, None, None


class SimpleRAGAgent(RAGEnhancedAgent):
    """
    Simplified RAG agent that provides direct question-answer responses.
    This is a drop-in replacement for the old knowledgebase_agent.
    """
    
    async def generate(self, message: List[dict], **kwargs) -> AsyncGenerator[Tuple[str, bool, float, bool, None, None], None]:
        """
        Generate a simple, direct response using RAG context.
        
        Args:
            message (List[dict]): Message history
            **kwargs: Additional arguments
            
        Yields:
            Tuple with response chunks
        """
        start_time = time.time()
        latency = -1
        
        try:
            if not message:
                yield "No query provided.", True, 0.0, False, None, None
                return
            
            user_query = message[-1].get('content', '').strip()
            if not user_query:
                yield "Empty query received.", True, 0.0, False, None, None
                return
            
            # Get RAG context
            context = await self._retrieve_context(user_query)
            
            if context:
                # Extract the most relevant information
                context_lines = context.split('\n')
                relevant_info = []
                
                for line in context_lines:
                    if line.strip() and not line.startswith('Context'):
                        relevant_info.append(line.strip())
                
                if relevant_info:
                    response = " ".join(relevant_info[:3])  # Take first 3 relevant lines
                else:
                    response = "I found some information but it may not be directly relevant to your question."
            else:
                response = "I don't have specific information about that topic in my knowledge base."
            
            # Stream response
            words = response.split()
            buffer = ""
            
            for i, word in enumerate(words):
                if latency < 0:
                    latency = time.time() - start_time
                
                buffer += word + " "
                
                if len(buffer.split()) >= self.buffer or i == len(words) - 1:
                    is_final = (i == len(words) - 1)
                    yield buffer.strip(), is_final, latency, False, None, None
                    buffer = ""
                    
        except Exception as e:
            logger.error(f"Error in simple RAG generation: {e}")
            yield f"Error: {str(e)}", True, time.time() - start_time, False, None, None