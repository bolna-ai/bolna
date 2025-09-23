import os
import time
import asyncio
import logging
import aiohttp
import json
from typing import List, Tuple, Generator, AsyncGenerator

from dotenv import load_dotenv, find_dotenv

from llama_index.core.llms import ChatMessage

from .base_agent import BaseAgent
from bolna.helpers.logger_config import configure_logger


load_dotenv(find_dotenv(), override=True)
logger = configure_logger(__name__)


class RAGAgent(BaseAgent):
    """
    Enhanced RAG (Retrieval-Augmented Generation) agent with reranker support.

    This class integrates with the RAG server for semantic search and retrieval,
    now supporting advanced reranking capabilities for improved accuracy.

    Attributes:
        buffer (int): Size of the token buffer for streaming responses.
        provider_config (dict): Provider configuration containing vector_id and settings.
        rag_server_url (str): URL of the RAG server for document retrieval.
        vector_id (str): Identifier for the vector index to search.
        similarity_top_k (int): Number of top similar documents to retrieve.
        score_threshold (float): Minimum relevance score for filtering results.
        reranker_enabled (bool): Whether to enable reranking for improved accuracy.
        reranker_model (str): Reranker model to use (bge-base, minilm-l6-v2, etc.).
        candidate_count (int): Number of candidates to retrieve before reranking.
        final_count (int): Final number of results after reranking.
    """

    def __init__(self,  provider_config: dict, temperature: float, model: str, buffer: int = 20, max_tokens: int = 100):
        """
        Initialize the RAG Agent instance.

        Args:
            provider_config (dict): Provider configuration containing vector_id and RAG settings.
            temperature (float): Temperature setting (not used in RAG-only mode).
            model (str): Model name (not used in RAG-only mode).
            buffer (int, optional): Size of the token buffer for streaming responses. Defaults to 20.
            max_tokens (int, optional): Max tokens (not used in RAG-only mode). Defaults to 100.
        """
        super().__init__()
        self.buffer = buffer
        self.provider_config = provider_config
        
        # RAG server configuration
        self.rag_server_url = os.getenv('RAG_SERVER_URL', 'http://localhost:8000')
        self.vector_id = provider_config['provider_config'].get('vector_id')
        self.similarity_top_k = provider_config['provider_config'].get('similarity_top_k', 5)
        self.score_threshold = provider_config['provider_config'].get('score_threshold', 0.1)
        
        # Reranker configuration
        reranker_config = provider_config['provider_config'].get('reranker', {})
        self.reranker_enabled = reranker_config.get('enabled', True)
        self.reranker_model = reranker_config.get('model_type', 'minilm-l6-v2')
        self.candidate_count = reranker_config.get('candidate_count', 20)
        self.final_count = reranker_config.get('final_count', 5)
        
        if not self.vector_id:
            raise ValueError("vector_id is required in provider_config")
            
        logger.info(f"RAG Agent initialized - Vector ID: {self.vector_id}, Server: {self.rag_server_url}")
        logger.info(f"Reranker: {'Enabled' if self.reranker_enabled else 'Disabled'} - Model: {self.reranker_model if self.reranker_enabled else 'None'}")

    def _setup_llm(self):
        """RAG agent doesn't need LLM setup - retrieval only."""
        pass

    def _setup_provider(self):
        """RAG agent uses HTTP client to RAG server - no direct DB setup needed."""
        pass

    async def async_word_generator(self, response):
        """
        An async generator that yields words from a response.
        Args:
            response (str): A string containing the response.
        Yields:
            str: A word from the response.
        """
        for word in response.split():
            yield word

    async def generate(self, message: List[dict], **kwargs) -> AsyncGenerator[Tuple[str, bool, float, bool], None]:
        """
        Retrieve relevant documents from RAG server and format as response.

        This method queries the RAG server for relevant documents and streams the results
        as formatted context information.

        Args:
            message (List[dict]): A list of dictionaries containing the message data and chat history.
            **kwargs: Additional keyword arguments (unused in this implementation).

        Yields:
            Tuple[str, bool, float, bool, None, None]: A tuple containing:
                - The retrieved context text chunk.
                - A boolean indicating if this is the final chunk.
                - The latency of the first response.
                - A boolean indicating if the response was truncated (always False).
                - None (unused parameter for compatibility)
                - None (unused parameter for compatibility)
        """
        buffer, latency, latest_message, start_time = "", -1, message[-1], time.time()
        query = latest_message['content']
        
        try:
            # Enhanced RAG server call with reranker support
            async with aiohttp.ClientSession() as session:
                # Build parameters with reranker configuration
                params = {
                    "query": query,
                    "top_k": self.final_count if self.reranker_enabled else self.similarity_top_k,
                    "score_threshold": self.score_threshold
                }
                
                # Add reranking parameters if enabled
                if self.reranker_enabled:
                    params.update({
                        "enable_reranking": True,
                        "reranker_model": self.reranker_model
                    })
                    logger.info(f"Using reranker: {self.reranker_model} for query: {query[:50]}...")
                
                async with session.get(
                    f"{self.rag_server_url}/retrieve/{self.vector_id}",
                    params=params
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        documents = result.get('documents', [])
                        reranking_applied = result.get('reranking_applied', False)
                        reranker_model = result.get('reranker_model')
                        
                        if documents:
                            # Enhanced context formatting with reranker info
                            context_header = "📚 Knowledge Base Results"
                            if reranking_applied:
                                context_header += f" (Reranked by {reranker_model})"
                            context_text = f"{context_header}:\n\n"
                            
                            for i, doc in enumerate(documents, 1):
                                context_text += f"[{i}] {doc['text']}\n"
                                if doc.get('score'):
                                    score_label = "Rerank Score" if reranking_applied else "Similarity"
                                    context_text += f"({score_label}: {doc['score']:.3f})\n"
                                context_text += "\n"
                                
                            # Add metadata for debugging
                            total_found = result.get('total_found', len(documents))
                            if total_found > len(documents):
                                context_text += f"📊 Retrieved {len(documents)} of {total_found} total matches\n"
                        else:
                            context_text = "❌ No relevant information found in the knowledge base."
                    else:
                        logger.error(f"RAG server error: {response.status}")
                        context_text = "❌ Error retrieving information from knowledge base."
                        
        except Exception as e:
            logger.error(f"RAG retrieval error: {str(e)}")
            context_text = "Error connecting to knowledge base."

        # FIXED: Stream the context immediately instead of fake word-by-word streaming
        if latency < 0:
            latency = time.time() - start_time
            
        # Stream the context in chunks
        async for token in self.async_word_generator(context_text):
            buffer += token + " "
            if len(buffer.split()) >= self.buffer or buffer[-1] in {'.', '!', '?'}:
                yield buffer.strip(), False, latency, False, None, None
                logger.info(f"RAG BUFFER OUTPUT: {buffer}")
                buffer = ""
                
        if buffer:
            yield buffer.strip(), True, latency, False, None, None
            logger.info(f"RAG FINAL BUFFER OUTPUT: {buffer}")
