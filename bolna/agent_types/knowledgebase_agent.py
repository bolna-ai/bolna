import os
import time
import asyncio
import logging
from typing import List, Tuple, Generator, AsyncGenerator

from dotenv import load_dotenv, find_dotenv

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.agent.openai import OpenAIAgent

from .base_agent import BaseAgent
from pymongo import MongoClient
from bolna.helpers.logger_config import configure_logger


load_dotenv(find_dotenv(), override=True)
logger = configure_logger(__name__)


class RAGAgent(BaseAgent):
    """
    A class that implements a RAG (Retrieval-Augmented Generation) agent.

    This class sets up and manages an OpenAI-based language model, a vector store, and an agent
    for performing document search and question answering tasks.

    Attributes:
        temperature (float): Temperature setting for the language model.
        model (str): The name of the OpenAI model to use.
        buffer (int): Size of the token buffer for streaming responses.
        max_tokens (int): Maximum number of tokens for the language model output.
        query_engine: Query engine for searching the vector index.
        provider_config (dict): Provider configuration for setting up RAG.
    """

    def __init__(self,  provider_config: dict, temperature: float, model: str, buffer: int = 20, max_tokens: int = 100):
        """
        Initialize the RAG Agent instance.

        Args:
            temperature (float): Temperature setting for the language model.
            model (str): The name of the OpenAI model to use.
            buffer (int, optional): Size of the token buffer for streaming responses. Defaults to 20.
            max_tokens (int, optional): Maximum number of tokens for the language model output. Defaults to 100.
            provider_config (dict, optional): Provider configuration for setting up RAG.
        """
        super().__init__()
        self.model = model
        self.buffer = buffer
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.provider_config = provider_config
        self.query_engine = None

        self.OPENAI_KEY = os.getenv('OPENAI_API_KEY')
        self.LANCE_DB_DIR = os.getenv('LANCEDB_DIR')

        self._setup_llm()
        self._setup_provider()

    def _setup_llm(self):
        """Set up the OpenAI language model."""
        self.llm = OpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=self.OPENAI_KEY,
            max_tokens=self.max_tokens
        )

    def _setup_provider(self):
        """Based on the relevant provider config, set up the provider."""

        provider_name = self.provider_config.get('provider')
        if provider_name == "lancedb":
            self.vector_store = LanceDBVectorStore(uri=self.LANCE_DB_DIR, table_name=self.provider_config['provider_config'].get('vector_id'))
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.vector_index = VectorStoreIndex([], storage_context=self.storage_context)
            self.query_engine = self.vector_index.as_query_engine(similarity_top_k=self.provider_config['provider_config'].get('similarity_top_k', 15))
            logger.info("LanceDB provider is initialized")

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
        Generate a response based on the input message.

        This method streams the generated response, yielding chunks of text along with metadata.

        Args:
            message (List[dict]): A list of dictionaries containing the message data and chat history.
            **kwargs: Additional keyword arguments (unused in this implementation).

        Yields:
            Tuple[str, bool, float, bool]: A tuple containing:
                - The generated text chunk.
                - A boolean indicating if this is the final chunk.
                - The latency of the first token.
                - A boolean indicating if the response was truncated (always False in this implementation).
        """
        buffer, latency, latest_message, start_time = "", -1, message[-1], time.time()
        message = ChatMessage(role=latest_message['role'], content=latest_message['content'])
        response = await self.query_engine.aquery(message.content)

        async for token in self.async_word_generator(response.response):
            if latency < 0:
                latency = time.time() - start_time
            buffer += token + " "
            if len(buffer.split()) >= self.buffer or buffer[-1] in {'.', '!', '?'}:
                yield buffer.strip(), False, latency, False
                logger.info(f"LLM BUFFER FULL BUFFER OUTPUT: {buffer}")
                buffer = ""
        if buffer:
            yield buffer.strip(), True, latency, False
            logger.info(f"LLM BUFFER FLUSH BUFFER OUTPUT: {buffer}")
