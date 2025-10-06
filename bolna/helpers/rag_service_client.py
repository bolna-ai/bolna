import aiohttp  # type: ignore
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class RAGContext:
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class RAGResponse:
    contexts: List[RAGContext]
    total_results: int
    processing_time: float

class RAGServiceClient:
    """
    Client for communicating with rag-proxy-server.
    This replaces all local RAG functionality in bolna agents.
    """
    
    def __init__(self, rag_server_url: str, timeout: int = 30):
        """
        Initialize the RAG service client.
        
        Args:
            rag_server_url: Base URL of the rag-proxy-server
            timeout: Request timeout in seconds
        """
        self.base_url = rag_server_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """Ensure session exists."""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the RAG service is healthy.
        
        Returns:
            Dict containing health status
        """
        await self._ensure_session()
        
        try:
            session = self.session
            assert session is not None
            async with session.get(f"{self.base_url}/") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "unhealthy", "error": f"HTTP {response.status}"}
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def check_collection_health(self, collection_id: str) -> Dict[str, Any]:
        """
        Check if a specific collection is accessible.
        
        Args:
            collection_id: ID of the collection to check
            
        Returns:
            Dict containing collection health status
        """
        await self._ensure_session()
        
        try:
            session = self.session
            assert session is not None
            async with session.get(
                f"{self.base_url}/collections/{collection_id}/health"
            ) as response:
                return await response.json()
        except Exception as e:
            self.logger.error(f"Collection health check failed: {e}")
            return {
                "collection_id": collection_id,
                "status": "error", 
                "accessible": False,
                "error": str(e)
            }
    
    async def query_for_conversation(
        self, 
        query: str, 
        collections: List[str],
        max_results: int = 15,
        similarity_threshold: float = 0.0
    ) -> RAGResponse:
        """
        Query multiple collections for conversation context.
        This is the main method used by bolna agents.
        
        Args:
            query: The user's query/message
            collections: List of collection IDs to search
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            RAGResponse with contextualized results
        """
        await self._ensure_session()
        
        payload = {
            "query": query,
            "collections": collections,
            "max_results": max_results,
            "similarity_threshold": similarity_threshold
        }
        
        try:
            session = self.session
            assert session is not None
            async with session.post(
                f"{self.base_url}/conversation/query",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"RAG query failed: {response.status} - {error_text}")
                    return RAGResponse(contexts=[], total_results=0, processing_time=0.0)
                
                data = await response.json()

                # rag-proxy-server returns { documents, total_retrieved, query_time_ms, ... }
                documents = data.get("documents", [])
                contexts = [
                    RAGContext(
                        text=doc.get("text", ""),
                        score=doc.get("score", 0.0),
                        metadata=doc.get("metadata", {})
                    )
                    for doc in documents
                ]

                total_results = data.get("total_retrieved", len(contexts))
                # Convert ms to seconds for consistency in logs
                processing_time = float(data.get("query_time_ms", 0.0)) / 1000.0

                return RAGResponse(
                    contexts=contexts,
                    total_results=total_results,
                    processing_time=processing_time
                )
                
        except asyncio.TimeoutError:
            self.logger.error(f"RAG query timeout for collections: {collections}")
            return RAGResponse(contexts=[], total_results=0, processing_time=0.0)
        except Exception as e:
            self.logger.error(f"RAG query error: {e}")
            return RAGResponse(contexts=[], total_results=0, processing_time=0.0)
    
    async def format_context_for_prompt(self, contexts: List[RAGContext]) -> str:
        """
        Format RAG contexts into a string suitable for LLM prompts.
        
        Args:
            contexts: List of RAG contexts
            
        Returns:
            Formatted context string
        """
        if not contexts:
            return ""
        
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            context_parts.append(f"Context {i} (relevance: {ctx.score:.3f}):")
            context_parts.append(ctx.text)
            context_parts.append("")  # Empty line for separation
            
        return "\n".join(context_parts)
    
    async def get_enhanced_prompt(
        self, 
        original_prompt: str, 
        user_query: str,
        collections: List[str],
        max_results: int = 5
    ) -> str:
        """
        Get an enhanced prompt with RAG context integrated.
        
        Args:
            original_prompt: The original system prompt
            user_query: The user's query
            collections: Collections to search
            max_results: Maximum contexts to include
            
        Returns:
            Enhanced prompt with RAG context
        """
        # Query RAG for context
        rag_response = await self.query_for_conversation(
            query=user_query,
            collections=collections,
            max_results=max_results
        )
        
        if not rag_response.contexts:
            return original_prompt
        
        # Format context
        context_text = await self.format_context_for_prompt(rag_response.contexts)
        
        # Integrate context into prompt
        enhanced_prompt = f"""{original_prompt}

RELEVANT CONTEXT:
{context_text}

Please respond to the user's query using the above context when relevant. If the context doesn't contain relevant information, respond based on your general knowledge."""

        return enhanced_prompt
    
    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None


class RAGServiceClientSingleton:
    """
    Singleton wrapper for RAG service client to avoid creating multiple sessions.
    """
    _instance = None
    _client = None
    
    @classmethod
    async def get_client(cls, rag_server_url: str) -> RAGServiceClient:
        """
        Get or create a RAG service client instance.
        
        Args:
            rag_server_url: Base URL of the rag-proxy-server
            
        Returns:
            RAGServiceClient instance
        """
        if cls._instance is None or cls._client is None:
            cls._client = RAGServiceClient(rag_server_url)
            cls._instance = cls
            
        return cls._client
    
    @classmethod
    async def close_client(cls):
        """Close the client if it exists."""
        if cls._client:
            await cls._client.close()
            cls._client = None
            cls._instance = None