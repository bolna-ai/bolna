import asyncio
import aiohttp
from typing import Dict, Optional
from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class ConnectionPoolManager:
    """Singleton connection pool manager for LLM HTTP connections"""
    _instance: Optional['ConnectionPoolManager'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._sessions: Dict[str, aiohttp.ClientSession] = {}
            self._connectors: Dict[str, aiohttp.TCPConnector] = {}
            self._initialized = True
    
    async def get_session(self, provider: str, **connector_kwargs) -> aiohttp.ClientSession:
        """Get or create a pooled session for the given provider"""
        async with self._lock:
            if provider not in self._sessions or self._sessions[provider].closed:
                connector = aiohttp.TCPConnector(
                    limit=100,  # Total connection pool size
                    limit_per_host=20,  # Max connections per host
                    ttl_dns_cache=300,  # DNS cache TTL
                    use_dns_cache=True,
                    keepalive_timeout=30,
                    enable_cleanup_closed=True,
                    **connector_kwargs
                )
                
                timeout = aiohttp.ClientTimeout(
                    total=300,  # 5 minutes total timeout
                    connect=10,  # 10 seconds connect timeout
                    sock_read=60  # 60 seconds read timeout
                )
                
                session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={'User-Agent': 'Bolna-AI/1.0'}
                )
                
                self._sessions[provider] = session
                self._connectors[provider] = connector
                logger.info(f"Created new connection pool for provider: {provider}")
                logger.info(f"Pool settings: {connector.limit} total connections, {connector.limit_per_host} per host, {connector._keepalive_timeout}s keepalive")
            
            return self._sessions[provider]
    
    async def close_all(self):
        """Close all pooled connections"""
        async with self._lock:
            for provider, session in self._sessions.items():
                if not session.closed:
                    await session.close()
                    logger.info(f"Closed connection pool for provider: {provider}")
            
            self._sessions.clear()
            self._connectors.clear()
    
    async def get_pool_stats(self) -> Dict[str, Dict]:
        """Get connection pool statistics"""
        stats = {}
        for provider, connector in self._connectors.items():
            if hasattr(connector, '_conns'):
                stats[provider] = {
                    'total_connections': len(connector._conns),
                    'limit': connector.limit,
                    'limit_per_host': connector.limit_per_host
                }
        return stats


class BaseLLM:
    _pool_manager: Optional[ConnectionPoolManager] = None
    
    def __init__(self, max_tokens=100, buffer_size=40):
        self.buffer_size = buffer_size
        self.max_tokens = max_tokens
        
        if BaseLLM._pool_manager is None:
            BaseLLM._pool_manager = ConnectionPoolManager()
    
    @property
    def pool_manager(self) -> ConnectionPoolManager:
        """Get the shared connection pool manager"""
        if BaseLLM._pool_manager is None:
            BaseLLM._pool_manager = ConnectionPoolManager()
        return BaseLLM._pool_manager
    
    async def get_pooled_session(self, provider: str, **kwargs) -> aiohttp.ClientSession:
        """Get a pooled HTTP session for the provider"""
        return await self.pool_manager.get_session(provider, **kwargs)

    async def respond_back_with_filler(self, messages):
        pass

    async def generate(self, messages, stream=True):
        pass
    
    async def cleanup_connections(self):
        """Cleanup method to close pooled connections"""
        if self.pool_manager:
            await self.pool_manager.close_all()
