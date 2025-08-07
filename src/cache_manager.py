"""Redis-based cache manager for repository validation and documentation caching."""

import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from urllib.parse import urlparse

import redis.asyncio as redis
from pydantic import BaseModel

from .config import Settings

logger = logging.getLogger(__name__)


class RepositoryInfo(BaseModel):
    """Repository information stored in cache."""
    
    url: str
    last_commit_sha: str
    last_commit_date: str
    last_documentation_generated: str
    documentation_hash: str
    analysis_summary: Dict[str, Any]


class CacheManager:
    """Redis-based cache manager for repository validation."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.redis_config = settings.redis
        self.redis: Optional[redis.Redis] = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to Redis server."""
        if not self.redis_config.enabled:
            logger.info("Redis caching is disabled")
            return False
            
        try:
            redis_url = self._build_redis_url()
            self.redis = await redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            # Test connection
            await self.redis.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.redis_config.host}:{self.redis_config.port}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Redis server."""
        if self.redis:
            await self.redis.close()
            self._connected = False
            logger.info("Disconnected from Redis")
    
    def _build_redis_url(self) -> str:
        """Build Redis connection URL."""
        if self.redis_config.password:
            return f"redis://:{self.redis_config.password}@{self.redis_config.host}:{self.redis_config.port}/{self.redis_config.db}"
        else:
            return f"redis://{self.redis_config.host}:{self.redis_config.port}/{self.redis_config.db}"
    
    def _get_repo_key(self, repo_url: str) -> str:
        """Generate Redis key for repository."""
        url_hash = hashlib.md5(repo_url.encode()).hexdigest()
        return f"repo:{url_hash}"
    
    def _get_doc_key(self, repo_url: str) -> str:
        """Generate Redis key for documentation."""
        url_hash = hashlib.md5(repo_url.encode()).hexdigest()
        return f"doc:{url_hash}"
    
    async def get_repository_info(self, repo_url: str) -> Optional[RepositoryInfo]:
        """Get cached repository information."""
        if not self._connected:
            return None
            
        try:
            key = self._get_repo_key(repo_url)
            data = await self.redis.get(key)
            
            if data:
                repo_data = json.loads(data)
                return RepositoryInfo(**repo_data)
                
        except Exception as e:
            logger.error(f"Failed to get repository info from cache: {e}")
        
        return None
    
    async def set_repository_info(
        self, 
        repo_url: str, 
        last_commit_sha: str, 
        last_commit_date: str,
        documentation_hash: str,
        analysis_summary: Dict[str, Any]
    ) -> bool:
        """Store repository information in cache."""
        if not self._connected:
            return False
            
        try:
            repo_info = RepositoryInfo(
                url=repo_url,
                last_commit_sha=last_commit_sha,
                last_commit_date=last_commit_date,
                last_documentation_generated=datetime.utcnow().isoformat(),
                documentation_hash=documentation_hash,
                analysis_summary=analysis_summary
            )
            
            key = self._get_repo_key(repo_url)
            await self.redis.setex(
                key, 
                self.redis_config.ttl, 
                json.dumps(repo_info.model_dump())
            )
            
            logger.debug(f"Stored repository info in cache: {repo_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store repository info in cache: {e}")
            return False
    
    async def get_cached_documentation(self, repo_url: str) -> Optional[str]:
        """Get cached documentation."""
        if not self._connected:
            return None
            
        try:
            key = self._get_doc_key(repo_url)
            return await self.redis.get(key)
            
        except Exception as e:
            logger.error(f"Failed to get documentation from cache: {e}")
            return None
    
    async def set_cached_documentation(self, repo_url: str, documentation: str) -> bool:
        """Store documentation in cache."""
        if not self._connected:
            return False
            
        try:
            key = self._get_doc_key(repo_url)
            await self.redis.setex(
                key, 
                self.redis_config.ttl, 
                documentation
            )
            
            logger.debug(f"Stored documentation in cache: {repo_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store documentation in cache: {e}")
            return False
    
    async def invalidate_repository(self, repo_url: str) -> bool:
        """Invalidate cached data for a repository."""
        if not self._connected:
            return False
            
        try:
            repo_key = self._get_repo_key(repo_url)
            doc_key = self._get_doc_key(repo_url)
            
            await self.redis.delete(repo_key, doc_key)
            logger.debug(f"Invalidated cache for repository: {repo_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")
            return False
    
    def generate_documentation_hash(self, documentation: str) -> str:
        """Generate hash for documentation content."""
        return hashlib.sha256(documentation.encode()).hexdigest()
    
    async def should_regenerate_documentation(
        self, 
        repo_url: str, 
        current_commit_sha: str,
        current_commit_date: str
    ) -> bool:
        """
        Check if documentation should be regenerated.
        
        Returns True if:
        - No cached data exists
        - Repository has new commits
        - Cache has expired
        """
        cached_info = await self.get_repository_info(repo_url)
        
        if not cached_info:
            logger.info(f"No cached data found for {repo_url}, regeneration needed")
            return True
        
        # Check if repository has new commits
        if cached_info.last_commit_sha != current_commit_sha:
            logger.info(f"New commits detected for {repo_url}, regeneration needed")
            return True
        
        # Check if cache has expired (additional safety check)
        try:
            last_generated = datetime.fromisoformat(cached_info.last_documentation_generated)
            if datetime.utcnow() - last_generated > timedelta(seconds=self.redis_config.ttl):
                logger.info(f"Cache expired for {repo_url}, regeneration needed")
                return True
        except Exception as e:
            logger.warning(f"Error parsing last generation time: {e}")
            return True
        
        logger.info(f"Documentation for {repo_url} is up to date")
        return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._connected:
            return {"status": "disconnected"}
            
        try:
            info = await self.redis.info()
            return {
                "status": "connected",
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_keys": await self.redis.dbsize(),
                "config": {
                    "host": self.redis_config.host,
                    "port": self.redis_config.port,
                    "db": self.redis_config.db,
                    "ttl": self.redis_config.ttl
                }
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def clear_cache(self) -> bool:
        """Clear all cached data."""
        if not self._connected:
            return False
            
        try:
            await self.redis.flushdb()
            logger.info("Cleared all cached data")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False


async def create_cache_manager(settings: Settings) -> CacheManager:
    """Factory function to create and connect cache manager."""
    cache_manager = CacheManager(settings)
    await cache_manager.connect()
    return cache_manager