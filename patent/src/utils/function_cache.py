import json
import logging
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import sqlite3
import os

logger = logging.getLogger(__name__)

@dataclass
class CachedResult:
    """Data class for cached journalist function results."""
    patent_id: str
    function_type: str  # 'analyze_patent_impact', 'generate_article_titles', 'generate_article_angles'
    result: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if the cached result has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

class JournalistFunctionCache:
    """Cache for journalist function results to avoid repeated expensive LLM calls."""
    
    def __init__(self, cache_file: str = "journalist_cache.db", ttl_hours: int = 24):
        """
        Initialize the cache.
        
        Args:
            cache_file: SQLite database file for persistent storage
            ttl_hours: Time-to-live in hours (None for no expiration)
        """
        self.cache_file = cache_file
        self.ttl_hours = ttl_hours
        self._init_database()
        
    def _init_database(self):
        """Initialize the SQLite database for caching."""
        try:
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            
            # Create cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS journalist_cache (
                    patent_id TEXT,
                    function_type TEXT,
                    result TEXT,
                    created_at TEXT,
                    expires_at TEXT,
                    PRIMARY KEY (patent_id, function_type)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Journalist function cache initialized: {self.cache_file}")
            
        except Exception as e:
            logger.error(f"Error initializing cache database: {e}")
    
    def _get_cache_key(self, patent_id: str, function_type: str) -> str:
        """Generate a cache key for the given patent and function."""
        return f"{patent_id}_{function_type}"
    
    def get(self, patent_id: str, function_type: str) -> Optional[Any]:
        """
        Get a cached result for the given patent and function.
        
        Args:
            patent_id: The patent identifier
            function_type: Type of journalist function
            
        Returns:
            Cached result if found and not expired, None otherwise
        """
        try:
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT result, created_at, expires_at 
                FROM journalist_cache 
                WHERE patent_id = ? AND function_type = ?
            ''', (patent_id, function_type))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                result_json, created_at_str, expires_at_str = row
                
                # Check if expired
                if expires_at_str:
                    expires_at = datetime.fromisoformat(expires_at_str)
                    if datetime.now() > expires_at:
                        logger.info(f"Cache expired for {patent_id} {function_type}")
                        self.delete(patent_id, function_type)
                        return None
                
                # Parse and return result
                result = json.loads(result_json)
                logger.info(f"Cache HIT for {patent_id} {function_type}")
                return result
            else:
                logger.info(f"Cache MISS for {patent_id} {function_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None
    
    def set(self, patent_id: str, function_type: str, result: Any) -> bool:
        """
        Store a result in the cache.
        
        Args:
            patent_id: The patent identifier
            function_type: Type of journalist function
            result: The result to cache
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Calculate expiration time
            expires_at = None
            if self.ttl_hours:
                expires_at = datetime.now() + timedelta(hours=self.ttl_hours)
            
            # Serialize result
            result_json = json.dumps(result, default=str)
            created_at = datetime.now()
            
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            
            # Insert or replace
            cursor.execute('''
                INSERT OR REPLACE INTO journalist_cache 
                (patent_id, function_type, result, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                patent_id, 
                function_type, 
                result_json, 
                created_at.isoformat(),
                expires_at.isoformat() if expires_at else None
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cached result for {patent_id} {function_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            return False
    
    def delete(self, patent_id: str, function_type: str) -> bool:
        """Delete a specific cached result."""
        try:
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM journalist_cache 
                WHERE patent_id = ? AND function_type = ?
            ''', (patent_id, function_type))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted cache for {patent_id} {function_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
    
    def clear_expired(self) -> int:
        """Clear all expired cache entries and return count of cleared items."""
        try:
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            
            # Delete expired entries
            cursor.execute('''
                DELETE FROM journalist_cache 
                WHERE expires_at IS NOT NULL AND expires_at < ?
            ''', (datetime.now().isoformat(),))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"Cleared {deleted_count} expired cache entries")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error clearing expired cache: {e}")
            return 0
    
    def clear_all(self) -> bool:
        """Clear all cached results."""
        try:
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM journalist_cache')
            conn.commit()
            conn.close()
            
            logger.info("Cleared all cache entries")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            
            # Total entries
            cursor.execute('SELECT COUNT(*) FROM journalist_cache')
            total_entries = cursor.fetchone()[0]
            
            # Expired entries
            cursor.execute('''
                SELECT COUNT(*) FROM journalist_cache 
                WHERE expires_at IS NOT NULL AND expires_at < ?
            ''', (datetime.now().isoformat(),))
            expired_entries = cursor.fetchone()[0]
            
            # Function type breakdown
            cursor.execute('''
                SELECT function_type, COUNT(*) 
                FROM journalist_cache 
                GROUP BY function_type
            ''')
            function_breakdown = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                "total_entries": total_entries,
                "expired_entries": expired_entries,
                "valid_entries": total_entries - expired_entries,
                "function_breakdown": function_breakdown
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}

# Global cache instance
journalist_cache = JournalistFunctionCache()

# Global chat cache instance (for chatbot responses)
chat_cache = JournalistFunctionCache(cache_file="chat_cache.db", ttl_hours=12) 