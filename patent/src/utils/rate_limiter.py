import time
import threading
from collections import defaultdict, deque
from typing import Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)

# Inline rate limiting settings
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "20"))  # Back to normal
MAX_REQUESTS_PER_HOUR = int(os.getenv("MAX_REQUESTS_PER_HOUR", "500"))     # Back to normal

class RateLimiter:
    """Thread-safe rate limiter with per-client tracking."""
    
    def __init__(self):
        self.client_requests = defaultdict(lambda: {"minute": deque(), "hour": deque()})
        self.lock = threading.Lock()
        self.max_requests_per_minute = MAX_REQUESTS_PER_MINUTE
        self.max_requests_per_hour = MAX_REQUESTS_PER_HOUR
    
    def is_allowed(self, client_id: str = "default") -> tuple[bool, Optional[str]]:
        """
        Check if a request is allowed based on rate limits.
        
        Args:
            client_id: Identifier for the client (IP, user ID, etc.)
            
        Returns:
            Tuple of (is_allowed, error_message)
        """
        current_time = time.time()
        
        with self.lock:
            # Clean old requests
            self._clean_old_requests(client_id, current_time)
            
            # Get current usage
            minute_requests = len(self.client_requests[client_id]["minute"])
            hour_requests = len(self.client_requests[client_id]["hour"])
            
            logger.info(f"Rate limit check for {client_id}: {minute_requests}/{self.max_requests_per_minute} per minute, {hour_requests}/{self.max_requests_per_hour} per hour")
            
            # Check minute limit
            if minute_requests >= self.max_requests_per_minute:
                wait_time = 60 - (current_time - self.client_requests[client_id]["minute"][0])
                logger.warning(f"Rate limit exceeded for {client_id}: {minute_requests} requests in current minute. Try again in {wait_time:.1f} seconds.")
                return False, f"Rate limit exceeded. Try again in {wait_time:.1f} seconds."
            
            # Check hour limit
            if hour_requests >= self.max_requests_per_hour:
                oldest_request = self.client_requests[client_id]["hour"][0]
                wait_time = 3600 - (current_time - oldest_request)
                logger.warning(f"Hourly rate limit exceeded for {client_id}: {hour_requests} requests in current hour. Try again in {wait_time/60:.1f} minutes.")
                return False, f"Hourly rate limit exceeded. Try again in {wait_time/60:.1f} minutes."
            
            # Record the request
            self.client_requests[client_id]["minute"].append(current_time)
            self.client_requests[client_id]["hour"].append(current_time)
            
            logger.info(f"Request allowed for {client_id}. "
                        f"Minute: {minute_requests + 1}/{self.max_requests_per_minute}, "
                        f"Hour: {hour_requests + 1}/{self.max_requests_per_hour}")
            
            return True, None
    
    def _clean_old_requests(self, client_id: str, current_time: float):
        """Remove requests older than the time windows."""
        # Clean minute window
        minute_cutoff = current_time - 60
        while (self.client_requests[client_id]["minute"] and 
               self.client_requests[client_id]["minute"][0] < minute_cutoff):
            self.client_requests[client_id]["minute"].popleft()
        
        # Clean hour window
        hour_cutoff = current_time - 3600
        while (self.client_requests[client_id]["hour"] and 
               self.client_requests[client_id]["hour"][0] < hour_cutoff):
            self.client_requests[client_id]["hour"].popleft()
    
    def get_stats(self, client_id: str = "default") -> Dict[str, int]:
        """Get current usage statistics for a client."""
        current_time = time.time()
        
        with self.lock:
            self._clean_old_requests(client_id, current_time)
            
            return {
                "requests_this_minute": len(self.client_requests[client_id]["minute"]),
                "requests_this_hour": len(self.client_requests[client_id]["hour"]),
                "minute_limit": self.max_requests_per_minute,
                "hour_limit": self.max_requests_per_hour,
                "minute_remaining": self.max_requests_per_minute - len(self.client_requests[client_id]["minute"]),
                "hour_remaining": self.max_requests_per_hour - len(self.client_requests[client_id]["hour"])
            }
    
    def reset_client(self, client_id: str):
        """Reset rate limits for a specific client (admin function)."""
        with self.lock:
            if client_id in self.client_requests:
                del self.client_requests[client_id]
            logger.info(f"Rate limits reset for client: {client_id}")

# Global rate limiter instance
rate_limiter = RateLimiter() 