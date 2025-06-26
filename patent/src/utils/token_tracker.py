import tiktoken
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    """Data class to track token usage for a single API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = "gpt-4o-mini"
    timestamp: datetime = field(default_factory=datetime.now)
    cost_usd: float = 0.0

class TokenTracker:
    """Tracks token usage and calculates costs for OpenAI API calls."""
    
    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "gpt-4o-mini": {
            "input": 0.00015,   # $0.15 per 1M tokens
            "output": 0.0006    # $0.60 per 1M tokens
        }
    }
    
    def __init__(self):
        self.usage_history: list[TokenUsage] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0
        
        # Initialize tokenizers
        self.encoders = {}
        
        # gpt-4o-mini uses cl100k_base encoding
        self.encoders["gpt-4o-mini"] = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str, model: str = "gpt-4o-mini") -> int:
        """Count tokens in text using the appropriate encoder."""
        try:
            encoder = self.encoders.get(model, self.encoders.get("gpt-4o-mini"))
            return len(encoder.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback: rough estimate (1 token ≈ 4 characters for English)
            return len(text) // 4
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini") -> float:
        """Calculate cost in USD for token usage."""
        try:
            pricing = self.PRICING.get(model, self.PRICING["gpt-4o-mini"])
            input_cost = (prompt_tokens / 1000) * pricing["input"]
            output_cost = (completion_tokens / 1000) * pricing["output"]
            return input_cost + output_cost
        except Exception as e:
            logger.error(f"Error calculating cost: {e}")
            return 0.0
    
    def track_usage(self, prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini") -> TokenUsage:
        """Track token usage for a single API call."""
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = self.calculate_cost(prompt_tokens, completion_tokens, model)
        
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model,
            cost_usd=cost_usd
        )
        
        # Update totals
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost_usd += cost_usd
        
        # Add to history
        self.usage_history.append(usage)
        
        logger.info(f"Token usage tracked: {prompt_tokens} prompt, {completion_tokens} completion, "
                   f"{total_tokens} total, ${cost_usd:.6f} cost")
        
        return usage
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session token usage."""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost_usd": self.total_cost_usd,
            "total_cost_formatted": f"${self.total_cost_usd:.4f}",
            "api_calls": len(self.usage_history)
        }
    
    def reset_session(self):
        """Reset session counters (keep history for debugging)."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0
        logger.info("Token tracker session reset")

# Global token tracker instance
token_tracker = TokenTracker() 