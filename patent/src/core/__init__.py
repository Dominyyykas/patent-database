"""
Core modules for the Patent RAG system.

Contains the main patent engine with consolidated chatbot and journalist functionality.
"""

from .patent_engine import (
    PatentChatbot,
    journalist_function,
    detect_journalist_function
)
from .prompts import (
    RAG_SYSTEM_PROMPT,
    RAG_HUMAN_PROMPT,
    IMPACT_ANALYSIS_SYSTEM_PROMPT,
    IMPACT_ANALYSIS_HUMAN_PROMPT,
    ARTICLE_TITLES_SYSTEM_PROMPT,
    ARTICLE_TITLES_HUMAN_PROMPT,
    ARTICLE_ANGLES_SYSTEM_PROMPT,
    ARTICLE_ANGLES_HUMAN_PROMPT
)

__all__ = [
    "PatentChatbot",
    "journalist_function",
    "detect_journalist_function",
    "RAG_SYSTEM_PROMPT",
    "RAG_HUMAN_PROMPT",
    "IMPACT_ANALYSIS_SYSTEM_PROMPT",
    "IMPACT_ANALYSIS_HUMAN_PROMPT",
    "ARTICLE_TITLES_SYSTEM_PROMPT",
    "ARTICLE_TITLES_HUMAN_PROMPT",
    "ARTICLE_ANGLES_SYSTEM_PROMPT",
    "ARTICLE_ANGLES_HUMAN_PROMPT"
]
