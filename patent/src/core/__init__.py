"""
Core modules for the Patent RAG system.

Contains the main chatbot logic, journalist functions, and LLM prompts.
"""

from .chatbot import PatentChatbot
from .journalist_functions import (
    analyze_patent_impact,
    generate_article_titles,
    generate_article_angles
)
from .prompts import (
    RAG_SYSTEM_PROMPT,
    get_rag_human_prompt,
    IMPACT_ANALYSIS_SYSTEM_PROMPT,
    ARTICLE_TITLES_SYSTEM_PROMPT,
    ARTICLE_ANGLES_SYSTEM_PROMPT
)

__all__ = [
    "PatentChatbot",
    "analyze_patent_impact",
    "generate_article_titles", 
    "generate_article_angles",
    "RAG_SYSTEM_PROMPT",
    "get_rag_human_prompt",
    "IMPACT_ANALYSIS_SYSTEM_PROMPT",
    "ARTICLE_TITLES_SYSTEM_PROMPT",
    "ARTICLE_ANGLES_SYSTEM_PROMPT"
]
