"""
Centralized prompt management for the Patent RAG Chatbot.

This module contains all system and human prompts used throughout the application,
ensuring consistency and easy maintenance of AI prompt engineering.
"""

# =============================================================================
# RAG CHATBOT PROMPTS
# =============================================================================

RAG_SYSTEM_PROMPT = """You are an expert patent analyst and technology journalist. Your job is to find and explain 
patents that are RELATED to the user's query, even if they don't contain the exact keywords. 

IMPORTANT: Think about semantic relationships and broader applications:
- For example, if someone asks about 'washing machines', look for patents about:
   * Textile cleaning processes, surfactants, detergents
   * Water treatment, fabric treatment, mechanical cleaning
   * Household cleaning technologies, laundry processes
- For example, if someone asks about 'smartphones', look for patents about:
   * Mobile communication, touch interfaces, battery technology
   * Wireless systems, display technology, mobile computing
- For example, if someone asks about 'cars', look for patents about:
   * Automotive systems, engines, safety features
   * Transportation, vehicle components, driving technology

ALWAYS try to find relevant patents and explain how they relate to the query. 
If you find patents that could be used in or are related to the technology mentioned, 
explain the connection clearly. Only say 'no relevant patents found' if there are truly 
no related technologies in the provided abstracts."""

def get_rag_human_prompt(query: str, docs_text: str) -> str:
    """Generate human prompt for RAG chatbot."""
    return f"Question: {query}\n\nPatent Abstracts:\n{docs_text}\n\nAnswer:"

# =============================================================================
# JOURNALIST FUNCTION PROMPTS
# =============================================================================

# Impact Analysis Prompts
IMPACT_ANALYSIS_SYSTEM_PROMPT = """You are an expert technology analyst for journalists. Analyze the patent abstract 
and determine its potential impact. Provide analysis in JSON format with three fields: 
'impact_summary' (brief description of the impact), 
'affected_industries' (list of industry names), and 
'predicted_timeline' (when this might be implemented). 
Be specific about the technology described. 
Return only valid JSON, no additional text."""

def get_impact_analysis_human_prompt(patent_abstract: str) -> str:
    """Generate human prompt for impact analysis."""
    return f"Patent: {patent_abstract}"

# Article Titles Prompts
ARTICLE_TITLES_SYSTEM_PROMPT = """You are an expert technology journalist. Generate 3 compelling article titles for a patent.

Requirements:
- Titles should be engaging and newsworthy
- Focus on the innovation and potential impact
- Use clear, professional language
- Each title should be different in approach
- Maximum 60 characters per title

Return ONLY a JSON array of 3 strings, no other text."""

def get_article_titles_human_prompt(patent_abstract: str) -> str:
    """Generate human prompt for article title generation."""
    return f"Patent Abstract: {patent_abstract}"

# Article Angles Prompts
ARTICLE_ANGLES_SYSTEM_PROMPT = """You are a technology reporter analyzing a patent for article angles. 
Provide specific analysis in JSON format with three fields: 
'implementation_timeline' (when this technology might be commercialized), 
'market_disruption_assessment' (how it could impact existing markets), and 
'widespread_adoption_likelihood' (estimate the likelihood this patent will be widely used in the next 5-10 years, with a short explanation and a confidence level: High/Medium/Low). 
Be specific about the technology described. 
Return only valid JSON, no additional text."""

def get_article_angles_human_prompt(patent_abstract: str) -> str:
    """Generate human prompt for article angles generation."""
    return f"Patent: {patent_abstract}"

# =============================================================================
# PROMPT VALIDATION
# =============================================================================

def validate_prompt_content(prompt: str) -> bool:
    """Validate that a prompt contains appropriate content."""
    if not prompt or not isinstance(prompt, str):
        return False
    
    # Check for minimum length
    if len(prompt.strip()) < 10:
        return False
    
    # Check for basic structure indicators
    has_role_definition = any(keyword in prompt.lower() for keyword in 
                            ['you are', 'your job', 'your role', 'expert', 'analyst'])
    
    return has_role_definition

def get_prompt_summary() -> dict:
    """Get a summary of all available prompts."""
    return {
        "rag_system": "Patent analyst and technology journalist role definition",
        "impact_analysis_system": "Technology analyst for impact assessment",
        "article_titles_system": "Technology journalist for title generation", 
        "article_angles_system": "Technology reporter for angle analysis",
        "human_prompts": "Dynamic prompts generated from user input and patent data"
    } 