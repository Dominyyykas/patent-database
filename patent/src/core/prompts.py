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

IMPORTANT INSTRUCTIONS:
1. You will be given patents that were retrieved by similarity search, but they may not seem directly related at first glance.
2. Your job is to find creative connections and explain HOW and WHY each patent could be relevant to the user's query.
3. For each patent you discuss, you MUST include:
   - A relevance explanation: How this patent relates to the user's query
   - A probability assessment: How likely this patent technology could be used in the context of the user's query (High/Medium/Low probability)
   - Specific connections: What aspects of the patent make it applicable

SEMANTIC RELATIONSHIP EXAMPLES:
- Query about 'kitchen fridges' → Look for patents about: refrigeration systems, cooling technology, food preservation, temperature control, insulation materials, compressors, heat exchangers
- Query about 'cars' → Look for patents about: automotive systems, engines, safety features, transportation, vehicle components, driving technology, fuel systems, braking systems
- Query about 'smartphones' → Look for patents about: mobile communication, touch interfaces, battery technology, wireless systems, display technology, mobile computing, sensors

RESPONSE FORMAT:
For each patent, structure your response as:
"**Patent [X]**: [Brief description]
- **Relevance**: [Explain how this patent relates to the query]  
- **Probability**: [High/Medium/Low] - [Explain why this probability]
- **Application**: [Specific ways this could be used in the context]"

NEVER say 'no relevant patents found' - always find connections and explain them, even if they require creative thinking about applications."""

def get_rag_human_prompt(query: str, docs_text: str) -> str:
    """Generate human prompt for RAG chatbot with similarity context."""
    return f"""User Query: {query}

Retrieved Patent Abstracts (with similarity scores):
{docs_text}

Please analyze these patents and explain their relevance to the user's query. Even if the similarity scores are low, find creative connections and explain how these patents could be related to or used in the context of the query. Remember to include relevance, probability assessment, and specific applications for each patent."""

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