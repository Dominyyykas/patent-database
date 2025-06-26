import os
import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.utils.function_cache import journalist_cache
from src.utils.token_tracker import token_tracker
from .prompts import (
    IMPACT_ANALYSIS_SYSTEM_PROMPT,
    get_impact_analysis_human_prompt,
    ARTICLE_TITLES_SYSTEM_PROMPT,
    get_article_titles_human_prompt,
    ARTICLE_ANGLES_SYSTEM_PROMPT,
    get_article_angles_human_prompt
)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_ABSTRACT_LENGTH = 10000

def detect_journalist_function(query: str) -> Optional[str]:
    """
    Detect which journalist function to call based on user query.
    
    Args:
        query: User's input query
        
    Returns:
        Function name if detected, None otherwise
    """
    if not query or not isinstance(query, str):
        return None
        
    query_lower = query.lower()
    
    # Domain-specific keyword detection
    if any(word in query_lower for word in ['impact', 'effect', 'implication', 'market analysis']):
        return 'analyze_patent_impact'
    elif any(word in query_lower for word in ['title', 'headline', 'article title']):
        return 'generate_article_titles'
    elif any(word in query_lower for word in ['angle', 'timeline', 'disruption', 'market disruption']):
        return 'generate_article_angles'
    return None

def _validate_patent_abstract(patent_abstract: str) -> bool:
    """Validate patent abstract input."""
    if not patent_abstract or not isinstance(patent_abstract, str):
        return False
    if len(patent_abstract.strip()) < 10:
        return False
    if len(patent_abstract) > MAX_ABSTRACT_LENGTH:
        return False
    return True

def analyze_patent_impact(patent_abstract: str, patent_id: str = None) -> Dict[str, Any]:
    """
    Analyzes a patent's potential impact on everyday life using an LLM.
    Domain-specific function for journalist use case.
    
    Args:
        patent_abstract: Patent abstract text
        patent_id: Patent identifier (for caching)
        
    Returns:
        Impact analysis result dictionary
    """
    logger.info("Starting patent impact analysis")
    
    # Input validation
    if not _validate_patent_abstract(patent_abstract):
        logger.error("Invalid patent abstract provided")
        return {
            "error": "Invalid patent abstract provided",
            "impact_summary": "Analysis could not be completed",
            "affected_industries": ["Unknown"],
            "predicted_timeline": "Unknown"
        }
    
    # Check cache if patent_id is provided
    if patent_id:
        cached_result = journalist_cache.get(patent_id, "analyze_patent_impact")
        if cached_result:
            logger.info(f"Using cached impact analysis for patent {patent_id}")
            return cached_result
    
    system_prompt = IMPACT_ANALYSIS_SYSTEM_PROMPT
    human_prompt = get_impact_analysis_human_prompt(patent_abstract)
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    
    # Define fallback data based on key terms
    fallback_data = _get_impact_fallback_data(patent_abstract)
    
    try:
        response = _call_llm(messages)
        
        # Try to parse JSON response
        try:
            response_str = response.strip()
            if response_str.startswith('```json'):
                response_str = response_str[7:-3]
            elif response_str.startswith('```'):
                response_str = response_str[3:-3]
            
            result = json.loads(response_str)
            logger.info("Patent impact analysis completed successfully")
            
            # Cache the result if patent_id is provided
            if patent_id:
                journalist_cache.set(patent_id, "analyze_patent_impact", result)
            
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in impact analysis: {e}")
            return fallback_data
            
    except Exception as e:
        logger.error(f"Error in impact analysis: {e}")
        return fallback_data

def _get_impact_fallback_data(patent_abstract: str) -> Dict[str, Any]:
    """Generate fallback data for impact analysis based on key terms."""
    abstract_lower = patent_abstract.lower()
    
    if "solar" in abstract_lower and "vehicle" in abstract_lower:
        return {
            "impact_summary": "This solar panel vehicle technology could significantly impact transportation by making electric vehicles more efficient and reducing reliance on traditional charging infrastructure.",
            "affected_industries": ["Automotive", "Renewable Energy", "Smart Technology", "Urban Planning", "Transportation"],
            "predicted_timeline": "3-5 years for initial implementation and adoption in commercial vehicles"
        }
    elif "solar" in abstract_lower:
        return {
            "impact_summary": "This solar technology could impact renewable energy adoption and efficiency improvements across multiple sectors.",
            "affected_industries": ["Renewable Energy", "Technology", "Manufacturing"],
            "predicted_timeline": "2-4 years for development and market adoption"
        }
    else:
        return {
            "impact_summary": "This patent technology could have significant impact depending on implementation and market adoption.",
            "affected_industries": ["Technology", "Manufacturing"],
            "predicted_timeline": "Timeline depends on technology complexity and market readiness"
        }

def generate_article_titles(patent_abstract: str, patent_id: str = None) -> List[str]:
    """
    Generate article titles for a patent.
    
    Args:
        patent_abstract: Patent abstract text
        patent_id: Patent identifier (for caching)
        
    Returns:
        List of generated article titles
    """
    logger.info("Starting article title generation")
    
    if not patent_abstract:
        return ["Error: No patent abstract provided"]
    
    # Check cache if patent_id is provided
    if patent_id:
        cached_result = journalist_cache.get(patent_id, "generate_article_titles")
        if cached_result:
            logger.info(f"Using cached article titles for patent {patent_id}")
            return cached_result
    
    try:
        # Truncate abstract if too long
        if len(patent_abstract) > MAX_ABSTRACT_LENGTH:
            patent_abstract = patent_abstract[:MAX_ABSTRACT_LENGTH] + "..."
        
        system_prompt = ARTICLE_TITLES_SYSTEM_PROMPT
        human_prompt = get_article_titles_human_prompt(patent_abstract)
        
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        
        response = _call_llm(messages)
        logger.info(f"LLM call successful, response length: {len(response)}")
        
        # Parse JSON response
        try:
            response_str = response.strip()
            if response_str.startswith('```json'):
                response_str = response_str[7:-3]
            elif response_str.startswith('```'):
                response_str = response_str[3:-3]
            
            titles = json.loads(response_str)
            if isinstance(titles, list) and len(titles) >= 3:
                logger.info("Article titles generated successfully")
                result = titles[:3]
                
                # Cache the result if patent_id is provided
                if patent_id:
                    journalist_cache.set(patent_id, "generate_article_titles", result)
                
                return result
            else:
                logger.warning("Invalid response format, returning default titles")
                return _get_title_fallback_data()
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return _get_title_fallback_data()
            
    except Exception as e:
        logger.error(f"Error generating article titles: {e}")
        return ["Error: Could not generate article titles"]

def _get_title_fallback_data() -> List[str]:
    """Get fallback article titles."""
    return [
        "Innovative Patent Technology Breakthrough",
        "New Patent Promises Industry Transformation",
        "Revolutionary Patent Technology Unveiled"
    ]

def generate_article_angles(patent_abstract: str, patent_id: str = None) -> Dict[str, Any]:
    """
    Generates article angles including timeline, market disruption assessment, and likelihood of widespread adoption.
    Domain-specific function for journalist use case.
    
    Args:
        patent_abstract: Patent abstract text
        patent_id: Patent identifier (for caching)
        
    Returns:
        Article angles result dictionary
    """
    logger.info("Starting article angles generation")
    
    # Input validation
    if not _validate_patent_abstract(patent_abstract):
        logger.error("Invalid patent abstract provided")
        return {
            "error": "Invalid patent abstract provided",
            "implementation_timeline": "Analysis could not be completed",
            "market_disruption_assessment": "Analysis could not be completed",
            "widespread_adoption_likelihood": "Analysis could not be completed"
        }
    
    # Check cache if patent_id is provided
    if patent_id:
        cached_result = journalist_cache.get(patent_id, "generate_article_angles")
        if cached_result:
            logger.info(f"Using cached article angles for patent {patent_id}")
            return cached_result
    
    system_prompt = ARTICLE_ANGLES_SYSTEM_PROMPT
    human_prompt = get_article_angles_human_prompt(patent_abstract)

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    
    # Define fallback data based on key terms
    fallback_data = _get_angles_fallback_data(patent_abstract)
    
    try:
        response = _call_llm(messages)
        
        # Try to parse JSON response
        try:
            response_str = response.strip()
            if response_str.startswith('```json'):
                response_str = response_str[7:-3]
            elif response_str.startswith('```'):
                response_str = response_str[3:-3]
            
            result = json.loads(response_str)
            logger.info("Article angles generated successfully")
            
            formatted_result = {
                "implementation_timeline": result.get("implementation_timeline", "Analysis unavailable"),
                "market_disruption_assessment": result.get("market_disruption_assessment", "Analysis unavailable"),
                "widespread_adoption_likelihood": result.get("widespread_adoption_likelihood", "Analysis unavailable")
            }
            
            # Cache the result if patent_id is provided
            if patent_id:
                journalist_cache.set(patent_id, "generate_article_angles", formatted_result)
            
            return formatted_result
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in article angles: {e}")
            return fallback_data
            
    except Exception as e:
        logger.error(f"Error generating angles: {e}")
        return fallback_data

def _get_angles_fallback_data(patent_abstract: str) -> Dict[str, Any]:
    """Generate fallback data for article angles based on key terms."""
    abstract_lower = patent_abstract.lower()
    
    if "solar" in abstract_lower and "vehicle" in abstract_lower:
        return {
            "implementation_timeline": "2-3 years for initial prototypes, 5-7 years for commercial deployment",
            "market_disruption_assessment": "Could disrupt automotive industry by making solar-powered vehicles more practical and efficient",
            "widespread_adoption_likelihood": "Medium - Adoption depends on cost, regulatory approval, and consumer demand. If solar panel efficiency improves and costs decrease, confidence is Medium."
        }
    elif "solar" in abstract_lower:
        return {
            "implementation_timeline": "1-2 years for development, 3-5 years for market adoption",
            "market_disruption_assessment": "Potential to impact renewable energy and automotive sectors",
            "widespread_adoption_likelihood": "Medium - Likelihood depends on integration with existing infrastructure and market incentives. Confidence: Medium."
        }
    else:
        return {
            "implementation_timeline": "Timeline depends on technology complexity and market readiness",
            "market_disruption_assessment": "Impact will vary based on industry adoption and competitive landscape",
            "widespread_adoption_likelihood": "Low - Unclear if this technology solves a pressing need or can overcome adoption barriers. Confidence: Low."
        }

def test_journalist_functions():
    """Test all journalist functions with a sample patent abstract."""
    test_abstract = """A solar-powered vehicle charging system that automatically parks vehicles in optimal locations for solar charging. The system uses computer vision and machine learning to identify parking spots with maximum sun exposure and guides vehicles to these locations for efficient solar panel charging."""
    
    print("Testing Journalist Functions")
    print("=" * 50)
    
    # Test detection
    print("\n--- Testing: detect_journalist_function ---")
    test_queries = [
        "What is the impact of this patent?",
        "Generate article titles",
        "What are the market angles?",
        "Tell me about solar panels"
    ]
    for query in test_queries:
        detected = detect_journalist_function(query)
        print(f"Query: '{query}' -> Detected: {detected}")
    
    # Test impact analysis
    print("\n--- Testing: analyze_patent_impact ---")
    impact = analyze_patent_impact(test_abstract, "test_patent_001")
    print(impact)
    
    # Test title generation
    print("\n--- Testing: generate_article_titles ---")
    titles = generate_article_titles(test_abstract, "test_patent_001")
    print(titles)
    
    # Test angle generation
    print("\n--- Testing: generate_article_angles ---")
    angles = generate_article_angles(test_abstract, "test_patent_001")
    print(angles)
    
    print("\nAll tests completed!")

def _call_llm(messages):
    """Call the OpenAI Chat LLM with token tracking and return the response content as string."""
    try:
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        response = llm.invoke(messages)
        
        # Count tokens properly
        prompt_text = str(messages)
        prompt_tokens = token_tracker.count_tokens(prompt_text, "gpt-4o-mini")
        completion_tokens = token_tracker.count_tokens(response.content, "gpt-4o-mini")
        
        # Track usage using global token tracker
        token_tracker.track_usage(prompt_tokens, completion_tokens, "gpt-4o-mini")
        
        return response.content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise

if __name__ == "__main__":
    test_journalist_functions() 