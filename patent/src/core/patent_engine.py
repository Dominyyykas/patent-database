import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('logs/patent_engine.log')  # File output
    ]
)
logger = logging.getLogger(__name__)

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from src.utils.token_tracker import token_tracker
from src.utils.rate_limiter import rate_limiter
from src.utils.function_cache import chat_cache, journalist_cache
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


# Load environment variables
load_dotenv()

#CONFIGURATION 
MAX_ABSTRACT_LENGTH = 10000
LLM_MODEL = "gpt-4o-mini"
VECTOR_DB_PATH = "db/chroma_db"
COLLECTION_NAME = "patent_chunks"

class SimpleEmbeddings:
    """Wrapper for SentenceTransformer to work with LangChain's Chroma."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()

def _call_llm(messages) -> str:
    """Unified LLM call with rate limiting and token tracking"""
    logger.info("Starting LLM call...")
    
    # Rate limit
    allowed, error_message = rate_limiter.is_allowed()
    if not allowed:
        logger.warning(f"Rate limit exceeded: {error_message}")
        raise Exception(f"Rate limit exceeded: {error_message}")
    
    logger.info("Rate limit check passed, making API call...")
    
    # API call
    llm = ChatOpenAI(temperature=0, model=LLM_MODEL)
    response = llm.invoke(messages)
    
    # Token tracking
    prompt_text = str(messages)
    prompt_tokens = token_tracker.count_tokens(prompt_text, LLM_MODEL)
    completion_tokens = token_tracker.count_tokens(response.content, LLM_MODEL)
    token_tracker.track_usage(prompt_tokens, completion_tokens, LLM_MODEL)
    
    logger.info(f"LLM call completed. Tokens: {prompt_tokens} prompt, {completion_tokens} completion")
    
    return response.content

def format_patent_context(patents: List[Dict[str, Any]]) -> str:
    """Format patents for LLM context with similarity scores"""
    return "\n\n".join([
        f"Patent {i+1} (Similarity Score: {p.get('similarity_score', 0):.3f}): {p['content']}" 
        for i, p in enumerate(patents)
    ])

# ========== JOURNALIST FUNCTIONS ==========

def detect_journalist_function(query: str) -> Optional[str]:
    """Detect which journalist function to call based on user query."""
    if not query or not isinstance(query, str):
        return None
        
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['impact', 'effect', 'implication', 'market analysis']):
        return 'analyze_patent_impact'
    elif any(word in query_lower for word in ['title', 'headline', 'article title']):
        return 'generate_article_titles'
    elif any(word in query_lower for word in ['angle', 'timeline', 'disruption', 'market disruption']):
        return 'generate_article_angles'
    return None

# Journalist function configurations
JOURNALIST_CONFIGS = {
    'analyze_patent_impact': {
        'system_prompt': IMPACT_ANALYSIS_SYSTEM_PROMPT,
        'human_prompt': IMPACT_ANALYSIS_HUMAN_PROMPT
    },
    'generate_article_titles': {
        'system_prompt': ARTICLE_TITLES_SYSTEM_PROMPT,
        'human_prompt': ARTICLE_TITLES_HUMAN_PROMPT
    },
    'generate_article_angles': {
        'system_prompt': ARTICLE_ANGLES_SYSTEM_PROMPT,
        'human_prompt': ARTICLE_ANGLES_HUMAN_PROMPT
    }
}

def journalist_function(function_name: str, patent_abstract: str, patent_id: str = None) -> Dict[str, Any]:
    """Unified journalist function handler"""
    # Validation
    if not patent_abstract or len(patent_abstract.strip()) < 10:
        return {"error": "Invalid patent abstract provided"}
    
    if function_name not in JOURNALIST_CONFIGS:
        return {"error": f"Unknown function: {function_name}"}
    
    # Check cache
    if patent_id:
        cache_key = f"{patent_id}_{function_name}"
        cached_result = journalist_cache.get(patent_id, function_name)
        if cached_result:
            logger.info(f"Journalist cache HIT for key: {cache_key}")
            return cached_result
        logger.info(f"Journalist cache MISS for key: {cache_key}")

    config = JOURNALIST_CONFIGS[function_name]
    
    try:
        # Prepare messages
        system_prompt = config['system_prompt']
        human_prompt = config['human_prompt'].format(patent_abstract=patent_abstract)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        
        # Call LLM
        response = _call_llm(messages)
        
        # Parse response
        response_str = response.strip()
        if response_str.startswith('```json'):
            response_str = response_str[7:-3]
        elif response_str.startswith('```'):
            response_str = response_str[3:-3]
        
        result = json.loads(response_str)
        
        # Cache result
        if patent_id:
            logger.info(f"Setting journalist cache for key: {cache_key}")
            journalist_cache.set(patent_id, function_name, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Journalist function error: {e}", exc_info=True)
        return {"error": f"Analysis failed: {str(e)}. Please try again."}

# ========== PATENT CHATBOT ==========

class PatentChatbot:
    """Patent chatbot with all core functionality"""
    
    def __init__(self):
        logger.info("Initializing PatentChatbot...")
        self.llm = ChatOpenAI(temperature=0, model=LLM_MODEL)
        self.vector_store = self._setup_vector_store()
    
    def _setup_vector_store(self):
        """Set up ChromaDB vector store"""
        try:
            logger.info(f"Setting up ChromaDB from path: {VECTOR_DB_PATH}")
            embeddings = SimpleEmbeddings()
            return Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME
            )
        except Exception as e:
            logger.error(f"Vector store setup error: {e}", exc_info=True)
            return None
    
    def _retrieve_patents(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve patents from vector store"""
        if not self.vector_store:
            return []
        
        try:
            logger.info(f"Retrieving top {k} patents for query: '{query[:50]}...'")
            results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
            
            patents = []
            
            for doc, score in results:
                patent_info = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': score
                }
                patents.append(patent_info)
            
            logger.info(f"Retrieved {len(patents)} patents.")
            return patents
        except Exception as e:
            logger.error(f"Patent retrieval error: {e}")
            return []
    
    def chat(self, user_input: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Main chat method"""
        # Input validation
        if not user_input or not isinstance(user_input, str):
            return {
                "response": "Please provide a valid query.",
                "error": "Invalid input",
                "patents": [],
                "journalist_analysis": None
            }
        
        user_input_clean = user_input.strip()
        if len(user_input_clean) < 3:
            return {
                "response": "Please provide a more detailed query (at least 3 characters).",
                "error": "Query too short",
                "patents": [],
                "journalist_analysis": None
            }
        
        if len(user_input) > 1000:
            return {
                "response": "Query too long. Please keep it under 1000 characters.",
                "error": "Query too long",
                "patents": [],
                "journalist_analysis": None
            }
        
        # Check for potentially malicious content (basic check)
        suspicious_patterns = ['<script', 'javascript:', 'data:text/html', 'vbscript:']
        if any(pattern in user_input.lower() for pattern in suspicious_patterns):
            return {
                "response": "Query contains potentially unsafe content. Please try a different query.",
                "error": "Suspicious content detected",
                "patents": [],
                "journalist_analysis": None
            }
        
        # Check cache
        cache_key = f"{user_input}_{hash(str(conversation_history))}"
        cached_response = chat_cache.get(cache_key)
        if cached_response:
            return cached_response
        logger.info(f"Chat cache MISS for key: '{user_input_clean}'")
        
        try:
            # Retrieve patents
            relevant_patents = self._retrieve_patents(user_input_clean)
            
            if not relevant_patents:
                return {
                    "response": "I couldn't access the patent database or no patents were found. Please try again later.",
                    "patents": [],
                    "journalist_analysis": None
                }
            
            # Check for journalist function
            detected_function = detect_journalist_function(user_input_clean)
            journalist_result = None
            
            if detected_function:
                patent_abstract = relevant_patents[0]['content']
                patent_id = relevant_patents[0]['metadata'].get('patent_id', 'unknown')
                journalist_result = journalist_function(detected_function, patent_abstract, patent_id)
            
            # Generate main response
            docs_text = format_patent_context(relevant_patents)
            
            messages = [
                SystemMessage(content=RAG_SYSTEM_PROMPT), 
                HumanMessage(content=RAG_HUMAN_PROMPT.format(query=user_input_clean, docs_text=docs_text))
            ]
            
            response = _call_llm(messages)
            
            result = {
                "response": response,
                "patents": relevant_patents,
                "journalist_analysis": journalist_result
            }
            
            # Cache result
            logger.info(f"Setting chat cache for key: '{user_input_clean}'")
            chat_cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            return {
                "response": "I encountered an error while processing your request. Please try again.",
                "error": str(e),
                "patents": [],
                "journalist_analysis": None
            }
    
    def get_token_usage(self) -> Dict[str, Any]:
        """Get token usage statistics"""
        return token_tracker.get_session_summary()
    
    def reset_token_tracker(self):
        """Reset token tracking"""
        token_tracker.reset_session()

# Global chatbot instance
chatbot = PatentChatbot() 