import os
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
# NOTE: Chroma import shows deprecation warning but works fine
# Future TODO: Migrate to langchain-chroma when version conflicts are resolved
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from src.utils.token_tracker import token_tracker
from src.utils.rate_limiter import rate_limiter
from src.utils.function_cache import chat_cache
from src.core.journalist_functions import (
    detect_journalist_function,
    analyze_patent_impact,
    generate_article_titles,
    generate_article_angles
)
from .prompts import (
    RAG_SYSTEM_PROMPT,
    get_rag_human_prompt
)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddings:
    """Wrapper for SentenceTransformer to work with LangChain's Chroma."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search documents."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        embedding = self.model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()

class PatentChatbot:
    """
    A chatbot for patent analysis with journalist-specific functions.
    """
    
    def __init__(self):
        """Initialize the chatbot with necessary components."""
        # Initialize token tracker (use global instance)
        self.token_tracker = token_tracker
        self.rate_limiter = rate_limiter
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        
        # Initialize vector store
        self._setup_vector_store()
        
        logger.info("PatentChatbot initialized successfully")
    
    def _call_llm_with_tracking(self, messages):
        """Call LLM with token tracking and rate limiting."""
        try:
            # Check rate limit for this specific API call
            allowed, error_message = self.rate_limiter.is_allowed()
            logger.info(f"Rate limiter called - allowed: {allowed}, current stats: {self.rate_limiter.get_stats()}")
            if not allowed:
                raise Exception(f"Rate limit exceeded: {error_message}")
            
            response = self.llm.invoke(messages)
            
            # Count tokens properly
            prompt_text = str(messages)
            prompt_tokens = self.token_tracker.count_tokens(prompt_text, "gpt-4o-mini")
            completion_tokens = self.token_tracker.count_tokens(response.content, "gpt-4o-mini")
            
            # Track usage
            self.token_tracker.track_usage(prompt_tokens, completion_tokens, "gpt-4o-mini")
            
            return response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _setup_vector_store(self):
        """Set up the ChromaDB vector store for patent retrieval."""
        try:
            embeddings = SentenceTransformerEmbeddings()
            self.vector_store = Chroma(
                persist_directory="db/chroma_db",
                embedding_function=embeddings,
                collection_name="patent_chunks"
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up vector store: {e}")
            self.vector_store = None
    
    def _retrieve_relevant_patents(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve patents from the vector store without score filtering.
        Returns the most similar patents regardless of score - let the LLM determine relevance.
        
        Args:
            query: User query
            k: Number of patents to retrieve
            
        Returns:
            List of patent documents (always returns results if available)
        """
        if not self.vector_store:
            logger.warning("Vector store not available, returning empty results")
            return []

        try:
            results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
            patents = []
            
            logger.info(f"Found {len(results)} results for query: '{query}'")
            
            # Return all results without score filtering - let LLM determine relevance
            for doc, score in results:
                logger.info(f"Including patent with similarity score: {score:.3f}")
                patent_info = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': score  # Keep score for LLM context
                }
                patents.append(patent_info)
            
            logger.info(f"Retrieved {len(patents)} patents (no filtering applied)")
            return patents
        except Exception as e:
            logger.error(f"Error retrieving patents: {e}")
            return []
    
    def _handle_journalist_function(self, query: str, patent_abstract: str, patent_id: str = None) -> Dict[str, Any]:
        """
        Handle journalist-specific function calls.
        
        Args:
            query: User query
            patent_abstract: Patent abstract text
            patent_id: Patent identifier
            
        Returns:
            Function result
        """
        detected_function = detect_journalist_function(query)
        
        if detected_function == 'analyze_patent_impact':
            return analyze_patent_impact(patent_abstract, patent_id)
        elif detected_function == 'generate_article_titles':
            return generate_article_titles(patent_abstract, patent_id)
        elif detected_function == 'generate_article_angles':
            return generate_article_angles(patent_abstract, patent_id)
        else:
            return {"error": "No journalist function detected"}
    
    def chat(self, user_input: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Main chat method for handling user queries.
        
        Args:
            user_input: User's message
            conversation_history: Previous conversation messages
            
        Returns:
            Chatbot response with relevant information
        """
        logger.info(f"Processing user input: {user_input[:100]}...")
        
        # Check cache
        cache_key = f"{user_input}_{hash(str(conversation_history))}"
        cached_response = chat_cache.get(cache_key, "chat")
        if cached_response:
            logger.info("Using cached response")
            return cached_response
        
        try:
            # Retrieve relevant patents (now always returns results if available)
            relevant_patents = self._retrieve_relevant_patents(user_input)
            
            # We now always process patents if any exist in the database
            # The LLM will explain relevance and probability
            if not relevant_patents:
                return {
                    "response": "I couldn't access the patent database or no patents were found. Please try again later.",
                    "patents": [],
                    "journalist_analysis": None
                }
            
            # Check if this is a journalist function query
            detected_function = detect_journalist_function(user_input)
            
            if detected_function and relevant_patents:
                # Use the first (most relevant) patent for journalist functions
                patent_abstract = relevant_patents[0]['content']
                patent_id = relevant_patents[0]['metadata'].get('patent_id', 'unknown')
                
                journalist_result = self._handle_journalist_function(
                    user_input, patent_abstract, patent_id
                )
                
                # Generate general response
                system_prompt = RAG_SYSTEM_PROMPT
                # Format patents for the prompt with similarity scores
                docs_text = "\n\n".join([
                    f"Patent {i+1} (Similarity Score: {patent.get('similarity_score', 0):.3f}): {patent['content']}" 
                    for i, patent in enumerate(relevant_patents)
                ])
                human_prompt = get_rag_human_prompt(user_input, docs_text)
                
                messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
                general_response = self._call_llm_with_tracking(messages)
                
                response_data = {
                    "response": general_response,
                    "patents": relevant_patents,
                    "journalist_analysis": journalist_result
                }
            else:
                # Standard chat response
                system_prompt = RAG_SYSTEM_PROMPT
                # Format patents for the prompt with similarity scores
                docs_text = "\n\n".join([
                    f"Patent {i+1} (Similarity Score: {patent.get('similarity_score', 0):.3f}): {patent['content']}" 
                    for i, patent in enumerate(relevant_patents)
                ])
                human_prompt = get_rag_human_prompt(user_input, docs_text)
                
                messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
                response = self._call_llm_with_tracking(messages)
                
                response_data = {
                    "response": response,
                    "patents": relevant_patents,
                    "journalist_analysis": None
                }
            
            # Cache the response
            chat_cache.set(cache_key, "chat", response_data)
            
            logger.info("Chat response generated successfully")
            return response_data
            
        except Exception as e:
            logger.error(f"Error in chat method: {e}")
            return {
                "response": "I encountered an error while processing your request. Please try again.",
                "error": str(e),
                "patents": [],
                "journalist_analysis": None
            }
    
    def get_token_usage(self) -> Dict[str, Any]:
        """Get current token usage statistics."""
        return self.token_tracker.get_session_summary()
    
    def reset_token_tracker(self):
        """Reset token usage tracking."""
        self.token_tracker.reset_session()
        logger.info("Token tracker reset")

# Global chatbot instance
chatbot = PatentChatbot()