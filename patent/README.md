# Patent RAG Chatbot - AI Engineering Project

A specialized Retrieval-Augmented Generation (RAG) chatbot for patent analysis, demonstrating advanced LangChain implementation with function calling capabilities. 

## Project Overview

This project implements a domain-specific chatbot focused on patent analysis using the [BIGPATENT dataset](https://huggingface.co/datasets/big_patent) from Hugging Face (1.4M+ patents). The system demonstrates uses RAG techniques, function calling, and practical AI application development.

### Course Requirements Met

✅ **Core RAG Implementation**
- Knowledge base: BIGPATENT dataset with semantic embeddings
- Document retrieval: ChromaDB vector database with similarity search
- Chunking strategies: 800-char semantic chunks with overlap

✅ **Function Calling (3+ Functions)**
- **Impact Analysis**: Assess patent market impact and commercial potential
- **Title Generation**: Create article titles based on patent content
- **Topic Angles**: Generate journalistic angles for patent coverage

✅ **Domain Specialization**
- **Domain**: Patent analysis and technology research
- **Focused Knowledge Base**: Technology-specific patent embeddings
- **Domain-Specific Prompts**: Tailored for patent analysis tasks
- **Security**: Rate limiting and API key management

✅ **Technical Implementation**
- **LangChain Integration**: OpenAI API with proper error handling
- **Logging & Monitoring**: Comprehensive token tracking and cost monitoring
- **Input Validation**: Query sanitization and validation
- **Rate Limiting**: 20 req/min, 500 req/hour protection

✅ **User Interface**
- **Streamlit Interface**: Intuitive chat interface with context display
- **Progress Indicators**: Loading states for long operations
- **Function Results**: Structured display of analysis outputs
- **Source Citations**: Patent references with similarity scores

## Technical Architecture

### RAG Pipeline
```
Query → SentenceTransformer (all-MiniLM-L6-v2) → 384-dim vector → ChromaDB similarity search → Top 3 patents → GPT-4o-mini response
```

### Key Components
- **Vector Database**: ChromaDB with 1.4M+ patent chunks
- **Embeddings**: SentenceTransformers for semantic search
- **LLM**: OpenAI gpt-4o-mini for response generation
- **Caching**: SQLite-based function cache with TTL
- **Rate Limiting**: Thread-safe API protection

## Optional Tasks Implemented

### Medium Difficulty
✅ **Advanced Caching Strategies**: Multi-level caching with TTL and cache invalidation
✅ **Token Usage & Cost Tracking**: Real-time monitoring with tiktoken precision
✅ **Visualization of Function Call Results**: Structured JSON output display

### Hard Difficulty
✅ **Advanced Analytics Dashboard**: Token usage, cost tracking, and performance metrics

## Code Organization

```
src/
├── core/
│   ├── patent_engine.py    # Main RAG system with function calling
│   └── prompts.py          # Domain-specific prompt engineering
├── data_pipeline/          # Data processing and embedding generation
└── utils/
    ├── function_cache.py   # Advanced caching with TTL
    ├── rate_limiter.py     # API protection
    └── token_tracker.py    # Cost monitoring and analytics
```

## Setup Instructions

```bash
# Install dependencies
poetry install

# Configure environment
cp .env.example .env
# Add OpenAI API key to .env

# Run application
poetry run streamlit run streamlit_app.py
```

**Note**: This system requires the full ChromaDB database (29GB) with 1.4M+ patent embeddings stored locally. The database is not included in this repository due to size constraints. You must run the complete data pipeline on the BIGPATENT dataset before using the application.

## Technical Specifications

- **Python Version**: 3.10+
- **Framework**: Streamlit + LangChain
- **Vector Database**: ChromaDB
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **LLM**: OpenAI gpt-4o-mini
- **Caching**: SQLite with TTL
- **Rate Limiting**: Thread-safe implementation

## Project Scope

This project demonstrates a production-ready RAG system with:
- 1.4M+ patent documents processed
- Advanced function calling capabilities
- Comprehensive error handling and monitoring
- Scalable architecture for real-world deployment

The implementation showcases mastery of modern AI engineering practices and provides a foundation for building specialized domain chatbots.