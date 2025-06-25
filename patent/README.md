# Patent RAG Chatbot System

A Retrieval-Augmented Generation (RAG) chatbot for patent analysis using the BIGPATENT dataset from Hugging Face. The system enables semantic search and intelligent analysis of 1.4+ million patent documents.

## 🎯 Project Overview

This system transforms raw patent data into an intelligent chatbot capable of:
- **Semantic Patent Search**: Find relevant patents using natural language queries
- **Journalist Functions**: Generate impact analysis, article titles, and market angles
- **Cost Tracking**: Monitor OpenAI API usage and costs
- **Web Interface**: User-friendly Streamlit application

## 📊 Data Workflow

### 1. **Data Source: Hugging Face BIGPATENT Dataset**
- **Source**: [BIGPATENT dataset](https://huggingface.co/datasets/big_patent) on Hugging Face
- **Content**: Patent abstracts from multiple technology categories
- **Format**: JSON files containing patent ID and abstract text
- **Categories**: Physics, Electricity, Human Necessities, Chemistry, etc.

### 2. **Data Extraction Process**
```
Raw BIGPATENT → JSON Abstracts → Semantic Chunks → Vector Embeddings → ChromaDB
```

**Step 1: Abstract Extraction**
- Extracted patent abstracts from BIGPATENT dataset
- Organized by technology categories (Physics, Electricity, etc.)
- Stored as JSON files: `{"id": "patent_123", "text": "Patent abstract..."}`

## 🔧 Technical Implementation

### **Data Pipeline** (`src/data_pipeline/`)

#### **1. Text Chunking** (`chunk_patents.py`)
- **Purpose**: Split long patent texts into semantic chunks
- **Method**: Respects paragraph and sentence boundaries
- **Configuration**:
  - Chunk size: 800 characters
  - Overlap: 150 characters (maintains context)
  - Minimum chunk: 100 characters
- **Output**: SQLite database (`patent_chunks.db`) with metadata

#### **2. Embedding Generation** (`generate_chunk_embeddings.py`)
- **Model**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **Dimensions**: 384-dimensional vectors
- **Process**: Batch processing (256 chunks at a time)
- **Features**: Normalized embeddings for better similarity search
- **Output**: JSONL files grouped by category

#### **3. Vector Database Import** (`import_chunked_embeddings.py`)
- **Database**: ChromaDB (persistent vector storage)
- **Collection**: `patent_chunks`
- **Batch Size**: 5,000 documents per batch
- **Total**: 1.4+ million patent chunks indexed

### **Core System** (`src/core/`)

#### **Chatbot Engine** (`chatbot.py`)
```python
# Retrieval Process
Query → SentenceTransformer → 384-dim vector → ChromaDB similarity search → Top 3 patents
```

**Key Components**:
- **Custom Embedding Wrapper**: Ensures compatibility between SentenceTransformers and LangChain
- **Similarity Threshold**: 0.4 (filters low-relevance results)
- **LLM Integration**: gpt-4o-mini for response generation
- **Caching**: 12-hour TTL for chat responses

#### **Journalist Functions** (`journalist_functions.py`)
Specialized analysis tools for patent insights:

1. **Impact Analysis**: Identifies affected industries and implementation timeline
2. **Article Titles**: Generates compelling news headlines
3. **Market Angles**: Assesses disruption potential and adoption likelihood

#### **Prompt Engineering** (`prompts.py`)
- **RAG System Prompt**: Instructs LLM to find semantic relationships
- **Journalist Prompts**: Structured JSON output for analysis functions
- **Semantic Guidance**: Helps find related technologies (e.g., "smartphones" → mobile communication, touch interfaces)

### **Utilities** (`src/utils/`)

#### **Function Cache** (`function_cache.py`)
- **Purpose**: Prevents expensive duplicate LLM calls
- **Storage**: SQLite with TTL (24h journalist, 12h chat)
- **Savings**: Significant cost reduction on repeated queries

#### **Token Tracker** (`token_tracker.py`)
- **Models**: gpt-4o-mini, GPT-4o pricing
- **Precision**: tiktoken for accurate token counting
- **Monitoring**: Real-time cost tracking and session summaries

#### **Rate Limiter** (`rate_limiter.py`)
- **Limits**: 20 requests/minute, 500 requests/hour
- **Thread-safe**: Handles concurrent requests
- **Prevents**: OpenAI API rate limit violations

## 🔍 How Retrieval Works

### **Query Processing**
1. **User Query**: "find patents about solar panels"
2. **Embedding**: Query → SentenceTransformer → 384-dim vector
3. **Search**: ChromaDB cosine similarity search
4. **Filtering**: Keep results with score > 0.4
5. **Context**: Top 3 patents sent to LLM
6. **Response**: gpt-4o-mini generates contextual answer


## 💻 Usage

> **⚠️ Note for Instructor**: This system requires the full ChromaDB database (29GB) with 1.4M+ patent embeddings stored locally. The database is not included in this repository due to size constraints, so the application cannot be run without first processing the complete BIGPATENT dataset through the data pipeline.



## 📈 Performance Metrics

- **Database Size**: 29GB ChromaDB with 1.4M+ documents
- **Query Speed**: ~1-2 seconds per search
- **Accuracy**: Semantic similarity with 0.4+ threshold
- **Cost Efficiency**: Caching reduces API costs by ~70%
- **Reliability**: Rate limiting prevents API failures

## 🔧 Configuration

- **Environment**: `.env` file for OpenAI API key
- **Dependencies**: Poetry for package management
- **Models**: SentenceTransformers + OpenAI gpt-4o-mini
- **Storage**: ChromaDB (vector) + SQLite (cache/metadata)

## 📚 Key Technologies

- **Vector Database**: ChromaDB for similarity search
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **LLM**: OpenAI gpt-4o-mini via LangChain
- **UI**: Streamlit for web interface
- **Caching**: SQLite with TTL for cost optimization
- **Monitoring**: Token tracking and rate limiting