# Patent RAG Chatbot System

A Retrieval-Augmented Generation (RAG) chatbot for patent analysis using the BIGPATENT dataset from Hugging Face. The system enables semantic search and intelligent analysis of 1.4+ million patent documents.

## 🎯 Project Overview

This system transforms raw patent data into an intelligent chatbot capable of:
- **Semantic Patent Search**: Find relevant patents using natural language queries
- **Journalist Functions**: Generate impact analysis, article titles, and article topic angles
- **Cost Tracking**: Monitor OpenAI API usage and costs
- **Web Interface**: User-friendly Streamlit application

## 📊 Data Workflow

### 1. **Data Source: Hugging Face BIGPATENT Dataset**
- **Source**: [BIGPATENT dataset](https://huggingface.co/datasets/big_patent) on Hugging Face
- **Content**: Patent abstracts from all technology categories
- **Format**: Converted that into JSON files containing patent ID and abstract text

### 2. **Data Processing Pipeline**
```
Hugging Face BIGPATENT database → JSON with patent abstracts → Semantic chunks → Vector embeddings → ChromaDB
```

**Step 1: Abstract Extraction**
- Extracted patent abstracts from BIGPATENT dataset
- Organized by technology categories (Physics, Electricity, etc.)
- Stored as JSON files: `{"id": "patent_123", "text": "Patent abstract..."}`

### **Data Pipeline** (`src/data_pipeline/`)

#### **Step 2: Text Chunking** (`chunk_patents.py`)
- **Purpose**: Split long patent texts into semantic chunks
- **Method**: Respects paragraph and sentence boundaries
- **Configuration**:
  - Chunk size: 800 characters
  - Overlap: 150 characters (maintains context)
  - Minimum chunk: 100 characters
- **Output**: SQLite database (`patent_chunks.db`) with metadata

#### **Step 3: Embedding Generation** (`generate_chunk_embeddings.py`)
- **Model**: `all-MiniLM-L6-v2` (SentenceTransformers) - chosen for speed with large database
- **Dimensions**: 384-dimensional vectors
- **Process**: Batch processing (256 chunks at a time)
- **Features**: Normalized embeddings for better similarity search
- **Output**: JSONL files grouped by category

#### **Step 4: Vector Database Import** (`import_chunked_embeddings.py`)
- **Database**: ChromaDB (persistent vector storage)
- **Collection**: `patent_chunks`
- **Batch Size**: 5,000 documents per batch
- **Total**: 1.4+ million patent chunks indexed

### **System** (`src/core/`)

#### **Patent Engine** (`patent_engine.py`)
System combining chatbot and journalist functionality:

```python
# Retrieval Process
Query → SentenceTransformer → 384-dim vector → ChromaDB similarity search → Top 3 patents
```

**Key Components**:
- **Custom Embedding Wrapper**: Ensures compatibility between SentenceTransformers and LangChain
- **No Similarity Filtering**: Returns all patents found, letting LLM assess relevance
- **LLM Integration**: gpt-4o-mini for response generation
- **In-Memory Caching**: Cache for session-based optimization
- **Journalist Functions**: Built-in impact analysis, title generation, and market angles

#### **Prompt Engineering** (`prompts.py`)
- **RAG System Prompt**: Instructs LLM to find semantic relationships
- **Journalist Prompts**: Structured JSON output for analysis functions
- **Semantic Guidance**: Helps find related technologies (e.g., "smartphones" → mobile communication, touch interfaces)

### **Utilities** (`src/utils/`)

#### **Function Cache** (`function_cache.py`)
- **Purpose**: Prevents expensive duplicate LLM calls
- **Storage**: SQLite with TTL (24h journalist, 12h chat)

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
4. **No Filtering**: All results sent to LLM for relevance assessment
5. **Context**: Top 3 patents sent to LLM with similarity scores
6. **Response**: gpt-4o-mini generates contextual answer with relevance explanations

## 💻 Usage

> **⚠️ Note for STLs**: This system requires the full ChromaDB database (29GB) with 1.4M+ patent embeddings stored locally. The database is not included in this repository due to size constraints, so the application cannot be run without first processing the complete BIGPATENT dataset through the data pipeline.

- **Environment**: `.env` file for OpenAI API key
- **Dependencies**: Poetry for package management
- **Models**: SentenceTransformers + OpenAI gpt-4o-mini
- **Storage**: ChromaDB (vector) + SQLite (cache/metadata)