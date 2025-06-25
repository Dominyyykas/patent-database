import os
import json
import re
import sqlite3
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import glob

# Chunking configuration
CHUNK_SIZE = 800  # Smaller chunks for more precise retrieval
CHUNK_OVERLAP = 150  # Overlap to maintain context
MIN_CHUNK_SIZE = 100  # Minimum chunk size to avoid tiny fragments

class PatentChunker:
    """Handles chunking of patent abstracts with semantic boundaries."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving important punctuation."""
        # Split on sentence endings, but keep the punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def create_semantic_chunks(self, text: str, patent_id: str, category: str) -> List[Dict[str, Any]]:
        """Create chunks that respect semantic boundaries with metadata."""
        chunks = []
        
        # First split into paragraphs
        paragraphs = self.split_into_paragraphs(text)
        
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for para in paragraphs:
            # If paragraph is too long, split into sentences
            if len(para) > self.chunk_size:
                sentences = self.split_into_sentences(para)
                for sentence in sentences:
                    if current_length + len(sentence) > self.chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text = ' '.join(current_chunk)
                        if len(chunk_text) >= MIN_CHUNK_SIZE:
                            chunks.append(self._create_chunk_metadata(
                                chunk_text, patent_id, category, chunk_index, len(chunks)
                            ))
                            chunk_index += 1
                        
                        # Start new chunk with overlap
                        overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                        current_chunk = current_chunk[overlap_start:]
                        current_length = sum(len(s) for s in current_chunk)
                    
                    current_chunk.append(sentence)
                    current_length += len(sentence)
            else:
                if current_length + len(para) > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text) >= MIN_CHUNK_SIZE:
                        chunks.append(self._create_chunk_metadata(
                            chunk_text, patent_id, category, chunk_index, len(chunks)
                        ))
                        chunk_index += 1
                    
                    # Start new chunk with overlap
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:]
                    current_length = sum(len(s) for s in current_chunk)
                
                current_chunk.append(para)
                current_length += len(para)
        
        # Add the last chunk if it exists
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= MIN_CHUNK_SIZE:
                chunks.append(self._create_chunk_metadata(
                    chunk_text, patent_id, category, chunk_index, len(chunks)
                ))
        
        return chunks
    
    def _create_chunk_metadata(self, chunk_text: str, patent_id: str, category: str, 
                              chunk_index: int, total_chunks: int) -> Dict[str, Any]:
        """Create metadata for a chunk."""
        chunk_id = f"{patent_id}_chunk{chunk_index+1}of{total_chunks+1}"
        
        return {
            'id': chunk_id,
            'text': chunk_text,
            'patent_id': patent_id,
            'category': category,
            'chunk_index': chunk_index,
            'total_chunks': total_chunks + 1,
            'chunk_size': len(chunk_text),
            'is_first_chunk': chunk_index == 0,
            'is_last_chunk': True  # Will be updated later
        }
    
    def process_patent_file(self, file_path: str, category: str) -> List[Dict[str, Any]]:
        """Process a single patent file and return chunked data."""
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                
            for item in batch_data:
                patent_id = item['id']
                text = item['text']
                
                # Create chunks for this patent
                patent_chunks = self.create_semantic_chunks(text, patent_id, category)
                
                # Update is_last_chunk flag
                for i, chunk in enumerate(patent_chunks):
                    chunk['is_last_chunk'] = (i == len(patent_chunks) - 1)
                
                chunks.extend(patent_chunks)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
        return chunks

def setup_chunked_database():
    """Create SQLite database for storing chunked patent data."""
    conn = sqlite3.connect('patent_chunks.db')
    cursor = conn.cursor()
    
    # Drop existing table if it exists
    cursor.execute('DROP TABLE IF EXISTS patent_chunks')
    
    # Create table for chunked data
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patent_chunks (
        id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        patent_id TEXT NOT NULL,
        category TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        total_chunks INTEGER NOT NULL,
        chunk_size INTEGER NOT NULL,
        is_first_chunk BOOLEAN NOT NULL,
        is_last_chunk BOOLEAN NOT NULL,
        full_patent_text TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create index for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_patent_id ON patent_chunks(patent_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON patent_chunks(category)')
    
    return conn, cursor

def process_category_chunks(category_code: str, category_name: str, chunker: PatentChunker) -> List[Dict[str, Any]]:
    """Process all files in a category and return chunked data."""
    all_chunks = []
    data_dir = f"data/raw/{category_code} {category_name}"
    
    if not os.path.exists(data_dir):
        print(f"Warning: Directory {data_dir} not found")
        return all_chunks
    
    # Get all batch files
    batch_files = glob.glob(f"{data_dir}/abstracts_batch_*.json")
    
    for batch_file in tqdm(batch_files, desc=f"Processing {category_name}"):
        # batch_file is already the full path, don't join again
        chunks = chunker.process_patent_file(batch_file, category_name)
        all_chunks.extend(chunks)
    
    return all_chunks

def store_chunks_in_database(chunks: List[Dict[str, Any]], conn: sqlite3.Connection, cursor: sqlite3.Cursor):
    """Store chunks in the database."""
    # First, get full patent texts for reconstruction
    patent_texts = {}
    for chunk in chunks:
        patent_id = chunk['patent_id']
        if patent_id not in patent_texts:
            patent_texts[patent_id] = []
        patent_texts[patent_id].append(chunk)
    
    # Reconstruct full texts and store chunks
    for patent_id, patent_chunks in patent_texts.items():
        # Sort chunks by index
        patent_chunks.sort(key=lambda x: x['chunk_index'])
        
        # Reconstruct full text
        full_text = ' '.join(chunk['text'] for chunk in patent_chunks)
        
        # Store each chunk with full text reference
        for chunk in patent_chunks:
            cursor.execute('''
                INSERT OR REPLACE INTO patent_chunks 
                (id, text, patent_id, category, chunk_index, total_chunks, 
                 chunk_size, is_first_chunk, is_last_chunk, full_patent_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                chunk['id'], chunk['text'], chunk['patent_id'], chunk['category'],
                chunk['chunk_index'], chunk['total_chunks'], chunk['chunk_size'],
                chunk['is_first_chunk'], chunk['is_last_chunk'], full_text
            ))
    
    conn.commit()

def main():
    """Main function to process all patent categories into chunks."""
    print("Starting patent chunking process...")
    
    # Initialize chunker
    chunker = PatentChunker()
    
    # Setup database
    conn, cursor = setup_chunked_database()
    
    # Categories to process
    CATEGORIES = {
        'g': 'Physics',
        'h': 'Electricity', 
        'y': 'General Technology',
        'a': 'Human Necessities',
        'b': 'Operations and Transporting',
        'c': 'Chemistry and Metallurgy',
        'f': 'Mechanical Engineering'
    }
    
    total_chunks = 0
    
    for category_code, category_name in CATEGORIES.items():
        print(f"\nProcessing category: {category_name}")
        
        # Process category
        chunks = process_category_chunks(category_code, category_name, chunker)
        
        # Store in database
        store_chunks_in_database(chunks, conn, cursor)
        
        total_chunks += len(chunks)
        print(f"Processed {len(chunks)} chunks for {category_name}")
    
    print(f"\nChunking complete! Total chunks: {total_chunks}")
    conn.close()

if __name__ == "__main__":
    main() 