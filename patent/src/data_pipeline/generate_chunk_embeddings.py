import os
import json
import sqlite3
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple

# Load environment variables
load_dotenv()

class ChunkEmbeddingGenerator:
    """Generates embeddings for chunked patent data."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 256):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        
    def load_chunks_from_database(self, limit: int = None) -> List[Dict[str, Any]]:
        """Load chunked data from the database."""
        conn = sqlite3.connect('patent_chunks.db')
        cursor = conn.cursor()
        
        if limit:
            cursor.execute('''
                SELECT id, text, patent_id, category, chunk_index, total_chunks, 
                       chunk_size, is_first_chunk, is_last_chunk, full_patent_text
                FROM patent_chunks 
                LIMIT ?
            ''', (limit,))
        else:
            cursor.execute('''
                SELECT id, text, patent_id, category, chunk_index, total_chunks, 
                       chunk_size, is_first_chunk, is_last_chunk, full_patent_text
                FROM patent_chunks
            ''')
        
        chunks = []
        for row in cursor.fetchall():
            chunks.append({
                'id': row[0],
                'text': row[1],
                'patent_id': row[2],
                'category': row[3],
                'chunk_index': row[4],
                'total_chunks': row[5],
                'chunk_size': row[6],
                'is_first_chunk': bool(row[7]),
                'is_last_chunk': bool(row[8]),
                'full_patent_text': row[9]
            })
        
        conn.close()
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for chunks in batches."""
        embeddings_data = []
        
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Generating embeddings"):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_texts = [chunk['text'] for chunk in batch_chunks]
            
            # Generate embeddings
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True)
            
            # Create records with embeddings
            for j, chunk in enumerate(batch_chunks):
                record = {
                    'id': chunk['id'],
                    'embedding': batch_embeddings[j].tolist(),
                    'text': chunk['text'],
                    'patent_id': chunk['patent_id'],
                    'category': chunk['category'],
                    'chunk_index': chunk['chunk_index'],
                    'total_chunks': chunk['total_chunks'],
                    'chunk_size': chunk['chunk_size'],
                    'is_first_chunk': chunk['is_first_chunk'],
                    'is_last_chunk': chunk['is_last_chunk'],
                    'full_patent_text': chunk['full_patent_text']
                }
                embeddings_data.append(record)
        
        return embeddings_data
    
    def save_embeddings_to_jsonl(self, embeddings_data: List[Dict[str, Any]], output_file: str):
        """Save embeddings to JSONL format."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in embeddings_data:
                f.write(json.dumps(record) + '\n')
        
        print(f"Saved {len(embeddings_data)} embeddings to {output_file}")
    
    def save_embeddings_by_category(self, embeddings_data: List[Dict[str, Any]], output_dir: str = "data/embeddings"):
        """Save embeddings grouped by category."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Group by category
        categories = {}
        for record in embeddings_data:
            category = record['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(record)
        
        # Save each category
        for category, records in categories.items():
            category_filename = category.replace(' ', '_').replace('&', 'and')
            output_file = os.path.join(output_dir, f"chunked_embedded_{category_filename}.jsonl")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record) + '\n')
            
            print(f"Saved {len(records)} embeddings for {category} to {output_file}")

def main():
    """Main function to generate embeddings for chunked patents."""
    print("Starting chunked embedding generation...")
    
    # Initialize generator
    generator = ChunkEmbeddingGenerator()
    
    # Load chunks from database
    print("Loading chunks from database...")
    chunks = generator.load_chunks_from_database()
    print(f"Loaded {len(chunks)} chunks")
    
    if not chunks:
        print("No chunks found in database. Please run chunk_patents.py first.")
        return
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings_data = generator.generate_embeddings(chunks)
    
    # Save embeddings by category (more space efficient)
    print("Saving embeddings by category...")
    generator.save_embeddings_by_category(embeddings_data)
    
    print("Embedding generation complete!")
    print("Note: All embeddings saved by category to avoid disk space issues.")

if __name__ == "__main__":
    main() 