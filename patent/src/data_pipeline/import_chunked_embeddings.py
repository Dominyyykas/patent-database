import os
import json
import chromadb
from typing import List, Dict, Any
from tqdm import tqdm

class ChunkedChromaImporter:
    """Imports chunked patent embeddings into ChromaDB."""
    
    def __init__(self, chroma_db_path: str = "db/chroma_db", collection_name: str = "patent_chunks"):
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.batch_size = 5000
        
        # Initialize ChromaDB client
        os.makedirs(chroma_db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Chunked patent abstracts with embeddings"}
        )
    
    def load_embeddings_from_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load embeddings from JSONL file."""
        records = []
        
        if not os.path.exists(file_path):
            print(f"File {file_path} not found")
            return records
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        return records
    
    def import_embeddings(self, embeddings_data: List[Dict[str, Any]]):
        """Import embeddings into ChromaDB in batches."""
        if not embeddings_data:
            print("No embeddings data to import")
            return
        
        total_added = 0
        
        # Process in batches
        for start_idx in tqdm(range(0, len(embeddings_data), self.batch_size), desc="Importing to ChromaDB"):
            end_idx = start_idx + self.batch_size
            batch = embeddings_data[start_idx:end_idx]
            
            # Prepare batch data
            ids = [record['id'] for record in batch]
            embeddings = [record['embedding'] for record in batch]
            documents = [record['text'] for record in batch]
            
            # Prepare metadata
            metadatas = []
            for record in batch:
                metadata = {
                    'patent_id': record['patent_id'],
                    'category': record['category'],
                    'chunk_index': record['chunk_index'],
                    'total_chunks': record['total_chunks'],
                    'chunk_size': record['chunk_size'],
                    'is_first_chunk': record['is_first_chunk'],
                    'is_last_chunk': record['is_last_chunk'],
                    'full_patent_text': record['full_patent_text']
                }
                metadatas.append(metadata)
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            total_added += len(batch)
        
        print(f"Successfully imported {total_added} chunked embeddings to ChromaDB")
    
    def import_from_file(self, file_path: str):
        """Import embeddings from a specific JSONL file."""
        print(f"Importing from {file_path}...")
        embeddings_data = self.load_embeddings_from_jsonl(file_path)
        self.import_embeddings(embeddings_data)
    
    def import_all_chunked_embeddings(self, embeddings_dir: str = "data/embeddings"):
        """Import all chunked embedding files."""
        # Look for chunked embedding files (exclude error files)
        chunked_files = []
        for filename in os.listdir(embeddings_dir):
            if (filename.startswith("chunked_embedded_") and 
                filename.endswith(".jsonl") and 
                not filename.endswith("_errors.jsonl")):
                chunked_files.append(os.path.join(embeddings_dir, filename))
        
        if not chunked_files:
            print("No chunked embedding files found")
            return
        
        print(f"Found {len(chunked_files)} chunked embedding files")
        
        total_imported = 0
        for file_path in chunked_files:
            print(f"\nProcessing {os.path.basename(file_path)}...")
            embeddings_data = self.load_embeddings_from_jsonl(file_path)
            self.import_embeddings(embeddings_data)
            total_imported += len(embeddings_data)
        
        print(f"\nTotal chunked embeddings imported: {total_imported}")
    
    def get_collection_stats(self):
        """Get statistics about the collection."""
        count = self.collection.count()
        print(f"Collection '{self.collection_name}' contains {count} documents")
        return count

def main():
    """Main function to import chunked embeddings."""
    print("Starting chunked embedding import to ChromaDB...")
    
    # Initialize importer
    importer = ChunkedChromaImporter()
    
    # Import all chunked embeddings
    importer.import_all_chunked_embeddings()
    
    # Show collection stats
    importer.get_collection_stats()
    
    print("Chunked embedding import complete!")

if __name__ == "__main__":
    main() 