import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# =========================
# CONFIG
# =========================
# Note: Added 'r' before the string to fix Windows Path escape errors
CSV_PATH = r"C:\Users\user\Downloads\archive\netflix_titles.csv"
DB_LOCATION = "./chroma_netflix_db"
COLLECTION_NAME = "netflix_catalog"

def get_vector_store(csv_path=CSV_PATH):
    """
    Initializes the ChromaDB vector store. 
    If the database is empty, it loads the CSV and performs ingestion.
    """
    
    # 1. Initialize Embedding Model (Requires: ollama pull mxbai-embed-large)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # 2. Initialize/Load Vector Store
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_LOCATION,
        embedding_function=embeddings
    )

    # 3. Check if we need to ingest data
    current_count = vector_store._collection.count()
    
    if current_count == 0:
        if not os.path.exists(csv_path):
            print(f"❌ Error: CSV file not found at {csv_path}")
            return vector_store

        print(f"🚀 Database is empty. Starting ingestion from: {csv_path}")
        
        # Load and clean data
        df = pd.read_csv(csv_path).fillna("Unknown")
        
        documents = []
        ids = []
        
        for i, row in df.iterrows():
            # Create a rich text summary for semantic search
            content = (
                f"Title: {row['title']}. "
                f"Type: {row['type']}. "
                f"Genres: {row['listed_in']}. "
                f"Release Year: {row['release_year']}. "
                f"Director: {row['director']}. "
                f"Cast: {row['cast']}. "
                f"Description: {row['description']}"
            )
            
            # Create the Document object with metadata for filtering
            doc = Document(
                page_content=content,
                metadata={
                    "show_id": str(row['show_id']),
                    "type": str(row['type']),
                    "year": int(row['release_year']),
                    "genre": str(row['listed_in']),
                    "title": str(row['title'])
                }
            )
            documents.append(doc)
            ids.append(str(row['show_id']))

        # 4. Batch Insertion (to prevent memory/timeout issues)
        batch_size = 500
        print(f"📦 Total documents to index: {len(documents)}")
        
        for i in range(0, len(documents), batch_size):
            end_idx = i + batch_size
            batch_docs = documents[i:end_idx]
            batch_ids = ids[i:end_idx]
            
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            print(f"✅ Ingested {min(end_idx, len(documents))} / {len(documents)} records...")

        print("✨ Vector Database initialization complete!")
    else:
        print(f"✅ Using existing Vector Database ({current_count} records).")
            
    return vector_store

# If you run this file directly, it will just initialize the DB
if __name__ == "__main__":
    get_vector_store()