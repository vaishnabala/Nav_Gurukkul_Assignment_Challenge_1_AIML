import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.parser import parse_pdf

# 1. Initialize local Qdrant DB
client = QdrantClient(path="qdrant_db")
COLLECTION_NAME = "pdf_knowledge_base"

print("Loading embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(pages, chunk_size=700, overlap=100):
    """Splits page text into overlapping chunks while maintaining metadata."""
    chunks = []
    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            chunks.append({
                "text": chunk_text,
                "metadata": metadata
            })
            start += chunk_size - overlap
            
    return chunks

def build_knowledge_base(data_dir):
    """Finds all PDFs in the folder, chunks them, and uploads everything to Qdrant."""
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in '{data_dir}/' folder.")
        return

    print(f"Found {len(pdf_files)} PDFs to index.")

    # Recreate the collection ONCE at the very beginning
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    global_point_idx = 0

    # Loop through each PDF in the data folder
    for filename in pdf_files:
        pdf_path = os.path.join(data_dir, filename)
        print(f"\n--- Processing: {filename} ---")
        
        # Parse and chunk this specific book
        pages = parse_pdf(pdf_path)
        chunks = chunk_text(pages)
        print(f"Created {len(chunks)} chunks for {filename}.")

        points = []
        for chunk in chunks:
            vector = embedding_model.encode(chunk["text"]).tolist()
            
            points.append(
                PointStruct(
                    id=global_point_idx,
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        "source": chunk["metadata"]["source"],
                        "page": chunk["metadata"]["page"]
                    }
                )
            )
            global_point_idx += 1
        
        # Upload points for this specific book before moving to the next
        print(f"Uploading {len(points)} chunks to Qdrant...")
        client.upload_points(collection_name=COLLECTION_NAME, points=points)

    print("\n🎉 All documents indexed and loaded into the vector database successfully!")

if __name__ == "__main__":
    DATA_DIR = "data"
    build_knowledge_base(DATA_DIR)