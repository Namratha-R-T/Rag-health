import os
import sys
import faiss
import pickle
from sentence_transformers import SentenceTransformer


def _load_resources():
    """Load model, FAISS index, and chunks with clear error messages."""
    index_path = "vectorstore/index.faiss"
    chunks_path = "vectorstore/chunks.pkl"

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        print(
            "Error: Vector store not found.\n"
            "Please run the ingestion pipeline first:\n"
            "  python ingest.py\n"
            "  python chunk.py\n"
            "  python embed.py\n"
            "\nMake sure you have placed PDF files in the data/ folder."
        )
        sys.exit(1)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(index_path)

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    return model, index, chunks


# Load embedding model
model = None
index = None
chunks = None


def _ensure_loaded():
    global model, index, chunks
    if model is None:
        model, index, chunks = _load_resources()


def search(query, top_k=3):
    _ensure_loaded()
    # Create embedding for the query, normalized to match the stored vectors
    query_embedding = model.encode([query], normalize_embeddings=True)

    # Search the index (higher score = more similar, since this is cosine/IP)
    scores, indices = index.search(query_embedding, top_k)

    results = []

    for idx in indices[0]:
        if idx != -1:
            results.append(chunks[idx])

    return results


if __name__ == "__main__":
    question = input("Ask a question: ")

    results = search(question)

    print("\nTop Retrieved Chunks:\n")

    for i, chunk in enumerate(results, 1):
        print("=" * 80)
        print(f"Chunk {i}\n")
        print(chunk[:700])  # Print first 700 characters
        print()