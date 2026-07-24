import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index (cosine similarity via normalized inner product)
index = faiss.read_index("vectorstore/index.faiss")

# Load chunks
with open("vectorstore/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def search(query, top_k=3):
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