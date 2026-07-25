import os
import sys
import faiss
import pickle
import ollama
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


# Lazy-loaded globals
embedding_model = None
index = None
chunks = None


def _ensure_loaded():
    global embedding_model, index, chunks
    if embedding_model is None:
        embedding_model, index, chunks = _load_resources()


def retrieve(query, top_k=5):
    _ensure_loaded()
    # Generate query embedding, normalized to match the stored vectors
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Search FAISS index (higher score = more similar)
    scores, indices = index.search(query_embedding, top_k)

    # Return only valid chunks
    return [chunks[i] for i in indices[0] if i != -1]


def answer_question(question):
    retrieved_chunks = retrieve(question)

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the information in the provided context to answer the question.

If the answer cannot be found in the context, reply exactly:

"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"], retrieved_chunks


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (type 'exit' to quit): ").strip()

        if question.lower() == "exit":
            break

        print("\nGenerating answer...\n")

        answer, retrieved_chunks = answer_question(question)

        print("Answer:")
        print(answer)

        print("\n" + "=" * 80)
        print("Retrieved Context")
        print("=" * 80)

        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"\nChunk {i}\n")
            print(chunk[:500])
            print("-" * 80)