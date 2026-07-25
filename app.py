import os
import streamlit as st
import faiss
import pickle
import ollama
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Health RAG Assistant",
    page_icon="🏥",
    layout="wide"
)


@st.cache_resource
def load_resources():
    index_path = "vectorstore/index.faiss"
    chunks_path = "vectorstore/chunks.pkl"

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return None, None, None

    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(index_path)

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    return model, index, chunks


model, index, chunks = load_resources()

st.title("🏥 Health RAG Assistant")

if model is None:
    st.error(
        "⚠️ Vector store not found. Please run the ingestion pipeline first:\n\n"
        "```bash\n"
        "python ingest.py    # Extract text from PDFs in data/\n"
        "python chunk.py     # Split text into chunks\n"
        "python embed.py     # Build FAISS index\n"
        "```\n\n"
        "Make sure you have placed your PDF files in the `data/` folder."
    )
    st.stop()

st.write("Ask questions about the uploaded health policy document.")

# --- Sidebar controls ---
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1, max_value=20, value=5, step=1,
        help="Higher values provide more context but may include less relevant chunks."
    )


def retrieve(query, k=5):
    # Normalized so inner product == cosine similarity, matching the stored vectors
    embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(embedding, k)
    return [chunks[i] for i in indices[0] if i != -1]


question = st.text_input("Enter your question")

if st.button("Get Answer") and question:

    retrieved_chunks = retrieve(question, k=top_k)

    if not retrieved_chunks:
        st.warning("No relevant chunks found for your query.")
        st.stop()

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.

If the answer is not available in the context, reply:

"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        st.subheader("Answer")
        st.write(response["message"]["content"])
    except Exception as e:
        st.error(
            f"❌ Failed to get response from Ollama. "
            f"Make sure Ollama is running and the `llama3.2` model is pulled.\n\n"
            f"Error: {e}"
        )

    with st.expander("Retrieved Context"):
        for i, chunk in enumerate(retrieved_chunks, 1):
            st.markdown(f"### Chunk {i}")
            st.write(chunk)