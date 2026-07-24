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
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("vectorstore/index.faiss")

    with open("vectorstore/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    return model, index, chunks

model, index, chunks = load_resources()


def retrieve(query, top_k=5):
    # Normalized so inner product == cosine similarity, matching the stored vectors
    embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(embedding, top_k)
    return [chunks[i] for i in indices[0] if i != -1]


st.title("🏥 Health RAG Assistant")
st.write("Ask questions about the uploaded health policy document.")

question = st.text_input("Enter your question")

if st.button("Get Answer") and question:

    retrieved_chunks = retrieve(question)

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

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    st.subheader("Answer")
    st.write(response["message"]["content"])

    with st.expander("Retrieved Context"):
        for i, chunk in enumerate(retrieved_chunks, 1):
            st.markdown(f"### Chunk {i}")
            st.write(chunk)