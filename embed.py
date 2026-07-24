import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = []

# Read all chunk files
chunk_files = sorted(
    [f for f in os.listdir("chunks") if f.endswith(".txt")],
    key=lambda x: int(x.split("_")[1].split(".")[0])
)

for file in chunk_files:
    with open(os.path.join("chunks", file), "r", encoding="utf-8") as f:
        chunks.append(f.read())

print(f"Loaded {len(chunks)} chunks")

# Generate embeddings
embeddings = model.encode(chunks, convert_to_numpy=True)

print("Embeddings created")

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
os.makedirs("vectorstore", exist_ok=True)

faiss.write_index(index, "vectorstore/index.faiss")

# Save chunk texts
with open("vectorstore/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Vector store saved successfully!")