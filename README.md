# Rag-health: Health Policy RAG Assistant

## Overview
**Rag-health** is a Retrieval-Augmented Generation (RAG) system designed to answer questions regarding India's health transformation based on the Press Information Bureau (PIB) backgrounder document. The system combines semantic vector search with a local LLM (Llama 3.2 via Ollama) to deliver accurate, grounded, and privacy-preserving answers without sending data to external APIs.

---

## Data Source
The system relies exclusively on a single source document:
- **Source Document**: PIB (Press Information Bureau) backgrounder page titled *"India's Health Transformation"* (June 2026).
- **Topics Covered**: Key flagship government health initiatives including Ayushman Bharat PM-JAY (AB-PMJAY), Ayushman Arogya Mandir, PM-ABHIM, Ayushman Bharat Digital Mission (ABDM), National Health Mission (NHM), Mission Indradhanush, Pradhan Mantri Bhartiya Janaushadhi Pariyanjana (Jan Aushadhi), Poshan Abhiyaan, and related healthcare schemes.
- **Storage & Policy**: The raw document is saved locally as `data/pib_document.pdf`. It is NOT committed to the repository (git-ignored) to ensure clean version tracking and compliance.
- **Data Constraint**: Only this PIB document is used across the entire application; no outside or unverified documents are ingested.

---

## Pipeline Architecture

The end-to-end processing pipeline consists of four modular stages:

### 1. Ingestion (`ingest.py`)
- **Method**: PDF text extraction using PyMuPDF (`fitz`).
- **Process**: Reads `data/pib_document.pdf` page-by-page, extracts raw text, and saves the consolidated text to `data/pib_document.txt`.
- **Rationale**: PyMuPDF was chosen because the source document is natively text-based and digitally generated, allowing ultra-fast extraction without requiring optical character recognition (OCR).

### 2. Chunking (`chunk.py`)
- **Method**: Section-aware topic chunking targeting 200–500 words per chunk.
- **Heading Detection**: Utilizes known topic keywords (e.g., *AB-PMJAY*, *Ayushman Arogya Mandir*, *PM-ABHIM*, *ABDM*, *NHM*, *Mission Indradhanush*, *Jan Aushadhi*, *Poshan Abhiyaan*) and title-case short line heuristics to identify section transitions.
- **Boundary Handling**: Sections shorter than 200 words are merged into adjacent blocks, while oversized sections are split at sentence boundaries to maintain context integrity. Reference or URL-only sections are automatically filtered out.
- **Output**: Each chunk is saved individually to `chunks/chunk_i.txt`, with its detected section header written on the first line `[Section Title]` to serve as inline context metadata.

### 3. Embedding & Storage (`embed.py`)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **Normalization**: Vectors are L2-normalized (`normalize_embeddings=True`) prior to indexing.
- **FAISS Indexing**: Uses FAISS `IndexFlatIP` (Inner Product). When input vectors are L2-normalized, inner product mathematically equals **cosine similarity**. An exact brute-force search is used, which is ideal for small corpus sizes (~14 chunks) and yields zero recall loss.
- **Vector Store Persistence**: Exports two main artifacts to `vectorstore/`:
  - `vectorstore/index.faiss`: Binary FAISS index storing the normalized chunk vectors.
  - `vectorstore/chunks.pkl`: Pickled Python list of text chunks aligned positionally with the FAISS vector indices (list index `i` matches FAISS vector `i`).

### 4. Retrieval & RAG (`search.py`, `rag.py`, `app.py`)
- **Query Vectorization**: User questions are embedded using `all-MiniLM-L6-v2` with the same L2 normalization, placing query and document embeddings in the identical vector space.
- **Cosine Retrieval**: Computes top-$k$ nearest neighbors against `vectorstore/index.faiss` and retrieves text chunks from `vectorstore/chunks.pkl`.
- **Context Injection & Prompting**: Retrieved chunks are concatenated into a context block. The LLM prompt instructs the model to answer **ONLY** using the provided context, with an explicit fallback string (`"I could not find the answer in the provided document."`) if the answer is missing.
- **Local LLM**: Answers are generated locally using **Llama 3.2** via Ollama (zero API costs or data leakage).
- **Interfaces**:
  - `app.py`: Streamlit web interface featuring an interactive sidebar with an adjustable `top_k` slider (1–20 chunks), real-time query handling, and expandable retrieved context.
  - `search.py`: Standalone CLI utility for testing semantic retrieval.
  - `rag.py`: Interactive CLI loop for end-to-end retrieval and RAG answer generation.

---

## Project Structure

```
rag-health-assistant1/
├── data/
│   ├── pib_document.pdf          # Source PDF document (git-ignored)
│   └── pib_document.txt          # Extracted plain text (git-ignored)
├── chunks/                       # Section-aware text chunks (git-ignored)
│   ├── chunk_0.txt
│   └── ...
├── vectorstore/                  # Vector index and chunk storage (git-ignored)
│   ├── index.faiss               # FAISS vector index
│   └── chunks.pkl                # Pickled text list positionally aligned with index
├── ingest.py                     # PDF text extraction script
├── chunk.py                      # Section-aware text chunker
├── embed.py                      # Vector embedding and FAISS index generation
├── search.py                     # Standalone CLI semantic search utility
├── rag.py                        # Interactive CLI RAG interface
├── app.py                        # Streamlit web frontend
├── requirements.txt              # Dependency specifications
├── .gitignore                    # Git ignore configuration
├── .env                          # Environment variables configuration
├── README.md                     # Comprehensive project documentation
└── IMPLEMENTATION_NOTE.md        # Technical design & architecture implementation note
```

---

## Setup & Usage

### Prerequisites
1. **Python 3.9+** installed.
2. **Ollama** installed and running locally.
3. Pull the **Llama 3.2** model:
   ```bash
   ollama pull llama3.2
   ```

### Quickstart Steps

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Data File**:
   Place the PIB backgrounder PDF (`pib_document.pdf`) into the `data/` directory.

3. **Run Pipeline**:
   ```bash
   # Step 1: Extract text from PDF
   python ingest.py

   # Step 2: Generate section-aware text chunks
   python chunk.py

   # Step 3: Embed chunks and build FAISS vector index
   python embed.py
   ```

4. **Launch Streamlit Web App**:
   ```bash
   streamlit run app.py
   ```

### Alternative CLI Interfaces
- **Semantic Search Only**:
  ```bash
  python search.py
  ```
- **Full RAG Terminal Loop**:
  ```bash
  python rag.py
  ```

### Graceful Error Handling
If you launch `app.py`, `search.py`, or `rag.py` before building the vector store, the system displays a clear, graceful error message stating that `vectorstore/index.faiss` and `vectorstore/chunks.pkl` are missing, alongside exact instructions on how to run `ingest.py`, `chunk.py`, and `embed.py`.
