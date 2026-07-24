# Rag-health: Health Policy RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions about a health policy document (PIB) using semantic search and a local LLM.

## Overview

This project extracts text from a PDF health policy document, chunks it, embeds the chunks into a vector store, and lets users ask natural-language questions through a Streamlit UI. Relevant chunks are retrieved via semantic similarity and passed to a local LLM (via Ollama) to generate grounded answers.

---

## Pipeline

### 1. Ingestion (`ingest.py`)

The source PDF (`data/pib_document.pdf`) is opened with **PyMuPDF (fitz)**. Text is extracted page by page and concatenated, then saved as a plain text file at `data/pib_document.txt`.

```
PDF → fitz.open() → page.get_text() per page → data/pib_document.txt
```

PyMuPDF was chosen because it preserves reading order well and is fast on text-based PDFs (the PIB document is not scanned/image-based, so OCR wasn't needed).

### 2. Chunking (`chunk.py`)

The extracted text is split using LangChain's `RecursiveCharacterTextSplitter`:

- **Chunk size:** 1800 characters
- **Overlap:** 200 characters
- **Separator priority:** `["\n\n", "\n", ".", " ", ""]` — tries to split on paragraph breaks first, then lines, then sentences, then words, only falling back to hard character cuts as a last resort.

Each chunk is written to its own file in `chunks/chunk_{i}.txt`. This "recursive" approach keeps semantically related sentences together rather than cutting mid-thought, and the 200-character overlap ensures context isn't lost at chunk boundaries.

### 3. Embedding & Storage (`embed.py`)

- All `chunks/*.txt` files are read back in **numeric order** (sorted by the index in the filename, not alphabetically, to avoid `chunk_10.txt` sorting before `chunk_2.txt`).
- Each chunk is embedded using **`sentence-transformers/all-MiniLM-L6-v2`**.
- Embeddings are stored in a **FAISS `IndexFlatL2`** index (exact L2/Euclidean nearest-neighbor search, no approximation).
- Two files are saved to `vectorstore/`:
  - `index.faiss` — the FAISS vector index
  - `chunks.pkl` — a pickled Python list of the original chunk text, where **list position `i` corresponds to vector `i`** in the FAISS index. This positional mapping is how a search result (a vector index) gets translated back into readable text.

### 4. Retrieval + RAG (`rag.py`, `search.py`, `app.py`)

At query time:

1. The user's question is embedded with the same `all-MiniLM-L6-v2` model (query and documents must share an embedding space).
2. FAISS performs a similarity search (`index.search`) and returns the `top_k` nearest chunk indices.
3. Those indices are used to look up the actual text from `chunks.pkl`.
4. The retrieved chunks are concatenated into a `context` block.
5. A prompt is constructed that instructs the LLM to answer **only** from the provided context, and to explicitly say `"I could not find the answer in the provided document."` if the context doesn't contain the answer — this reduces hallucination.
6. The prompt is sent to a local **Llama 3.2** model via **Ollama**, and the generated answer is returned along with the source chunks (shown in an expandable "Retrieved Context" section in the UI, for transparency/verifiability).

`app.py` wraps this entire flow in a **Streamlit** interface: a text input for the question, a button to trigger retrieval + generation, the answer displayed prominently, and the retrieved chunks shown underneath so a user can verify the answer against the source text.

---

## Project Structure

```
Rag-health/
├── data/
│   └── pib_document.pdf       # source document (not committed)
├── chunks/                    # generated chunk .txt files
├── vectorstore/
│   ├── index.faiss            # FAISS vector index
│   └── chunks.pkl             # chunk text, aligned by position to the index
├── ingest.py                  # PDF → text
├── chunk.py                   # text → chunks
├── embed.py                   # chunks → embeddings → FAISS index
├── search.py                  # standalone CLI semantic search
├── rag.py                     # CLI RAG (search + LLM answer)
├── app.py                     # Streamlit web app
└── requirements.txt
```

## Setup & Usage

```bash
pip install -r requirements.txt

# 1. Extract text from PDF
python ingest.py

# 2. Chunk the extracted text
python chunk.py

# 3. Generate embeddings and build the FAISS index
python embed.py

# 4. Run the Streamlit app
streamlit run app.py
```

**Prerequisite:** [Ollama](https://ollama.com) must be installed and running locally with the `llama3.2` model pulled (`ollama pull llama3.2`).

You can also test retrieval alone (`python search.py`) or the full RAG loop from the CLI (`python rag.py`) without the Streamlit UI.
