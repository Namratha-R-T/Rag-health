# Implementation Note — Health Policy RAG Assistant

## 1. Embedding Model Choice

**Selected Model**: `sentence-transformers/all-MiniLM-L6-v2`

### Selection Rationale
- **Lightweight & High Efficiency**: Produces compact 384-dimensional dense vectors with approximately 22 million parameters. It executes rapidly on CPU without requiring GPU hardware, ensuring fast embedding generation and low search latency.
- **Strong General-Purpose Semantic Quality**: Ranks high among lightweight embedding models on standard benchmarks (such as MTEB) for semantic similarity and information retrieval.
- **Local & Offline Execution**: Integrates natively via the `sentence-transformers` package. It operates fully offline with no API key requirements, external API costs, or network dependency, keeping health policy data local and private.
- **Symmetric Model Usage**: The exact same model and L2 normalization are applied to both document chunks and user queries, maintaining vector space parity.

### Trade-Off Analysis
Larger embedding models (e.g., `bge-base-en-v1.5` or cloud APIs such as OpenAI `text-embedding-3-small`) can offer marginal improvements in semantic retrieval for complex queries. However, they introduce higher memory usage, slower CPU latency, recurring costs, and cloud privacy trade-offs. For a small target document (~14 chunks), `all-MiniLM-L6-v2` delivers an optimal balance of accuracy, speed, and local privacy.

---

## 2. Storage / Index Choice

**Selected Storage**: FAISS `IndexFlatIP` with L2-Normalized Embeddings

### Selection Rationale
- **Cosine Similarity Equivalence**: Embeddings are L2-normalized (`normalize_embeddings=True`) prior to indexing. For unit vectors, inner product matches cosine similarity:
  $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = u \cdot v$$
- **Exact Search (No Approximation)**: Given a small collection (~14 chunks), approximate nearest neighbor search (e.g., IVF or HNSW) is unnecessary. `IndexFlatIP` performs exact brute-force search in milliseconds with zero recall degradation.
- **No Clustering or Training Required**: Unlike `IndexIVFFlat`, which requires training cluster centroids on sufficient vector samples, `IndexFlatIP` needs no training step.
- **Simple Disk Persistence**: Simple file serialization via `faiss.write_index` to `vectorstore/index.faiss` and `faiss.read_index` for loading.
- **Positional Chunk Mapping**: Document text chunks are stored in a parallel pickled list (`vectorstore/chunks.pkl`). List position `i` directly corresponds to vector index `i` in FAISS. This parallel array approach avoids the operational overhead of heavy vector databases while remaining fast and sufficient at this scale.

---

## 3. LLM and Prompt Design

**Selected LLM**: Llama 3.2 (via Ollama local inference)

### Selection Rationale
- **Privacy & Security**: Runs completely on the local host machine via Ollama. No health document content or user queries are transmitted to external servers.
- **Zero API Expenses**: Eliminates token costs, rate limits, and external service downtime.
- **Context-Grounded Reasoning**: Llama 3.2 reliably adheres to strict system prompts and context constraints.

### Prompt Design & Anti-Hallucination Guardrails
The prompt structure explicitly isolates context from question and mandates an exact fallback string for out-of-context queries:

```text
You are a helpful AI assistant.

Answer ONLY using the context below.

If the answer is not available in the context, reply:

"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
```

- **Strict Context Boundary**: Explicitly instructs the model to utilize only the provided context block, mitigating parametric memory hallucinations.
- **Exact Fallback String**: Requires the exact string `"I could not find the answer in the provided document."` when context lacks the required information, facilitating programmatic verification in downstream code.
- **Delimited Sections**: Uses clear labels (`Context:`, `Question:`, `Answer:`) to prevent prompt confusion.

### Example Q&A (Expected Output)

> **Note:** The following represents the expected model output based on the provided document context. Exact phrasing will vary depending on the local LLM (e.g., Llama 3.2) used during live execution.

**Question 1**: *"What is AB-PMJAY?"*
- **Retrieved Chunk Titles**: `[Ayushman Bharat]`, `[AB-PMJAY]`, `[Health and Wellness Centres]`
- **Answer Output**: "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana (AB-PMJAY) is the world's largest health insurance scheme fully financed by the government. It provides health coverage of up to Rs. 5 lakh per family per year for secondary and tertiary care hospitalization to over 12 crore poor and vulnerable families."

**Question 2**: *"What is PM-ABHIM?"*
- **Retrieved Chunk Titles**: `[PM-ABHIM]`, `[Infrastructure Mission]`, `[National Health Mission]`
- **Answer Output**: "The Pradhan Mantri Ayushman Bharat Health Infrastructure Mission (PM-ABHIM) is a pan-India scheme launched to strengthen healthcare infrastructure across the country. It focuses on developing capacities of primary, secondary, and tertiary care health systems, strengthening surveillance capabilities, and creating a network of public health labs."

---

## 4. What I Had to Learn / Research

- **FAISS Index Architectures**: Studied differences between `IndexFlatL2` (Euclidean distance), `IndexFlatIP` (Inner Product), `IndexIVFFlat` (Inverted File Index), and `IndexHNSW` (Graph-based ANN). Confirmed that `IndexFlatIP` implements cosine similarity only when vectors are L2-normalized.
- **Dense Embedding Normalization**: Researched vector normalization principles in `sentence-transformers` and why query vectors must undergo identical L2 normalization as document vectors.
- **Section-Aware Chunking**: Researched heading detection heuristics (known scheme keywords like *AB-PMJAY*, *ABDM*, *NHM* plus title-case line checks) to chunk documents logically by section (200–500 words) instead of arbitrary character cuts.
- **Local Ollama Integration**: Learned setup and API integration with Ollama Python bindings (`ollama.chat`) for local open-source LLM inference.
- **Retrieve-Then-Generate Pattern**: Formulated positional index-to-chunk mapping, candidate retrieval, context concatenation, and prompt construction.

---

## 5. Limitations & What I'd Improve With Two More Days

### Recent Fixes
- **User-Adjustable Top-K Slider**: Previously, `top_k` was fixed at a static value. This has been **FIXED** in `app.py` with an interactive UI sidebar slider allowing users to adjust `top_k` between 1 and 20 chunks dynamically.
- **Reference & URL Chunk Filtering**: Updated `chunk.py` logic to identify and filter out reference-only or URL-heavy sections, preventing uninformative chunks from entering the vector store.

### Current Limitations
1. **No Fine-Grained Source Citations**: The model's answer does not cite specific chunk IDs or page numbers directly inside the text response.
2. **No Re-ranking Step**: Retrieval relies solely on bi-encoder cosine similarity without a secondary cross-encoder re-ranker.
3. **Occasional Heading Misdetection**: Heuristic line inspection can occasionally mistake short non-heading lines for section titles.
4. **No Incremental Indexing**: Updating or appending documents requires re-ingesting and re-embedding the entire vector store.
5. **No Automated Evaluation Harness**: Assessment relies on manual spot-checking without systematic metrics (e.g., recall@k, RAGAS faithfulness).
6. **Single-Document Focus**: Hardcoded pipeline geared for a single document (`pib_document.pdf`) without multi-document collection metadata.
7. **No Response Streaming**: The Streamlit interface waits for the entire LLM response to finish before rendering.

### Two-Day Improvement Roadmap
1. **Metadata & Citation Engine**: Store page numbers and section headers with each chunk, prompting the LLM to output explicit source citations (e.g., `[Page 4, AB-PMJAY Section]`).
2. **Cross-Encoder Re-ranking**: Add a second-stage re-ranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over top-20 FAISS candidates to select the top-5 most relevant context snippets.
3. **Evaluation Question Benchmark**: Construct a test set of 30+ question-answer pairs to benchmark retrieval recall and answer accuracy programmatically.
4. **Multi-Document Support**: Expand schema to support indexing multiple PDF documents with per-document metadata filters.
5. **Streamlit Token Streaming**: Use `ollama.chat(..., stream=True)` to stream generated response tokens to the UI in real-time.
6. **Environment Configuration**: Move chunk sizes, model names, and system paths into `.env` and a centralized `config.py`.
