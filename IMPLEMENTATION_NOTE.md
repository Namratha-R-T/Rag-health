# Implementation Note — Health RAG Assistant

## 1. Embedding Model Choice

**Chosen: `sentence-transformers/all-MiniLM-L6-v2`**

Reasons:
- **Lightweight and fast** — only 384 dimensions and ~22M parameters, so it embeds chunks and queries quickly on CPU with no GPU dependency, which matters since the rest of the stack (Ollama/Llama 3.2) is also expected to run locally.
- **Strong general-purpose semantic quality for its size** — it's one of the most widely benchmarked sentence-embedding models and performs well on semantic similarity/retrieval tasks despite being small.
- **Easy integration** — a single line via the `sentence-transformers` library, no API key or external service required, which keeps the whole pipeline self-contained and offline-capable (important for a *health* document where data privacy is a reasonable concern).
- **Consistency requirement** — the same model is used to embed both the document chunks and the incoming query, which is required for the vector space to be comparable; using a smaller/faster model made it cheap to re-embed everything during iteration.

Trade-off acknowledged: larger models (e.g., `bge-base`, OpenAI embeddings) would likely give modestly better retrieval accuracy, but at higher compute/latency cost and, for API-based ones, loss of the fully-local/offline property.

## 2. Storage / Index Choice

**Chosen: FAISS `IndexFlatIP` with L2-normalized embeddings (cosine similarity)**

Reasons:
- **Cosine similarity, as required** — embeddings are normalized to unit length (`normalize_embeddings=True`) before being added to the index, so the inner product FAISS computes is mathematically equivalent to cosine similarity. The same normalization is applied to the query embedding at search time, keeping both in the same comparable space.
- **Exact search** — for a single moderately-sized document (a few dozen chunks), an approximate index (IVF, HNSW) isn't necessary; brute-force inner-product search over this many vectors is fast enough (milliseconds) and guarantees the true nearest neighbors, avoiding any recall trade-off.
- **Simplicity** — `IndexFlatIP` requires no training/clustering step, unlike `IndexIVFFlat`, which needs enough data to train cluster centroids — overkill and potentially unreliable at this small scale.
- **Persistence is trivial** — `faiss.write_index` / `faiss.read_index` gives simple file-based persistence with no external database or service to run.
- **Chunk-to-vector mapping via parallel storage** — rather than a more complex metadata store, the original chunk text is kept in a pickled list where index position aligns with the FAISS vector's index. This is simple and sufficient at this scale, though it's a design point worth flagging (see Limitations).

## 3. LLM and Prompt Design

**Chosen: Llama 3.2 via Ollama (local inference)**

Reasons:
- Runs fully locally — no API cost, no data leaving the machine, appropriate for a health-related document.
- Ollama provides a simple, consistent local serving interface (`ollama.chat`) without needing to manage model weights or a custom inference server directly.

**Prompt design:**
- The prompt explicitly instructs the model to answer **ONLY** using the supplied context, directly addressing the core RAG failure mode of the model falling back on parametric/world knowledge instead of the source document.
- It specifies an **exact fallback string** (`"I could not find the answer in the provided document."`) for out-of-context questions, rather than letting the model paraphrase a refusal freely — this makes it easy to detect/handle "no answer found" cases programmatically if needed later (e.g., to trigger a different UI state).
- Context and question are clearly delimited with labeled sections (`Context:` / `Question:` / `Answer:`) to reduce ambiguity for the model about which part of the prompt is reference material versus the actual task.

## 4. What I Had to Learn / Research

- How FAISS indices work at a basic level (flat vs. IVF vs. HNSW indices) and why a flat index is appropriate for small-scale, single-document RAG rather than reaching for something more complex.
- The difference between `IndexFlatL2` (Euclidean distance) and `IndexFlatIP` (inner product), and that inner product only equals cosine similarity once vectors are L2-normalized — an easy mistake to miss since both indices "work" in the sense of returning nearest neighbors, but only one actually implements cosine similarity.
- How `sentence-transformers` produces fixed-size dense embeddings and why query and document embeddings must come from the same model (and the same normalization) to be comparable in the same vector space.
- How to detect natural section boundaries in a semi-structured document (heading-like lines, known topic keywords) and enforce a target word-count range per chunk, instead of relying on a fixed-character-count splitter — and the trade-offs involved (heuristic heading detection can occasionally misfire on a stray sentence).
- Setting up and running Ollama locally, and how to pull/serve open models like Llama 3.2 for local inference instead of relying on a hosted LLM API.
- The general shape of the "retrieve-then-generate" pattern: how retrieval indices (integer positions from FAISS) need to be mapped back to actual text via a parallel data structure, and how a grounding/anti-hallucination instruction should be embedded directly into the prompt.

## 5. Limitations & What I'd Improve With Two More Days

**Current limitations:**
- **No source citations per answer** — the answer text doesn't indicate *which* retrieved chunk(s) it actually drew from; the user has to manually compare the answer against the "Retrieved Context" chunks.
- **No re-ranking step** — chunks are returned purely by cosine similarity from the initial embedding search; a second-stage cross-encoder re-ranker would likely improve precision, especially for borderline-relevant chunks.
- **Occasional heading misdetection** — a small number of chunks inherit a stray sentence fragment as their section title instead of a clean heading (e.g., "NHM"), since heading detection is heuristic (short, title-case lines, or a known keyword match). Doesn't affect retrieval quality — only the cosmetic section label shown alongside retrieved chunks in the UI.
- **Static top_k** — `top_k` is a fixed value (5 in `app.py`/`rag.py`, 3 in `search.py`) regardless of query complexity or how many chunks are actually relevant.
- **No handling for documents changing** — running `embed.py` again requires wiping/regenerating the whole `vectorstore/`; there's no incremental update or chunk-versioning logic.
- **No evaluation harness** — there's no way currently to measure retrieval quality (e.g., recall@k against a labeled question set) or answer quality other than manual spot-checking.
- **Single-document assumption** — the pipeline hardcodes a single PDF path and doesn't support multiple documents, per-document filtering, or metadata (e.g., section/page number) attached to chunks.
- **No streaming responses** — the Streamlit app waits for the full LLM response before displaying anything, which can feel slow for longer answers.

**What I'd improve with two more days:**
1. **Add citations** — track page/section metadata alongside each chunk during ingestion so the final answer can cite "(Page X)" style references, which matters a lot for a health-policy use case where verifiability is important.
2. **Add a re-ranking step** using a small cross-encoder (e.g., `ms-marco-MiniLM`) on top of the FAISS candidates to improve the quality of what gets passed to the LLM.
3. **Build a small evaluation set** of question/expected-answer pairs to measure retrieval recall and spot-check answer faithfulness, so changes to chunking/embedding can be validated instead of eyeballed.
4. **Support multiple documents** with per-chunk source metadata (filename, page) instead of a single hardcoded PDF path.
5. **Stream the LLM response** token-by-token in the Streamlit UI for better perceived responsiveness.
6. **Store `.env`-driven configuration** (model names, chunk size, top_k) instead of hardcoded values scattered across files, so the pipeline is easier to tune without editing source.
