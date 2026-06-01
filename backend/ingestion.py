import json
import os

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "vector_store/faiss_index"
CHUNKS_PATH = "vector_store/faiss_index_chunks.json"

model = SentenceTransformer(MODEL_NAME)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks of roughly chunk_size characters."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def ingest(chunks: list[str]) -> None:
    """Embed chunks and persist FAISS index + raw chunks for BM25."""
    if not chunks:
        raise ValueError("No chunks to ingest.")

    os.makedirs("vector_store", exist_ok=True)

    # --- Dense embeddings (FAISS) ---
    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]

    # Load existing index if present, otherwise create fresh
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH) as f:
            existing_chunks = json.load(f)
    else:
        index = faiss.IndexFlatIP(dim)
        existing_chunks = []

    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

    # --- Persist raw chunks for BM25 ---
    all_chunks = existing_chunks + chunks
    with open(CHUNKS_PATH, "w") as f:
        json.dump(all_chunks, f, ensure_ascii=False)


def load_bm25(chunks: list[str]) -> BM25Okapi:
    tokenized = [c.lower().split() for c in chunks]
    return BM25Okapi(tokenized)