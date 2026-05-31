import json
import os

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "/tmp/vector_store/faiss_index"
CHUNKS_PATH = "/tmp/vector_store/faiss_index_chunks.json"

model = SentenceTransformer(MODEL_NAME)


def load_artifacts() -> tuple[faiss.Index, BM25Okapi, list[str]]:
    """Load FAISS index, BM25 index, and raw chunks from disk."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            "No index found. Please upload a document first via POST /upload."
        )

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)

    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    return index, bm25, chunks


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[int]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.
    score(doc) = sum over lists of: weight * 1 / (k + rank)
    k=60 is the standard default from the original RRF paper.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranked_list, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


def hybrid_retrieve(
    query: str,
    index: faiss.Index,
    bm25: BM25Okapi,
    chunks: list[str],
    top_k: int = 5,
    candidate_k: int = 20,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[str]:
    """
    Retrieve top_k chunks using hybrid BM25 + FAISS search.
    Each retriever fetches candidate_k results; RRF merges the ranked lists.
    """
    if not query.strip():
        raise ValueError("Query must not be empty.")

    candidate_k = min(candidate_k, index.ntotal)

    # --- Dense retrieval (FAISS) ---
    q_emb = model.encode([query])
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    _, dense_ids = index.search(q_emb, candidate_k)
    dense_ranking = [int(i) for i in dense_ids[0] if i >= 0]

    # --- Sparse retrieval (BM25) ---
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranking = np.argsort(bm25_scores)[::-1][:candidate_k].tolist()

    # --- Fuse and return top chunks ---
    fused_ids = reciprocal_rank_fusion(
        [dense_ranking, bm25_ranking],
        weights=[dense_weight, sparse_weight],
    )
    return [chunks[i] for i in fused_ids[:top_k]]