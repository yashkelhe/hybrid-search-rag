"""
utils/pinecone_db.py
Manages the Pinecone index and provides the alpha-scaling helper required
for convex-combination hybrid search (Pinecone does not scale dense/sparse
vectors for you - you must do it before querying).
"""

from pinecone import Pinecone, ServerlessSpec
from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_DIM,
)

_pc = Pinecone(api_key=PINECONE_API_KEY)


def ensure_index():
    """Creates the index if it doesn't exist yet. Must use metric='dotproduct' for hybrid search."""
    existing = _pc.list_indexes().names()
    if PINECONE_INDEX_NAME not in existing:
        print(f"[pinecone_db] Creating index '{PINECONE_INDEX_NAME}'...")
        _pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="dotproduct",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    return _pc.Index(PINECONE_INDEX_NAME)


def get_index():
    return _pc.Index(PINECONE_INDEX_NAME)


def hybrid_scale(dense: list[float], sparse: dict, alpha: float):
    """
    Scales dense and sparse vectors by alpha / (1 - alpha) before a hybrid query.
    alpha = 1.0 -> pure dense search
    alpha = 0.0 -> pure sparse (keyword/BM25) search
    alpha = 0.5 -> balanced hybrid (default)
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    hsparse = {
        "indices": sparse["indices"],
        "values": [v * (1 - alpha) for v in sparse["values"]],
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse
