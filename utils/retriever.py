"""
utils/retriever.py
Ties together query embedding (dense + sparse) and Pinecone hybrid search.
"""

from utils.embeddings import embed_query
from utils.sparse_encoder import BM25Sparse
from utils.pinecone_db import get_index, hybrid_scale
from config import BM25_PARAMS_PATH, TOP_K_DENSE, ALPHA

_bm25 = BM25Sparse()
_bm25.load(BM25_PARAMS_PATH)  # must have been created by ingest.py first


def retrieve(query: str, top_k: int = TOP_K_DENSE, alpha: float = ALPHA) -> list[dict]:
    dense = embed_query(query)
    sparse = _bm25.encode_query(query)
    hdense, hsparse = hybrid_scale(dense, sparse, alpha)

    index = get_index()
    results = index.query(
        vector=hdense,
        sparse_vector=hsparse,
        top_k=top_k,
        include_metadata=True,
    )

    chunks = []
    for match in results["matches"]:
        metadata = match["metadata"] or {}
        chunks.append({
            "id": match["id"],
            "score": match["score"],
            "text": metadata.get("text", ""),
            "metadata": metadata,
        })
    return chunks
