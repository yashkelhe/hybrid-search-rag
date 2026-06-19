"""
utils/embeddings.py
Generates dense embeddings using Google Gemini's embedding model.
Documents and queries use different task_type values, which Gemini
uses internally to optimize the embedding for retrieval.
"""

from google import genai
from google.genai import types
from config import GOOGLE_API_KEY, EMBEDDING_MODEL

_client = genai.Client(api_key=GOOGLE_API_KEY)


def embed_documents(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    """Embeds a list of document chunks. Batches requests to stay under API limits."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = _client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        all_embeddings.extend([e.values for e in result.embeddings])
    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Embeds a single user query."""
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values
