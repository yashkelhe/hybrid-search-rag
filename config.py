"""
config.py
Central place for API keys, model names, and tunable parameters.
Everything else in the project imports from here so you only change values once.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = str(BASE_DIR / "data")
BM25_PARAMS_PATH = str(BASE_DIR / "data" / "bm25_params.json")

# ---------------- API Keys ----------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# ---------------- Pinecone ----------------
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hybrid-search-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

#  gemini-embedding-001 outputs 768-dim vectors. If you switch embedding models,
# update this to match (and you'll need to delete + recreate the Pinecone index).
EMBEDDING_DIM = 768

# ---------------- Models ----------------
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
RERANK_MODEL = "rerank-v3.5"

# ---------------- Chunking ----------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ---------------- Retrieval ----------------
TOP_K_DENSE = 20      # how many candidates to pull from Pinecone hybrid search
TOP_K_RERANK = 5       # how many chunks survive Cohere rerank
ALPHA = 0.5             # hybrid weight: 1.0 = pure dense, 0.0 = pure sparse (BM25)

# Sparse vector dimensionality (hash space for BM25 tokens). 2^18 keeps collisions low.
SPARSE_DIM = 2 ** 18

# ---------------- Sanity checks ----------------
for _name, _val in [
    ("GOOGLE_API_KEY", GOOGLE_API_KEY),
    ("PINECONE_API_KEY", PINECONE_API_KEY),
    ("COHERE_API_KEY", COHERE_API_KEY),
]:
    if not _val:
        print(f"[config] WARNING: {_name} is not set. Add it to your .env file.")
