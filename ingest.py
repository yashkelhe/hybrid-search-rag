"""
ingest.py
Run this once (or whenever you add new PDFs) to populate Pinecone.

Flow:
PDF -> extract text -> chunk -> dense embeddings (Gemini) -> sparse vectors (BM25)
    -> upsert (dense + sparse + metadata + original text) into Pinecone

Usage:
    python ingest.py
(reads every .pdf file in the data/ folder)
"""

import glob
import os

from tqdm import tqdm

from config import DATA_DIR, BM25_PARAMS_PATH
from utils.pdf_loader import load_pdfs
from utils.chunker import chunk_pages
from utils.embeddings import embed_documents
from utils.sparse_encoder import BM25Sparse
from utils.pinecone_db import ensure_index


def main():
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DATA_DIR}. Drop some .pdf files there and re-run.")
        return

    print(f"Found {len(pdf_files)} PDF(s): {[os.path.basename(p) for p in pdf_files]}")

    print("Extracting text...")
    pages = load_pdfs(pdf_files)
    print(f"Extracted {len(pages)} non-empty pages.")

    print("Chunking...")
    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks.")

    texts = [c["text"] for c in chunks]

    print("Fitting BM25 sparse encoder on the corpus...")
    bm25 = BM25Sparse()
    bm25.fit(texts)
    bm25.save(BM25_PARAMS_PATH)
    print(f"Saved BM25 params to {BM25_PARAMS_PATH}")

    print("Generating dense embeddings (Gemini)...")
    dense_vectors = embed_documents(texts)

    print("Generating sparse vectors (BM25)...")
    sparse_vectors = bm25.encode_documents(texts)

    print("Ensuring Pinecone index exists...")
    index = ensure_index()

    print("Upserting vectors to Pinecone...")
    vectors = []
    for i, chunk in enumerate(chunks):
        vectors.append({
            "id": chunk["metadata"]["chunk_id"],
            "values": dense_vectors[i],
            "sparse_values": sparse_vectors[i],
            "metadata": {**chunk["metadata"], "text": chunk["text"]},
        })

    batch_size = 50
    for i in tqdm(range(0, len(vectors), batch_size)):
        index.upsert(vectors=vectors[i:i + batch_size])

    print(f"\nDone. Upserted {len(vectors)} chunks into '{index}'.")


if __name__ == "__main__":
    main()
