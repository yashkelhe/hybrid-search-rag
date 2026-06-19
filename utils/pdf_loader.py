"""
utils/pdf_loader.py
Extracts text from a PDF, page by page, using PyMuPDF (fitz).
"""

import os
import fitz  # PyMuPDF


def load_pdf(path: str) -> list[dict]:
    """
    Reads a PDF and returns one dict per non-empty page:
    {"page_num": int, "text": str, "source": filename}
    """
    doc = fitz.open(path)
    pages = []
    filename = os.path.basename(path)

    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text and text.strip():
            pages.append({
                "page_num": i + 1,
                "text": text,
                "source": filename,
            })

    doc.close()
    return pages


def load_pdfs(paths: list[str]) -> list[dict]:
    """Convenience wrapper to load multiple PDFs into a single flat page list."""
    all_pages = []
    for p in paths:
        all_pages.extend(load_pdf(p))
    return all_pages
