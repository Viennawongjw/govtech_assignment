"""PDF text extraction. Born-digital only; OCR is out of scope."""
from __future__ import annotations

import fitz  # PyMuPDF


def extract_text(pdf_path: str) -> str:
    """Return page-tagged text. Page tags help the LLM keep multi-page questions together."""
    doc = fitz.open(pdf_path)
    chunks: list[str] = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        chunks.append(f"<<PAGE {i}>>\n{text}")
    doc.close()
    return "\n\n".join(chunks)


def has_text_layer(pdf_path: str, min_chars: int = 200) -> bool:
    """True if the PDF has an extractable text layer (i.e. not a scan)."""
    doc = fitz.open(pdf_path)
    total = sum(len(page.get_text("text")) for page in doc)
    doc.close()
    return total >= min_chars
