"""
Document ingestion service.
Reads different file formats and converts them into plain text
so the rest of the pipeline (chunking, embedding) doesn't need
to know or care what format the original file was.
"""

import os
import pandas as pd
from pypdf import PdfReader


class UnsupportedFileTypeError(Exception):
    """Raised when a file extension isn't one we know how to read."""
    pass


def load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text_parts = []
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
    return "\n\n".join(text_parts)


def load_csv(file_path: str) -> str:
    df = pd.read_csv(file_path)
    # Convert the table into a readable text block so an LLM can reason over it
    return df.to_string(index=False)


def load_excel(file_path: str) -> str:
    # Read every sheet, since enterprise spreadsheets often have more than one
    sheets = pd.read_excel(file_path, sheet_name=None)
    text_parts = []
    for sheet_name, df in sheets.items():
        text_parts.append(f"[Sheet: {sheet_name}]\n{df.to_string(index=False)}")
    return "\n\n".join(text_parts)


LOADERS = {
    ".txt": load_txt,
    ".pdf": load_pdf,
    ".csv": load_csv,
    ".xlsx": load_excel,
    ".xls": load_excel,
}


def load_document(file_path: str) -> str:
    """
    Dispatch to the right loader based on file extension.
    Raises UnsupportedFileTypeError for anything we don't handle,
    so callers (like the API layer) can return a clean 400 error
    instead of an ugly stack trace.
    """
    ext = os.path.splitext(file_path)[1].lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise UnsupportedFileTypeError(
            f"'{ext}' files are not supported. Allowed types: {list(LOADERS.keys())}"
        )
    text = loader(file_path)
    if not text or not text.strip():
        raise ValueError(f"No extractable text found in {file_path}")
    return text
