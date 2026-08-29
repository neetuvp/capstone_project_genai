"""
Guardrails and reliability controls.
Two jobs here:
  1. Input validation - reject bad uploads before they cause problems downstream.
  2. Output validation - catch likely hallucinations / unsafe answers
     before they're returned to the user.
These are intentionally simple, rule-based checks (not another LLM call)
so they're fast, predictable, and easy to explain in your documentation.
"""

import os

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE_MB = 20

REFUSAL_PHRASES = [
    "i don't have enough information",
    "cannot answer",
    "not contained in the context",
]


class ValidationError(Exception):
    """Raised when an input fails validation. Caught by the API layer
    and turned into a clean 400 response instead of a 500 crash."""
    pass


def validate_upload(file_path: str, file_size_bytes: int) -> None:
    """Raises ValidationError if the upload doesn't meet basic requirements."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"File type '{ext}' is not allowed. Allowed types: {sorted(ALLOWED_EXTENSIONS)}"
        )

    size_mb = file_size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValidationError(
            f"File is {size_mb:.1f} MB, which exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )


def validate_question(question: str) -> None:
    """Raises ValidationError for empty or suspiciously long questions."""
    if not question or not question.strip():
        raise ValidationError("Question cannot be empty.")
    if len(question) > 2000:
        raise ValidationError("Question is too long (max 2000 characters).")


def validate_answer(answer: str, retrieved_chunks: list[dict]) -> dict:
    """
    Post-generation check. Doesn't block the answer, but flags signals that suggest
    the answer might be ungrounded, so the UI can show a warning.
    """
    flags = []

    if not answer or not answer.strip():
        flags.append("empty_answer")

    answer_lower = answer.lower() if answer else ""
    is_refusal = any(phrase in answer_lower for phrase in REFUSAL_PHRASES)

    if not retrieved_chunks and not is_refusal:
        # No context was found, but the model didn't admit it — possible hallucination
        flags.append("no_context_but_answered")

    return {
        "passed": len(flags) == 0,
        "flags": flags,
        "is_refusal": is_refusal,
    }
