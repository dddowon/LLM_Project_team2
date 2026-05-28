from __future__ import annotations

# text-embedding-3-* API limit is 8192 tokens per input.
EMBEDDING_MAX_TOKENS = 8191
# Dense OCR/table text can tokenize heavily; keep a conservative char cap without tiktoken.
EMBEDDING_MAX_CHARS_FALLBACK = 12_000


def truncate_text_for_embedding(
    text: str,
    *,
    model: str = "text-embedding-3-small",
    max_tokens: int = EMBEDDING_MAX_TOKENS,
) -> tuple[str, bool]:
    """Return text safe for OpenAI embeddings and whether truncation occurred."""
    normalized = text.replace("\n", " ").strip()
    if not normalized:
        return normalized, False

    token_truncated = _truncate_with_tiktoken(normalized, model=model, max_tokens=max_tokens)
    if token_truncated is not None:
        return token_truncated

    if len(normalized) <= EMBEDDING_MAX_CHARS_FALLBACK:
        return normalized, False
    return normalized[:EMBEDDING_MAX_CHARS_FALLBACK], True


def _truncate_with_tiktoken(
    text: str,
    *,
    model: str,
    max_tokens: int,
) -> tuple[str, bool] | None:
    try:
        import tiktoken
    except ImportError:
        return None

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text, False
    return encoding.decode(tokens[:max_tokens]), True
