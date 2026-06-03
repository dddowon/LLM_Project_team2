from __future__ import annotations

# OpenAI embeddings API hard limit is 8192 tokens; the API tokenizer can differ from tiktoken.
EMBEDDING_API_MAX_TOKENS = 8192
# Default cap with margin so cl100k_base under-counting does not trigger 400 errors.
EMBEDDING_SAFE_MAX_TOKENS = 6000
# Dense OCR/table lines can be >1 token/char; keep a strict char fallback without tiktoken.
EMBEDDING_MAX_CHARS_FALLBACK = 5_000
EMBEDDING_RETRY_TOKEN_LIMITS = (EMBEDDING_SAFE_MAX_TOKENS, 4500, 3000, 2000, 1000)
EMBEDDING_RETRY_CHAR_LIMITS = (EMBEDDING_MAX_CHARS_FALLBACK, 3500, 2000, 1000)


def truncate_text_for_embedding(
    text: str,
    *,
    model: str = "text-embedding-3-small",
    max_tokens: int = EMBEDDING_SAFE_MAX_TOKENS,
) -> tuple[str, bool]:
    """Return text safe for OpenAI embeddings and whether truncation occurred."""
    normalized = text.replace("\n", " ").strip()
    if not normalized:
        return normalized, False

    token_truncated = _truncate_with_tiktoken(normalized, model=model, max_tokens=max_tokens)
    if token_truncated is not None:
        return token_truncated

    if len(normalized) <= EMBEDDING_MAX_CHARS_FALLBACK:
        return normalized, len(normalized) < len(text.replace("\n", " ").strip())
    return normalized[:EMBEDDING_MAX_CHARS_FALLBACK], True


def truncate_text_for_embedding_retry(
    text: str,
    *,
    model: str = "text-embedding-3-small",
) -> list[str]:
    """Candidate inputs from strictest to original-safe (for progressive embed retries)."""
    normalized = text.replace("\n", " ").strip()
    if not normalized:
        return [normalized]

    seen: set[str] = set()
    candidates: list[str] = []

    def add(candidate: str) -> None:
        value = candidate.strip()
        if not value or value in seen:
            return
        seen.add(value)
        candidates.append(value)

    for max_chars in EMBEDDING_RETRY_CHAR_LIMITS:
        add(normalized[:max_chars])

    for max_tokens in EMBEDDING_RETRY_TOKEN_LIMITS:
        truncated = _truncate_with_tiktoken(normalized, model=model, max_tokens=max_tokens)
        if truncated is not None:
            add(truncated[0])

    add(normalized)
    return candidates


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
