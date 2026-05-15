from __future__ import annotations

from typing import Any


def evaluate_retrieval(contexts: list[Any], ground_truth_keywords: list[str]) -> float:
    """Return 1.0 if any keyword appears in retrieved context text, else 0.0."""
    if not ground_truth_keywords:
        return 0.0
    extracted: list[str] = []
    for c in contexts:
        if isinstance(c, dict):
            extracted.append(str(c.get("text", c.get("content", ""))))
        else:
            extracted.append(str(c))
    all_text = " ".join(extracted)
    is_hit = any(kw in all_text for kw in ground_truth_keywords)
    return 1.0 if is_hit else 0.0
