"""검색 성능 지표 계산 (doc hit, keyword hit, context precision)."""
from __future__ import annotations

from typing import Any

from src.engine.doc_scope import doc_keys_match


def context_text(context: Any) -> str:
    if isinstance(context, dict):
        return str(context.get("text", context.get("content", "")))
    return str(context)


def keyword_list_from_row(row: dict[str, Any]) -> list[str]:
    keywords = row.get("ground_truth_keywords")
    if keywords is None or not isinstance(keywords, list):
        return []
    return [str(keyword) for keyword in keywords if str(keyword).strip()]


def source_file_names(contexts: list[Any]) -> list[str]:
    files: list[str] = []
    for context in contexts:
        if not isinstance(context, dict):
            continue
        meta = context.get("metadata") or {}
        name = str(meta.get("file_name") or meta.get("source_file") or "").strip()
        if name and name not in files:
            files.append(name)
    return files


def source_chunk_ids(contexts: list[Any]) -> list[str]:
    ids: list[str] = []
    for context in contexts:
        if not isinstance(context, dict):
            continue
        chunk_id = str(context.get("chunk_id") or "").strip()
        if chunk_id:
            ids.append(chunk_id)
    return ids


def gold_chunk_ids_from_row(row: dict[str, Any]) -> list[str]:
    gold_chunk_ids = row.get("gold_chunk_ids")
    if not isinstance(gold_chunk_ids, list):
        return []
    return [str(chunk_id) for chunk_id in gold_chunk_ids if str(chunk_id).strip()]


def contains_keyword(text: str, ground_truth_keywords: list[str]) -> bool:
    return any(keyword in text for keyword in ground_truth_keywords)


def evaluate_keyword_hit(contexts: list[Any], ground_truth_keywords: list[str]) -> float | None:
    if not ground_truth_keywords:
        return None
    all_text = " ".join(context_text(context) for context in contexts)
    return 1.0 if contains_keyword(all_text, ground_truth_keywords) else 0.0


def evaluate_context_precision(contexts: list[Any], ground_truth_keywords: list[str]) -> float | None:
    if not contexts or not ground_truth_keywords:
        return None
    relevant_count = sum(
        1 for context in contexts if contains_keyword(context_text(context), ground_truth_keywords)
    )
    return relevant_count / len(contexts)


def evaluate_doc_hit(expected_doc_id: str, contexts: list[Any]) -> float | None:
    expected = str(expected_doc_id or "").strip()
    if not expected:
        return None
    files = source_file_names(contexts)
    if not files:
        return 0.0
    return 1.0 if any(doc_keys_match(expected, name) for name in files) else 0.0


def evaluate_recall_at_k(
    contexts: list[Any],
    gold_chunk_ids: list[str],
    *,
    k: int = 5,
) -> float | None:
    if not gold_chunk_ids:
        return None
    retrieved = set(source_chunk_ids(contexts)[:k])
    gold = {str(chunk_id) for chunk_id in gold_chunk_ids if str(chunk_id).strip()}
    if not gold:
        return None
    return len(retrieved & gold) / len(gold)


def evaluate_mrr(contexts: list[Any], gold_chunk_ids: list[str]) -> float | None:
    gold = {str(chunk_id) for chunk_id in gold_chunk_ids if str(chunk_id).strip()}
    if not gold:
        return None
    for index, chunk_id in enumerate(source_chunk_ids(contexts), start=1):
        if chunk_id in gold:
            return 1.0 / index
    return 0.0


def evaluate_retrieval_metrics(row: dict[str, Any], contexts: list[Any]) -> dict[str, Any]:
    keywords = keyword_list_from_row(row)
    gold_chunk_ids = gold_chunk_ids_from_row(row)
    files = source_file_names(contexts)
    doc_hit = evaluate_doc_hit(str(row.get("doc_id") or ""), contexts)
    keyword_hit = evaluate_keyword_hit(contexts, keywords)
    precision = evaluate_context_precision(contexts, keywords)
    recall_at_5 = evaluate_recall_at_k(contexts, gold_chunk_ids, k=5)
    mrr = evaluate_mrr(contexts, gold_chunk_ids)

    return {
        "doc_hit": doc_hit,
        "retrieval_keyword_hit": keyword_hit,
        "context_precision": precision,
        "recall_at_5": recall_at_5,
        "mrr": mrr,
        "retrieved_files": " | ".join(files[:5]) + (" ..." if len(files) > 5 else ""),
    }


# Legacy alias
evaluate_retrieval = evaluate_keyword_hit
