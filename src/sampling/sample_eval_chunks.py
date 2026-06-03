#!/usr/bin/env python3
"""Sample evaluation chunks from a slim RAG chunk JSONL.

This is a standalone script. It does not import project-local Python files.

Expected input row shape:
  {
    "chunk_id": "...",
    "chunk_type": "section_text",
    "chunk_text": "...",
    "metadata": {
      "file_name": "...",
      "section_path_text": "...",
      "section_type": "requirements",
      "heading": "...",
      "part_index": 1
    }
  }

Default per-document quotas:
  overview=1, requirements=4, evaluation=2, bid_contract=2, security=1

appendix_form is optional:
  --appendix-mode never   : never sample appendix_form
  --appendix-mode auto    : use appendix_form only when core samples are sparse
  --appendix-mode always  : sample up to appendix_form=1 for every document
  
  metadata.section_type 기준으로 아래 quota만큼 샘플링
overview: 1
requirements: 4
evaluation: 2
bid_contract: 2
security: 있으면 1
appendix_form: 기본은 auto, 핵심 chunk가 부족한 문서에서만 1개 보충
  
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("eda/hwp_text_chunks_slim.jsonl")
DEFAULT_OUTPUT = Path("eda/eval_sample_chunks.jsonl")

SECTION_ORDER = [
    "overview",
    "requirements",
    "evaluation",
    "bid_contract",
    "security",
    "appendix_form",
]

DEFAULT_QUOTAS = {
    "overview": 1,
    "requirements": 4,
    "evaluation": 2,
    "bid_contract": 2,
    "security": 1,
    "appendix_form": 1,
}


@dataclass(frozen=True)
class Candidate:
    index: int
    row: dict[str, Any]
    doc_id: str
    section_type: str
    section_path: str
    chunk_id: str
    text_len: int
    part_index: int
    score: int


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_int(value: Any, default: int = 9999) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_doc_id(row: dict[str, Any]) -> str:
    meta = metadata(row)
    for key in ("file_name", "source_file", "doc_id"):
        value = meta.get(key) or row.get(key)
        if value:
            return str(value)
    return "unknown_document"


def get_section_type(row: dict[str, Any]) -> str:
    meta = metadata(row)
    value = meta.get("section_type") or row.get("section_type")
    return str(value or "body")


def get_section_path(row: dict[str, Any]) -> str:
    meta = metadata(row)
    value = meta.get("section_path_text") or row.get("section_path_text")
    if value:
        return str(value)
    path = meta.get("section_path") or row.get("section_path")
    if isinstance(path, list):
        return " > ".join(str(item).strip() for item in path if str(item).strip())
    return str(path or "")


def get_chunk_id(row: dict[str, Any], index: int) -> str:
    value = row.get("chunk_id") or row.get("id")
    return str(value or f"chunk_{index:08d}")


def score_candidate(row: dict[str, Any]) -> int:
    meta = metadata(row)
    text = clean_text(row.get("chunk_text") or row.get("text"))
    text_len = len(text)
    chunk_type = str(row.get("chunk_type") or "")
    part_index = parse_int(meta.get("part_index"), default=9999)
    section_path = get_section_path(row)

    score = 0

    if chunk_type == "section_text":
        score += 30
    elif chunk_type == "cover_text":
        score -= 20

    if 250 <= text_len <= 1600:
        score += 25
    elif 120 <= text_len < 250:
        score += 12
    elif 1600 < text_len <= 2800:
        score += 8
    elif text_len < 120:
        score -= 35
    else:
        score -= 10

    if section_path:
        score += 8

    if part_index == 1:
        score += 8
    else:
        score -= min(part_index, 15)

    return score


def build_candidates(rows: list[dict[str, Any]], *, min_chars: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for index, row in enumerate(rows):
        text = clean_text(row.get("chunk_text") or row.get("text"))
        if len(text) < min_chars:
            continue
        candidates.append(
            Candidate(
                index=index,
                row=row,
                doc_id=get_doc_id(row),
                section_type=get_section_type(row),
                section_path=get_section_path(row),
                chunk_id=get_chunk_id(row, index),
                text_len=len(text),
                part_index=parse_int(metadata(row).get("part_index"), default=9999),
                score=score_candidate(row),
            )
        )
    return candidates


def parse_quota_config(raw: str | None) -> dict[str, int]:
    quotas = dict(DEFAULT_QUOTAS)
    if not raw:
        return quotas

    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid quota item: {item!r}. Use section_type=count.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid quota key in item: {item!r}")
        quotas[key] = int(value.strip())
    return quotas


def select_diverse(candidates: list[Candidate], quota: int, used_ids: set[str]) -> list[Candidate]:
    if quota <= 0:
        return []

    pool = [candidate for candidate in candidates if candidate.chunk_id not in used_ids]
    pool.sort(
        key=lambda candidate: (
            -candidate.score,
            candidate.section_path,
            candidate.part_index,
            candidate.index,
        )
    )

    selected: list[Candidate] = []
    seen_paths: set[str] = set()

    for candidate in pool:
        path_key = candidate.section_path or f"__row_{candidate.index}"
        if path_key in seen_paths:
            continue
        selected.append(candidate)
        seen_paths.add(path_key)
        if len(selected) >= quota:
            return selected

    selected_ids = {candidate.chunk_id for candidate in selected}
    for candidate in pool:
        if candidate.chunk_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.chunk_id)
        if len(selected) >= quota:
            return selected

    return selected


def sample_document(
    doc_candidates: list[Candidate],
    *,
    quotas: dict[str, int],
    appendix_mode: str,
    min_per_doc: int,
    fallback_body: int,
) -> list[Candidate]:
    by_type: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in doc_candidates:
        by_type[candidate.section_type].append(candidate)

    selected: list[Candidate] = []
    used_ids: set[str] = set()

    for section in SECTION_ORDER:
        if section == "appendix_form":
            continue
        picked = select_diverse(by_type.get(section, []), quotas.get(section, 0), used_ids)
        selected.extend(picked)
        used_ids.update(candidate.chunk_id for candidate in picked)

    should_add_appendix = False
    if appendix_mode == "always":
        should_add_appendix = True
    elif appendix_mode == "auto" and len(selected) < min_per_doc:
        should_add_appendix = True

    if should_add_appendix:
        picked = select_diverse(by_type.get("appendix_form", []), quotas.get("appendix_form", 1), used_ids)
        selected.extend(picked)
        used_ids.update(candidate.chunk_id for candidate in picked)

    if fallback_body > 0 and len(selected) < min_per_doc:
        needed = min(fallback_body, min_per_doc - len(selected))
        picked = select_diverse(by_type.get("body", []), needed, used_ids)
        selected.extend(picked)

    selected.sort(
        key=lambda candidate: (
            SECTION_ORDER.index(candidate.section_type)
            if candidate.section_type in SECTION_ORDER
            else len(SECTION_ORDER),
            candidate.section_path,
            candidate.part_index,
            candidate.index,
        )
    )
    return selected


def normalize_output_row(
    candidate: Candidate,
    *,
    add_sampling_metadata: bool,
    sample_rank_in_doc: int,
    sample_rank_in_section: int,
) -> dict[str, Any]:
    row = candidate.row
    meta = dict(metadata(row))
    if add_sampling_metadata:
        meta.update(
            {
                "sample_strategy": "section_type_quota",
                "sample_section_type": candidate.section_type,
                "sample_rank_in_doc": sample_rank_in_doc,
                "sample_rank_in_section": sample_rank_in_section,
            }
        )

    return {
        "chunk_id": candidate.chunk_id,
        "chunk_type": str(row.get("chunk_type") or candidate.section_type),
        "chunk_text": clean_text(row.get("chunk_text") or row.get("text")),
        "metadata": meta,
    }


def sample_rows(
    rows: list[dict[str, Any]],
    *,
    quotas: dict[str, int],
    appendix_mode: str,
    min_chars: int,
    min_per_doc: int,
    fallback_body: int,
    limit_docs: int | None,
    add_sampling_metadata: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = build_candidates(rows, min_chars=min_chars)

    doc_order: list[str] = []
    by_doc: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.doc_id not in by_doc:
            doc_order.append(candidate.doc_id)
        by_doc[candidate.doc_id].append(candidate)

    if limit_docs is not None:
        doc_order = doc_order[:limit_docs]

    sampled_rows: list[dict[str, Any]] = []
    doc_counts: Counter[str] = Counter()
    section_counts: Counter[str] = Counter()

    for doc_id in doc_order:
        selected = sample_document(
            by_doc[doc_id],
            quotas=quotas,
            appendix_mode=appendix_mode,
            min_per_doc=min_per_doc,
            fallback_body=fallback_body,
        )

        rank_by_section: Counter[str] = Counter()
        for rank_in_doc, candidate in enumerate(selected, start=1):
            rank_by_section[candidate.section_type] += 1
            sampled_rows.append(
                normalize_output_row(
                    candidate,
                    add_sampling_metadata=add_sampling_metadata,
                    sample_rank_in_doc=rank_in_doc,
                    sample_rank_in_section=rank_by_section[candidate.section_type],
                )
            )
            doc_counts[doc_id] += 1
            section_counts[candidate.section_type] += 1

    summary = {
        "input_rows": len(rows),
        "candidate_rows": len(candidates),
        "documents": len(doc_order),
        "sampled_rows": len(sampled_rows),
        "section_counts": dict(section_counts),
        "min_doc_samples": min(doc_counts.values()) if doc_counts else 0,
        "max_doc_samples": max(doc_counts.values()) if doc_counts else 0,
        "avg_doc_samples": round(sum(doc_counts.values()) / len(doc_counts), 2) if doc_counts else 0,
    }
    return sampled_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample per-document evaluation chunks by section_type quota."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input slim chunk JSONL.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output sampled JSONL.")
    parser.add_argument(
        "--quotas",
        default=None,
        help=(
            "Comma-separated quota override. Example: "
            "overview=1,requirements=4,evaluation=2,bid_contract=2,security=1,appendix_form=1"
        ),
    )
    parser.add_argument(
        "--appendix-mode",
        choices=("auto", "always", "never"),
        default="auto",
        help="How to sample appendix_form chunks.",
    )
    parser.add_argument(
        "--min-per-doc",
        type=int,
        default=9,
        help="Auto appendix/body fallback target. Does not force duplicate chunks.",
    )
    parser.add_argument(
        "--fallback-body",
        type=int,
        default=0,
        help="If a document is sparse, sample up to this many body chunks as fallback.",
    )
    parser.add_argument("--min-chars", type=int, default=80, help="Drop chunks shorter than this.")
    parser.add_argument("--limit-docs", type=int, default=None, help="Debug only: first N documents.")
    parser.add_argument(
        "--add-sampling-metadata",
        action="store_true",
        help="Add sample_strategy/sample_rank fields to metadata.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    quotas = parse_quota_config(args.quotas)
    rows = read_jsonl(args.input)
    sampled_rows, summary = sample_rows(
        rows,
        quotas=quotas,
        appendix_mode=args.appendix_mode,
        min_chars=args.min_chars,
        min_per_doc=args.min_per_doc,
        fallback_body=args.fallback_body,
        limit_docs=args.limit_docs,
        add_sampling_metadata=args.add_sampling_metadata,
    )
    write_jsonl(args.output, sampled_rows)

    print(f"input_rows: {summary['input_rows']}")
    print(f"candidate_rows: {summary['candidate_rows']}")
    print(f"documents: {summary['documents']}")
    print(f"sampled_rows: {summary['sampled_rows']}")
    print(f"doc_samples: min={summary['min_doc_samples']} avg={summary['avg_doc_samples']} max={summary['max_doc_samples']}")
    print(f"section_counts: {json.dumps(summary['section_counts'], ensure_ascii=False, sort_keys=True)}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
