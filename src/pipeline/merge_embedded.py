"""Merge per-document embedded JSONL files into one file and build unified Chroma index."""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_OCR_EMBEDDED_REL = Path("ocr_rag/ocr_input_embedded.jsonl")

from src.utils.jsonl import read_jsonl, write_jsonl


@dataclass(frozen=True)
class MergeEmbeddedResult:
    source_files: int
    total_rows: int
    duplicate_chunk_ids: int
    merged_path: Path
    index_dir: Path
    chunks_in_index: int


def discover_embedded_jsonl(
    root: Path,
    *,
    pattern: str = "*_embedded.jsonl",
    recursive: bool = True,
) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"입력 디렉터리가 없습니다: {root}")

    if recursive:
        paths = sorted(root.rglob(pattern))
    else:
        paths = sorted(root.glob(pattern))

    return [path for path in paths if path.is_file()]


def merge_embedded_jsonl(
    input_paths: list[Path],
    output_path: Path,
    *,
    dedupe_chunk_ids: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    """Merge embedded rows; return merged rows and count of chunk_id collisions rewritten."""
    if not input_paths:
        raise RuntimeError("병합할 embedded JSONL 파일이 없습니다.")

    seen_ids: set[str] = set()
    merged: list[dict[str, Any]] = []
    collisions = 0
    expected_dim: int | None = None

    for source_path in input_paths:
        rows = read_jsonl(source_path)
        if not rows:
            continue

        for row in rows:
            embedding = row.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise ValueError(f"{source_path}에 embedding이 없는 row가 있습니다.")

            dim = len(embedding)
            if expected_dim is None:
                expected_dim = dim
            elif dim != expected_dim:
                raise ValueError(
                    f"embedding 차원이 파일마다 다릅니다: {source_path}={dim}, expected={expected_dim}"
                )

            out = dict(row)
            chunk_id = str(out.get("chunk_id") or out.get("id") or "").strip()
            if not chunk_id:
                chunk_id = f"chunk_{len(merged):08d}"

            if dedupe_chunk_ids:
                unique_id = _unique_chunk_id(chunk_id, seen_ids, str(source_path))
                if unique_id != chunk_id:
                    collisions += 1
                    out["chunk_id"] = unique_id
                else:
                    out["chunk_id"] = chunk_id
                    seen_ids.add(chunk_id)
            else:
                if chunk_id in seen_ids:
                    raise ValueError(f"중복 chunk_id: {chunk_id} ({source_path})")
                seen_ids.add(chunk_id)
                out["chunk_id"] = chunk_id

            merged.append(out)

    if not merged:
        raise RuntimeError("병합 결과 row가 0개입니다. embedded JSONL 내용을 확인하세요.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, merged)
    return merged, collisions


def build_unified_chroma_index(
    *,
    input_dir: Path,
    index_dir: Path,
    merged_output: Path | None = None,
    pattern: str = "*_embedded.jsonl",
    recursive: bool = True,
    dedupe_chunk_ids: bool = True,
    skip_chroma_build: bool = False,
) -> MergeEmbeddedResult:
    from src.pipeline.embedding_pipeline import build_chroma_from_embedded_jsonl

    sources = discover_embedded_jsonl(input_dir, pattern=pattern, recursive=recursive)
    if not sources:
        raise RuntimeError(f"embedded JSONL을 찾지 못했습니다: {input_dir} ({pattern})")

    ocr_embedded = (input_dir / DEFAULT_OCR_EMBEDDED_REL).resolve()
    if not ocr_embedded.is_file():
        warnings.warn(
            f"OCR embedded가 없습니다: {ocr_embedded}\n"
            "  → ./scripts/run_ocr_stage.sh 후 ./scripts/run_rag_stage.sh 실행 후 merge-embedded 재실행",
            UserWarning,
            stacklevel=2,
        )
    elif ocr_embedded not in {path.resolve() for path in sources}:
        warnings.warn(
            f"OCR embedded가 병합 목록에 없습니다: {ocr_embedded} (pattern={pattern})",
            UserWarning,
            stacklevel=2,
        )

    out_path = merged_output or (index_dir.parent / "all_embedded.jsonl")
    _, collisions = merge_embedded_jsonl(sources, out_path, dedupe_chunk_ids=dedupe_chunk_ids)

    chunks_in_index = 0
    if not skip_chroma_build:
        chunks_in_index = build_chroma_from_embedded_jsonl(
            input_path=out_path,
            index_dir=index_dir,
        )

    return MergeEmbeddedResult(
        source_files=len(sources),
        total_rows=len(read_jsonl(out_path)),
        duplicate_chunk_ids=collisions,
        merged_path=out_path,
        index_dir=index_dir,
        chunks_in_index=chunks_in_index,
    )


def _unique_chunk_id(chunk_id: str, seen_ids: set[str], source_hint: str) -> str:
    if chunk_id not in seen_ids:
        return chunk_id

    digest = hashlib.sha1(f"{source_hint}:{chunk_id}".encode("utf-8")).hexdigest()[:8]
    candidate = f"{chunk_id}_{digest}"
    counter = 1
    while candidate in seen_ids:
        candidate = f"{chunk_id}_{digest}_{counter}"
        counter += 1
    seen_ids.add(candidate)
    return candidate
