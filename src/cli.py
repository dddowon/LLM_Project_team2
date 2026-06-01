from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from pathlib import Path

DEFAULT_OCR_ENGINE_MATRIX = (
    "pp_ocrv5",
    "pp_ocrv5_transformers",
    "pp_structurev3",
    "table_recognition_v2",
    "paddleocr_vl",
)


def _normalize_ocr_engine(ocr_engine: str) -> str:
    key = str(ocr_engine).strip().lower()
    if key in DEFAULT_OCR_ENGINE_MATRIX:
        return key
    supported = ", ".join(DEFAULT_OCR_ENGINE_MATRIX)
    raise ValueError(
        f"Unsupported OCR engine: {ocr_engine}. Use one of: {supported}"
    )


def _require_exactly_one(*, a, b, a_name: str, b_name: str) -> None:
    if bool(a) == bool(b):
        raise SystemExit(f"Provide exactly one of {a_name} or {b_name}.")


def _discover_hwp_in_dir(input_dir: str | Path) -> list[Path]:
    from src.Parsing.parsing import discover_hwp_files

    input_files = discover_hwp_files(Path(input_dir), glob_pattern="*.hwp", recursive=False)
    if not input_files:
        raise SystemExit(f"No HWP files found in: {input_dir}")
    return input_files


def _discover_hwp_pdf_in_dir(
    input_dir: str | Path,
    *,
    glob_pattern: str = "*",
    recursive: bool = False,
    limit_files: int = 0,
) -> list[Path]:
    from src.pipeline.mixed_slim_pipeline import discover_source_files

    input_files = discover_source_files(Path(input_dir), glob_pattern=glob_pattern, recursive=recursive)
    if limit_files:
        input_files = input_files[:limit_files]
    if not input_files:
        raise SystemExit(f"No HWP/PDF files found in: {input_dir}")
    return input_files


def ingest(config_path: str) -> None:
    from tqdm import tqdm

    from src.config import load_config
    from src.dataset.loaders import load_documents
    from src.engine.vector_store import ChromaVectorStore
    from src.models.openai_client import OpenAIModelClient
    from src.preprocessing.chunker import chunk_documents

    config = load_config(config_path)
    documents = load_documents(config.paths.raw_data_dir, config.paths.metadata_csv)
    chunks = chunk_documents(
        documents,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        min_chunk_chars=config.chunking.min_chunk_chars,
    )
    if not chunks:
        raise RuntimeError("생성된 청크가 없습니다. raw_data_dir와 metadata_csv 경로를 확인해 주세요.")

    client = OpenAIModelClient()
    texts = [chunk.text for chunk in chunks]
    embeddings = []
    for start in tqdm(range(0, len(texts), 64), desc="Embedding chunks"):
        embeddings.extend(
            client.embed_texts(texts[start : start + 64], config.openai.embedding_model, batch_size=64)
        )
    store = ChromaVectorStore.build(chunks, embeddings)
    store.save(config.paths.index_dir)
    print(f"Indexed {len(documents)} documents / {len(chunks)} chunks -> {config.paths.index_dir}")


def query(config_path: str, question: str) -> None:
    from src.config import load_config
    from src.engine.rag import RagEngine
    from src.engine.vector_store import ChromaVectorStore

    config = load_config(config_path)
    store = ChromaVectorStore.load(config.paths.index_dir)
    engine = RagEngine(config, store)
    result = engine.answer(question)
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source['chunk_id']} score={source['score']:.4f}")


def evaluate(config_path: str) -> None:
    from tqdm import tqdm

    from src.config import load_config
    from src.engine.rag import RagEngine
    from src.engine.vector_store import ChromaVectorStore
    from src.utils.jsonl import read_jsonl, write_jsonl

    config = load_config(config_path)
    questions = read_jsonl(config.paths.evaluation_set)
    if not questions:
        raise RuntimeError(f"평가 질문셋이 없습니다: {config.paths.evaluation_set}")

    store = ChromaVectorStore.load(config.paths.index_dir)
    engine = RagEngine(config, store)
    rows = []
    for item in tqdm(questions, desc="Evaluating"):
        question = item["question"]
        doc_id = str(item.get("doc_id") or "").strip() or None
        question_type = str(item.get("question_type") or "").strip() or None
        category = str(item.get("category") or "").strip() or None
        result = engine.answer(
            question,
            doc_id=doc_id,
            question_type=question_type,
            category=category,
        )
        rows.append({**item, **result})
    write_jsonl(config.paths.evaluation_output, rows)
    print(f"Wrote evaluation results -> {config.paths.evaluation_output}")


def evaluate_harness(
    config_path: str,
    *,
    output_path: str | None,
    evaluation_set: str | None,
    judge_model: str,
    no_llm_judge: bool,
    no_correctness_judge: bool,
    no_langsmith_feedback: bool,
) -> None:
    from src.evaluation.langsmith_harness import run_eval_harness

    out, summary = run_eval_harness(
        config_path,
        evaluation_set=Path(evaluation_set) if evaluation_set else None,
        output_path=Path(output_path) if output_path else None,
        judge_model=judge_model,
        run_llm_judge=not no_llm_judge,
        run_correctness_judge=not no_correctness_judge,
        langsmith_feedback=not no_langsmith_feedback,
    )
    print(f"Wrote harness evaluation -> {out}")
    print(f"Summary: {summary}")


def embed_jsonl(
    input_path: str,
    output_path: str,
    model: str,
    batch_size: int = 64,
    force_real: bool = False,
) -> None:
    from src.pipeline.embedding_pipeline import embed_prechunked_jsonl

    count = embed_prechunked_jsonl(
        input_path=Path(input_path),
        output_path=Path(output_path),
        model=model,
        batch_size=batch_size,
        force_real=force_real,
    )
    print(f"Embedded {count} rows -> {output_path}")


def resolve_index_dir(config_path: str, index_dir: str | None):
    if index_dir:
        return Path(index_dir)

    from src.config import load_config

    return load_config(config_path).paths.index_dir


def build_chroma_index(input_path: str, index_dir: str, doc_id: str | None = None) -> None:
    from src.pipeline.embedding_pipeline import build_chroma_from_embedded_jsonl

    count = build_chroma_from_embedded_jsonl(
        input_path=Path(input_path),
        index_dir=Path(index_dir),
        doc_id=doc_id,
    )
    print(f"Built Chroma index with {count} chunks -> {index_dir}")


def merge_embedded_checkpoint(
    config_path: str,
    *,
    input_dir: str,
    index_dir: str | None,
    merged_output: str | None,
    pattern: str,
    recursive: bool,
    merge_only: bool,
) -> None:
    from src.pipeline.merge_embedded import build_unified_chroma_index

    resolved_index = Path(resolve_index_dir(config_path, index_dir))
    result = build_unified_chroma_index(
        input_dir=Path(input_dir),
        index_dir=resolved_index,
        merged_output=Path(merged_output) if merged_output else None,
        pattern=pattern,
        recursive=recursive,
        skip_chroma_build=merge_only,
    )
    print(f"merged_sources: {result.source_files}")
    print(f"merged_rows: {result.total_rows}")
    print(f"duplicate_chunk_ids_rewritten: {result.duplicate_chunk_ids}")
    print(f"merged_jsonl: {result.merged_path}")
    if merge_only:
        print("skip_chroma_build: True (merge JSONL only)")
    else:
        print(f"index_dir: {result.index_dir}")
        print(f"chunks_in_index: {result.chunks_in_index}")


def parse_hwp(
    output_path: str,
    *,
    input_path: str | None = None,
    input_dir: str | None = None,
    debug_headings: str | None = None,
    limit: int = 0,
    group_size: int = 8,
) -> None:
    from src.Parsing.parsing import (
        build_prechunk_records,
        parse_hwp_files,
        write_jsonl,
    )

    _require_exactly_one(
        a=input_path,
        b=input_dir,
        a_name="--input (single HWP)",
        b_name="--input-dir (folder)",
    )

    debug_path = Path(debug_headings) if debug_headings else None
    if input_dir:
        input_files = _discover_hwp_in_dir(input_dir)
        records, error_count = parse_hwp_files(input_files, group_size=group_size, stop_on_error=False)
        print(f"target_files: {len(input_files)}")
        print(f"error_files: {error_count}")
    else:
        records = build_prechunk_records(
            Path(input_path),  # type: ignore[arg-type]
            group_size=group_size,
            debug_headings_path=debug_path,
        )

    write_limit = None if limit == 0 else limit
    write_jsonl(Path(output_path), records, limit=write_limit)
    written = len(records) if write_limit is None else min(write_limit, len(records))
    print(f"parsed_records: {len(records)}")
    print(f"written_records: {written}")
    print(f"output: {output_path}")


def chunk_jsonl(
    input_path: str,
    output_path: str,
    summary_output: str | None = None,
    sample_output: str | None = None,
    sample_size: int = 20,
    text_chunk_size: int = 900,
    text_overlap: int = 180,
    table_chunk_size: int = 1000,
    max_table_rows: int = 6,
    min_text_chars: int = 40,
    short_context_chars: int = 140,
    include_cover: bool = True,
    include_toc: bool = False,
    include_debug_metadata: bool = False,
) -> None:
    from argparse import Namespace

    from src.chunking.chunking import (
        build_rag_chunks,
        compact_output_metadata,
        default_sample_output,
        default_summary_output,
        read_jsonl,
        write_jsonl,
        write_sample_jsonl,
        write_summary_csv,
    )

    args = Namespace(
        text_chunk_size=text_chunk_size,
        text_overlap=text_overlap,
        table_chunk_size=table_chunk_size,
        max_table_rows=max_table_rows,
        min_text_chars=min_text_chars,
        short_context_chars=short_context_chars,
        include_cover=include_cover,
        include_toc=include_toc,
        include_debug_metadata=include_debug_metadata,
    )

    output = Path(output_path)
    records = read_jsonl(Path(input_path))
    chunks = build_rag_chunks(records, args)
    if not include_debug_metadata:
        compact_output_metadata(chunks)

    write_jsonl(output, chunks)
    summary = Path(summary_output) if summary_output else default_summary_output(output)
    sample = Path(sample_output) if sample_output else default_sample_output(output)
    write_summary_csv(summary, chunks)
    write_sample_jsonl(sample, chunks, sample_size=sample_size)

    print(f"input_records: {len(records)}")
    print(f"output_chunks: {len(chunks)}")
    print(f"output: {output}")
    print(f"summary_output: {summary}")
    print(f"sample_output: {sample}")


def _load_chunks_for_sampling(
    *,
    input_path: str | None,
    input_dir: str | None,
    pattern: str,
    recursive: bool,
) -> tuple[list[dict], list[Path]]:
    from src.sampling.sample_eval_chunks import read_jsonl

    _require_exactly_one(
        a=input_path,
        b=input_dir,
        a_name="--input (one JSONL)",
        b_name="--input-dir (many JSONL files)",
    )

    if input_path:
        path = Path(input_path)
        return read_jsonl(path), [path]

    root = Path(input_dir)  # type: ignore[arg-type]
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    paths = sorted(path for path in iterator if path.is_file())
    if not paths:
        raise SystemExit(f"No chunk JSONL files found: {input_dir} ({pattern})")
    rows: list[dict] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows, paths


def chunk_hwp_dir(
    input_dir: str,
    output_dir: str,
    limit: int = 0,
    group_size: int = 8,
    sample_size: int = 10,
    text_chunk_size: int = 900,
    text_overlap: int = 180,
) -> None:
    from pathlib import Path

    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    files = sorted(input_root.rglob("*.hwp"))
    if limit > 0:
        files = files[:limit]
    if not files:
        raise RuntimeError(f"No HWP files found under {input_root}")

    manifest: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for index, input_file in enumerate(files, start=1):
        stem = input_file.stem
        safe_stem = re.sub(r"[\\/:*?\"<>|&\s]+", "_", stem).strip("._")
        safe_stem = re.sub(r"_+", "_", safe_stem) or f"document_{index:04d}"
        doc_dir = output_root / safe_stem
        doc_dir.mkdir(parents=True, exist_ok=True)

        prechunk = doc_dir / "prechunk.jsonl"
        headings = doc_dir / "heading_debug.jsonl"
        chunks = doc_dir / "chunks.jsonl"
        summary = doc_dir / "chunks_summary.csv"
        sample = doc_dir / "chunks_sample.jsonl"

        print(f"[{index}/{len(files)}] {input_file.name}")
        try:
            parse_hwp(
                input_path=str(input_file),
                output_path=str(prechunk),
                debug_headings=str(headings),
                group_size=group_size,
            )
            chunk_jsonl(
                input_path=str(prechunk),
                output_path=str(chunks),
                summary_output=str(summary),
                sample_output=str(sample),
                sample_size=sample_size,
                text_chunk_size=text_chunk_size,
                text_overlap=text_overlap,
            )
            manifest.append(
                {
                    "source": str(input_file),
                    "output_dir": str(doc_dir),
                    "prechunk": str(prechunk),
                    "chunks": str(chunks),
                    "summary": str(summary),
                    "sample": str(sample),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "source": str(input_file),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"FAILED: {type(exc).__name__}: {exc}")

    manifest_path = output_root / "chunk_manifest.json"
    failure_path = output_root / "chunk_failures.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    print("chunk_hwp_dir_done")
    print(f"processed: {len(manifest)}")
    print(f"failed: {len(failures)}")
    print(f"manifest: {manifest_path}")
    print(f"failures: {failure_path}")


def chunk_hwp_slim(
    input_dir: str,
    input_files: list[str] | None,
    output_path: str | None,
    errors_path: str | None,
    tables_output_path: str | None,
    glob_pattern: str,
    recursive: bool,
    limit_files: int,
    group_size: int,
    text_chunk_size: int,
    text_overlap: int,
    table_chunk_size: int,
    max_table_rows: int,
    include_toc: bool,
    exclude_cover: bool,
    stop_on_error: bool,
) -> None:
    from pathlib import Path

    from src.pipeline.hwp_slim_pipeline import chunk_hwp_dir_to_slim_jsonl, safe_output_stem

    selected_input_files = [Path(path) for path in input_files] if input_files else None
    if output_path is None:
        if selected_input_files and len(selected_input_files) == 1:
            stem = safe_output_stem(selected_input_files[0].stem)
            resolved_output_path = Path("data/v2/samples") / f"{stem}_slim_with_tables.jsonl"
            resolved_tables_output_path = (
                Path(tables_output_path) if tables_output_path else Path("data/v2/samples") / f"{stem}_tables_raw.jsonl"
            )
        else:
            resolved_output_path = Path("data/v2/hwp_chunks_slim.jsonl")
            resolved_tables_output_path = Path(tables_output_path) if tables_output_path else None
    else:
        resolved_output_path = Path(output_path)
        resolved_tables_output_path = Path(tables_output_path) if tables_output_path else None

    result = chunk_hwp_dir_to_slim_jsonl(
        input_dir=Path(input_dir),
        input_files=selected_input_files,
        output_path=resolved_output_path,
        errors_path=Path(errors_path) if errors_path else None,
        tables_output_path=resolved_tables_output_path,
        glob_pattern=glob_pattern,
        recursive=recursive,
        limit_files=limit_files,
        group_size=group_size,
        text_chunk_size=text_chunk_size,
        text_overlap=text_overlap,
        table_chunk_size=table_chunk_size,
        max_table_rows=max_table_rows,
        include_cover=not exclude_cover,
        include_toc=include_toc,
        stop_on_error=stop_on_error,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


def chunk_mixed_slim(
    input_dir: str,
    input_files: list[str] | None,
    output_path: str | None,
    errors_path: str | None,
    tables_output_path: str | None,
    glob_pattern: str,
    recursive: bool,
    limit_files: int,
    group_size: int,
    text_chunk_size: int,
    text_overlap: int,
    table_chunk_size: int,
    max_table_rows: int,
    include_toc: bool,
    exclude_cover: bool,
    stop_on_error: bool,
    pdf_backend: str,
    pdf_no_tables: bool,
) -> None:
    from pathlib import Path

    from src.pipeline.mixed_slim_pipeline import chunk_mixed_dir_to_slim_jsonl, safe_output_stem

    selected_input_files = [Path(path) for path in input_files] if input_files else None
    if output_path is None:
        if selected_input_files and len(selected_input_files) == 1:
            stem = safe_output_stem(selected_input_files[0].stem)
            resolved_output_path = Path("data/v2/samples") / f"{stem}_mixed_slim.jsonl"
            resolved_tables_output_path = (
                Path(tables_output_path) if tables_output_path else Path("data/v2/samples") / f"{stem}_mixed_tables_raw.jsonl"
            )
        else:
            resolved_output_path = Path("data/v2/mixed_chunks_slim.jsonl")
            resolved_tables_output_path = Path(tables_output_path) if tables_output_path else None
    else:
        resolved_output_path = Path(output_path)
        resolved_tables_output_path = Path(tables_output_path) if tables_output_path else None

    result = chunk_mixed_dir_to_slim_jsonl(
        input_dir=Path(input_dir),
        input_files=selected_input_files,
        output_path=resolved_output_path,
        errors_path=Path(errors_path) if errors_path else None,
        tables_output_path=resolved_tables_output_path,
        glob_pattern=glob_pattern,
        recursive=recursive,
        limit_files=limit_files,
        group_size=group_size,
        text_chunk_size=text_chunk_size,
        text_overlap=text_overlap,
        table_chunk_size=table_chunk_size,
        max_table_rows=max_table_rows,
        include_cover=not exclude_cover,
        include_toc=include_toc,
        stop_on_error=stop_on_error,
        pdf_backend=pdf_backend,
        pdf_extract_tables=not pdf_no_tables,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


def sampling(
    output_path: str,
    *,
    input_path: str | None = None,
    input_dir: str | None = None,
    pattern: str = "*_chunks.jsonl",
    recursive: bool = True,
    quotas: str | None = None,
    appendix_mode: str = "auto",
    min_per_doc: int = 9,
    fallback_body: int = 0,
    min_chars: int = 80,
    limit_docs: int | None = None,
    add_sampling_metadata: bool = False,
) -> None:
    from src.sampling.sample_eval_chunks import (
        parse_quota_config,
        sample_rows,
        write_jsonl,
    )

    quota_config = parse_quota_config(quotas)
    rows, source_paths = _load_chunks_for_sampling(
        input_path=input_path,
        input_dir=input_dir,
        pattern=pattern,
        recursive=recursive,
    )
    if source_paths:
        print(f"chunk_files: {len(source_paths)}")
        for path in source_paths:
            print(f"  - {path}")
    sampled_rows, summary = sample_rows(
        rows,
        quotas=quota_config,
        appendix_mode=appendix_mode,
        min_chars=min_chars,
        min_per_doc=min_per_doc,
        fallback_body=fallback_body,
        limit_docs=limit_docs,
        add_sampling_metadata=add_sampling_metadata,
    )
    write_jsonl(Path(output_path), sampled_rows)

    print(f"input_rows: {summary['input_rows']}")
    print(f"candidate_rows: {summary['candidate_rows']}")
    print(f"documents: {summary['documents']}")
    print(f"sampled_rows: {summary['sampled_rows']}")
    print(
        "doc_samples: "
        f"min={summary['min_doc_samples']} "
        f"avg={summary['avg_doc_samples']} "
        f"max={summary['max_doc_samples']}"
    )
    section_counts = json.dumps(summary["section_counts"], ensure_ascii=False, sort_keys=True)
    print(f"section_counts: {section_counts}")
    print(f"output: {output_path}")


def convert_embedding_input(input_path: str, output_path: str, doc_id: str | None = None) -> None:
    from src.Parsing.convert_prechunk_to_embedding_input import convert

    count = convert(Path(input_path), Path(output_path), doc_id)
    print(f"Converted {count} rows -> {output_path}")


def export_ocr_rag_handoff(
    *,
    ocr_eval_root: str,
    output_manifest: str,
    output_chunks: str,
    engine: str | None,
    doc_key: str | None,
    include_review_required: bool,
    include_html_chunk: bool,
    html_chunk_max_chars: int,
    allow_inference_only: bool,
    images_tag: str | None,
    curated_root: str | None,
    curated_file_name: str,
    use_merge_manifest: bool,
    curated_only: bool,
    input_version: str | None,
    ocr_engine_version: str | None,
    ocr_output_version: str | None,
    ocr_curated_version: str | None,
    rag_index_version: str | None,
) -> None:
    from src.pipeline.ocr_rag_bridge import export_ocr_eval_to_rag_inputs

    manifest_count, chunk_count = export_ocr_eval_to_rag_inputs(
        ocr_eval_root=Path(ocr_eval_root),
        output_manifest=Path(output_manifest),
        output_chunks=Path(output_chunks),
        engine=engine,
        doc_key=doc_key,
        include_review_required=include_review_required,
        include_html_chunk=include_html_chunk,
        html_chunk_max_chars=html_chunk_max_chars,
        allow_inference_only=allow_inference_only,
        images_tag=images_tag,
        curated_root=Path(curated_root) if curated_root else None,
        curated_file_name=curated_file_name,
        use_merge_manifest=use_merge_manifest,
        curated_only=curated_only,
        input_version=input_version,
        ocr_engine_version=ocr_engine_version,
        ocr_output_version=ocr_output_version,
        ocr_curated_version=ocr_curated_version,
        rag_index_version=rag_index_version,
    )
    print("\n=== OCR Export Summary ===")
    print("1. status: ocr_export_rag_done")
    print(f"2. manifest_rows: {manifest_count}")
    print(f"3. chunk_rows: {chunk_count}")
    print(f"4. manifest_output: {output_manifest}")
    print(f"5. chunks_output: {output_chunks}")


def extract_ocr_images(
    input_dir: str,
    output_dir: str,
    *,
    limit: int = 0,
    source_type: str = "all",
    recursive: bool = False,
    pdf_min_width: int = 100,
    pdf_min_height: int = 40,
    pdf_min_area: int = 10_000,
    pdf_min_bytes: int = 1_000,
) -> None:
    from pathlib import Path

    from src.Parsing.ocr.inference.extract_hwp_pdf_images import extract_images_in_dir

    include_hwp = source_type in {"all", "hwp"}
    include_pdf = source_type in {"all", "pdf"}
    saved = extract_images_in_dir(
        Path(input_dir),
        Path(output_dir),
        limit=limit,
        include_hwp=include_hwp,
        include_pdf=include_pdf,
        recursive=recursive,
        pdf_min_width=pdf_min_width,
        pdf_min_height=pdf_min_height,
        pdf_min_area=pdf_min_area,
        pdf_min_bytes=pdf_min_bytes,
    )
    print(f"saved_images: {len(saved)}")
    print(f"output_dir: {output_dir}")


def _to_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [value]
    try:
        return list(value)  # type: ignore[arg-type]
    except TypeError:
        return [value]


def _make_json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _make_json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _make_json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _result_item_to_dict(item: object) -> dict:
    candidate: object
    if hasattr(item, "to_dict"):
        try:
            candidate = item.to_dict()  # type: ignore[no-any-return]
            safe = _make_json_safe(candidate)
            if isinstance(safe, dict):
                return safe
            return {"raw": str(safe)}
        except Exception:
            pass
    if isinstance(item, dict):
        safe = _make_json_safe(item)
        if isinstance(safe, dict):
            return safe
        return {"raw": str(safe)}
    return {"raw": str(_make_json_safe(item))}


def _collect_texts(value: object) -> list[str]:
    texts: list[str] = []
    if value is None:
        return texts
    if isinstance(value, str):
        text = value.strip()
        if text:
            texts.append(text)
        return texts
    if isinstance(value, dict):
        preferred_keys = [
            "text",
            "texts",
            "rec_texts",
            "markdown",
            "content",
            "result",
            "answer",
            "output",
            "prediction",
            "html",
            "table_html",
            "table_markdown",
        ]
        ignore_keys = {"image", "image_path", "input_path", "query", "prompt"}
        used_keys: set[str] = set()
        for key in preferred_keys:
            if key in value:
                texts.extend(_collect_texts(value.get(key)))
                used_keys.add(key)
        for key, nested in value.items():
            if key in ignore_keys or key in used_keys:
                continue
            texts.extend(_collect_texts(nested))
        return texts
    if isinstance(value, (list, tuple)):
        for item in value:
            texts.extend(_collect_texts(item))
        return texts
    return texts


def _dedupe_text_rows(rows: list[dict]) -> list[dict]:
    deduped_rows: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("text", "")).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped_rows.append({"text": key, "score": row.get("score"), "poly": row.get("poly")})
    return deduped_rows


def _extract_generic_rows_and_raw(results: object) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    raw_items: list[dict] = []
    for item in _to_list(results):
        item_dict = _result_item_to_dict(item)
        raw_items.append(item_dict)
        extracted = _collect_texts(item_dict)
        for text_block in extracted:
            for line in (seg.strip() for seg in str(text_block).splitlines()):
                if line:
                    rows.append({"text": line, "score": None, "poly": None})
    return _dedupe_text_rows(rows), raw_items


def _extract_ppocr_rows_and_raw(results: object) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    raw_items: list[dict] = []
    for item in _to_list(results):
        item_dict = _result_item_to_dict(item)
        raw_items.append(item_dict)
        rec_texts = _to_list(item_dict.get("rec_texts"))
        rec_scores = _to_list(item_dict.get("rec_scores"))
        rec_polys = _to_list(item_dict.get("rec_polys"))
        for idx, text in enumerate(rec_texts):
            poly = rec_polys[idx] if idx < len(rec_polys) else None
            rows.append(
                {
                    "text": str(text),
                    "score": float(rec_scores[idx]) if idx < len(rec_scores) and rec_scores[idx] is not None else None,
                    "poly": poly.tolist() if hasattr(poly, "tolist") else poly,
                }
            )
    if rows:
        return _dedupe_text_rows(rows), raw_items
    generic_rows, _ = _extract_generic_rows_and_raw(raw_items)
    return generic_rows, raw_items


def _write_ocr_payload(output_path: str, payload: dict) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _make_json_safe(payload)
    output.write_text(json.dumps(safe_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ocr_lines: {len(payload.get('ocr_lines', []))}")
    print(f"per_image_latency_ms: {payload.get('latency_ms')}")
    print(f"output: {output}")


def _build_pp_ocrv5_model(*, lang: str, device: str, engine: str = "paddle") -> object:
    from paddleocr import PaddleOCR

    candidates = [
        {
            "lang": lang,
            "device": device,
            "engine": engine,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {
            "lang": lang,
            "engine": engine,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {"lang": lang, "engine": engine},
        {},
    ]
    last_error: Exception | None = None
    for kwargs in candidates:
        try:
            return PaddleOCR(**kwargs)
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    return PaddleOCR()


def _build_pp_structurev3_model(*, lang: str, device: str) -> object:
    from paddleocr import PPStructureV3

    candidates = [
        {
            "device": device,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
        },
        {"use_doc_orientation_classify": False, "use_doc_unwarping": False},
        {"device": device},
        {},
    ]
    last_error: Exception | None = None
    for kwargs in candidates:
        try:
            return PPStructureV3(**kwargs)
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    return PPStructureV3()


def _build_table_recognition_v2_model(*, lang: str, device: str) -> object:
    from paddleocr import TableRecognitionPipelineV2

    candidates = [
        {"device": device, "use_doc_orientation_classify": False, "use_doc_unwarping": False},
        {"use_doc_orientation_classify": False, "use_doc_unwarping": False},
        {"device": device},
        {},
    ]
    last_error: Exception | None = None
    for kwargs in candidates:
        try:
            return TableRecognitionPipelineV2(**kwargs)
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    return TableRecognitionPipelineV2()


def _build_paddleocr_vl_model(
    *,
    device: str,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_chart_recognition: bool = False,
) -> object:
    from paddleocr import PaddleOCRVL

    candidates = [
        {
            "device": device,
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "use_chart_recognition": use_chart_recognition,
        },
        {"device": device},
        {},
    ]
    last_error: Exception | None = None
    for kwargs in candidates:
        try:
            return PaddleOCRVL(**kwargs)
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    return PaddleOCRVL()


def run_paddle_ocr(
    image_path: str,
    output_path: str,
    lang: str = "korean",
    device: str = "gpu:0",
    backend_engine: str = "paddle",
    ocr_model: object | None = None,
) -> None:
    import time

    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    model = (
        ocr_model
        if ocr_model is not None
        else _build_pp_ocrv5_model(lang=lang, device=device, engine=backend_engine)
    )
    start = time.perf_counter()
    results = model.predict(str(image))
    latency_ms = (time.perf_counter() - start) * 1000.0
    rows, raw_items = _extract_ppocr_rows_and_raw(results)
    payload = {
        "image_path": str(image),
        "model": f"pp_ocrv5:{backend_engine}",
        "lang": lang,
        "status": "success",
        "latency_ms": round(latency_ms, 2),
        "ocr_lines": rows,
        "raw_pipeline_output": raw_items,
    }
    _write_ocr_payload(output_path, payload)


def run_pp_structurev3_ocr(
    image_path: str,
    output_path: str,
    lang: str = "korean",
    device: str = "gpu:0",
    structure_model: object | None = None,
) -> None:
    import time

    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    model = structure_model if structure_model is not None else _build_pp_structurev3_model(lang=lang, device=device)
    start = time.perf_counter()
    results = model.predict(
        input=str(image),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    rows, raw_items = _extract_generic_rows_and_raw(results)
    payload = {
        "image_path": str(image),
        "model": "pp_structurev3",
        "lang": lang,
        "status": "success",
        "latency_ms": round(latency_ms, 2),
        "ocr_lines": rows,
        "raw_pipeline_output": raw_items,
    }
    _write_ocr_payload(output_path, payload)


def run_paddleocr_vl_ocr(
    image_path: str,
    output_path: str,
    device: str = "gpu:0",
    batch_size: int = 1,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_chart_recognition: bool = False,
    ocr_model: object | None = None,
) -> None:
    import time

    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    model = (
        ocr_model
        if ocr_model is not None
        else _build_paddleocr_vl_model(
            device=device,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_chart_recognition=use_chart_recognition,
        )
    )

    start = time.perf_counter()
    results = model.predict(
        input=str(image),
        batch_size=batch_size,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_chart_recognition=use_chart_recognition,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    rows, raw_items = _extract_generic_rows_and_raw(results)
    payload = {
        "image_path": str(image),
        "model": "paddleocr_vl",
        "lang": "multilingual",
        "status": "success",
        "latency_ms": round(latency_ms, 2),
        "ocr_lines": rows,
        "raw_pipeline_output": raw_items,
    }
    _write_ocr_payload(output_path, payload)


def run_table_recognition_v2_ocr(
    image_path: str,
    output_path: str,
    lang: str = "korean",
    device: str = "gpu:0",
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_table_orientation_classify: bool = True,
    use_ocr_results_with_table_cells: bool = True,
    text_det_limit_side_len: int | None = None,
    table_model: object | None = None,
) -> None:
    import time

    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    model = table_model if table_model is not None else _build_table_recognition_v2_model(lang=lang, device=device)
    start = time.perf_counter()
    predict_kwargs = {
        "input": str(image),
        "use_doc_orientation_classify": use_doc_orientation_classify,
        "use_doc_unwarping": use_doc_unwarping,
        "use_layout_detection": True,
        "use_ocr_model": True,
        "use_table_orientation_classify": use_table_orientation_classify,
        "use_ocr_results_with_table_cells": use_ocr_results_with_table_cells,
    }
    if text_det_limit_side_len is not None:
        predict_kwargs["text_det_limit_side_len"] = text_det_limit_side_len
    results = model.predict(**predict_kwargs)
    latency_ms = (time.perf_counter() - start) * 1000.0
    rows, raw_items = _extract_generic_rows_and_raw(results)
    payload = {
        "image_path": str(image),
        "model": "table_recognition_v2",
        "lang": lang,
        "status": "success",
        "latency_ms": round(latency_ms, 2),
        "ocr_lines": rows,
        "raw_pipeline_output": raw_items,
    }
    _write_ocr_payload(output_path, payload)


def _build_shared_ocr_model(
    *,
    ocr_engine: str,
    lang: str,
    ocr_device: str,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_chart_recognition: bool = False,
) -> object | None:
    engine = _normalize_ocr_engine(ocr_engine)
    if engine == "pp_ocrv5":
        print(f"[PP-OCRv5] initialize once: lang={lang}, device={ocr_device}")
        return _build_pp_ocrv5_model(lang=lang, device=ocr_device)
    if engine == "pp_ocrv5_transformers":
        print(f"[PP-OCRv5 Transformers] initialize once: lang={lang}, device={ocr_device}")
        return _build_pp_ocrv5_model(lang=lang, device=ocr_device, engine="transformers")
    if engine == "pp_structurev3":
        print(f"[PP-StructureV3] initialize once: lang={lang}, device={ocr_device}")
        return _build_pp_structurev3_model(lang=lang, device=ocr_device)
    if engine == "paddleocr_vl":
        print(f"[PaddleOCR-VL] initialize once: device={ocr_device}")
        return _build_paddleocr_vl_model(
            device=ocr_device,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_chart_recognition=use_chart_recognition,
        )
    if engine == "table_recognition_v2":
        print(f"[table_recognition_v2] initialize once: lang={lang}, device={ocr_device}")
        return _build_table_recognition_v2_model(lang=lang, device=ocr_device)
    return None


def build_pred_structured(
    gt_path: str,
    pred_raw_path: str,
    item_id: str,
    output_path: str,
    score_threshold: float = 0.0,
) -> None:
    from pathlib import Path

    from src.Parsing.ocr.inference.build_pred_structured import (
        build_pred_structured as build_structured_item,
        load_item_by_id,
        load_pred_raw,
    )

    gt_item = load_item_by_id(Path(gt_path), item_id)
    pred_raw_item = load_pred_raw(Path(pred_raw_path), item_id)
    structured = build_structured_item(gt_item, pred_raw_item, score_threshold=score_threshold)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output: {output}")
    print(f"pred_text: {structured['pred_text']}")


def _compute_required_field_gaps(
    required_fields: list[object],
    field_metrics: list[object],
) -> tuple[list[str], list[str]]:
    required = [str(field).strip() for field in required_fields if str(field).strip()]
    per_field: dict[str, dict] = {}
    for metric in field_metrics:
        if not isinstance(metric, dict):
            continue
        field_path = str(metric.get("field_path", "")).strip()
        if not field_path:
            continue
        per_field[field_path] = metric

    if not required:
        required = [
            field_name
            for field_name, metric in per_field.items()
            if int(metric.get("gt_total", 0)) > 0
        ]

    missing_required: list[str] = []
    for field_name in required:
        metric = per_field.get(field_name)
        if not metric:
            missing_required.append(field_name)
            continue
        if int(metric.get("matched", 0)) <= 0:
            missing_required.append(field_name)
    return required, missing_required


def _build_eval_summary_payload(
    *,
    eval_payload: dict[str, object],
    gt_path: str,
    pred_structured_path: str,
) -> dict[str, object]:
    text = eval_payload.get("text", {}) if isinstance(eval_payload.get("text"), dict) else {}
    structure = eval_payload.get("structure", {}) if isinstance(eval_payload.get("structure"), dict) else {}
    aggregate = structure.get("aggregate", {}) if isinstance(structure.get("aggregate"), dict) else {}
    field_metrics = structure.get("field_metrics", []) if isinstance(structure.get("field_metrics"), list) else []

    required_fields, missing_required_fields = _compute_required_field_gaps(
        required_fields=eval_payload.get("required_fields", []) if isinstance(eval_payload.get("required_fields"), list) else [],
        field_metrics=field_metrics,
    )

    table_html = eval_payload.get("table_html", {}) if isinstance(eval_payload.get("table_html"), dict) else {}
    table_rows = eval_payload.get("table_rows", {}) if isinstance(eval_payload.get("table_rows"), dict) else {}
    table_html_exists = bool(table_html.get("exists", False))
    table_rows_exists = bool(table_rows.get("exists", False))

    review_reasons: list[str] = []
    if str(eval_payload.get("status", "")).lower() != "success":
        review_reasons.append("status_not_success")

    structure_micro_recall = eval_payload.get("structure_micro_recall")
    if structure_micro_recall is not None and float(structure_micro_recall) < 0.8:
        review_reasons.append("structure_micro_recall<0.8")
    if missing_required_fields:
        review_reasons.append("required_field_missing")
    if str(eval_payload.get("type", "")).lower() == "table" and not table_html_exists:
        review_reasons.append("table_html_missing")
    if str(eval_payload.get("type", "")).lower() == "table" and not table_rows_exists:
        review_reasons.append("table_rows_missing")

    return {
        "schema_version": "ocr_eval_summary.v1",
        "id": eval_payload.get("id"),
        "type": eval_payload.get("type"),
        "status": eval_payload.get("status"),
        "latency_ms": eval_payload.get("latency_ms"),
        "text": {
            "exact_match": text.get("exact_match"),
            "cer": text.get("cer"),
            "wer": text.get("wer"),
            "char_similarity_pct": text.get("char_similarity_pct"),
        },
        "structure": {
            "mode": structure.get("mode", "field_path_value_match"),
            "threshold": structure.get("threshold"),
            "aggregate": {
                "field_count": aggregate.get("field_count"),
                "matched": aggregate.get("matched"),
                "gt_total": aggregate.get("gt_total"),
                "pred_total": aggregate.get("pred_total"),
                "micro_precision": aggregate.get("micro_precision"),
                "micro_recall": aggregate.get("micro_recall"),
                "micro_f1": aggregate.get("micro_f1"),
                "macro_f1": aggregate.get("macro_f1"),
            },
        },
        "structure_micro_recall": eval_payload.get("structure_micro_recall"),
        "structure_macro_f1": eval_payload.get("structure_macro_f1"),
        "required_fields": required_fields,
        "missing_required_fields": missing_required_fields,
        "review_required": bool(review_reasons),
        "review_reasons": review_reasons,
        "table_html": table_html,
        "table_rows": table_rows,
        "refs": {
            "gt_path": gt_path,
            "pred_structured_path": pred_structured_path,
        },
    }


def _build_eval_debug_payload(
    *,
    eval_payload: dict[str, object],
    summary_payload: dict[str, object],
) -> dict[str, object]:
    structure = eval_payload.get("structure", {}) if isinstance(eval_payload.get("structure"), dict) else {}
    field_metrics = structure.get("field_metrics", []) if isinstance(structure.get("field_metrics"), list) else []
    include_structures = bool(summary_payload.get("missing_required_fields"))
    debug_payload: dict[str, object] = {
        "schema_version": "ocr_eval_debug.v1",
        "id": eval_payload.get("id"),
        "type": eval_payload.get("type"),
        "status": eval_payload.get("status"),
        "review_required": summary_payload.get("review_required", False),
        "review_reasons": summary_payload.get("review_reasons", []),
        "missing_required_fields": summary_payload.get("missing_required_fields", []),
        "field_metrics": field_metrics,
        "refs": summary_payload.get("refs", {}),
    }
    if include_structures:
        debug_payload["gt_structure"] = structure.get("gt_structure", {})
        debug_payload["pred_structure"] = structure.get("pred_structure", {})
    return debug_payload


def eval_pred_structured_vs_gt(
    gt_path: str,
    pred_structured_path: str,
    item_id: str,
    output_path: str,
    structure_match_threshold: float = 0.65,
    table_html_path: str | None = None,
    table_rows_path: str | None = None,
) -> None:
    from pathlib import Path

    from src.Parsing.ocr.evaluation.eval_pred_structured_vs_gt import (
        build_eval_report,
        load_item_by_id,
    )

    gt_item = load_item_by_id(Path(gt_path), item_id)
    pred_item = load_item_by_id(Path(pred_structured_path), item_id)
    eval_payload = build_eval_report(
        item_id=item_id,
        gt_item=gt_item,
        pred_item=pred_item,
        threshold=structure_match_threshold,
    )
    if table_html_path:
        table_path = Path(table_html_path)
        table_exists = table_path.exists() and table_path.stat().st_size > 0
        eval_payload["table_html"] = {
            "path": str(table_path),
            "exists": table_exists,
            "byte_size": table_path.stat().st_size if table_exists else 0,
        }
    if table_rows_path:
        rows_path = Path(table_rows_path)
        rows_exists = rows_path.exists() and rows_path.stat().st_size > 0
        eval_payload["table_rows"] = {
            "path": str(rows_path),
            "exists": rows_exists,
            "byte_size": rows_path.stat().st_size if rows_exists else 0,
        }

    summary_payload = _build_eval_summary_payload(
        eval_payload=eval_payload,
        gt_path=gt_path,
        pred_structured_path=pred_structured_path,
    )
    debug_payload = _build_eval_debug_payload(
        eval_payload=eval_payload,
        summary_payload=summary_payload,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    debug_output = output.with_name("gt_eval_debug.json")
    if bool(summary_payload.get("review_required", False)):
        debug_output.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif debug_output.exists():
        debug_output.unlink()

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    print(f"saved_summary: {output}")
    if bool(summary_payload.get("review_required", False)):
        print(f"saved_debug: {debug_output}")


def _save_pred_table_html(
    *,
    pred_raw_path: str,
    output_path: str,
) -> dict[str, object]:
    from src.Parsing.ocr.postprocess.ocr_table_normalizer import save_ocr_table_outputs

    return save_ocr_table_outputs(
        pred_raw_path=pred_raw_path,
        output_path=output_path,
    )


def _merge_vl_and_ppocr_raw(
    *,
    vl_raw_path: str,
    ppocr_raw_path: str,
    output_path: str,
) -> None:
    vl_payload = json.loads(Path(vl_raw_path).read_text(encoding="utf-8"))
    pp_payload = json.loads(Path(ppocr_raw_path).read_text(encoding="utf-8"))
    vl_lines = vl_payload.get("ocr_lines", []) if isinstance(vl_payload.get("ocr_lines"), list) else []
    pp_lines = pp_payload.get("ocr_lines", []) if isinstance(pp_payload.get("ocr_lines"), list) else []
    merged_lines = _dedupe_text_rows([*vl_lines, *pp_lines])

    merged_payload = dict(vl_payload)
    merged_payload["ocr_lines"] = merged_lines
    merged_payload["model"] = "paddleocr_vl+pp_ocrv5"
    merged_payload["latency_ms"] = round(
        float(vl_payload.get("latency_ms") or 0.0) + float(pp_payload.get("latency_ms") or 0.0),
        2,
    )
    merged_payload["meta"] = {
        "dual_pass": True,
        "first_pass_model": vl_payload.get("model", "paddleocr_vl"),
        "second_pass_model": pp_payload.get("model", "pp_ocrv5:paddle"),
        "first_pass_path": str(vl_raw_path),
        "second_pass_path": str(ppocr_raw_path),
    }
    _write_ocr_payload(output_path, merged_payload)


def _has_table_label_in_vl_raw(vl_raw_path: str) -> bool:
    payload = json.loads(Path(vl_raw_path).read_text(encoding="utf-8"))
    raw_output = payload.get("raw_pipeline_output")
    if not isinstance(raw_output, list):
        return False
    for page in raw_output:
        if not isinstance(page, dict):
            continue
        parsing_items = page.get("parsing_res_list")
        if not isinstance(parsing_items, list):
            continue
        for item in parsing_items:
            if not isinstance(item, str):
                continue
            if re.search(r"label:\s*table\b", item, flags=re.IGNORECASE):
                return True
    return False


def _ocr_batch_log(message: str, *, quiet: bool) -> None:
    if not quiet:
        print(message)


def _collect_ocr_batch_image_jobs(
    doc_dirs: list[Path],
) -> tuple[list[tuple[str, Path]], list[dict[str, str]]]:
    jobs: list[tuple[str, Path]] = []
    skip_details: list[dict[str, str]] = []
    for doc_dir in doc_dirs:
        try:
            for path in _list_image_paths(doc_dir):
                jobs.append((doc_dir.name, path))
        except FileNotFoundError as exc:
            skip_details.append({"doc_key": doc_dir.name, "image_name": "-", "reason": str(exc)})
    return jobs, skip_details


def ocr_run_all(
    image_path: str,
    gt_path: str | None,
    item_id: str | None,
    pred_raw_output: str,
    pred_structured_output: str | None,
    pred_table_html_output: str,
    eval_output: str | None,
    lang: str = "korean",
    score_threshold: float = 0.0,
    ocr_engine: str = "pp_ocrv5",
    ocr_device: str = "gpu:0",
    ocr_batch_size: int = 1,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_chart_recognition: bool = False,
    use_table_orientation_classify: bool = True,
    use_ocr_results_with_table_cells: bool = True,
    text_det_limit_side_len: int | None = None,
    ocr_model: object | None = None,
    structure_match_threshold: float = 0.65,
    table_dual_pass: bool = False,
    table_second_engine: str = "pp_ocrv5",
    require_gt: bool = True,
    quiet: bool = False,
) -> None:
    normalized_engine = _normalize_ocr_engine(ocr_engine)
    _ocr_batch_log("[1/4] run-ocr", quiet=quiet)
    if normalized_engine == "pp_structurev3":
        run_pp_structurev3_ocr(
            image_path=image_path,
            output_path=pred_raw_output,
            lang=lang,
            device=ocr_device,
            structure_model=ocr_model,
        )
    elif normalized_engine == "paddleocr_vl":
        if table_dual_pass:
            pred_raw_path = Path(pred_raw_output)
            pred_raw_vl_output = str(pred_raw_path.with_name("pred_raw_vl.json"))
            pred_raw_ppocr_output = str(pred_raw_path.with_name("pred_raw_ppocr.json"))
            run_paddleocr_vl_ocr(
                image_path=image_path,
                output_path=pred_raw_vl_output,
                device=ocr_device,
                batch_size=ocr_batch_size,
                use_doc_orientation_classify=use_doc_orientation_classify,
                use_doc_unwarping=use_doc_unwarping,
                use_chart_recognition=use_chart_recognition,
                ocr_model=ocr_model,
            )
            if _has_table_label_in_vl_raw(pred_raw_vl_output):
                second_engine = _normalize_ocr_engine(table_second_engine)
                run_paddle_ocr(
                    image_path=image_path,
                    output_path=pred_raw_ppocr_output,
                    lang=lang,
                    device=ocr_device,
                    backend_engine="transformers" if second_engine == "pp_ocrv5_transformers" else "paddle",
                    ocr_model=None,
                )
                _merge_vl_and_ppocr_raw(
                    vl_raw_path=pred_raw_vl_output,
                    ppocr_raw_path=pred_raw_ppocr_output,
                    output_path=pred_raw_output,
                )
            else:
                vl_payload = json.loads(Path(pred_raw_vl_output).read_text(encoding="utf-8"))
                _write_ocr_payload(pred_raw_output, vl_payload)
        else:
            run_paddleocr_vl_ocr(
                image_path=image_path,
                output_path=pred_raw_output,
                device=ocr_device,
                batch_size=ocr_batch_size,
                use_doc_orientation_classify=use_doc_orientation_classify,
                use_doc_unwarping=use_doc_unwarping,
                use_chart_recognition=use_chart_recognition,
                ocr_model=ocr_model,
            )
    elif normalized_engine == "table_recognition_v2":
        run_table_recognition_v2_ocr(
            image_path=image_path,
            output_path=pred_raw_output,
            lang=lang,
            device=ocr_device,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_table_orientation_classify=use_table_orientation_classify,
            use_ocr_results_with_table_cells=use_ocr_results_with_table_cells,
            text_det_limit_side_len=text_det_limit_side_len,
            table_model=ocr_model,
        )
    else:
        run_paddle_ocr(
            image_path=image_path,
            output_path=pred_raw_output,
            lang=lang,
            device=ocr_device,
            backend_engine="transformers" if normalized_engine == "pp_ocrv5_transformers" else "paddle",
            ocr_model=ocr_model,
        )
    if require_gt:
        if not gt_path or not item_id or not pred_structured_output or not eval_output:
            raise ValueError("GT mode requires gt_path, item_id, pred_structured_output, and eval_output.")
        _ocr_batch_log("[2/4] build-pred-structured", quiet=quiet)
        build_pred_structured(
            gt_path=gt_path,
            pred_raw_path=pred_raw_output,
            item_id=item_id,
            output_path=pred_structured_output,
            score_threshold=score_threshold,
        )
    else:
        _ocr_batch_log("[2/4] skip-gt-structured (inference-only)", quiet=quiet)

    _ocr_batch_log("[3/4] extract-table-html", quiet=quiet)
    table_outputs = _save_pred_table_html(
        pred_raw_path=pred_raw_output,
        output_path=pred_table_html_output,
    )
    has_table_html = bool(table_outputs.get("saved", False))
    _ocr_batch_log(f"table_html_saved: {has_table_html}", quiet=quiet)
    if table_outputs.get("pred_table_rows_path"):
        _ocr_batch_log(f"pred_table_rows_path: {table_outputs.get('pred_table_rows_path')}", quiet=quiet)

    if require_gt:
        _ocr_batch_log("[4/4] eval-pred-structured", quiet=quiet)
        eval_pred_structured_vs_gt(
            gt_path=gt_path,
            pred_structured_path=pred_structured_output,
            item_id=item_id,
            output_path=eval_output,
            structure_match_threshold=structure_match_threshold,
            table_html_path=pred_table_html_output,
            table_rows_path=table_outputs.get("pred_table_rows_path"),
        )
    else:
        _ocr_batch_log("[4/4] skip-gt-eval (inference-only)", quiet=quiet)
    _ocr_batch_log("ocr_run_all_done", quiet=quiet)


def _infer_item_id_from_gt(gt_path: Path, image_name: str | None = None) -> str:
    payload = _load_gt_payload(gt_path)
    if isinstance(payload, dict):
        item_id = payload.get("id")
        if not item_id:
            raise ValueError(f"`id` not found in GT JSON: {gt_path}")
        return str(item_id)

    if isinstance(payload, list):
        records: list[dict] = []
        for item in payload:
            if isinstance(item, dict) and item.get("id"):
                records.append(item)

        if image_name:
            image_stem = Path(image_name).stem.lower()
            matched_ids: list[str] = []
            for record in records:
                record_id = str(record.get("id", ""))
                original_name = str(record.get("original_image_file_name", ""))
                derived_name = str(record.get("image_file_name", ""))
                candidate_names = [original_name, derived_name]
                candidate_stems = [Path(name).stem.lower() for name in candidate_names if name]
                if image_name in candidate_names or image_stem in candidate_stems:
                    if record_id and record_id not in matched_ids:
                        matched_ids.append(record_id)
            if len(matched_ids) == 1:
                return matched_ids[0]
            if len(matched_ids) > 1:
                raise ValueError(
                    "Multiple ids matched by image name. Pass --id explicitly. "
                    f"image={image_name}, ids={matched_ids}"
                )

        ids: list[str] = []
        for record in records:
            item_id = str(record["id"])
            if item_id not in ids:
                ids.append(item_id)
        if not ids:
            raise ValueError(f"No `id` found in GT JSON list: {gt_path}")
        if len(ids) > 1:
            raise ValueError(
                f"Multiple ids found in GT JSON ({len(ids)}). Pass --id explicitly."
                f"{' image=' + image_name + ',' if image_name else ''} ids={ids}"
            )
        return ids[0]

    raise ValueError(f"Unsupported GT JSON shape: {type(payload).__name__}")


def _load_gt_payload(gt_path: Path) -> dict | list[dict]:
    raw = gt_path.read_text(encoding="utf-8")
    stripped = raw.strip()
    if not stripped:
        return []
    decoder = json.JSONDecoder()
    pos = 0
    top_level: list[dict | list] = []
    while pos < len(raw):
        while pos < len(raw) and raw[pos].isspace():
            pos += 1
        if pos >= len(raw):
            break
        item, next_pos = decoder.raw_decode(raw, pos)
        top_level.append(item)
        pos = next_pos

    if len(top_level) == 1:
        only = top_level[0]
        if isinstance(only, dict):
            return only
        if isinstance(only, list):
            return only
        raise ValueError(f"Unsupported GT JSON shape in {gt_path}: {type(only).__name__}")

    rows: list[dict] = []
    for idx, block in enumerate(top_level, start=1):
        if isinstance(block, dict):
            rows.append(block)
            continue
        if isinstance(block, list):
            for row in block:
                if isinstance(row, dict):
                    rows.append(row)
                    continue
                raise ValueError(
                    f"Unsupported item type in GT block {idx} at {gt_path}: {type(row).__name__}"
                )
            continue
        raise ValueError(f"Unsupported GT block type {idx} in {gt_path}: {type(block).__name__}")
    return rows


def _resolve_gt_path(gt_root: str, doc_key: str) -> Path:
    gt_root_path = Path(gt_root)
    jsonl_path = gt_root_path / f"{doc_key}.jsonl"
    if jsonl_path.exists():
        return jsonl_path
    json_path = gt_root_path / f"{doc_key}.json"
    if json_path.exists():
        return json_path
    for path in sorted(gt_root_path.glob("*.jsonl")) + sorted(gt_root_path.glob("*.json")):
        try:
            payload = _load_gt_payload(path)
        except Exception:
            continue
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            if not isinstance(record, dict):
                continue
            if str(record.get("source_doc_key", "")) == doc_key:
                return path
            record_id = str(record.get("id", ""))
            if record_id.startswith(f"{doc_key}_"):
                return path
    # Prefer JSONL by default for new runs.
    return jsonl_path


def _infer_ocr_images_tag(images_root: str) -> str:
    """Infer dataset/version tag from paths like .../ocr_images/<tag>/... ."""
    parts = Path(images_root).parts
    for idx, part in enumerate(parts):
        if part == "ocr_images" and idx + 1 < len(parts):
            candidate = str(parts[idx + 1]).strip()
            if candidate:
                return candidate
    return ""


def _resolve_ocr_engine_output_root(*, output_root: str, ocr_engine: str, images_root: str) -> Path:
    engine_root = Path(output_root) / _engine_dir_name(ocr_engine)
    images_tag = _infer_ocr_images_tag(images_root)
    if images_tag:
        return engine_root / images_tag
    return engine_root


def _resolve_ocr_doc_paths(
    *,
    doc_key: str,
    images_root: str,
    gt_root: str,
    output_root: str,
    image_name: str,
    ocr_engine: str,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    normalized_engine = _normalize_ocr_engine(ocr_engine)
    image_path = Path(images_root) / doc_key / image_name
    gt_path = _resolve_gt_path(gt_root, doc_key)
    image_stem = Path(image_name).stem
    out_dir = _resolve_ocr_engine_output_root(
        output_root=output_root,
        ocr_engine=normalized_engine,
        images_root=images_root,
    ) / doc_key / image_stem
    inference_dir = out_dir / "inference"
    eval_dir = out_dir / "eval"
    pred_raw = inference_dir / "pred_raw.json"
    pred_structured = eval_dir / "gt_pred_structured.json"
    pred_table_html = inference_dir / "pred_table_raw.html"
    pred_table_rows = inference_dir / "pred_table_layout.json"
    eval_summary_json = eval_dir / "gt_eval_summary.json"
    return image_path, gt_path, pred_raw, pred_structured, pred_table_html, pred_table_rows, eval_summary_json


def _engine_dir_name(ocr_engine: str) -> str:
    return _normalize_ocr_engine(ocr_engine)


def _resolve_image_path(doc_dir: Path, image_name: str) -> Path:
    if not doc_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {doc_dir}")

    # 1) Exact path first.
    exact = doc_dir / image_name
    if exact.exists():
        return exact

    # 2) Try same stem with common image extensions.
    stem = Path(image_name).stem
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]:
        candidate = doc_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    # 3) Fallback: first image in folder.
    image_files = sorted(
        [
            path
            for path in doc_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        ],
        key=lambda p: p.name,
    )
    if image_files:
        return image_files[0]

    raise FileNotFoundError(f"No image files found in: {doc_dir}")


def _list_image_paths(doc_dir: Path) -> list[Path]:
    if not doc_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {doc_dir}")
    image_files = sorted(
        [
            path
            for path in doc_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        ],
        key=lambda p: p.name,
    )
    if not image_files:
        raise FileNotFoundError(f"No image files found in: {doc_dir}")
    return image_files


def _prepare_resized_image_if_needed(image_path: Path, max_long_side: int | None, *, quiet: bool = False) -> tuple[Path, Path | None]:
    # [Design Intent]
    # OOM 완화를 위해 긴 변 기준으로만 비율 유지 축소한다.
    # 원본은 유지하고 필요 시 임시 파일로만 추론한다.
    if not max_long_side or max_long_side <= 0:
        return image_path, None

    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to load image for resize check: {image_path}")

    height, width = image.shape[:2]
    long_side = max(width, height)
    if long_side <= max_long_side:
        return image_path, None

    scale = float(max_long_side) / float(long_side)
    resized = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    suffix = image_path.suffix if image_path.suffix else ".jpg"
    with tempfile.NamedTemporaryFile(prefix="ocr_resized_", suffix=suffix, delete=False) as tmp:
        temp_path = Path(tmp.name)
    if not cv2.imwrite(str(temp_path), resized):
        raise RuntimeError(f"Failed to write resized temp image: {temp_path}")
    _ocr_batch_log(
        f"resize_applied: {width}x{height} -> {resized.shape[1]}x{resized.shape[0]} (max_long_side={max_long_side})",
        quiet=quiet,
    )
    return temp_path, temp_path


def ocr_run_image(
    *,
    doc_key: str,
    item_id: str | None,
    images_root: str,
    gt_root: str,
    output_root: str,
    image_name: str,
    lang: str,
    score_threshold: float,
    structure_match_threshold: float,
    ocr_engine: str,
    ocr_device: str,
    ocr_batch_size: int,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_chart_recognition: bool = False,
    use_table_orientation_classify: bool = True,
    use_ocr_results_with_table_cells: bool = True,
    text_det_limit_side_len: int | None = None,
    ocr_model: object | None = None,
    table_dual_pass: bool = False,
    table_second_engine: str = "pp_ocrv5",
    require_gt: bool = True,
    max_long_side: int | None = 2000,
    quiet: bool = False,
) -> bool:
    (
        image_path,
        gt_path,
        pred_raw,
        pred_structured,
        pred_table_html,
        _pred_table_rows,
        eval_summary_json,
    ) = _resolve_ocr_doc_paths(
        doc_key=doc_key,
        images_root=images_root,
        gt_root=gt_root,
        output_root=output_root,
        image_name=image_name,
        ocr_engine=ocr_engine,
    )
    image_path = _resolve_image_path(Path(images_root) / doc_key, image_name)
    if require_gt and not gt_path.exists():
        raise FileNotFoundError(f"GT not found: {gt_path}")

    if require_gt:
        final_item_id = item_id or _infer_item_id_from_gt(gt_path, image_path.name)
        gt_payload = _load_gt_payload(gt_path)
        gt_records = gt_payload if isinstance(gt_payload, list) else [gt_payload]
        gt_item = next(
            (record for record in gt_records if isinstance(record, dict) and record.get("id") == final_item_id),
            {},
        )
        use_eval = bool(gt_item.get("use_eval", True))
    else:
        final_item_id = item_id or f"{doc_key}_{Path(image_path.name).stem}"
        gt_path = None
        use_eval = False

    pred_raw.parent.mkdir(parents=True, exist_ok=True)
    prepared_image_path, temp_image_path = _prepare_resized_image_if_needed(
        image_path, max_long_side=max_long_side, quiet=quiet
    )
    try:
        _ocr_batch_log(f"doc_key: {doc_key}", quiet=quiet)
        _ocr_batch_log(f"image: {image_path}", quiet=quiet)
        _ocr_batch_log(f"image_input: {prepared_image_path}", quiet=quiet)
        _ocr_batch_log(f"gt: {gt_path if gt_path else '<disabled>'}", quiet=quiet)
        _ocr_batch_log(f"id: {final_item_id}", quiet=quiet)
        ocr_run_all(
            image_path=str(prepared_image_path),
            gt_path=str(gt_path) if gt_path else None,
            item_id=final_item_id,
            pred_raw_output=str(pred_raw),
            pred_structured_output=str(pred_structured) if require_gt else None,
            pred_table_html_output=str(pred_table_html),
            eval_output=str(eval_summary_json) if require_gt else None,
            lang=lang,
            score_threshold=score_threshold,
            structure_match_threshold=structure_match_threshold,
            ocr_engine=ocr_engine,
            ocr_device=ocr_device,
            ocr_batch_size=ocr_batch_size,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_chart_recognition=use_chart_recognition,
            use_table_orientation_classify=use_table_orientation_classify,
            use_ocr_results_with_table_cells=use_ocr_results_with_table_cells,
            text_det_limit_side_len=text_det_limit_side_len,
            ocr_model=ocr_model,
            table_dual_pass=table_dual_pass,
            table_second_engine=table_second_engine,
            require_gt=require_gt,
            quiet=quiet,
        )
    finally:
        if temp_image_path and temp_image_path.exists():
            temp_image_path.unlink(missing_ok=True)
    if require_gt and not use_eval:
        _ocr_batch_log(f"[SKIP_EVAL_SUMMARY] use_eval=false: {final_item_id}", quiet=quiet)
    return use_eval


def ocr_run_batch(
    *,
    images_root: str,
    gt_root: str,
    output_root: str,
    image_name: str,
    doc_key: str | None,
    limit: int,
    stop_on_error: bool,
    lang: str,
    score_threshold: float,
    structure_match_threshold: float,
    ocr_engine: str,
    ocr_device: str,
    ocr_batch_size: int,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_chart_recognition: bool = False,
    use_table_orientation_classify: bool = True,
    use_ocr_results_with_table_cells: bool = True,
    text_det_limit_side_len: int | None = None,
    table_dual_pass: bool = False,
    table_second_engine: str = "pp_ocrv5",
    require_gt: bool = True,
    max_long_side: int | None = 2000,
    quiet: bool = True,
    show_progress: bool = True,
) -> None:
    normalized_engine = _normalize_ocr_engine(ocr_engine)
    engine_out_root = _resolve_ocr_engine_output_root(
        output_root=output_root,
        ocr_engine=normalized_engine,
        images_root=images_root,
    )
    images_tag = _infer_ocr_images_tag(images_root)
    shared_model = _build_shared_ocr_model(
        ocr_engine=normalized_engine,
        lang=lang,
        ocr_device=ocr_device,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_chart_recognition=use_chart_recognition,
    )

    image_root_path = Path(images_root)
    if not image_root_path.exists():
        raise FileNotFoundError(f"images_root not found: {image_root_path}")

    doc_dirs = sorted([path for path in image_root_path.iterdir() if path.is_dir()], key=lambda p: p.name)
    if doc_key:
        doc_dirs = [path for path in doc_dirs if path.name == doc_key]
        if not doc_dirs:
            raise FileNotFoundError(f"doc_key folder not found under images_root: {doc_key}")
    if limit > 0:
        doc_dirs = doc_dirs[:limit]

    ok_count = 0
    skip_count = 0
    skip_image_count = 0
    skip_gt_count = 0
    fail_count = 0
    skip_image_details: list[dict[str, str]] = []
    fail_details: list[dict[str, str]] = []
    eval_rows: list[dict[str, object]] = []
    review_queue_rows: list[dict[str, object]] = []

    def _notice(message: str) -> None:
        if quiet:
            if show_progress:
                from tqdm import tqdm

                tqdm.write(message)
            return
        print(message)

    jobs, pre_skip_details = _collect_ocr_batch_image_jobs(doc_dirs)
    skip_image_details.extend(pre_skip_details)
    skip_count += len(pre_skip_details)
    skip_image_count += len(pre_skip_details)

    if not quiet:
        print(f"ocr_batch_images: {len(jobs)} across {len(doc_dirs)} docs")

    progress_bar = None
    job_iter: object = jobs
    if show_progress and jobs:
        from tqdm import tqdm

        mininterval = 0.3 if sys.stderr.isatty() else 10.0
        progress_bar = tqdm(
            jobs,
            desc="OCR",
            unit="img",
            file=sys.stderr,
            mininterval=mininterval,
            dynamic_ncols=sys.stderr.isatty(),
        )
        job_iter = progress_bar

    try:
        for doc_key, path in job_iter:
            try:
                include_in_eval = ocr_run_image(
                    doc_key=doc_key,
                    item_id=None,
                    images_root=images_root,
                    gt_root=gt_root,
                    output_root=output_root,
                    image_name=path.name,
                    lang=lang,
                    score_threshold=score_threshold,
                    structure_match_threshold=structure_match_threshold,
                    ocr_engine=ocr_engine,
                    ocr_device=ocr_device,
                    ocr_batch_size=ocr_batch_size,
                    use_doc_orientation_classify=use_doc_orientation_classify,
                    use_doc_unwarping=use_doc_unwarping,
                    use_chart_recognition=use_chart_recognition,
                    use_table_orientation_classify=use_table_orientation_classify,
                    use_ocr_results_with_table_cells=use_ocr_results_with_table_cells,
                    text_det_limit_side_len=text_det_limit_side_len,
                    ocr_model=shared_model,
                    table_dual_pass=table_dual_pass,
                    table_second_engine=table_second_engine,
                    require_gt=require_gt,
                    max_long_side=max_long_side,
                    quiet=quiet,
                )
                ok_count += 1
                if not include_in_eval:
                    if progress_bar is not None:
                        progress_bar.set_postfix(ok=ok_count, fail=fail_count, skip=skip_count, refresh=False)
                    continue
            except FileNotFoundError as e:
                message = str(e)
                if "Image folder not found" in message or "No image files found" in message or "Image not found" in message:
                    _notice(f"[SKIP][이미지 없음] {doc_key}/{path.name}: {message}")
                    skip_image_count += 1
                    skip_image_details.append({"doc_key": doc_key, "image_name": path.name, "reason": message})
                elif "GT not found" in message:
                    _notice(f"[SKIP][GT 없음] {doc_key}/{path.name}: {message}")
                    skip_gt_count += 1
                else:
                    _notice(f"[SKIP] {doc_key}/{path.name}: {message}")
                skip_count += 1
                if progress_bar is not None:
                    progress_bar.set_postfix(ok=ok_count, fail=fail_count, skip=skip_count, refresh=False)
                continue
            except Exception as e:
                _notice(f"[FAIL] {doc_key}/{path.name}: {e}")
                fail_count += 1
                fail_details.append({"doc_key": doc_key, "image_name": path.name, "reason": str(e)})
                if progress_bar is not None:
                    progress_bar.set_postfix(ok=ok_count, fail=fail_count, skip=skip_count, refresh=False)
                if stop_on_error:
                    raise
                continue

            eval_path = engine_out_root / doc_key / Path(path.name).stem / "eval" / "gt_eval_summary.json"
            if eval_path.exists():
                result = json.loads(eval_path.read_text(encoding="utf-8"))
                text_metrics = result.get("text", {}) if isinstance(result, dict) else {}
                structure = result.get("structure", {}) if isinstance(result, dict) else {}
                structure_aggregate = structure.get("aggregate", {}) if isinstance(structure, dict) else {}
                field_metrics = structure.get("field_metrics", []) if isinstance(structure, dict) else []
                char_similarity_pct = text_metrics.get("char_similarity_pct")
                similarity_ratio = (
                    float(char_similarity_pct) / 100.0 if char_similarity_pct is not None else None
                )
                structure_micro_recall = result.get("structure_micro_recall")
                structure_macro_f1 = result.get("structure_macro_f1")
                matched_fields = structure_aggregate.get("matched")
                total_fields = structure_aggregate.get("gt_total")
                table_html_info = result.get("table_html", {}) if isinstance(result, dict) else {}
                table_html_exists = bool(table_html_info.get("exists", False))
                table_rows_info = result.get("table_rows", {}) if isinstance(result, dict) else {}
                table_rows_exists = bool(table_rows_info.get("exists", False))

                required_fields = result.get("required_fields", [])
                if not isinstance(required_fields, list):
                    required_fields = []
                per_field = {
                    str(metric.get("field_path", "")): metric
                    for metric in field_metrics
                    if isinstance(metric, dict)
                }

                missing_required_fields = result.get("missing_required_fields", [])
                if not isinstance(missing_required_fields, list):
                    missing_required_fields = []
                if not missing_required_fields and required_fields and per_field:
                    for field_name in required_fields:
                        metric = per_field.get(str(field_name))
                        if not metric:
                            missing_required_fields.append(str(field_name))
                            continue
                        if int(metric.get("matched", 0)) <= 0:
                            missing_required_fields.append(str(field_name))

                review_reasons = result.get("review_reasons", [])
                if not isinstance(review_reasons, list):
                    review_reasons = []
                if not review_reasons:
                    if structure_micro_recall is not None and float(structure_micro_recall) < 0.8:
                        review_reasons.append("structure_micro_recall<0.8")
                    if missing_required_fields:
                        review_reasons.append("required_field_missing")
                    if str(result.get("type", "")).lower() == "table" and not table_html_exists:
                        review_reasons.append("table_html_missing")
                    if str(result.get("type", "")).lower() == "table" and not table_rows_exists:
                        review_reasons.append("table_rows_missing")

                review_required = bool(result.get("review_required", bool(review_reasons)))
                row = {
                    "id": result.get("id"),
                    "doc_key": doc_key,
                    "type": result.get("type"),
                    "status": result.get("status"),
                    "text_similarity": similarity_ratio,
                    "cer": text_metrics.get("cer"),
                    "wer": text_metrics.get("wer"),
                    "structure_micro_recall": structure_micro_recall,
                    "structure_macro_f1": structure_macro_f1,
                    "matched_fields": matched_fields,
                    "total_fields": total_fields,
                    "table_html_exists": table_html_exists,
                    "table_rows_exists": table_rows_exists,
                    "review_required": review_required,
                    "review_reasons": "|".join(review_reasons),
                    "missing_required_fields": "|".join(missing_required_fields),
                    "latency_ms": result.get("latency_ms"),
                }
                eval_rows.append(row)
                if review_required:
                    review_queue_rows.append(
                        {
                            "id": result.get("id"),
                            "doc_key": doc_key,
                            "image_name": path.name,
                            "type": result.get("type"),
                            "review_reasons": review_reasons,
                            "missing_required_fields": missing_required_fields,
                            "structure_micro_recall": structure_micro_recall,
                            "structure_macro_f1": structure_macro_f1,
                            "eval_path": str(eval_path),
                            "pred_structured_path": str(eval_path.parent / "gt_pred_structured.json"),
                            "pred_table_html_path": str(eval_path.parent.parent / "inference" / "pred_table_raw.html"),
                            "pred_table_rows_path": str(eval_path.parent.parent / "inference" / "pred_table_layout.json"),
                        }
                    )

                if not quiet:
                    similarity_pct = (
                        float(row["text_similarity"]) * 100.0 if row.get("text_similarity") is not None else 0.0
                    )
                    print("=" * 80)
                    print(row["id"])
                    print(f"type: {row['type']}")
                    print(f"status: {row['status']}")
                    print(f"문자 유사도: {round(similarity_pct, 2)} %")
                    print(f"CER: {round(float(row['cer']), 4) if row.get('cer') is not None else 'N/A'}")
                    print(f"WER: {round(float(row['wer']), 4) if row.get('wer') is not None else 'N/A'}")
                    if row.get("structure_micro_recall") is not None:
                        print(
                            "구조 micro recall: "
                            f"{round(float(row['structure_micro_recall']) * 100.0, 2)} % "
                            f"({row.get('matched_fields')}/{row.get('total_fields')})"
                        )
                    if row.get("structure_macro_f1") is not None:
                        print(f"구조 macro f1: {round(float(row['structure_macro_f1']), 4)}")
                    print(f"table_html_exists: {row.get('table_html_exists')}")
                    print(f"table_rows_exists: {row.get('table_rows_exists')}")
                    if row.get("review_required"):
                        print(
                            "review_required: True "
                            f"reasons={row.get('review_reasons')} "
                            f"missing_required={row.get('missing_required_fields')}"
                        )
                    print(f"latency_ms: {row['latency_ms']}")

            if progress_bar is not None:
                progress_bar.set_postfix(ok=ok_count, fail=fail_count, skip=skip_count, refresh=False)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    print("\n=== OCR Batch Summary ===")
    print(f"1. total_docs: {len(doc_dirs)}")
    print(f"1-1. total_images: {len(jobs)}")
    print(f"2. ok: {ok_count}")
    print(f"3. skip: {skip_count}")
    print(f"3-1. skip_image_missing: {skip_image_count}")
    print(f"3-2. skip_gt_missing: {skip_gt_count}")
    print(f"3-3. images_root: {images_root}")
    print(f"3-4. output_root: {engine_out_root}")
    if images_tag:
        print(f"3-5. images_tag: {images_tag}")
    if skip_image_details:
        print("3-6. skip_image_missing_details:")
        for idx, detail in enumerate(skip_image_details, start=1):
            print(f"  {idx}) doc_key={detail['doc_key']} image={detail['image_name']} reason={detail['reason']}")
    print(f"4. fail: {fail_count}")
    if fail_details:
        print("4-1. fail_details:")
        for idx, detail in enumerate(fail_details, start=1):
            print(f"  {idx}) doc_key={detail['doc_key']} image={detail['image_name']} reason={detail['reason']}")

    if eval_rows:
        engine_out_root.mkdir(parents=True, exist_ok=True)

        summary_csv_path = engine_out_root / "ocr_eval_summary.csv"
        with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "doc_key",
                    "type",
                    "status",
                    "text_similarity",
                    "cer",
                    "wer",
                    "structure_micro_recall",
                    "structure_macro_f1",
                    "matched_fields",
                    "total_fields",
                    "table_html_exists",
                    "table_rows_exists",
                    "review_required",
                    "review_reasons",
                    "missing_required_fields",
                    "latency_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(eval_rows)

        summary_json_path = engine_out_root / "ocr_eval_summary.json"
        summary_json_path.write_text(json.dumps(eval_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        avg_similarity = sum(float(row["text_similarity"]) for row in eval_rows if row.get("text_similarity") is not None)
        avg_similarity /= max(1, len([row for row in eval_rows if row.get("text_similarity") is not None]))

        structure_recalls = [
            float(row["structure_micro_recall"]) for row in eval_rows if row.get("structure_micro_recall") is not None
        ]
        avg_structure_micro_recall = sum(structure_recalls) / len(structure_recalls) if structure_recalls else 0.0
        structure_macro_f1_values = [
            float(row["structure_macro_f1"]) for row in eval_rows if row.get("structure_macro_f1") is not None
        ]
        avg_structure_macro_f1 = (
            sum(structure_macro_f1_values) / len(structure_macro_f1_values) if structure_macro_f1_values else 0.0
        )
        review_required_count = sum(1 for row in eval_rows if bool(row.get("review_required")))
        fail_rate = (
            sum(1 for row in eval_rows if str(row.get("status", "")) != "success") / len(eval_rows) * 100.0
            if eval_rows
            else 0.0
        )

        summary_txt_path = engine_out_root / "ocr_eval_summary.txt"
        lines: list[str] = []
        for row in eval_rows:
            similarity_pct = float(row["text_similarity"]) * 100.0 if row.get("text_similarity") is not None else 0.0
            lines.append("=" * 80)
            lines.append(str(row.get("id", "")))
            lines.append(f"type: {row.get('type', '')}")
            lines.append(f"status: {row.get('status', '')}")
            lines.append(f"문자 유사도: {round(similarity_pct, 2)} %")
            lines.append(f"CER: {round(float(row['cer']), 4) if row.get('cer') is not None else 'N/A'}")
            lines.append(f"WER: {round(float(row['wer']), 4) if row.get('wer') is not None else 'N/A'}")
            if row.get("structure_micro_recall") is not None:
                lines.append(
                    "구조 micro recall: "
                    f"{round(float(row['structure_micro_recall']) * 100.0, 2)} % "
                    f"({row.get('matched_fields')}/{row.get('total_fields')})"
                )
            if row.get("structure_macro_f1") is not None:
                lines.append(f"구조 macro f1: {round(float(row['structure_macro_f1']), 4)}")
            lines.append(f"table_html_exists: {row.get('table_html_exists')}")
            lines.append(f"table_rows_exists: {row.get('table_rows_exists')}")
            if row.get("review_required"):
                lines.append(
                    "review_required: True "
                    f"reasons={row.get('review_reasons')} "
                    f"missing_required={row.get('missing_required_fields')}"
                )
            lines.append(f"latency_ms: {row.get('latency_ms')}")

        lines.append("")
        lines.append("[SUMMARY]")
        lines.append(f"평가 이미지 수: {len(eval_rows)}")
        lines.append(f"평균 문자 유사도: {round(avg_similarity * 100.0, 2)} %")
        lines.append(f"평균 구조 micro recall: {round(avg_structure_micro_recall * 100.0, 2)} %")
        lines.append(f"평균 구조 macro f1: {round(avg_structure_macro_f1, 4)}")
        lines.append(f"리뷰 큐 건수: {review_required_count}")
        lines.append(f"실패율: {round(fail_rate, 2)} %")
        summary_txt_path.write_text("\n".join(lines), encoding="utf-8")

        print("\n[SUMMARY]")
        print(f"평가 이미지 수: {len(eval_rows)}")
        print(f"평균 문자 유사도: {round(avg_similarity * 100.0, 2)} %")
        print(f"평균 구조 micro recall: {round(avg_structure_micro_recall * 100.0, 2)} %")
        print(f"평균 구조 macro f1: {round(avg_structure_macro_f1, 4)}")
        print(f"리뷰 큐 건수: {review_required_count}")
        print(f"실패율: {round(fail_rate, 2)} %")
        print(f"saved_eval_summary_csv: {summary_csv_path}")
        print(f"saved_eval_summary_json: {summary_json_path}")
        print(f"saved_eval_summary_txt: {summary_txt_path}")
        review_queue_path = engine_out_root / "review_queue.jsonl"
        if review_queue_rows:
            with review_queue_path.open("w", encoding="utf-8") as f:
                for row in review_queue_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            review_queue_path.write_text("", encoding="utf-8")
        print(f"saved_review_queue_jsonl: {review_queue_path}")
    elif not require_gt:
        engine_out_root.mkdir(parents=True, exist_ok=True)
        inference_summary = {
            "mode": "inference_only",
            "engine": normalized_engine,
            "images_root": images_root,
            "images_tag": images_tag,
            "engine_output_root": str(engine_out_root),
            "total_docs": len(doc_dirs),
            "ok": ok_count,
            "skip": skip_count,
            "skip_image_missing": skip_image_count,
            "skip_gt_missing": skip_gt_count,
            "fail": fail_count,
            "skip_image_missing_details": skip_image_details,
            "fail_details": fail_details,
            "message": "GT-based eval summaries are skipped.",
        }
        summary_json_path = engine_out_root / "ocr_batch_inference_summary.json"
        summary_txt_path = engine_out_root / "ocr_batch_inference_summary.txt"
        summary_json_path.write_text(json.dumps(inference_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        txt_lines = [
            "=== OCR Batch Summary (Inference-Only) ===",
            f"engine: {normalized_engine}",
            f"total_docs: {len(doc_dirs)}",
            f"ok: {ok_count}",
            f"skip: {skip_count}",
            f"skip_image_missing: {skip_image_count}",
            f"skip_gt_missing: {skip_gt_count}",
            f"fail: {fail_count}",
            "[SUMMARY] inference-only mode: GT-based eval summaries are skipped.",
        ]
        if skip_image_details:
            txt_lines.append("skip_image_missing_details:")
            for idx, detail in enumerate(skip_image_details, start=1):
                txt_lines.append(
                    f"  {idx}. doc_key={detail['doc_key']} image={detail['image_name']} reason={detail['reason']}"
                )
        if fail_details:
            txt_lines.append("fail_details:")
            for idx, detail in enumerate(fail_details, start=1):
                txt_lines.append(f"  {idx}. doc_key={detail['doc_key']} image={detail['image_name']} reason={detail['reason']}")
        summary_txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

        print("5. mode: inference-only (GT-based eval summaries are skipped)")
        print("6. output")
        print(f"6-1. saved_inference_summary_json: {summary_json_path}")
        print(f"6-2. saved_inference_summary_txt: {summary_txt_path}")


def _pipeline_paths_for_input(
    input_file: Path,
    output_dir: str,
    *,
    prechunk_output: str | None,
    chunks_output: str | None,
    embedded_output: str | None,
    debug_headings_output: str | None,
) -> tuple[Path, Path | None, Path, Path, Path]:
    explicit = [prechunk_output, chunks_output, embedded_output]
    if any(explicit) and not all(explicit):
        raise SystemExit(
            "For explicit paths, pass all of --prechunk-output, --chunks-output, --embedded-output."
        )

    if all(explicit):
        prechunk = Path(prechunk_output)  # type: ignore[arg-type]
        chunks = Path(chunks_output)  # type: ignore[arg-type]
        embedded = Path(embedded_output)  # type: ignore[arg-type]
        heading_debug = Path(debug_headings_output) if debug_headings_output else None
        metadata_sample = embedded.with_name(f"{embedded.stem}_chroma_metadata_sample.json")
        doc_dir = prechunk.parent
        doc_dir.mkdir(parents=True, exist_ok=True)
        return prechunk, heading_debug, chunks, embedded, metadata_sample

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_file.stem
    safe_stem = re.sub(r"[\\/:*?\"<>|&\s]+", "_", stem).strip("._")
    safe_stem = re.sub(r"_+", "_", safe_stem) or "document"
    doc_dir = out_dir / safe_stem
    doc_dir.mkdir(parents=True, exist_ok=True)
    prechunk = doc_dir / f"{stem}_prechunk.jsonl"
    heading_debug = (
        Path(debug_headings_output)
        if debug_headings_output
        else doc_dir / f"{stem}_heading_debug.jsonl"
    )
    chunks = doc_dir / f"{stem}_chunks.jsonl"
    embedded = doc_dir / f"{stem}_embedded.jsonl"
    metadata_sample = doc_dir / f"{stem}_chroma_metadata_sample.json"
    return prechunk, heading_debug, chunks, embedded, metadata_sample


def _parse_pipeline_input_file(
    input_file: Path,
    *,
    prechunk: Path,
    heading_debug: Path | None,
    group_size: int,
    debug_headings: bool,
    pdf_backend: str,
    pdf_no_tables: bool,
) -> None:
    suffix = input_file.suffix.lower()
    if suffix == ".hwp":
        parse_hwp(
            str(prechunk),
            input_path=str(input_file),
            debug_headings=str(heading_debug) if debug_headings and heading_debug else None,
            group_size=group_size,
        )
        return

    if suffix == ".pdf":
        from src.Parsing.pdf_parsing import build_prechunk_records, write_jsonl

        records = build_prechunk_records(
            input_file,
            group_size=group_size,
            debug_headings_path=heading_debug if debug_headings else None,
            backend=pdf_backend,
            extract_tables=not pdf_no_tables,
        )
        write_jsonl(prechunk, records)
        print(f"parsed_records: {len(records)}")
        print(f"written_records: {len(records)}")
        print(f"output: {prechunk}")
        return

    raise SystemExit(f"Unsupported run-pipeline input extension: {input_file.suffix}")


def _run_pipeline_for_file(
    input_file: Path,
    *,
    output_dir: str,
    index_dir: str | None,
    doc_id: str | None,
    model: str,
    batch_size: int,
    force_real: bool,
    group_size: int,
    debug_headings: bool,
    dump_metadata_sample: bool,
    dump_limit: int,
    prechunk_output: str | None = None,
    chunks_output: str | None = None,
    embedded_output: str | None = None,
    debug_headings_output: str | None = None,
    pdf_backend: str = "auto",
    pdf_no_tables: bool = False,
) -> None:
    import chromadb

    prechunk, heading_debug, chunks, embedded, metadata_sample = _pipeline_paths_for_input(
        input_file,
        output_dir,
        prechunk_output=prechunk_output,
        chunks_output=chunks_output,
        embedded_output=embedded_output,
        debug_headings_output=debug_headings_output,
    )
    resolved_index_dir = index_dir or str(prechunk.parent / "chroma_index")

    print(f"pipeline_input: {input_file}")
    print("[1/4] parse-document")
    _parse_pipeline_input_file(
        input_file,
        prechunk=prechunk,
        heading_debug=heading_debug,
        group_size=group_size,
        debug_headings=debug_headings,
        pdf_backend=pdf_backend,
        pdf_no_tables=pdf_no_tables,
    )

    print("[2/4] chunk-jsonl")
    chunk_jsonl(input_path=str(prechunk), output_path=str(chunks))

    print("[3/4] embed-jsonl")
    embed_jsonl(
        input_path=str(chunks),
        output_path=str(embedded),
        model=model,
        batch_size=batch_size,
        force_real=force_real,
    )
    print("[4/4] build-chroma")
    build_chroma_index(input_path=str(embedded), index_dir=resolved_index_dir, doc_id=doc_id)
    if dump_metadata_sample:
        client = chromadb.PersistentClient(path=str(resolved_index_dir))
        col = client.get_collection("rfp_chunks")
        result = col.get(limit=max(1, dump_limit), include=["metadatas"])
        keys = [
            "file_name",
            "section_path_text",
            "section_type",
            "table_type",
            "table_id",
            "row_range",
            "chunk_id",
            "doc_id",
            "row_idx",
        ]
        rows = [{k: m.get(k) for k in keys} for m in result.get("metadatas", [])]
        metadata_sample.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("pipeline_done")
    print(f"prechunk: {prechunk}")
    print(f"chunks: {chunks}")
    print(f"embedded: {embedded}")
    print(f"index_dir: {resolved_index_dir}")
    if dump_metadata_sample:
        print(f"metadata_sample: {metadata_sample}")


def _dump_chroma_metadata_sample(index_dir: str, output_path: Path, *, dump_limit: int) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(index_dir))
    col = client.get_collection("rfp_chunks")
    result = col.get(limit=max(1, dump_limit), include=["metadatas"])
    keys = [
        "file_name",
        "section_path_text",
        "section_type",
        "table_type",
        "table_id",
        "row_range",
        "chunk_id",
        "doc_id",
        "row_idx",
    ]
    rows = [{k: m.get(k) for k in keys} for m in result.get("metadatas", [])]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_pipeline_for_mixed_dir(
    input_dir: Path,
    *,
    output_dir: str,
    index_dir: str | None,
    doc_id: str | None,
    model: str,
    batch_size: int,
    force_real: bool,
    group_size: int,
    dump_metadata_sample: bool,
    dump_limit: int,
    glob_pattern: str,
    recursive: bool,
    limit_files: int,
    pdf_backend: str,
    pdf_no_tables: bool,
) -> None:
    from src.pipeline.mixed_slim_pipeline import chunk_mixed_dir_to_slim_jsonl

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    chunks = output_root / "mixed_chunks_slim.jsonl"
    tables = output_root / "mixed_tables_raw.jsonl"
    errors = output_root / "mixed_chunks_slim_errors.json"
    embedded = output_root / "mixed_embedded.jsonl"
    metadata_sample = output_root / "mixed_chroma_metadata_sample.json"
    resolved_index_dir = index_dir or str(output_root / "chroma_index")

    print(f"pipeline_input_dir: {input_dir}")
    print("[1/3] parse/chunk-mixed-slim")
    result = chunk_mixed_dir_to_slim_jsonl(
        input_dir=input_dir,
        output_path=chunks,
        errors_path=errors,
        tables_output_path=tables,
        glob_pattern=glob_pattern,
        recursive=recursive,
        limit_files=limit_files,
        group_size=group_size,
        pdf_backend=pdf_backend,
        pdf_extract_tables=not pdf_no_tables,
    )
    for key, value in result.items():
        print(f"{key}: {value}")

    print("[2/3] embed-jsonl")
    embed_jsonl(
        input_path=str(chunks),
        output_path=str(embedded),
        model=model,
        batch_size=batch_size,
        force_real=force_real,
    )

    print("[3/3] build-chroma")
    build_chroma_index(input_path=str(embedded), index_dir=resolved_index_dir, doc_id=doc_id)
    if dump_metadata_sample:
        _dump_chroma_metadata_sample(resolved_index_dir, metadata_sample, dump_limit=dump_limit)

    print("pipeline_done")
    print(f"chunks: {chunks}")
    print(f"tables: {tables}")
    print(f"embedded: {embedded}")
    print(f"index_dir: {resolved_index_dir}")
    if dump_metadata_sample:
        print(f"metadata_sample: {metadata_sample}")


def run_pipeline(
    *,
    input_path: str | None = None,
    input_dir: str | None = None,
    output_dir: str = "data/v2",
    index_dir: str | None = None,
    doc_id: str | None = None,
    model: str = "text-embedding-3-small",
    batch_size: int = 64,
    force_real: bool = False,
    group_size: int = 8,
    debug_headings: bool = True,
    dump_metadata_sample: bool = False,
    dump_limit: int = 20,
    prechunk_output: str | None = None,
    chunks_output: str | None = None,
    embedded_output: str | None = None,
    debug_headings_output: str | None = None,
    glob_pattern: str = "*",
    recursive: bool = False,
    limit_files: int = 0,
    pdf_backend: str = "auto",
    pdf_no_tables: bool = False,
) -> None:
    _require_exactly_one(
        a=input_path,
        b=input_dir,
        a_name="--input (single HWP/PDF)",
        b_name="--input-dir (folder)",
    )

    explicit_outputs = any([prechunk_output, chunks_output, embedded_output])
    if input_dir and explicit_outputs:
        raise SystemExit("Explicit --*-output paths work only with --input (single file).")

    if input_path:
        _run_pipeline_for_file(
            Path(input_path),
            output_dir=output_dir,
            index_dir=index_dir,
            doc_id=doc_id,
            model=model,
            batch_size=batch_size,
            force_real=force_real,
            group_size=group_size,
            debug_headings=debug_headings,
            dump_metadata_sample=dump_metadata_sample,
            dump_limit=dump_limit,
            prechunk_output=prechunk_output,
            chunks_output=chunks_output,
            embedded_output=embedded_output,
            debug_headings_output=debug_headings_output,
            pdf_backend=pdf_backend,
            pdf_no_tables=pdf_no_tables,
        )
        return

    _discover_hwp_pdf_in_dir(
        input_dir,  # type: ignore[arg-type]
        glob_pattern=glob_pattern,
        recursive=recursive,
        limit_files=limit_files,
    )
    _run_pipeline_for_mixed_dir(
        Path(input_dir),  # type: ignore[arg-type]
        output_dir=output_dir,
        index_dir=index_dir,
        doc_id=doc_id,
        model=model,
        batch_size=batch_size,
        force_real=force_real,
        group_size=group_size,
        dump_metadata_sample=dump_metadata_sample,
        dump_limit=dump_limit,
        glob_pattern=glob_pattern,
        recursive=recursive,
        limit_files=limit_files,
        pdf_backend=pdf_backend,
        pdf_no_tables=pdf_no_tables,
    )


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Bidmate RAG scenario B baseline")
    parser.add_argument("--config", default="configs/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Load RFP files, chunk, embed, and build Chroma index")

    query_parser = subparsers.add_parser("query", help="Ask a question against the built index")
    query_parser.add_argument("question")

    subparsers.add_parser("evaluate", help="Run questions from the evaluation JSONL file")

    harness_parser = subparsers.add_parser(
        "evaluate-harness",
        help="RAG eval with LLM-as-judge and LangSmith traces (default; set LANGSMITH_* in .env)",
    )
    harness_parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: outputs/eval_harness_results.jsonl)",
    )
    harness_parser.add_argument(
        "--evaluation-set",
        default=None,
        help="Eval questions JSONL (default: paths.evaluation_set in config)",
    )
    harness_parser.add_argument(
        "--judge-model",
        default="gpt-5-mini",
        help="OpenAI chat model for faithfulness/relevance scoring",
    )
    harness_parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip GPT judge (retrieval keyword metrics only if keywords are present)",
    )
    harness_parser.add_argument(
        "--no-langsmith-feedback",
        action="store_true",
        help="Do not attach LangSmith Client feedback scores (tracing still follows LANGSMITH_* env)",
    )
    harness_parser.add_argument(
        "--no-correctness-judge",
        action="store_true",
        help="Skip expected_answer correctness judge (retrieval + f/r/s only)",
    )

    check_parser = subparsers.add_parser(
        "check-setup",
        help="Validate local settings, dependencies, paths, and optional OpenAI connectivity",
    )
    check_parser.add_argument(
        "--check-openai",
        action="store_true",
        help="Make a real OpenAI API request to verify the key and network connection",
    )
    subparsers.add_parser(
        "check-ocr3-setup",
        help="Validate PaddleOCR 3.x package/module installation status",
    )

    embed_parser = subparsers.add_parser("embed-jsonl", help="Embed prepared JSONL rows")
    embed_parser.add_argument("--input", required=True, help="Input JSONL path")
    embed_parser.add_argument("--output", required=True, help="Output JSONL path")
    embed_parser.add_argument("--model", default="text-embedding-3-small", help="Embedding model")
    embed_parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    embed_parser.add_argument("--force-real", action="store_true", help="Fail if OPENAI_API_KEY is missing")

    chroma_parser = subparsers.add_parser("build-chroma", help="Build Chroma checkpoint from embedded JSONL")
    chroma_parser.add_argument("--input", required=True, help="Embedded JSONL input path")
    chroma_parser.add_argument(
        "--index-dir",
        default=None,
        help="Chroma checkpoint output directory; defaults to config paths.index_dir",
    )
    chroma_parser.add_argument("--doc-id", default=None, help="Optional doc id override")

    merge_parser = subparsers.add_parser(
        "merge-embedded",
        help="Merge data/v2 *_embedded.jsonl files and build unified Chroma checkpoint",
    )
    merge_parser.add_argument(
        "--input-dir",
        default="data/v2",
        help="Root directory to search for embedded JSONL files",
    )
    merge_parser.add_argument(
        "--index-dir",
        default=None,
        help="Unified Chroma output directory; defaults to config paths.index_dir",
    )
    merge_parser.add_argument(
        "--merged-output",
        default=None,
        help="Merged embedded JSONL path (default: checkpoints/all_embedded.jsonl)",
    )
    merge_parser.add_argument(
        "--pattern",
        default="*_embedded.jsonl",
        help="Glob pattern for embedded files under --input-dir",
    )
    merge_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Search only --input-dir itself, not subfolders",
    )
    merge_parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Write merged JSONL only; skip build-chroma",
    )

    parse_parser = subparsers.add_parser("parse-hwp", help="Parse HWP into prechunk JSONL")
    parse_input = parse_parser.add_mutually_exclusive_group(required=True)
    parse_input.add_argument("--input", help="Single HWP file path")
    parse_input.add_argument(
        "--input-dir",
        help="Folder of HWP files (*.hwp in that folder only; merged into one prechunk JSONL)",
    )
    parse_parser.add_argument("--output", required=True, help="Prechunk JSONL output path")
    parse_parser.add_argument("--debug-headings", default=None, help="Optional heading debug JSONL path")
    parse_parser.add_argument("--limit", type=int, default=0, help="Write first N records only; 0=all")
    parse_parser.add_argument("--group-size", type=int, default=8, help="Table row group size for parser")

    chunk_parser = subparsers.add_parser("chunk-jsonl", help="Chunk prechunk JSONL into RAG chunk JSONL")
    chunk_parser.add_argument("--input", required=True, help="Prechunk JSONL input path")
    chunk_parser.add_argument("--output", required=True, help="RAG chunk JSONL output path")
    chunk_parser.add_argument("--summary-output", default=None, help="Optional chunk summary CSV path")
    chunk_parser.add_argument("--sample-output", default=None, help="Optional sample JSONL path")
    chunk_parser.add_argument("--sample-size", type=int, default=20, help="Sample chunk count")
    chunk_parser.add_argument("--text-chunk-size", type=int, default=900, help="Text chunk body size")
    chunk_parser.add_argument("--text-overlap", type=int, default=180, help="Text overlap chars")
    chunk_parser.add_argument("--table-chunk-size", type=int, default=1000, help="Table chunk body size")
    chunk_parser.add_argument("--max-table-rows", type=int, default=6, help="Max table rows per chunk")
    chunk_parser.add_argument("--min-text-chars", type=int, default=40, help="Min text chars")
    chunk_parser.add_argument("--short-context-chars", type=int, default=140, help="Short context threshold")
    chunk_parser.add_argument("--include-toc", action="store_true", help="Include TOC records")
    chunk_parser.add_argument("--exclude-cover", action="store_true", help="Exclude cover_text records")
    chunk_parser.add_argument("--include-debug-metadata", action="store_true", help="Keep debug metadata")

    sampling_parser = subparsers.add_parser(
        "sampling",
        help="Sample evaluation chunks from slim RAG chunk JSONL (one file or many under a folder)",
    )
    sampling_input = sampling_parser.add_mutually_exclusive_group(required=True)
    sampling_input.add_argument("--input", help="Single chunks JSONL path")
    sampling_input.add_argument(
        "--input-dir",
        help="Directory to search for chunk JSONL files (e.g. after run-pipeline --input-dir)",
    )
    sampling_parser.add_argument(
        "--pattern",
        default="*_chunks.jsonl",
        help="Glob for --input-dir (default: *_chunks.jsonl)",
    )
    sampling_parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search subfolders under --input-dir (default: on)",
    )
    sampling_parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Only files directly under --input-dir",
    )
    sampling_parser.add_argument("--output", required=True, help="Output sampled JSONL")
    sampling_parser.add_argument(
        "--quotas",
        default=None,
        help=(
            "Comma-separated quota override. Example: "
            "overview=1,requirements=4,evaluation=2,bid_contract=2,security=1,appendix_form=1"
        ),
    )
    sampling_parser.add_argument(
        "--appendix-mode",
        choices=("auto", "always", "never"),
        default="auto",
        help="How to sample appendix_form chunks",
    )
    sampling_parser.add_argument(
        "--min-per-doc",
        type=int,
        default=9,
        help="Auto appendix/body fallback target; does not force duplicate chunks",
    )
    sampling_parser.add_argument(
        "--fallback-body",
        type=int,
        default=0,
        help="If a document is sparse, sample up to this many body chunks as fallback",
    )
    sampling_parser.add_argument(
        "--min-chars",
        type=int,
        default=80,
        help="Drop chunks shorter than this",
    )
    sampling_parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Debug only: first N documents",
    )
    sampling_parser.add_argument(
        "--add-sampling-metadata",
        action="store_true",
        help="Add sample_strategy/sample_rank fields to metadata",
    )

    chunk_dir_parser = subparsers.add_parser(
        "chunk-hwp-dir",
        help="Parse and chunk HWP files in a directory without embedding or vector indexing",
    )
    chunk_dir_parser.add_argument("--input-dir", default="data/v1/raw", help="Directory containing HWP files")
    chunk_dir_parser.add_argument("--output-dir", default="data/v2/chunked_hwp", help="Chunk output directory")
    chunk_dir_parser.add_argument("--limit", type=int, default=0, help="Process first N HWP files; 0=all")
    chunk_dir_parser.add_argument("--group-size", type=int, default=8, help="Table row group size for parser")
    chunk_dir_parser.add_argument("--sample-size", type=int, default=10, help="Sample chunks per document")
    chunk_dir_parser.add_argument("--text-chunk-size", type=int, default=900, help="Text chunk body size")
    chunk_dir_parser.add_argument("--text-overlap", type=int, default=180, help="Text overlap chars")

    chunk_slim_parser = subparsers.add_parser(
        "chunk-hwp-slim",
        help="Parse HWP files into one slim JSONL including section and table chunks",
    )
    chunk_slim_parser.add_argument("--input-dir", default="data/v1/raw", help="Directory containing HWP files")
    chunk_slim_parser.add_argument(
        "--input-file",
        action="append",
        default=None,
        help="Specific HWP file to process; can be passed multiple times. Overrides directory discovery.",
    )
    chunk_slim_parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL. If omitted with one --input-file, uses data/v2/samples/<input_name>_slim_with_tables.jsonl",
    )
    chunk_slim_parser.add_argument("--errors-output", default=None, help="Optional errors JSON path")
    chunk_slim_parser.add_argument(
        "--tables-output",
        default=None,
        help="Optional table-only raw JSONL path; defaults to <output_stem>_tables_raw.jsonl",
    )
    chunk_slim_parser.add_argument("--glob", default="*.hwp", help="HWP filename glob")
    chunk_slim_parser.add_argument("--recursive", action="store_true", help="Search input directory recursively")
    chunk_slim_parser.add_argument("--limit-files", type=int, default=0, help="Process first N HWP files; 0=all")
    chunk_slim_parser.add_argument("--group-size", type=int, default=8, help="Table row group size")
    chunk_slim_parser.add_argument("--text-chunk-size", type=int, default=900, help="Text chunk body size")
    chunk_slim_parser.add_argument("--text-overlap", type=int, default=150, help="Text overlap chars")
    chunk_slim_parser.add_argument("--table-chunk-size", type=int, default=1000, help="Table chunk body size")
    chunk_slim_parser.add_argument("--max-table-rows", type=int, default=6, help="Max table rows per chunk")
    chunk_slim_parser.add_argument("--include-toc", action="store_true", help="Include TOC records")
    chunk_slim_parser.add_argument("--exclude-cover", action="store_true", help="Exclude cover_text records")
    chunk_slim_parser.add_argument("--stop-on-error", action="store_true", help="Stop when one file fails")

    chunk_mixed_parser = subparsers.add_parser(
        "chunk-mixed-slim",
        help="Parse HWP/PDF files into one slim JSONL including section and table chunks",
    )
    chunk_mixed_parser.add_argument("--input-dir", default="data/v1/raw", help="Directory containing HWP/PDF files")
    chunk_mixed_parser.add_argument(
        "--input-file",
        action="append",
        default=None,
        help="Specific HWP/PDF file to process; can be passed multiple times. Overrides directory discovery.",
    )
    chunk_mixed_parser.add_argument("--output", default=None, help="Output slim chunk JSONL")
    chunk_mixed_parser.add_argument("--errors-output", default=None, help="Optional errors JSON path")
    chunk_mixed_parser.add_argument(
        "--tables-output",
        default=None,
        help="Optional table-only raw JSONL path; defaults to <output_stem>_tables_raw.jsonl",
    )
    chunk_mixed_parser.add_argument("--glob", default="*", help="Filename glob before extension filtering")
    chunk_mixed_parser.add_argument("--recursive", action="store_true", help="Search input directory recursively")
    chunk_mixed_parser.add_argument("--limit-files", type=int, default=0, help="Process first N files; 0=all")
    chunk_mixed_parser.add_argument("--group-size", type=int, default=8, help="Table row group size")
    chunk_mixed_parser.add_argument("--text-chunk-size", type=int, default=900, help="Text chunk body size")
    chunk_mixed_parser.add_argument("--text-overlap", type=int, default=150, help="Text overlap chars")
    chunk_mixed_parser.add_argument("--table-chunk-size", type=int, default=1000, help="Table chunk body size")
    chunk_mixed_parser.add_argument("--max-table-rows", type=int, default=6, help="Max table rows per chunk")
    chunk_mixed_parser.add_argument("--include-toc", action="store_true", help="Include TOC records")
    chunk_mixed_parser.add_argument("--exclude-cover", action="store_true", help="Exclude cover_text records")
    chunk_mixed_parser.add_argument("--stop-on-error", action="store_true", help="Stop when one file fails")
    chunk_mixed_parser.add_argument(
        "--pdf-backend",
        choices=("auto", "pymupdf", "pypdf"),
        default="auto",
        help="PDF parser backend. auto prefers PyMuPDF, then pypdf fallback.",
    )
    chunk_mixed_parser.add_argument(
        "--pdf-no-tables",
        action="store_true",
        help="Disable PyMuPDF PDF table extraction and keep PDF text only.",
    )

    convert_parser = subparsers.add_parser(
        "convert-embedding-input",
        help="Convert prechunk/chunk JSONL to embedding input JSONL",
    )
    convert_parser.add_argument("--input", required=True, help="Input JSONL path")
    convert_parser.add_argument("--output", required=True, help="Embedding input JSONL output path")
    convert_parser.add_argument("--doc-id", default=None, help="Optional doc id override")

    ocr_export_parser = subparsers.add_parser(
        "ocr-export-rag",
        help="Export OCR eval outputs into RAG handoff manifest/chunk JSONL",
    )
    ocr_export_parser.add_argument(
        "--ocr-eval-root",
        default="data/v2/ocr_outputs",
        help="Root folder containing OCR eval outputs",
    )
    ocr_export_parser.add_argument(
        "--output-manifest",
        default="data/v2/ocr_rag/ocr_handoff_manifest.jsonl",
        help="Output JSONL manifest for OCR->RAG handoff",
    )
    ocr_export_parser.add_argument(
        "--output-chunks",
        default="data/v2/ocr_rag/ocr_handoff_chunks.jsonl",
        help="Output chunk JSONL usable by embed-jsonl/build-chroma",
    )
    ocr_export_parser.add_argument(
        "--engine",
        default=None,
        help="Optional OCR engine folder filter (e.g. paddleocr_vl)",
    )
    ocr_export_parser.add_argument(
        "--doc-key",
        default=None,
        help="Optional document key folder filter",
    )
    ocr_export_parser.add_argument(
        "--exclude-review-required",
        action="store_true",
        help="Exclude review_required=true items from exported handoff",
    )
    ocr_export_parser.add_argument(
        "--include-html-chunk",
        action="store_true",
        help="Include table HTML snippet chunks in output-chunks",
    )
    ocr_export_parser.add_argument(
        "--html-chunk-max-chars",
        type=int,
        default=1200,
        help="Maximum chars for HTML snippet chunk when --include-html-chunk is enabled",
    )
    ocr_export_parser.add_argument(
        "--allow-inference-only",
        action="store_true",
        help="Allow exporting RAG handoff from inference-only OCR outputs (pred_raw/pred_table_layout) without GT eval files.",
    )
    ocr_export_parser.add_argument(
        "--images-tag",
        default=None,
        help="Optional OCR images version tag filter (e.g. v4_table_filtered_260531).",
    )
    ocr_export_parser.add_argument(
        "--curated-root",
        default=None,
        help="Optional curated root. If set, {curated_root}/{engine}/{doc_key}/{image_stem}/{curated-file-name} is preferred over raw table layout.",
    )
    ocr_export_parser.add_argument(
        "--curated-file-name",
        default="pred_table_layout.curated.json",
        help="Curated table layout file name under curated root.",
    )
    ocr_export_parser.add_argument(
        "--use-merge-manifest",
        action="store_true",
        help="When set, read merge_manifest.json under curated doc folders and export merged units (e.g., img_005_006).",
    )
    ocr_export_parser.add_argument(
        "--curated-only",
        action="store_true",
        help="Use only curated dataset as export source (do not enumerate ocr_outputs image units).",
    )
    ocr_export_parser.add_argument(
        "--input-version",
        default=None,
        help="Input document-set version tag (lineage metadata).",
    )
    ocr_export_parser.add_argument(
        "--ocr-engine-version",
        default=None,
        help="OCR engine/model version tag (lineage metadata).",
    )
    ocr_export_parser.add_argument(
        "--ocr-output-version",
        default=None,
        help="Raw OCR output version tag (lineage metadata).",
    )
    ocr_export_parser.add_argument(
        "--ocr-curated-version",
        default=None,
        help="Curated OCR output version tag (lineage metadata).",
    )
    ocr_export_parser.add_argument(
        "--rag-index-version",
        default=None,
        help="Planned RAG index version tag (lineage metadata).",
    )

    ocr_parser = subparsers.add_parser(
        "extract-ocr-images",
        help="Extract embedded images from HWP/PDF files for OCR preparation",
    )
    ocr_parser.add_argument("--input-dir", required=True, help="Directory containing HWP/PDF files")
    ocr_parser.add_argument(
        "--output-dir",
        default="data/v2/ocr_images",
        help="Directory to save extracted images (default: data/v2/ocr_images)",
    )
    ocr_parser.add_argument("--limit", type=int, default=0, help="Process first N files only; 0=all")
    ocr_parser.add_argument(
        "--source-type",
        choices=["all", "hwp", "pdf"],
        default="all",
        help="Source file types to process",
    )
    ocr_parser.add_argument("--recursive", action="store_true", help="Recursively search input-dir")
    ocr_parser.add_argument("--pdf-min-width", type=int, default=100, help="PDF image min width")
    ocr_parser.add_argument("--pdf-min-height", type=int, default=40, help="PDF image min height")
    ocr_parser.add_argument("--pdf-min-area", type=int, default=10_000, help="PDF image min area")
    ocr_parser.add_argument("--pdf-min-bytes", type=int, default=1_000, help="PDF image min byte size")

    build_pred_parser = subparsers.add_parser(
        "build-pred-structured",
        help="Convert raw OCR JSON to GT-aligned pred structured JSON",
    )
    build_pred_parser.add_argument("--gt", required=True, help="GT path (.json/.jsonl)")
    build_pred_parser.add_argument("--pred-raw", required=True, help="Raw OCR JSON path")
    build_pred_parser.add_argument("--id", required=True, help="Target item id")
    build_pred_parser.add_argument("--output", required=True, help="Pred structured JSON output path")
    build_pred_parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Minimum OCR confidence threshold",
    )

    eval_pred_parser = subparsers.add_parser(
        "eval-pred-structured",
        help="Evaluate pred structured JSON against GT (.json/.jsonl)",
    )
    eval_pred_parser.add_argument("--gt", required=True, help="GT path (.json/.jsonl)")
    eval_pred_parser.add_argument("--pred-structured", required=True, help="Pred structured JSON path")
    eval_pred_parser.add_argument("--id", required=True, help="Target item id")
    eval_pred_parser.add_argument("--output", required=True, help="Evaluation JSON output path")
    eval_pred_parser.add_argument(
        "--structure-threshold",
        type=float,
        default=0.65,
        help="Value similarity threshold for structure matching",
    )

    ocr_image_parser = subparsers.add_parser(
        "ocr-run-image",
        help="Run OCR/eval for one image with doc_key-based path resolution",
    )
    ocr_image_parser.add_argument(
        "--doc-key",
        required=True,
        help="Document key (folder name under paths.images_root from ocr-config)",
    )
    ocr_image_parser.add_argument(
        "--id",
        default=None,
        help="Target item id. If omitted, auto-detected from GT JSON when unique.",
    )
    ocr_image_parser.add_argument("--image-name", default="img_001.jpg", help="Image filename inside doc folder")
    ocr_image_parser.add_argument("--lang", default="korean", help="PaddleOCR language fallback")
    ocr_image_parser.add_argument(
        "--ocr-config",
        default="configs/ocr_default.yaml",
        help="OCR config path (engine/device/model settings)",
    )
    ocr_image_parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Override minimum OCR confidence threshold for structured prediction (default from ocr-config)",
    )
    ocr_image_parser.add_argument(
        "--structure-threshold",
        type=float,
        default=None,
        help="Override structure match similarity threshold for eval (default from ocr-config)",
    )
    ocr_image_parser.add_argument(
        "--all-engines",
        action="store_true",
        help="Run default OCR engine matrix (pp_ocrv5, pp_ocrv5_transformers, pp_structurev3, table_recognition_v2, paddleocr_vl)",
    )
    ocr_image_parser.add_argument("--use-doc-orientation-classify", action="store_true")
    ocr_image_parser.add_argument("--use-doc-unwarping", action="store_true")
    ocr_image_parser.add_argument("--use-chart-recognition", action="store_true")
    ocr_image_parser.add_argument("--use-table-orientation-classify", action="store_true")
    ocr_image_parser.add_argument("--use-ocr-results-with-table-cells", action="store_true")
    ocr_image_parser.add_argument("--text-det-limit-side-len", type=int, default=None)
    ocr_image_parser.add_argument(
        "--table-dual-pass",
        action="store_true",
        help="For table images, run VL first and then PP-OCRv5 to merge non-table text.",
    )
    ocr_image_parser.add_argument(
        "--table-second-engine",
        default="pp_ocrv5",
        choices=["pp_ocrv5", "pp_ocrv5_transformers"],
        help="Second pass OCR engine used when --table-dual-pass is enabled.",
    )
    ocr_image_parser.add_argument(
        "--no-gt",
        action="store_true",
        help="Run inference-only mode without GT build/eval.",
    )

    ocr_batch_parser = subparsers.add_parser(
        "ocr-run-batch",
        help="Run OCR/eval for all doc folders under ocr_images",
    )
    ocr_batch_parser.add_argument(
        "--doc-key",
        default=None,
        help="Run batch only for one document key folder under paths.images_root from ocr-config",
    )
    ocr_batch_parser.add_argument("--image-name", default="img_001.jpg", help="Image filename inside doc folder")
    ocr_batch_parser.add_argument("--limit", type=int, default=0, help="Process first N doc folders only; 0=all")
    ocr_batch_parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop batch immediately on first failure",
    )
    ocr_batch_parser.add_argument("--lang", default="korean", help="PaddleOCR language fallback")
    ocr_batch_parser.add_argument(
        "--ocr-config",
        default="configs/ocr_default.yaml",
        help="OCR config path (engine/device/model settings)",
    )
    ocr_batch_parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Override minimum OCR confidence threshold for structured prediction (default from ocr-config)",
    )
    ocr_batch_parser.add_argument(
        "--structure-threshold",
        type=float,
        default=None,
        help="Override structure match similarity threshold for eval (default from ocr-config)",
    )
    ocr_batch_parser.add_argument(
        "--all-engines",
        action="store_true",
        help="Run default OCR engine matrix (pp_ocrv5, pp_ocrv5_transformers, pp_structurev3, table_recognition_v2, paddleocr_vl)",
    )
    ocr_batch_parser.add_argument("--use-doc-orientation-classify", action="store_true")
    ocr_batch_parser.add_argument("--use-doc-unwarping", action="store_true")
    ocr_batch_parser.add_argument("--use-chart-recognition", action="store_true")
    ocr_batch_parser.add_argument("--use-table-orientation-classify", action="store_true")
    ocr_batch_parser.add_argument("--use-ocr-results-with-table-cells", action="store_true")
    ocr_batch_parser.add_argument("--text-det-limit-side-len", type=int, default=None)
    ocr_batch_parser.add_argument(
        "--table-dual-pass",
        action="store_true",
        help="For table images, run VL first and then PP-OCRv5 to merge non-table text.",
    )
    ocr_batch_parser.add_argument(
        "--table-second-engine",
        default="pp_ocrv5",
        choices=["pp_ocrv5", "pp_ocrv5_transformers"],
        help="Second pass OCR engine used when --table-dual-pass is enabled.",
    )
    ocr_batch_parser.add_argument(
        "--no-gt",
        action="store_true",
        help="Run inference-only mode without GT build/eval.",
    )
    ocr_batch_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image OCR step logs (default: tqdm progress only).",
    )
    ocr_batch_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar for batch OCR.",
    )

    pipeline_parser = subparsers.add_parser(
        "run-pipeline",
        help="Run parse->chunk->embed->build-chroma in one command",
    )
    pipeline_input = pipeline_parser.add_mutually_exclusive_group(required=True)
    pipeline_input.add_argument("--input", help="Single HWP/PDF file path")
    pipeline_input.add_argument(
        "--input-dir",
        help="Folder of HWP/PDF files; builds one unified chunk/embedding/index output",
    )
    pipeline_parser.add_argument(
        "--output-dir",
        default="data/v2",
        help="Base output directory when --*-output paths are omitted",
    )
    pipeline_parser.add_argument("--prechunk-output", default=None, help="Prechunk JSONL path (--input only)")
    pipeline_parser.add_argument("--chunks-output", default=None, help="Chunks JSONL path (--input only)")
    pipeline_parser.add_argument("--embedded-output", default=None, help="Embedded JSONL path (--input only)")
    pipeline_parser.add_argument(
        "--debug-headings-output",
        default=None,
        help="Heading debug JSONL path (--input only)",
    )
    pipeline_parser.add_argument(
        "--index-dir",
        default=None,
        help="Chroma checkpoint output directory; defaults to config paths.index_dir",
    )
    pipeline_parser.add_argument("--doc-id", default=None, help="Optional doc id override")
    pipeline_parser.add_argument("--model", default="text-embedding-3-small", help="Embedding model")
    pipeline_parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    pipeline_parser.add_argument("--force-real", action="store_true", help="Fail if OPENAI_API_KEY is missing")
    pipeline_parser.add_argument("--group-size", type=int, default=8, help="Table row group size for parser")
    pipeline_parser.add_argument("--glob", default="*", help="Input-dir filename glob before extension filtering")
    pipeline_parser.add_argument("--recursive", action="store_true", help="Search input-dir recursively")
    pipeline_parser.add_argument("--limit-files", type=int, default=0, help="Process first N input-dir files; 0=all")
    pipeline_parser.add_argument(
        "--pdf-backend",
        choices=("auto", "pymupdf", "pypdf"),
        default="auto",
        help="PDF parser backend. auto prefers PyMuPDF, then pypdf fallback.",
    )
    pipeline_parser.add_argument(
        "--pdf-no-tables",
        action="store_true",
        help="Disable PDF table extraction and keep PDF text only.",
    )
    pipeline_parser.add_argument("--no-debug-headings", action="store_true", help="Skip heading debug JSONL")
    pipeline_parser.add_argument(
        "--dump-metadata-sample",
        action="store_true",
        help="Write Chroma metadata sample JSON (disabled by default)",
    )
    pipeline_parser.add_argument(
        "--dump-limit",
        type=int,
        default=20,
        help="Number of metadata rows to dump into sample JSON",
    )

    args = parser.parse_args()
    if args.command == "ingest":
        ingest(args.config)
    elif args.command == "query":
        query(args.config, args.question)
    elif args.command == "evaluate":
        evaluate(args.config)
    elif args.command == "evaluate-harness":
        evaluate_harness(
            args.config,
            output_path=args.output,
            evaluation_set=args.evaluation_set,
            judge_model=args.judge_model,
            no_llm_judge=args.no_llm_judge,
            no_correctness_judge=args.no_correctness_judge,
            no_langsmith_feedback=args.no_langsmith_feedback,
        )
    elif args.command == "check-setup":
        from src.utils.setup_check import run_setup_check

        sys.exit(run_setup_check(args.config, check_openai=args.check_openai))
    elif args.command == "check-ocr3-setup":
        from src.utils.ocr3_setup_check import run_ocr3_setup_check

        sys.exit(run_ocr3_setup_check())
    elif args.command == "embed-jsonl":
        embed_jsonl(
            input_path=args.input,
            output_path=args.output,
            model=args.model,
            batch_size=args.batch_size,
            force_real=args.force_real,
        )
    elif args.command == "build-chroma":
        build_chroma_index(
            input_path=args.input,
            index_dir=str(resolve_index_dir(args.config, args.index_dir)),
            doc_id=args.doc_id,
        )
    elif args.command == "merge-embedded":
        merge_embedded_checkpoint(
            args.config,
            input_dir=args.input_dir,
            index_dir=args.index_dir,
            merged_output=args.merged_output,
            pattern=args.pattern,
            recursive=not args.no_recursive,
            merge_only=args.merge_only,
        )
    elif args.command == "parse-hwp":
        parse_hwp(
            args.output,
            input_path=args.input,
            input_dir=args.input_dir,
            debug_headings=args.debug_headings,
            limit=args.limit,
            group_size=args.group_size,
        )
    elif args.command == "chunk-jsonl":
        chunk_jsonl(
            input_path=args.input,
            output_path=args.output,
            summary_output=args.summary_output,
            sample_output=args.sample_output,
            sample_size=args.sample_size,
            text_chunk_size=args.text_chunk_size,
            text_overlap=args.text_overlap,
            table_chunk_size=args.table_chunk_size,
            max_table_rows=args.max_table_rows,
            min_text_chars=args.min_text_chars,
            short_context_chars=args.short_context_chars,
            include_cover=not args.exclude_cover,
            include_toc=args.include_toc,
            include_debug_metadata=args.include_debug_metadata,
        )
    elif args.command == "sampling":
        sampling(
            args.output,
            input_path=args.input,
            input_dir=args.input_dir,
            pattern=args.pattern,
            recursive=args.recursive,
            quotas=args.quotas,
            appendix_mode=args.appendix_mode,
            min_per_doc=args.min_per_doc,
            fallback_body=args.fallback_body,
            min_chars=args.min_chars,
            limit_docs=args.limit_docs,
            add_sampling_metadata=args.add_sampling_metadata,
        )
    elif args.command == "chunk-hwp-dir":
        chunk_hwp_dir(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            limit=args.limit,
            group_size=args.group_size,
            sample_size=args.sample_size,
            text_chunk_size=args.text_chunk_size,
            text_overlap=args.text_overlap,
        )
    elif args.command == "chunk-hwp-slim":
        chunk_hwp_slim(
            input_dir=args.input_dir,
            input_files=args.input_file,
            output_path=args.output,
            errors_path=args.errors_output,
            tables_output_path=args.tables_output,
            glob_pattern=args.glob,
            recursive=args.recursive,
            limit_files=args.limit_files,
            group_size=args.group_size,
            text_chunk_size=args.text_chunk_size,
            text_overlap=args.text_overlap,
            table_chunk_size=args.table_chunk_size,
            max_table_rows=args.max_table_rows,
            include_toc=args.include_toc,
            exclude_cover=args.exclude_cover,
            stop_on_error=args.stop_on_error,
        )
    elif args.command == "chunk-mixed-slim":
        chunk_mixed_slim(
            input_dir=args.input_dir,
            input_files=args.input_file,
            output_path=args.output,
            errors_path=args.errors_output,
            tables_output_path=args.tables_output,
            glob_pattern=args.glob,
            recursive=args.recursive,
            limit_files=args.limit_files,
            group_size=args.group_size,
            text_chunk_size=args.text_chunk_size,
            text_overlap=args.text_overlap,
            table_chunk_size=args.table_chunk_size,
            max_table_rows=args.max_table_rows,
            include_toc=args.include_toc,
            exclude_cover=args.exclude_cover,
            stop_on_error=args.stop_on_error,
            pdf_backend=args.pdf_backend,
            pdf_no_tables=args.pdf_no_tables,
        )
    elif args.command == "convert-embedding-input":
        convert_embedding_input(input_path=args.input, output_path=args.output, doc_id=args.doc_id)
    elif args.command == "ocr-export-rag":
        export_ocr_rag_handoff(
            ocr_eval_root=args.ocr_eval_root,
            output_manifest=args.output_manifest,
            output_chunks=args.output_chunks,
            engine=args.engine,
            doc_key=args.doc_key,
            include_review_required=not args.exclude_review_required,
            include_html_chunk=args.include_html_chunk,
            html_chunk_max_chars=args.html_chunk_max_chars,
            allow_inference_only=args.allow_inference_only,
            images_tag=args.images_tag,
            curated_root=args.curated_root,
            curated_file_name=args.curated_file_name,
            use_merge_manifest=args.use_merge_manifest,
            curated_only=args.curated_only,
            input_version=args.input_version,
            ocr_engine_version=args.ocr_engine_version,
            ocr_output_version=args.ocr_output_version,
            ocr_curated_version=args.ocr_curated_version,
            rag_index_version=args.rag_index_version,
        )
    elif args.command == "extract-ocr-images":
        extract_ocr_images(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            limit=args.limit,
            source_type=args.source_type,
            recursive=args.recursive,
            pdf_min_width=args.pdf_min_width,
            pdf_min_height=args.pdf_min_height,
            pdf_min_area=args.pdf_min_area,
            pdf_min_bytes=args.pdf_min_bytes,
        )
    elif args.command == "build-pred-structured":
        build_pred_structured(
            gt_path=args.gt,
            pred_raw_path=args.pred_raw,
            item_id=args.id,
            output_path=args.output,
            score_threshold=args.score_threshold,
        )
    elif args.command == "eval-pred-structured":
        eval_pred_structured_vs_gt(
            gt_path=args.gt,
            pred_structured_path=args.pred_structured,
            item_id=args.id,
            output_path=args.output,
            structure_match_threshold=args.structure_threshold,
        )
    elif args.command == "ocr-run-image":
        from src.config_ocr import load_ocr_config

        ocr_app_cfg = load_ocr_config(args.ocr_config)
        ocr_cfg = ocr_app_cfg.ocr
        ocr_paths = ocr_app_cfg.paths
        effective_score_threshold = (
            args.score_threshold if args.score_threshold is not None else ocr_cfg.score_threshold
        )
        effective_structure_threshold = (
            args.structure_threshold
            if args.structure_threshold is not None
            else ocr_cfg.structure_match_threshold
        )
        print(f"[OCR CONFIG] images_root={ocr_paths.images_root}")
        print(f"[OCR CONFIG] gt_root={ocr_paths.gt_root}")
        print(f"[OCR CONFIG] output_root={ocr_paths.output_root}")
        engines = list(DEFAULT_OCR_ENGINE_MATRIX) if args.all_engines else [_normalize_ocr_engine(ocr_cfg.engine)]
        failed_engines: list[str] = []
        for engine_name in engines:
            print(f"\n=== OCR Image Engine: {engine_name} ===")
            try:
                ocr_run_image(
                    doc_key=args.doc_key,
                    item_id=args.id,
                    images_root=ocr_paths.images_root,
                    gt_root=ocr_paths.gt_root,
                    output_root=ocr_paths.output_root,
                    image_name=args.image_name,
                    lang=ocr_cfg.lang or args.lang,
                    score_threshold=effective_score_threshold,
                    structure_match_threshold=effective_structure_threshold,
                    ocr_engine=engine_name,
                    ocr_device=ocr_cfg.device,
                    ocr_batch_size=ocr_cfg.batch_size,
                    use_doc_orientation_classify=args.use_doc_orientation_classify,
                    use_doc_unwarping=args.use_doc_unwarping,
                    use_chart_recognition=args.use_chart_recognition,
                    use_table_orientation_classify=args.use_table_orientation_classify,
                    use_ocr_results_with_table_cells=args.use_ocr_results_with_table_cells,
                    text_det_limit_side_len=args.text_det_limit_side_len,
                    table_dual_pass=args.table_dual_pass,
                    table_second_engine=args.table_second_engine,
                    require_gt=not args.no_gt,
                )
            except Exception as exc:
                failed_engines.append(engine_name)
                print(f"[FAIL_ENGINE] {engine_name}: {exc}")
                if not args.all_engines:
                    raise
        if failed_engines:
            raise SystemExit(f"Failed engines: {', '.join(failed_engines)}")
    elif args.command == "ocr-run-batch":
        from src.config_ocr import load_ocr_config

        ocr_app_cfg = load_ocr_config(args.ocr_config)
        ocr_cfg = ocr_app_cfg.ocr
        ocr_paths = ocr_app_cfg.paths
        effective_score_threshold = (
            args.score_threshold if args.score_threshold is not None else ocr_cfg.score_threshold
        )
        effective_structure_threshold = (
            args.structure_threshold
            if args.structure_threshold is not None
            else ocr_cfg.structure_match_threshold
        )
        print(f"[OCR CONFIG] images_root={ocr_paths.images_root}")
        print(f"[OCR CONFIG] gt_root={ocr_paths.gt_root}")
        print(f"[OCR CONFIG] output_root={ocr_paths.output_root}")
        engines = list(DEFAULT_OCR_ENGINE_MATRIX) if args.all_engines else [_normalize_ocr_engine(ocr_cfg.engine)]
        failed_engines: list[str] = []
        for engine_name in engines:
            print(f"\n=== OCR Batch Engine: {engine_name} ===")
            try:
                ocr_run_batch(
                    images_root=ocr_paths.images_root,
                    gt_root=ocr_paths.gt_root,
                    output_root=ocr_paths.output_root,
                    image_name=args.image_name,
                    doc_key=args.doc_key,
                    limit=args.limit,
                    stop_on_error=args.stop_on_error,
                    lang=ocr_cfg.lang or args.lang,
                    score_threshold=effective_score_threshold,
                    structure_match_threshold=effective_structure_threshold,
                    ocr_engine=engine_name,
                    ocr_device=ocr_cfg.device,
                    ocr_batch_size=ocr_cfg.batch_size,
                    use_doc_orientation_classify=args.use_doc_orientation_classify,
                    use_doc_unwarping=args.use_doc_unwarping,
                    use_chart_recognition=args.use_chart_recognition,
                    use_table_orientation_classify=args.use_table_orientation_classify,
                    use_ocr_results_with_table_cells=args.use_ocr_results_with_table_cells,
                    text_det_limit_side_len=args.text_det_limit_side_len,
                    table_dual_pass=args.table_dual_pass,
                    table_second_engine=args.table_second_engine,
                    require_gt=not args.no_gt,
                    quiet=not args.verbose,
                    show_progress=not args.no_progress,
                )
            except Exception as exc:
                failed_engines.append(engine_name)
                print(f"[FAIL_ENGINE] {engine_name}: {exc}")
                if not args.all_engines:
                    raise
        if failed_engines:
            raise SystemExit(f"Failed engines: {', '.join(failed_engines)}")
    elif args.command == "run-pipeline":
        resolved_index = str(resolve_index_dir(args.config, args.index_dir))
        run_pipeline(
            input_path=args.input,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            index_dir=resolved_index,
            doc_id=args.doc_id,
            model=args.model,
            batch_size=args.batch_size,
            force_real=args.force_real,
            group_size=args.group_size,
            debug_headings=not args.no_debug_headings,
            dump_metadata_sample=args.dump_metadata_sample,
            dump_limit=args.dump_limit,
            prechunk_output=args.prechunk_output,
            chunks_output=args.chunks_output,
            embedded_output=args.embedded_output,
            debug_headings_output=args.debug_headings_output,
            glob_pattern=args.glob,
            recursive=args.recursive,
            limit_files=args.limit_files,
            pdf_backend=args.pdf_backend,
            pdf_no_tables=args.pdf_no_tables,
        )


if __name__ == "__main__":
    main()
