from __future__ import annotations

import argparse
import csv
import json
import re
import sys
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
        result = engine.answer(question)
        rows.append({**item, **result})
    write_jsonl(config.paths.evaluation_output, rows)
    print(f"Wrote evaluation results -> {config.paths.evaluation_output}")


def evaluate_mlflow(
    config_path: str,
    *,
    output_path: str | None,
    judge_model: str,
    no_llm_judge: bool,
    tracking_uri: str | None,
    experiment_name: str,
    run_name: str | None,
    evaluation_set: str | None,
) -> None:
    from dotenv import load_dotenv

    from pathlib import Path

    try:
        from src.evaluation.mlflow_harness import run_eval_harness_mlflow
    except ImportError as exc:
        raise SystemExit(
            "evaluate-mlflow requires MLflow. Install with: pip install -e \".[mlflow]\""
        ) from exc

    load_dotenv()
    out, summary, run_id = run_eval_harness_mlflow(
        config_path,
        evaluation_set=Path(evaluation_set) if evaluation_set else None,
        output_path=Path(output_path) if output_path else None,
        judge_model=judge_model,
        run_llm_judge=not no_llm_judge,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
    )
    print(f"Wrote harness evaluation -> {out}")
    print(f"Summary: {summary}")
    if run_id:
        print(f"MLflow run_id: {run_id}")


def evaluate_harness(
    config_path: str,
    *,
    output_path: str | None,
    judge_model: str,
    no_llm_judge: bool,
    no_langsmith_feedback: bool,
) -> None:
    from src.evaluation.langsmith_harness import run_eval_harness

    out, summary = run_eval_harness(
        config_path,
        output_path=Path(output_path) if output_path else None,
        judge_model=judge_model,
        run_llm_judge=not no_llm_judge,
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
                Path(tables_output_path) if tables_output_path else Path("data/v2/samples") / f"{stem}_tables_markdown.jsonl"
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


def extract_ocr_images(input_dir: str, output_dir: str, limit: int = 0) -> None:
    from pathlib import Path

    from src.Parsing.ocr.inference.extract_hwp_images import extract_images_in_dir

    saved = extract_images_in_dir(Path(input_dir), Path(output_dir), limit=limit)
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
    print(f"latency_ms: {payload.get('latency_ms')}")
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


def _build_paddleocr_vl_model(*, device: str) -> object:
    from paddleocr import PaddleOCRVL

    candidates = [{"device": device}, {}]
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
    ocr_model: object | None = None,
) -> None:
    import time

    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    model = ocr_model if ocr_model is not None else _build_paddleocr_vl_model(device=device)

    start = time.perf_counter()
    results = model.predict(input=str(image), batch_size=batch_size)
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
    table_model: object | None = None,
) -> None:
    import time

    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    model = table_model if table_model is not None else _build_table_recognition_v2_model(lang=lang, device=device)
    start = time.perf_counter()
    results = model.predict(
        input=str(image),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=True,
        use_ocr_model=True,
    )
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
        return _build_paddleocr_vl_model(device=ocr_device)
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


def eval_pred_structured_vs_gt(
    gt_path: str,
    pred_structured_path: str,
    item_id: str,
    output_path: str,
    relaxed_threshold: float = 0.65,
) -> None:
    from difflib import SequenceMatcher
    from pathlib import Path

    from src.Parsing.ocr.evaluation.eval_pred_structured_vs_gt import (
        levenshtein,
        levenshtein_tokens,
        load_item_by_id,
        normalize,
    )

    gt_item = load_item_by_id(Path(gt_path), item_id)
    pred_item = load_item_by_id(Path(pred_structured_path), item_id)
    if "pred_structure" not in pred_item:
        raise ValueError("Pred structured JSON missing key: pred_structure")

    gt_text = str(gt_item.get("gt_text", ""))
    pred_text = str(pred_item.get("pred_text", ""))
    gt_norm = normalize(gt_text)
    pred_norm = normalize(pred_text)
    text_dist = levenshtein(gt_norm, pred_norm)
    text_cer = text_dist / max(1, len(gt_norm))
    char_similarity = max(0.0, 1.0 - (text_dist / max(1, len(gt_norm), len(pred_norm)))) * 100.0
    gt_words = gt_norm.split() if gt_norm else []
    pred_words = pred_norm.split() if pred_norm else []
    word_dist = levenshtein_tokens(gt_words, pred_words)
    text_wer = word_dist / max(1, len(gt_words))

    gt_structure = gt_item.get("gt_structure", {}) if isinstance(gt_item.get("gt_structure", {}), dict) else {}

    def normalize_text_relaxed(text: str) -> str:
        text = str(text).lower()
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def compact_text(text: str) -> str:
        text = normalize_text_relaxed(text)
        return re.sub(r"[^0-9a-z가-힣]+", "", text)

    def flatten_structure(obj: object, path: str = "") -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                next_path = f"{path}.{key}" if path else str(key)
                rows.extend(flatten_structure(value, next_path))
            return rows
        if isinstance(obj, list):
            for value in obj:
                rows.extend(flatten_structure(value, path))
            return rows
        value = str(obj).strip()
        if value:
            rows.append({"field_path": path, "expected_value": value})
        return rows

    def value_match_score(expected: str, pred: str, threshold: float) -> tuple[float, bool]:
        expected_norm = compact_text(expected)
        pred_norm = compact_text(pred)
        if not expected_norm:
            return 0.0, False
        if expected_norm in pred_norm:
            return 1.0, True
        score = SequenceMatcher(None, expected_norm, pred_norm).ratio()
        return score, score >= threshold

    expected_fields = flatten_structure(gt_structure)
    matched_count = 0
    for field in expected_fields:
        score, matched = value_match_score(field["expected_value"], pred_text, relaxed_threshold)
        if matched:
            matched_count += 1

    field_hit_rate = matched_count / len(expected_fields) if expected_fields else 0.0

    eval_payload = {
        "id": item_id,
        "type": pred_item.get("type", gt_item.get("image_type", "unknown")),
        "status": pred_item.get("status", "success"),
        "engine": pred_item.get("model", "unknown"),
        "latency_ms": pred_item.get("latency_ms"),
        "text_metrics": {
            "exact_match": gt_norm == pred_norm,
            "cer": text_cer,
            "wer": text_wer,
            "char_similarity_pct": round(char_similarity, 2),
        },
        "field_match": {
            "rate_pct": round(field_hit_rate * 100.0, 2),
            "matched": matched_count,
            "total": len(expected_fields),
            "threshold": relaxed_threshold,
        },
        "macro_f1": field_hit_rate,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(eval_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(eval_payload, ensure_ascii=False, indent=2))
    print(f"saved: {output}")

def ocr_run_all(
    image_path: str,
    gt_path: str,
    item_id: str,
    pred_raw_output: str,
    pred_structured_output: str,
    eval_output: str,
    lang: str = "korean",
    score_threshold: float = 0.0,
    ocr_engine: str = "pp_ocrv5",
    ocr_device: str = "gpu:0",
    ocr_batch_size: int = 1,
    ocr_model: object | None = None,
    relaxed_threshold: float = 0.65,
) -> None:
    normalized_engine = _normalize_ocr_engine(ocr_engine)
    print("[1/3] run-ocr")
    if normalized_engine == "pp_structurev3":
        run_pp_structurev3_ocr(
            image_path=image_path,
            output_path=pred_raw_output,
            lang=lang,
            device=ocr_device,
            structure_model=ocr_model,
        )
    elif normalized_engine == "paddleocr_vl":
        run_paddleocr_vl_ocr(
            image_path=image_path,
            output_path=pred_raw_output,
            device=ocr_device,
            batch_size=ocr_batch_size,
            ocr_model=ocr_model,
        )
    elif normalized_engine == "table_recognition_v2":
        run_table_recognition_v2_ocr(
            image_path=image_path,
            output_path=pred_raw_output,
            lang=lang,
            device=ocr_device,
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
    print("[2/3] build-pred-structured")
    build_pred_structured(
        gt_path=gt_path,
        pred_raw_path=pred_raw_output,
        item_id=item_id,
        output_path=pred_structured_output,
        score_threshold=score_threshold,
    )
    print("[3/3] eval-pred-structured")
    eval_pred_structured_vs_gt(
        gt_path=gt_path,
        pred_structured_path=pred_structured_output,
        item_id=item_id,
        output_path=eval_output,
        relaxed_threshold=relaxed_threshold,
    )
    print("ocr_run_all_done")


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


def _resolve_ocr_doc_paths(
    *,
    doc_key: str,
    images_root: str,
    gt_root: str,
    output_root: str,
    image_name: str,
    ocr_engine: str,
) -> tuple[Path, Path, Path, Path, Path]:
    normalized_engine = _normalize_ocr_engine(ocr_engine)
    image_path = Path(images_root) / doc_key / image_name
    gt_path = _resolve_gt_path(gt_root, doc_key)
    engine_dir_name = _engine_dir_name(normalized_engine)
    image_stem = Path(image_name).stem
    out_dir = Path(output_root) / engine_dir_name / doc_key / image_stem
    pred_raw = out_dir / "pred_raw.json"
    pred_structured = out_dir / "pred_structured.json"
    eval_json = out_dir / "eval.json"
    return image_path, gt_path, pred_raw, pred_structured, eval_json


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
    ocr_engine: str,
    ocr_device: str,
    ocr_batch_size: int,
    ocr_model: object | None = None,
) -> bool:
    image_path, gt_path, pred_raw, pred_structured, eval_json = _resolve_ocr_doc_paths(
        doc_key=doc_key,
        images_root=images_root,
        gt_root=gt_root,
        output_root=output_root,
        image_name=image_name,
        ocr_engine=ocr_engine,
    )
    image_path = _resolve_image_path(Path(images_root) / doc_key, image_name)
    if not gt_path.exists():
        raise FileNotFoundError(f"GT not found: {gt_path}")

    final_item_id = item_id or _infer_item_id_from_gt(gt_path, image_path.name)
    gt_payload = _load_gt_payload(gt_path)
    gt_records = gt_payload if isinstance(gt_payload, list) else [gt_payload]
    gt_item = next(
        (record for record in gt_records if isinstance(record, dict) and record.get("id") == final_item_id),
        {},
    )
    pred_raw.parent.mkdir(parents=True, exist_ok=True)
    print(f"doc_key: {doc_key}")
    print(f"image: {image_path}")
    print(f"gt: {gt_path}")
    print(f"id: {final_item_id}")
    use_eval = bool(gt_item.get("use_eval", True))
    ocr_run_all(
        image_path=str(image_path),
        gt_path=str(gt_path),
        item_id=final_item_id,
        pred_raw_output=str(pred_raw),
        pred_structured_output=str(pred_structured),
        eval_output=str(eval_json),
        lang=lang,
        score_threshold=score_threshold,
        ocr_engine=ocr_engine,
        ocr_device=ocr_device,
        ocr_batch_size=ocr_batch_size,
        ocr_model=ocr_model,
    )
    if not use_eval:
        print(f"[SKIP_EVAL_SUMMARY] use_eval=false: {final_item_id}")
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
    ocr_engine: str,
    ocr_device: str,
    ocr_batch_size: int,
) -> None:
    normalized_engine = _normalize_ocr_engine(ocr_engine)
    shared_model = _build_shared_ocr_model(
        ocr_engine=normalized_engine,
        lang=lang,
        ocr_device=ocr_device,
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
    eval_rows: list[dict[str, object]] = []

    for doc_dir in doc_dirs:
        doc_key = doc_dir.name
        print(f"\n=== OCR Batch: {doc_key} ===")
        try:
            doc_image_paths = _list_image_paths(doc_dir)
            print(f"images_in_doc: {len(doc_image_paths)}")
            for path in doc_image_paths:
                include_in_eval = ocr_run_image(
                    doc_key=doc_key,
                    item_id=None,
                    images_root=images_root,
                    gt_root=gt_root,
                    output_root=output_root,
                    image_name=path.name,
                    lang=lang,
                    score_threshold=score_threshold,
                    ocr_engine=ocr_engine,
                    ocr_device=ocr_device,
                    ocr_batch_size=ocr_batch_size,
                    ocr_model=shared_model,
                )
                ok_count += 1
                if not include_in_eval:
                    continue

                eval_path = (
                    Path(output_root) / _engine_dir_name(normalized_engine) / doc_key / Path(path.name).stem / "eval.json"
                )
                if eval_path.exists():
                    result = json.loads(eval_path.read_text(encoding="utf-8"))
                    text_metrics = result.get("text_metrics", {}) if isinstance(result, dict) else {}
                    field_metrics = result.get("field_match", {}) if isinstance(result, dict) else {}
                    char_similarity_pct = text_metrics.get("char_similarity_pct")
                    similarity_ratio = (
                        float(char_similarity_pct) / 100.0 if char_similarity_pct is not None else None
                    )
                    row = {
                        "id": result.get("id"),
                        "doc_key": doc_key,
                        "type": result.get("type"),
                        "status": result.get("status"),
                        "text_similarity": similarity_ratio,
                        "cer": text_metrics.get("cer"),
                        "wer": text_metrics.get("wer"),
                        "field_hit_rate_pct": field_metrics.get("rate_pct"),
                        "matched_fields": field_metrics.get("matched"),
                        "total_fields": field_metrics.get("total"),
                        "latency_ms": result.get("latency_ms"),
                    }
                    eval_rows.append(row)

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
                    if row.get("field_hit_rate_pct") is not None:
                        print(
                            "필드값 매칭률: "
                            f"{round(float(row['field_hit_rate_pct']), 2)} % "
                            f"({row.get('matched_fields')}/{row.get('total_fields')})"
                        )
                    print(f"latency_ms: {row['latency_ms']}")
        except FileNotFoundError as e:
            message = str(e)
            if "Image folder not found" in message or "No image files found" in message or "Image not found" in message:
                print(f"[SKIP][이미지 없음] {message}")
                skip_image_count += 1
            elif "GT not found" in message:
                print(f"[SKIP][GT 없음] {message}")
                skip_gt_count += 1
            else:
                print(f"[SKIP] {message}")
            skip_count += 1
        except Exception as e:
            print(f"[FAIL] {doc_key}: {e}")
            fail_count += 1
            if stop_on_error:
                raise

    print("\n=== OCR Batch Summary ===")
    print(f"total_docs: {len(doc_dirs)}")
    print(f"ok: {ok_count}")
    print(f"skip: {skip_count}")
    print(f"skip_image_missing: {skip_image_count}")
    print(f"skip_gt_missing: {skip_gt_count}")
    print(f"fail: {fail_count}")

    if eval_rows:
        engine_out_root = Path(output_root) / _engine_dir_name(normalized_engine)
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
                    "field_hit_rate_pct",
                    "matched_fields",
                    "total_fields",
                    "latency_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(eval_rows)

        summary_json_path = engine_out_root / "ocr_eval_summary.json"
        summary_json_path.write_text(json.dumps(eval_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        avg_similarity = sum(float(row["text_similarity"]) for row in eval_rows if row.get("text_similarity") is not None)
        avg_similarity /= max(1, len([row for row in eval_rows if row.get("text_similarity") is not None]))

        field_rates = [float(row["field_hit_rate_pct"]) for row in eval_rows if row.get("field_hit_rate_pct") is not None]
        avg_field_hit_rate = sum(field_rates) / len(field_rates) if field_rates else 0.0
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
            if row.get("field_hit_rate_pct") is not None:
                lines.append(
                    "필드값 매칭률: "
                    f"{round(float(row['field_hit_rate_pct']), 2)} % "
                    f"({row.get('matched_fields')}/{row.get('total_fields')})"
                )
            lines.append(f"latency_ms: {row.get('latency_ms')}")

        lines.append("")
        lines.append("[SUMMARY]")
        lines.append(f"평가 이미지 수: {len(eval_rows)}")
        lines.append(f"평균 문자 유사도: {round(avg_similarity * 100.0, 2)} %")
        lines.append(f"평균 필드값 매칭률: {round(avg_field_hit_rate, 2)} %")
        lines.append(f"실패율: {round(fail_rate, 2)} %")
        summary_txt_path.write_text("\n".join(lines), encoding="utf-8")

        print("\n[SUMMARY]")
        print(f"평가 이미지 수: {len(eval_rows)}")
        print(f"평균 문자 유사도: {round(avg_similarity * 100.0, 2)} %")
        print(f"평균 필드값 매칭률: {round(avg_field_hit_rate, 2)} %")
        print(f"실패율: {round(fail_rate, 2)} %")
        print(f"saved_eval_summary_csv: {summary_csv_path}")
        print(f"saved_eval_summary_json: {summary_json_path}")
        print(f"saved_eval_summary_txt: {summary_txt_path}")


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
    print("[1/4] parse-hwp")
    parse_hwp(
        str(prechunk),
        input_path=str(input_file),
        debug_headings=str(heading_debug) if debug_headings and heading_debug else None,
        group_size=group_size,
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
) -> None:
    _require_exactly_one(
        a=input_path,
        b=input_dir,
        a_name="--input (single HWP)",
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
        )
        return

    input_files = _discover_hwp_in_dir(input_dir)  # type: ignore[arg-type]

    resolved_index = index_dir
    for index, input_file in enumerate(input_files, start=1):
        print(f"\n=== [{index}/{len(input_files)}] {input_file.name} ===")
        _run_pipeline_for_file(
            input_file,
            output_dir=output_dir,
            index_dir=resolved_index,
            doc_id=doc_id,
            model=model,
            batch_size=batch_size,
            force_real=force_real,
            group_size=group_size,
            debug_headings=debug_headings,
            dump_metadata_sample=dump_metadata_sample and index == len(input_files),
            dump_limit=dump_limit,
        )
    print(f"\nall_done: {len(input_files)} file(s)")


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
        "--judge-model",
        default="gpt-4o",
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

    mlflow_parser = subparsers.add_parser(
        "evaluate-mlflow",
        help="Same as evaluate-harness but logs to MLflow (requires pip install -e '.[mlflow]')",
    )
    mlflow_parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: outputs/eval_harness_results.jsonl)",
    )
    mlflow_parser.add_argument(
        "--evaluation-set",
        default=None,
        help="Evaluation questions JSONL (default: config paths.evaluation_set)",
    )
    mlflow_parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="OpenAI chat model for faithfulness/relevance scoring",
    )
    mlflow_parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip GPT judge (retrieval keyword metrics only if keywords are present)",
    )
    mlflow_parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (overrides MLFLOW_TRACKING_URI env; default: local ./mlruns)",
    )
    mlflow_parser.add_argument(
        "--experiment-name",
        default="bidmate-rag-eval",
        help="MLflow experiment name",
    )
    mlflow_parser.add_argument(
        "--run-name",
        default=None,
        help="MLflow run name (default: eval_YYYYMMDD_HHMMSS UTC)",
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
        help="Optional table-only JSONL path with Markdown tables; defaults to <output_stem>_tables.jsonl",
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

    convert_parser = subparsers.add_parser(
        "convert-embedding-input",
        help="Convert prechunk/chunk JSONL to embedding input JSONL",
    )
    convert_parser.add_argument("--input", required=True, help="Input JSONL path")
    convert_parser.add_argument("--output", required=True, help="Embedding input JSONL output path")
    convert_parser.add_argument("--doc-id", default=None, help="Optional doc id override")

    ocr_parser = subparsers.add_parser(
        "extract-ocr-images",
        help="Extract embedded images from HWP files for OCR ground-truth preparation",
    )
    ocr_parser.add_argument("--input-dir", required=True, help="Directory containing HWP files")
    ocr_parser.add_argument("--output-dir", required=True, help="Directory to save extracted images")
    ocr_parser.add_argument("--limit", type=int, default=0, help="Process first N files only; 0=all")

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

    ocr_image_parser = subparsers.add_parser(
        "ocr-run-image",
        help="Run OCR/eval for one image with doc_key-based path resolution",
    )
    ocr_image_parser.add_argument("--doc-key", required=True, help="Document key (folder name under ocr_images)")
    ocr_image_parser.add_argument(
        "--id",
        default=None,
        help="Target item id. If omitted, auto-detected from GT JSON when unique.",
    )
    ocr_image_parser.add_argument("--images-root", default="data/v2/ocr_images", help="Root folder of OCR images")
    ocr_image_parser.add_argument(
        "--gt-root",
        default="data/v2/ocr_eval/incoming_gt",
        help="Root folder of GT files (.jsonl preferred, .json supported)",
    )
    ocr_image_parser.add_argument("--output-root", default="data/v2/ocr_eval", help="Root folder of OCR outputs")
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
        default=0.0,
        help="Minimum OCR confidence threshold for structured prediction",
    )
    ocr_image_parser.add_argument(
        "--all-engines",
        action="store_true",
        help="Run default OCR engine matrix (pp_ocrv5, pp_ocrv5_transformers, pp_structurev3, table_recognition_v2, paddleocr_vl)",
    )

    ocr_batch_parser = subparsers.add_parser(
        "ocr-run-batch",
        help="Run OCR/eval for all doc folders under ocr_images",
    )
    ocr_batch_parser.add_argument("--images-root", default="data/v2/ocr_images", help="Root folder of OCR images")
    ocr_batch_parser.add_argument(
        "--gt-root",
        default="data/v2/ocr_eval/incoming_gt",
        help="Root folder of GT files (.jsonl preferred, .json supported)",
    )
    ocr_batch_parser.add_argument("--output-root", default="data/v2/ocr_eval", help="Root folder of OCR outputs")
    ocr_batch_parser.add_argument(
        "--doc-key",
        default=None,
        help="Run batch only for one document key folder under images-root",
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
        default=0.0,
        help="Minimum OCR confidence threshold for structured prediction",
    )
    ocr_batch_parser.add_argument(
        "--all-engines",
        action="store_true",
        help="Run default OCR engine matrix (pp_ocrv5, pp_ocrv5_transformers, pp_structurev3, table_recognition_v2, paddleocr_vl)",
    )

    pipeline_parser = subparsers.add_parser(
        "run-pipeline",
        help="Run parse->chunk->embed->build-chroma in one command",
    )
    pipeline_input = pipeline_parser.add_mutually_exclusive_group(required=True)
    pipeline_input.add_argument("--input", help="Single HWP file path")
    pipeline_input.add_argument(
        "--input-dir",
        help="Folder of HWP files (*.hwp in that folder only; one pipeline per file)",
    )
    pipeline_parser.add_argument(
        "--output-dir",
        default="data/v2",
        help="Base output directory when --*-output paths are omitted (per-file subfolder)",
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
            judge_model=args.judge_model,
            no_llm_judge=args.no_llm_judge,
            no_langsmith_feedback=args.no_langsmith_feedback,
        )
    elif args.command == "evaluate-mlflow":
        evaluate_mlflow(
            args.config,
            output_path=args.output,
            judge_model=args.judge_model,
            no_llm_judge=args.no_llm_judge,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            evaluation_set=args.evaluation_set,
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
    elif args.command == "convert-embedding-input":
        convert_embedding_input(input_path=args.input, output_path=args.output, doc_id=args.doc_id)
    elif args.command == "extract-ocr-images":
        extract_ocr_images(input_dir=args.input_dir, output_dir=args.output_dir, limit=args.limit)
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
        )
    elif args.command == "ocr-run-image":
        from src.config_ocr import load_ocr_config

        ocr_cfg = load_ocr_config(args.ocr_config).ocr
        engines = list(DEFAULT_OCR_ENGINE_MATRIX) if args.all_engines else [_normalize_ocr_engine(ocr_cfg.engine)]
        failed_engines: list[str] = []
        for engine_name in engines:
            print(f"\n=== OCR Image Engine: {engine_name} ===")
            try:
                ocr_run_image(
                    doc_key=args.doc_key,
                    item_id=args.id,
                    images_root=args.images_root,
                    gt_root=args.gt_root,
                    output_root=args.output_root,
                    image_name=args.image_name,
                    lang=ocr_cfg.lang or args.lang,
                    score_threshold=ocr_cfg.score_threshold if ocr_cfg.score_threshold is not None else args.score_threshold,
                    ocr_engine=engine_name,
                    ocr_device=ocr_cfg.device,
                    ocr_batch_size=ocr_cfg.batch_size,
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

        ocr_cfg = load_ocr_config(args.ocr_config).ocr
        engines = list(DEFAULT_OCR_ENGINE_MATRIX) if args.all_engines else [_normalize_ocr_engine(ocr_cfg.engine)]
        failed_engines: list[str] = []
        for engine_name in engines:
            print(f"\n=== OCR Batch Engine: {engine_name} ===")
            try:
                ocr_run_batch(
                    images_root=args.images_root,
                    gt_root=args.gt_root,
                    output_root=args.output_root,
                    image_name=args.image_name,
                    doc_key=args.doc_key,
                    limit=args.limit,
                    stop_on_error=args.stop_on_error,
                    lang=ocr_cfg.lang or args.lang,
                    score_threshold=ocr_cfg.score_threshold if ocr_cfg.score_threshold is not None else args.score_threshold,
                    ocr_engine=engine_name,
                    ocr_device=ocr_cfg.device,
                    ocr_batch_size=ocr_cfg.batch_size,
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
        )


if __name__ == "__main__":
    main()
