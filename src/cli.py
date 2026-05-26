from __future__ import annotations

import argparse
import json
import re
import sys


def ingest(config_path: str) -> None:
    from dotenv import load_dotenv
    from tqdm import tqdm

    from src.config import load_config
    from src.dataset.loaders import load_documents
    from src.engine.vector_store import ChromaVectorStore
    from src.models.openai_client import OpenAIModelClient
    from src.preprocessing.chunker import chunk_documents

    load_dotenv()
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
    from dotenv import load_dotenv

    from src.config import load_config
    from src.engine.rag import RagEngine
    from src.engine.vector_store import ChromaVectorStore

    load_dotenv()
    config = load_config(config_path)
    store = ChromaVectorStore.load(config.paths.index_dir)
    engine = RagEngine(config, store)
    result = engine.answer(question)
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source['chunk_id']} score={source['score']:.4f}")


def evaluate(config_path: str) -> None:
    from dotenv import load_dotenv
    from tqdm import tqdm

    from src.config import load_config
    from src.engine.rag import RagEngine
    from src.engine.vector_store import ChromaVectorStore
    from src.utils.jsonl import read_jsonl, write_jsonl

    load_dotenv()
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


def evaluate_harness(
    config_path: str,
    *,
    output_path: str | None,
    judge_model: str,
    no_llm_judge: bool,
    no_langsmith_feedback: bool,
) -> None:
    from dotenv import load_dotenv

    from pathlib import Path

    from src.evaluation.langsmith_harness import run_eval_harness

    load_dotenv()
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
    from pathlib import Path

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
    from pathlib import Path

    if index_dir:
        return Path(index_dir)

    from src.config import load_config

    return load_config(config_path).paths.index_dir


def build_chroma_index(input_path: str, index_dir: str, doc_id: str | None = None) -> None:
    from pathlib import Path

    from src.pipeline.embedding_pipeline import build_chroma_from_embedded_jsonl

    count = build_chroma_from_embedded_jsonl(
        input_path=Path(input_path),
        index_dir=Path(index_dir),
        doc_id=doc_id,
    )
    print(f"Built Chroma index with {count} chunks -> {index_dir}")


def parse_hwp(
    input_path: str,
    output_path: str,
    debug_headings: str | None = None,
    limit: int = 0,
    group_size: int = 8,
) -> None:
    from pathlib import Path

    from src.Parsing.parsing import build_prechunk_records, write_jsonl

    debug_path = Path(debug_headings) if debug_headings else None
    records = build_prechunk_records(Path(input_path), group_size=group_size, debug_headings_path=debug_path)
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
    from pathlib import Path

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


def convert_embedding_input(input_path: str, output_path: str, doc_id: str | None = None) -> None:
    from pathlib import Path

    from src.Parsing.convert_prechunk_to_embedding_input import convert

    count = convert(Path(input_path), Path(output_path), doc_id)
    print(f"Converted {count} rows -> {output_path}")


def run_pipeline(
    input_path: str,
    output_dir: str,
    index_dir: str | None,
    doc_id: str | None,
    model: str,
    batch_size: int,
    force_real: bool,
    group_size: int = 8,
    debug_headings: bool = True,
    dump_metadata_sample: bool = False,
    dump_limit: int = 20,
) -> None:
    from pathlib import Path
    import chromadb

    input_file = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = input_file.stem
    safe_stem = re.sub(r"[\\/:*?\"<>|&\s]+", "_", stem).strip("._")
    safe_stem = re.sub(r"_+", "_", safe_stem) or "document"
    doc_dir = out_dir / safe_stem
    doc_dir.mkdir(parents=True, exist_ok=True)

    prechunk = doc_dir / f"{stem}_prechunk.jsonl"
    heading_debug = doc_dir / f"{stem}_heading_debug.jsonl"
    chunks = doc_dir / f"{stem}_chunks.jsonl"
    embedded = doc_dir / f"{stem}_embedded.jsonl"
    metadata_sample = doc_dir / f"{stem}_chroma_metadata_sample.json"
    resolved_index_dir = index_dir or str(doc_dir / "chroma_index")

    print("[1/4] parse-hwp")
    parse_hwp(
        input_path=str(input_file),
        output_path=str(prechunk),
        debug_headings=str(heading_debug) if debug_headings else None,
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
    print(f"doc_output_dir: {doc_dir}")
    print(f"embedded_output: {embedded}")
    print(f"index_dir: {resolved_index_dir}")
    if dump_metadata_sample:
        print(f"metadata_sample: {metadata_sample}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bidmate RAG scenario B baseline")
    parser.add_argument("--config", default="configs/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Load RFP files, chunk, embed, and build Chroma index")

    query_parser = subparsers.add_parser("query", help="Ask a question against the built index")
    query_parser.add_argument("question")

    subparsers.add_parser("evaluate", help="Run questions from the evaluation JSONL file")

    harness_parser = subparsers.add_parser(
        "evaluate-harness",
        help="RAG eval with optional keyword retrieval hit, LLM-as-judge, LangSmith traces/feedback",
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

    check_parser = subparsers.add_parser(
        "check-setup",
        help="Validate local settings, dependencies, paths, and optional OpenAI connectivity",
    )
    check_parser.add_argument(
        "--check-openai",
        action="store_true",
        help="Make a real OpenAI API request to verify the key and network connection",
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
    parse_parser.add_argument("--input", required=True, help="HWP input path")
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

    convert_parser = subparsers.add_parser(
        "convert-embedding-input",
        help="Convert prechunk/chunk JSONL to embedding input JSONL",
    )
    convert_parser.add_argument("--input", required=True, help="Input JSONL path")
    convert_parser.add_argument("--output", required=True, help="Embedding input JSONL output path")
    convert_parser.add_argument("--doc-id", default=None, help="Optional doc id override")

    pipeline_parser = subparsers.add_parser(
        "run-pipeline",
        help="Run parse->chunk->embed->build-chroma in one command",
    )
    pipeline_parser.add_argument("--input", required=True, help="HWP input path")
    pipeline_parser.add_argument("--output-dir", default="data/v2", help="Output directory for pipeline artifacts")
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
    elif args.command == "check-setup":
        from src.utils.setup_check import run_setup_check

        sys.exit(run_setup_check(args.config, check_openai=args.check_openai))
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
            input_path=args.input,
            output_path=args.output,
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
    elif args.command == "run-pipeline":
        run_pipeline(
            input_path=args.input,
            output_dir=args.output_dir,
            index_dir=str(resolve_index_dir(args.config, args.index_dir)) if args.index_dir else None,
            doc_id=args.doc_id,
            model=args.model,
            batch_size=args.batch_size,
            force_real=args.force_real,
            group_size=args.group_size,
            debug_headings=not args.no_debug_headings,
            dump_metadata_sample=args.dump_metadata_sample,
            dump_limit=args.dump_limit,
        )


if __name__ == "__main__":
    main()
