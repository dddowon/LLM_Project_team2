from __future__ import annotations

import argparse
import sys


def ingest(config_path: str) -> None:
    from dotenv import load_dotenv
    from tqdm import tqdm

    from src.config import load_config
    from src.dataset.loaders import load_documents
    from src.engine.vector_store import FaissVectorStore
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
    store = FaissVectorStore.build(chunks, embeddings)
    store.save(config.paths.index_dir)
    print(f"Indexed {len(documents)} documents / {len(chunks)} chunks -> {config.paths.index_dir}")


def query(config_path: str, question: str) -> None:
    from dotenv import load_dotenv

    from src.config import load_config
    from src.engine.rag import RagEngine
    from src.engine.vector_store import FaissVectorStore

    load_dotenv()
    config = load_config(config_path)
    store = FaissVectorStore.load(config.paths.index_dir)
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
    from src.engine.vector_store import FaissVectorStore
    from src.utils.jsonl import read_jsonl, write_jsonl

    load_dotenv()
    config = load_config(config_path)
    questions = read_jsonl(config.paths.evaluation_set)
    if not questions:
        raise RuntimeError(f"평가 질문셋이 없습니다: {config.paths.evaluation_set}")

    store = FaissVectorStore.load(config.paths.index_dir)
    engine = RagEngine(config, store)
    rows = []
    for item in tqdm(questions, desc="Evaluating"):
        question = item["question"]
        result = engine.answer(question)
        rows.append({**item, **result})
    write_jsonl(config.paths.evaluation_output, rows)
    print(f"Wrote evaluation results -> {config.paths.evaluation_output}")

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


def build_faiss_index(input_path: str, index_dir: str, doc_id: str | None = None) -> None:
    from pathlib import Path

    from src.pipeline.embedding_pipeline import build_faiss_from_embedded_jsonl

    count = build_faiss_from_embedded_jsonl(
        input_path=Path(input_path),
        index_dir=Path(index_dir),
        doc_id=doc_id,
    )
    print(f"Built FAISS index with {count} chunks -> {index_dir}")


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


def convert_embedding_input(input_path: str, output_path: str, doc_id: str | None = None) -> None:
    from pathlib import Path

    from src.Parsing.convert_prechunk_to_embedding_input import convert

    count = convert(Path(input_path), Path(output_path), doc_id)
    print(f"Converted {count} rows -> {output_path}")


def run_pipeline(
    input_path: str,
    output_dir: str,
    index_dir: str,
    doc_id: str | None,
    model: str,
    batch_size: int,
    force_real: bool,
    group_size: int = 8,
    debug_headings: bool = True,
) -> None:
    from pathlib import Path

    input_file = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = input_file.stem
    prechunk = out_dir / f"{stem}_prechunk.jsonl"
    heading_debug = out_dir / f"{stem}_heading_debug.jsonl"
    chunks = out_dir / f"{stem}_chunks.jsonl"
    embedded = out_dir / f"{stem}_embedded.jsonl"

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
    print("[4/4] build-faiss")
    build_faiss_index(input_path=str(embedded), index_dir=index_dir, doc_id=doc_id)
    print("pipeline_done")
    print(f"embedded_output: {embedded}")
    print(f"index_dir: {index_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bidmate RAG scenario B baseline")
    parser.add_argument("--config", default="configs/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Load RFP files, chunk, embed, and build FAISS index")

    query_parser = subparsers.add_parser("query", help="Ask a question against the built index")
    query_parser.add_argument("question")

    subparsers.add_parser("evaluate", help="Run questions from the evaluation JSONL file")

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

    faiss_parser = subparsers.add_parser("build-faiss", help="Build FAISS checkpoint from embedded JSONL")
    faiss_parser.add_argument("--input", required=True, help="Embedded JSONL input path")
    faiss_parser.add_argument(
        "--index-dir",
        default=None,
        help="FAISS checkpoint output directory; defaults to config paths.index_dir",
    )
    faiss_parser.add_argument("--doc-id", default=None, help="Optional doc id override")

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

    convert_parser = subparsers.add_parser(
        "convert-embedding-input",
        help="Convert prechunk/chunk JSONL to embedding input JSONL",
    )
    convert_parser.add_argument("--input", required=True, help="Input JSONL path")
    convert_parser.add_argument("--output", required=True, help="Embedding input JSONL output path")
    convert_parser.add_argument("--doc-id", default=None, help="Optional doc id override")

    pipeline_parser = subparsers.add_parser(
        "run-pipeline",
        help="Run parse->chunk->embed->build-faiss in one command",
    )
    pipeline_parser.add_argument("--input", required=True, help="HWP input path")
    pipeline_parser.add_argument("--output-dir", default="data/v2", help="Output directory for pipeline artifacts")
    pipeline_parser.add_argument(
        "--index-dir",
        default=None,
        help="FAISS checkpoint output directory; defaults to config paths.index_dir",
    )
    pipeline_parser.add_argument("--doc-id", default=None, help="Optional doc id override")
    pipeline_parser.add_argument("--model", default="text-embedding-3-small", help="Embedding model")
    pipeline_parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    pipeline_parser.add_argument("--force-real", action="store_true", help="Fail if OPENAI_API_KEY is missing")
    pipeline_parser.add_argument("--group-size", type=int, default=8, help="Table row group size for parser")
    pipeline_parser.add_argument("--no-debug-headings", action="store_true", help="Skip heading debug JSONL")

    args = parser.parse_args()
    if args.command == "ingest":
        ingest(args.config)
    elif args.command == "query":
        query(args.config, args.question)
    elif args.command == "evaluate":
        evaluate(args.config)
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
    elif args.command == "build-faiss":
        build_faiss_index(
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
    elif args.command == "convert-embedding-input":
        convert_embedding_input(input_path=args.input, output_path=args.output, doc_id=args.doc_id)
    elif args.command == "run-pipeline":
        run_pipeline(
            input_path=args.input,
            output_dir=args.output_dir,
            index_dir=str(resolve_index_dir(args.config, args.index_dir)),
            doc_id=args.doc_id,
            model=args.model,
            batch_size=args.batch_size,
            force_real=args.force_real,
            group_size=args.group_size,
            debug_headings=not args.no_debug_headings,
        )


if __name__ == "__main__":
    main()
