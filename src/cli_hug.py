from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset.schema import Chunk
from src.pipeline.embedding_pipeline import build_chroma_from_embedded_jsonl
from src.pipeline.merge_embedded import build_unified_chroma_index
from src.utils.jsonl import read_jsonl, write_jsonl


DEFAULT_EMBEDDING_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
DEFAULT_INDEX_DIR = "checkpoints/chroma_hug"

EMBEDDING_MODEL_ALIASES = {
    "snowflake-ko": "dragonkue/snowflake-arctic-embed-l-v2.0-ko",
    "kure": "nlpai-lab/KURE-v1",
    "bge-m3": "BAAI/bge-m3",
}

LLM_MODEL_ALIASES = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen": "Qwen/Qwen3-8B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-2-9b": "google/gemma-2-9b-it",
    "gemma": "google/gemma-2-9b-it",
    "exaone": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "exaone-3.5-7.8b": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
}

SYSTEM_POLICY = """당신은 공공입찰 RFP 분석을 돕는 입찰메이트 사내 RAG 어시스턴트입니다.

[근거]
- 반드시 아래 "문서 컨텍스트"에 있는 내용만 근거로 답하세요.
- 추측이나 일반 상식으로 내용을 채우지 마세요.
- 컨텍스트에 없는 항목은 짧게 "문서에서 확인되지 않습니다"라고 답하세요.
- 질문 범위를 벗어난 항목을 임의로 추가하지 마세요.

[문체]
- 질문에 바로 답하세요.
- 단순 사실 질문은 핵심만 1~3문장으로 답하세요.
- 요약, 비교, 목록 질문은 불릿으로 구조화하세요.
- source_id나 chunk_id는 답변 본문에 붙이지 마세요."""


@dataclass(frozen=True)
class QueryResult:
    chunk: Chunk
    score: float


def resolve_embedding_model(model: str) -> str:
    return EMBEDDING_MODEL_ALIASES.get(model.strip().lower(), model)


def resolve_llm_model(model: str | None) -> str | None:
    if not model:
        return None
    return LLM_MODEL_ALIASES.get(model.strip().lower(), model)


def require_exactly_one(*, a: Any, b: Any, a_name: str, b_name: str) -> None:
    if bool(a) == bool(b):
        raise SystemExit(f"Provide exactly one of {a_name} or {b_name}.")


def safe_output_stem(value: str) -> str:
    stem = re.sub(r"[\\/:*?\"<>|&\s]+", "_", value).strip("._")
    return re.sub(r"_+", "_", stem) or "document"


def extract_chunk_text(row: dict[str, Any]) -> str:
    text = str(row.get("chunk_text", "")).strip()
    if text:
        return text
    return str(row.get("text", "")).strip()


def normalize_metadata(metadata: Any) -> dict[str, Any]:
    return dict(metadata) if isinstance(metadata, dict) else {}


class HuggingFaceEmbeddingClient:
    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        trust_remote_code: bool = False,
        normalize_embeddings: bool = True,
        passage_prefix: str = "",
        query_prefix: str = "",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers가 설치되어 있지 않습니다. "
                'python -m pip install -e ".[huggingface]" 를 먼저 실행하세요.'
            ) from exc

        self.model_name = resolve_embedding_model(model_name)
        self.normalize_embeddings = normalize_embeddings
        self.passage_prefix = passage_prefix
        self.query_prefix = query_prefix
        kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if device:
            kwargs["device"] = device
        self.model = SentenceTransformer(self.model_name, **kwargs)

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return self._encode(texts, prefix=self.passage_prefix, batch_size=batch_size)

    def embed_queries(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return self._encode(texts, prefix=self.query_prefix, batch_size=batch_size)

    def _encode(self, texts: list[str], *, prefix: str, batch_size: int) -> list[list[float]]:
        prepared = [f"{prefix}{text}" if prefix else text for text in texts]
        vectors = self.model.encode(
            prepared,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )
        return [[float(value) for value in vector] for vector in vectors]


class HuggingFaceLLMClient:
    def __init__(
        self,
        model_name: str,
        *,
        device_map: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers/torch가 설치되어 있지 않습니다. "
                'python -m pip install -e ".[huggingface]" 를 먼저 실행하세요.'
            ) from exc

        self.model_name = resolve_llm_model(model_name) or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
        )

        torch_dtype: Any
        if dtype == "auto":
            torch_dtype = "auto"
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            raise ValueError("--dtype must be one of: auto, float16, bfloat16, float32")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    def generate(
        self,
        *,
        question: str,
        context: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        strip_thinking: bool = True,
    ) -> str:
        import torch

        user_content = (
            "문서 컨텍스트:\n"
            f"{context or '(검색된 컨텍스트 없음)'}\n\n"
            "사용자 질문:\n"
            f"{question}\n\n"
            "위 지침을 따르고, 문서 컨텍스트에 근거해서만 답하세요."
        )
        messages = [
            {"role": "system", "content": SYSTEM_POLICY},
            {"role": "user", "content": user_content},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"{SYSTEM_POLICY}\n\n{user_content}\n\n답변:"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        do_sample = temperature > 0
        generate_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        with torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)

        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if strip_thinking:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text


def validate_chunk_rows(rows: list[dict[str, Any]]) -> None:
    for idx, row in enumerate(rows):
        for key in ("chunk_id", "chunk_type", "metadata"):
            if key not in row:
                raise ValueError(f"{idx}번째 row에 필수 필드가 없습니다: {key}")
        if not extract_chunk_text(row):
            raise ValueError(f"{idx}번째 row에 chunk_text/text가 없습니다.")
        if not isinstance(row["metadata"], dict):
            raise ValueError(f"{idx}번째 row의 metadata는 dict 여야 합니다.")


def embed_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    embedding_model: str,
    batch_size: int,
    device: str | None,
    normalize_embeddings: bool,
    passage_prefix: str,
    query_prefix: str,
    trust_remote_code: bool,
) -> int:
    rows = read_jsonl(input_path)
    if not rows:
        raise RuntimeError(f"입력 JSONL이 비어있습니다: {input_path}")
    validate_chunk_rows(rows)

    client = HuggingFaceEmbeddingClient(
        embedding_model,
        device=device,
        trust_remote_code=trust_remote_code,
        normalize_embeddings=normalize_embeddings,
        passage_prefix=passage_prefix,
        query_prefix=query_prefix,
    )
    texts = [extract_chunk_text(row) for row in rows]
    embeddings = client.embed_passages(texts, batch_size=batch_size)

    output_rows: list[dict[str, Any]] = []
    resolved_model = resolve_embedding_model(embedding_model)
    for row, embedding in zip(rows, embeddings, strict=False):
        out = dict(row)
        metadata = normalize_metadata(out.get("metadata"))
        metadata["embedding_source"] = "huggingface"
        metadata["embedding_model"] = resolved_model
        metadata["embedding_normalized"] = str(normalize_embeddings)
        if passage_prefix:
            metadata["embedding_passage_prefix"] = passage_prefix
        if query_prefix:
            metadata["embedding_query_prefix"] = query_prefix
        out["metadata"] = metadata
        out["embedding"] = embedding
        output_rows.append(out)

    write_jsonl(output_path, output_rows)
    return len(output_rows)


def build_chroma(*, input_path: Path, index_dir: Path, doc_id: str | None = None) -> int:
    return build_chroma_from_embedded_jsonl(input_path=input_path, index_dir=index_dir, doc_id=doc_id)


def format_context(results: list[QueryResult], max_context_chars: int) -> str:
    blocks: list[str] = []
    total = 0
    for item in results:
        chunk = item.chunk
        title = chunk.metadata.get("file_name") or chunk.metadata.get("title") or chunk.doc_id
        section = chunk.metadata.get("section_path_text", "")
        body = (
            f"[source_id: {chunk.chunk_id} | score: {item.score:.4f} | title: {title}]\n"
            f"section: {section}\n"
            f"{chunk.text}"
        )
        if total + len(body) > max_context_chars:
            break
        blocks.append(body)
        total += len(body)
    return "\n\n---\n\n".join(blocks)


def retrieve(
    *,
    index_dir: Path,
    question: str,
    embedding_model: str,
    top_k: int,
    score_threshold: float,
    doc_id: str | None,
    batch_size: int,
    device: str | None,
    normalize_embeddings: bool,
    passage_prefix: str,
    query_prefix: str,
    trust_remote_code: bool,
) -> list[QueryResult]:
    from src.engine.vector_store import ChromaVectorStore

    store = ChromaVectorStore.load(index_dir)
    client = HuggingFaceEmbeddingClient(
        embedding_model,
        device=device,
        trust_remote_code=trust_remote_code,
        normalize_embeddings=normalize_embeddings,
        passage_prefix=passage_prefix,
        query_prefix=query_prefix,
    )
    query_embedding = client.embed_queries([question], batch_size=batch_size)[0]
    raw_results = store.search(query_embedding, top_k, doc_id=doc_id)
    return [
        QueryResult(chunk=chunk, score=score)
        for chunk, score in raw_results
        if score >= score_threshold
    ]


def query_index(
    *,
    index_dir: Path,
    question: str,
    embedding_model: str,
    llm_model: str | None,
    top_k: int,
    score_threshold: float,
    doc_id: str | None,
    max_context_chars: int,
    batch_size: int,
    device: str | None,
    normalize_embeddings: bool,
    passage_prefix: str,
    query_prefix: str,
    trust_remote_code: bool,
    llm_device_map: str,
    dtype: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    strip_thinking: bool,
    include_source_text: bool,
    output_path: Path | None,
) -> dict[str, Any]:
    results = retrieve(
        index_dir=index_dir,
        question=question,
        embedding_model=embedding_model,
        top_k=top_k,
        score_threshold=score_threshold,
        doc_id=doc_id,
        batch_size=batch_size,
        device=device,
        normalize_embeddings=normalize_embeddings,
        passage_prefix=passage_prefix,
        query_prefix=query_prefix,
        trust_remote_code=trust_remote_code,
    )
    context = format_context(results, max_context_chars)

    answer: str | None = None
    resolved_llm = resolve_llm_model(llm_model)
    if resolved_llm:
        llm = HuggingFaceLLMClient(
            resolved_llm,
            device_map=llm_device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        answer = llm.generate(
            question=question,
            context=context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            strip_thinking=strip_thinking,
        )

    sources: list[dict[str, Any]] = []
    for item in results:
        row = {
            "chunk_id": item.chunk.chunk_id,
            "score": item.score,
            "metadata": item.chunk.metadata,
        }
        if include_source_text:
            row["text"] = item.chunk.text
        sources.append(row)

    payload: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "embedding_model": resolve_embedding_model(embedding_model),
        "llm_model": resolved_llm,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def print_query_result(payload: dict[str, Any]) -> None:
    answer = payload.get("answer")
    if answer:
        print(answer)
        print()
    else:
        print("(retrieval only: --llm-model을 지정하면 답변 생성까지 수행합니다.)")
        print()
    print("Sources:")
    for source in payload.get("sources", []):
        print(f"- {source['chunk_id']} score={source['score']:.4f}")
        metadata = source.get("metadata") or {}
        file_name = metadata.get("file_name")
        section = metadata.get("section_path_text")
        if file_name:
            print(f"  file: {file_name}")
        if section:
            print(f"  section: {section}")
        if source.get("text"):
            print("  text:")
            print(indent_block(str(source["text"]), "    "))


def indent_block(text: str, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def parse_hwp_to_chunks(
    *,
    input_file: Path,
    prechunk_path: Path,
    chunks_path: Path,
    group_size: int,
    text_chunk_size: int,
    text_overlap: int,
    table_chunk_size: int,
    max_table_rows: int,
    include_toc: bool,
    exclude_cover: bool,
) -> None:
    from src.cli import chunk_jsonl, parse_hwp

    parse_hwp(
        str(prechunk_path),
        input_path=str(input_file),
        group_size=group_size,
    )
    chunk_jsonl(
        input_path=str(prechunk_path),
        output_path=str(chunks_path),
        text_chunk_size=text_chunk_size,
        text_overlap=text_overlap,
        table_chunk_size=table_chunk_size,
        max_table_rows=max_table_rows,
        include_toc=include_toc,
        include_cover=not exclude_cover,
    )


def discover_hwp_files(input_dir: Path) -> list[Path]:
    from src.Parsing.parsing import discover_hwp_files as discover

    return discover(input_dir, glob_pattern="*.hwp", recursive=False)


def pipeline_paths(input_file: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    safe_stem = safe_output_stem(input_file.stem)
    doc_dir = output_dir / safe_stem
    doc_dir.mkdir(parents=True, exist_ok=True)
    return (
        doc_dir / f"{input_file.stem}_prechunk.jsonl",
        doc_dir / f"{input_file.stem}_chunks.jsonl",
        doc_dir / f"{input_file.stem}_embedded_hug.jsonl",
    )


def run_pipeline(
    *,
    input_path: Path | None,
    input_dir: Path | None,
    output_dir: Path,
    index_dir: Path,
    embedding_model: str,
    batch_size: int,
    device: str | None,
    normalize_embeddings: bool,
    passage_prefix: str,
    query_prefix: str,
    trust_remote_code: bool,
    group_size: int,
    text_chunk_size: int,
    text_overlap: int,
    table_chunk_size: int,
    max_table_rows: int,
    include_toc: bool,
    exclude_cover: bool,
    stop_on_error: bool,
) -> None:
    require_exactly_one(a=input_path, b=input_dir, a_name="--input", b_name="--input-dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path:
        input_files = [input_path]
    else:
        input_files = discover_hwp_files(input_dir)  # type: ignore[arg-type]
        if not input_files:
            raise SystemExit(f"No HWP files found in: {input_dir}")

    embedded_paths: list[Path] = []
    failures: list[dict[str, str]] = []
    for index, input_file in enumerate(input_files, start=1):
        print(f"\n=== [{index}/{len(input_files)}] {input_file.name} ===")
        prechunk_path, chunks_path, embedded_path = pipeline_paths(input_file, output_dir)
        try:
            print("[1/3] parse/chunk")
            parse_hwp_to_chunks(
                input_file=input_file,
                prechunk_path=prechunk_path,
                chunks_path=chunks_path,
                group_size=group_size,
                text_chunk_size=text_chunk_size,
                text_overlap=text_overlap,
                table_chunk_size=table_chunk_size,
                max_table_rows=max_table_rows,
                include_toc=include_toc,
                exclude_cover=exclude_cover,
            )
            print("[2/3] huggingface embed")
            count = embed_jsonl(
                input_path=chunks_path,
                output_path=embedded_path,
                embedding_model=embedding_model,
                batch_size=batch_size,
                device=device,
                normalize_embeddings=normalize_embeddings,
                passage_prefix=passage_prefix,
                query_prefix=query_prefix,
                trust_remote_code=trust_remote_code,
            )
            embedded_paths.append(embedded_path)
            print(f"embedded_rows: {count}")
        except Exception as exc:
            failures.append(
                {
                    "source": str(input_file),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"FAILED: {type(exc).__name__}: {exc}")
            if stop_on_error:
                raise

    failure_path = output_dir / "hug_pipeline_failures.json"
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    if not embedded_paths:
        raise RuntimeError(f"No embedded JSONL files were created. failures={failure_path}")

    print("[3/3] build unified chroma")
    if len(embedded_paths) == 1:
        chunks_in_index = build_chroma(input_path=embedded_paths[0], index_dir=index_dir)
        merged_path = embedded_paths[0]
        duplicate_ids = 0
    else:
        merged_path = output_dir / "all_embedded_hug.jsonl"
        result = build_unified_chroma_index(
            input_dir=output_dir,
            index_dir=index_dir,
            merged_output=merged_path,
            pattern="*_embedded_hug.jsonl",
            recursive=True,
        )
        chunks_in_index = result.chunks_in_index
        duplicate_ids = result.duplicate_chunk_ids

    print("pipeline_done")
    print(f"files: {len(input_files)}")
    print(f"failed: {len(failures)}")
    print(f"merged_jsonl: {merged_path}")
    print(f"duplicate_chunk_ids_rewritten: {duplicate_ids}")
    print(f"chunks_in_index: {chunks_in_index}")
    print(f"index_dir: {index_dir}")
    print(f"failures: {failure_path}")


def list_models() -> None:
    print("Embedding aliases:")
    for alias, model in EMBEDDING_MODEL_ALIASES.items():
        print(f"- {alias}: {model}")
    print("\nLLM aliases:")
    for alias, model in LLM_MODEL_ALIASES.items():
        print(f"- {alias}: {model}")


def add_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, cpu")
    parser.add_argument("--no-normalize-embeddings", action="store_true")
    parser.add_argument("--passage-prefix", default="")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--trust-remote-code", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hugging Face-only CLI for Bidmate RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-models", help="Show built-in model aliases")

    embed_parser = subparsers.add_parser("embed-jsonl", help="Embed chunk JSONL with a HF embedding model")
    embed_parser.add_argument("--input", required=True)
    embed_parser.add_argument("--output", required=True)
    add_embedding_args(embed_parser)

    chroma_parser = subparsers.add_parser("build-chroma", help="Build Chroma from embedded JSONL")
    chroma_parser.add_argument("--input", required=True)
    chroma_parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    chroma_parser.add_argument("--doc-id", default=None)

    merge_parser = subparsers.add_parser("merge-embedded", help="Merge *_embedded_hug.jsonl and build Chroma")
    merge_parser.add_argument("--input-dir", default="data/v2")
    merge_parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    merge_parser.add_argument("--merged-output", default=None)
    merge_parser.add_argument("--pattern", default="*_embedded_hug.jsonl")
    merge_parser.add_argument("--no-recursive", action="store_true")
    merge_parser.add_argument("--merge-only", action="store_true")

    query_parser = subparsers.add_parser("query", help="Retrieve and optionally answer with a HF LLM")
    query_parser.add_argument("question")
    query_parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    query_parser.add_argument("--llm-model", default=None, help="Example: qwen3-8b, llama, gemma, exaone")
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--score-threshold", type=float, default=0.0)
    query_parser.add_argument("--doc-id", default=None)
    query_parser.add_argument("--max-context-chars", type=int, default=12000)
    query_parser.add_argument("--llm-device-map", default="auto")
    query_parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    query_parser.add_argument("--max-new-tokens", type=int, default=512)
    query_parser.add_argument("--temperature", type=float, default=0.2)
    query_parser.add_argument("--top-p", type=float, default=0.9)
    query_parser.add_argument("--keep-thinking", action="store_true")
    query_parser.add_argument("--include-source-text", action="store_true")
    query_parser.add_argument("--output", default=None)
    add_embedding_args(query_parser)

    pipeline_parser = subparsers.add_parser(
        "run-pipeline",
        help="Run HWP parse -> chunk -> HF embed -> unified Chroma",
    )
    pipeline_input = pipeline_parser.add_mutually_exclusive_group(required=True)
    pipeline_input.add_argument("--input")
    pipeline_input.add_argument("--input-dir")
    pipeline_parser.add_argument("--output-dir", default="data/v2_hug")
    pipeline_parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    pipeline_parser.add_argument("--group-size", type=int, default=8)
    pipeline_parser.add_argument("--text-chunk-size", type=int, default=900)
    pipeline_parser.add_argument("--text-overlap", type=int, default=180)
    pipeline_parser.add_argument("--table-chunk-size", type=int, default=1000)
    pipeline_parser.add_argument("--max-table-rows", type=int, default=6)
    pipeline_parser.add_argument("--include-toc", action="store_true")
    pipeline_parser.add_argument("--exclude-cover", action="store_true")
    pipeline_parser.add_argument("--stop-on-error", action="store_true")
    add_embedding_args(pipeline_parser)

    args = parser.parse_args()

    if args.command == "list-models":
        list_models()
    elif args.command == "embed-jsonl":
        count = embed_jsonl(
            input_path=Path(args.input),
            output_path=Path(args.output),
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            normalize_embeddings=not args.no_normalize_embeddings,
            passage_prefix=args.passage_prefix,
            query_prefix=args.query_prefix,
            trust_remote_code=args.trust_remote_code,
        )
        print(f"Embedded {count} rows -> {args.output}")
    elif args.command == "build-chroma":
        count = build_chroma(
            input_path=Path(args.input),
            index_dir=Path(args.index_dir),
            doc_id=args.doc_id,
        )
        print(f"Built Chroma index with {count} chunks -> {args.index_dir}")
    elif args.command == "merge-embedded":
        result = build_unified_chroma_index(
            input_dir=Path(args.input_dir),
            index_dir=Path(args.index_dir),
            merged_output=Path(args.merged_output) if args.merged_output else None,
            pattern=args.pattern,
            recursive=not args.no_recursive,
            skip_chroma_build=args.merge_only,
        )
        print(f"merged_sources: {result.source_files}")
        print(f"merged_rows: {result.total_rows}")
        print(f"duplicate_chunk_ids_rewritten: {result.duplicate_chunk_ids}")
        print(f"merged_jsonl: {result.merged_path}")
        print(f"index_dir: {result.index_dir}")
        print(f"chunks_in_index: {result.chunks_in_index}")
    elif args.command == "query":
        payload = query_index(
            index_dir=Path(args.index_dir),
            question=args.question,
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            doc_id=args.doc_id,
            max_context_chars=args.max_context_chars,
            batch_size=args.batch_size,
            device=args.device,
            normalize_embeddings=not args.no_normalize_embeddings,
            passage_prefix=args.passage_prefix,
            query_prefix=args.query_prefix,
            trust_remote_code=args.trust_remote_code,
            llm_device_map=args.llm_device_map,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            strip_thinking=not args.keep_thinking,
            include_source_text=args.include_source_text,
            output_path=Path(args.output) if args.output else None,
        )
        print_query_result(payload)
    elif args.command == "run-pipeline":
        run_pipeline(
            input_path=Path(args.input) if args.input else None,
            input_dir=Path(args.input_dir) if args.input_dir else None,
            output_dir=Path(args.output_dir),
            index_dir=Path(args.index_dir),
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            normalize_embeddings=not args.no_normalize_embeddings,
            passage_prefix=args.passage_prefix,
            query_prefix=args.query_prefix,
            trust_remote_code=args.trust_remote_code,
            group_size=args.group_size,
            text_chunk_size=args.text_chunk_size,
            text_overlap=args.text_overlap,
            table_chunk_size=args.table_chunk_size,
            max_table_rows=args.max_table_rows,
            include_toc=args.include_toc,
            exclude_cover=args.exclude_cover,
            stop_on_error=args.stop_on_error,
        )
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
