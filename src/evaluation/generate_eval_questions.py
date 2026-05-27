from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.models.openai_client import supports_chat_temperature
from src.utils.jsonl import read_jsonl, write_jsonl


_GENERIC_DOC_PARENT_NAMES = frozenset({"data", "v2", "raw", "processed", "outputs", "ocr_rag"})
DEFAULT_OCR_CHUNKS_REL = Path("ocr_rag/ocr_input_chunks.jsonl")


def is_ocr_handoff_chunk_file(path: Path) -> bool:
    """OCR→RAG handoff 단일 파일은 glob 대상에서 제외하고 전용 경로로만 로드한다."""
    return path.name == "ocr_input_chunks.jsonl" and path.parent.name == "ocr_rag"


def find_chunk_files(
    input_dir: Path,
    pattern: str = "*_chunks.jsonl",
    *,
    recursive: bool = True,
) -> list[Path]:
    """run-pipeline 산출물처럼 하위 폴더에 있는 chunks JSONL도 찾는다."""
    iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    return sorted(
        path for path in iterator if path.is_file() and not is_ocr_handoff_chunk_file(path)
    )


def default_ocr_chunks_path(input_dir: Path) -> Path:
    return input_dir / DEFAULT_OCR_CHUNKS_REL


def load_chunk_files(
    input_dir: Path,
    pattern: str = "*_chunks.jsonl",
    *,
    recursive: bool = True,
    extra_chunk_paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()

    def ingest(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen_paths or not path.is_file():
            return
        seen_paths.add(resolved)
        for row in read_jsonl(path):
            item = dict(row)
            item["_source_chunk_file"] = str(path)
            rows.append(item)

    for path in find_chunk_files(input_dir, pattern, recursive=recursive):
        ingest(path)

    ocr_path = default_ocr_chunks_path(input_dir)
    paths_to_add: list[Path] = [ocr_path, *(extra_chunk_paths or [])]
    if not ocr_path.is_file():
        warnings.warn(
            f"OCR handoff가 없습니다: {ocr_path}\n"
            "  → ./scripts/run_ocr_stage.sh 후 ./scripts/run_rag_stage.sh (또는 ocr-export-rag + embed-jsonl)",
            UserWarning,
            stacklevel=2,
        )

    for path in paths_to_add:
        ingest(path)

    return rows


def chunk_text(row: dict[str, Any], max_chars: int) -> str:
    text = str(row.get("chunk_text") or row.get("text") or "").strip()
    return text[:max_chars]


def chunk_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def is_ocr_chunk(row: dict[str, Any]) -> bool:
    chunk_type = str(row.get("chunk_type") or chunk_metadata(row).get("chunk_type") or "").lower()
    metadata = chunk_metadata(row)
    if chunk_type.startswith("ocr_"):
        return True
    if str(metadata.get("source") or "").lower() == "ocr":
        return True
    text = str(row.get("chunk_text") or row.get("text") or "")
    return "이미지:" in text and "OCR ID:" in text


def resolve_doc_id(row: dict[str, Any]) -> str:
    """문서별 하위 폴더(run-pipeline) 또는 메타데이터 기준으로 doc_id를 정한다."""
    metadata = chunk_metadata(row)
    source = str(row.get("_source_chunk_file", "")).strip()
    default_from_path = "document"
    if source:
        source_path = Path(source)
        folder_name = source_path.parent.name
        if folder_name and folder_name.lower() not in _GENERIC_DOC_PARENT_NAMES:
            default_from_path = folder_name
        else:
            stem = source_path.stem
            default_from_path = stem[: -len("_chunks")] if stem.endswith("_chunks") else stem
    return str(
        row.get("doc_id")
        or metadata.get("doc_id")
        or metadata.get("file_name")
        or metadata.get("source_file")
        or default_from_path
    )


def group_chunks_by_doc(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in chunks:
        grouped.setdefault(resolve_doc_id(row), []).append(row)
    return grouped


def _stride_sample(rows: list[dict[str, Any]], max_chunks: int) -> list[dict[str, Any]]:
    if len(rows) <= max_chunks:
        return list(rows)
    if max_chunks <= 0:
        return []
    if max_chunks == 1:
        return [rows[0]]
    step = (len(rows) - 1) / (max_chunks - 1)
    indices = [round(i * step) for i in range(max_chunks)]
    return [rows[index] for index in indices]


def _dedupe_chunks_by_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        chunk_id = str(row.get("chunk_id") or row.get("id") or "").strip()
        if chunk_id and chunk_id in seen:
            continue
        if chunk_id:
            seen.add(chunk_id)
        out.append(row)
    return out


def sample_chunks(rows: list[dict[str, Any]], max_chunks: int) -> list[dict[str, Any]]:
    """문서당 샘플. OCR 청크가 있으면 최소 1개는 컨텍스트에 포함한다."""
    if len(rows) <= max_chunks:
        return rows
    ocr_rows = [row for row in rows if is_ocr_chunk(row)]
    other_rows = [row for row in rows if not is_ocr_chunk(row)]
    picked: list[dict[str, Any]] = []
    if ocr_rows:
        ocr_slots = 1 if max_chunks == 1 else min(len(ocr_rows), max(1, max_chunks // 3))
        picked.extend(_stride_sample(ocr_rows, ocr_slots))
    remain = max_chunks - len(picked)
    if remain > 0:
        picked_ids = {str(row.get("chunk_id") or row.get("id") or "") for row in picked}
        pool = other_rows if other_rows else [
            row for row in ocr_rows if str(row.get("chunk_id") or row.get("id") or "") not in picked_ids
        ]
        if pool:
            picked.extend(_stride_sample(pool, remain))
    return _dedupe_chunks_by_id(picked)


def build_generation_inputs(
    chunks: list[dict[str, Any]],
    *,
    max_docs: int,
    max_chunks_per_doc: int,
    max_chars_per_chunk: int,
    questions_per_doc: int,
) -> list[dict[str, Any]]:
    grouped = group_chunks_by_doc(chunks)
    doc_ids = sorted(grouped)
    if max_docs > 0:
        doc_ids = doc_ids[:max_docs]
    rows: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        sampled = sample_chunks(grouped[doc_id], max_chunks_per_doc)
        context_chunks = []
        source_files = sorted({str(row.get("_source_chunk_file", "")) for row in sampled if row.get("_source_chunk_file")})
        for row in sampled:
            metadata = chunk_metadata(row)
            ocr = is_ocr_chunk(row)
            context_chunks.append(
                {
                    "chunk_id": str(row.get("chunk_id") or row.get("id") or ""),
                    "chunk_type": str(row.get("chunk_type") or metadata.get("chunk_type") or ""),
                    "source": "ocr" if ocr else str(metadata.get("source") or "hwp"),
                    "image_stem": str(metadata.get("image_stem") or ""),
                    "ocr_type": str(metadata.get("ocr_type") or metadata.get("type") or ""),
                    "metadata": metadata,
                    "text": chunk_text(row, max_chars_per_chunk),
                }
            )
        prompt = build_question_generation_prompt(doc_id, context_chunks, questions_per_doc)
        rows.append(
            {
                "doc_id": doc_id,
                "source_files": source_files,
                "chunk_count": len(grouped[doc_id]),
                "sampled_chunk_count": len(context_chunks),
                "questions_per_doc": questions_per_doc,
                "chunks": context_chunks,
                "prompt": prompt,
            }
        )
    return rows


def build_question_generation_prompt(
    doc_id: str,
    chunks: list[dict[str, Any]],
    questions_per_doc: int,
) -> str:
    ocr_chunk_count = sum(1 for chunk in chunks if str(chunk.get("source") or "") == "ocr")
    has_ocr = ocr_chunk_count > 0
    ocr_min = 1 if has_ocr and questions_per_doc >= 2 else (1 if has_ocr and questions_per_doc == 1 else 0)

    context = json.dumps(chunks, ensure_ascii=False, indent=2)
    ocr_note = ""
    if has_ocr:
        ocr_note = (
            f"\n\n[OCR/이미지 청크 안내] 이 문서에는 PaddleOCR 등으로 추출한 이미지 청크가 "
            f"{ocr_chunk_count}개 포함되어 있습니다(source=ocr, chunk_type=ocr_*). "
            f"최소 {ocr_min}개 질문은 반드시 OCR 청크만으로 답할 수 있게 만드세요 "
            f"(스캔 별지·검토결과서·표 이미지의 사업기간·검토의견·서식명·수치 등)."
        )

    return f"""당신은 RAG 성능평가용 질문셋 생성자입니다.
아래 문서 청크만 근거로 평가 질문을 생성하세요. (다른 문서·외부 지식·추측 금지)

문서 ID (doc_id는 반드시 이 값과 동일):
{doc_id}

문서 청크:
{context}
{ocr_note}

총 {questions_per_doc}개의 질문을 생성하세요.
출력은 반드시 {{"questions": [...]}} 형태의 JSON 객체로만 작성하세요.

각 항목 필수 필드:
- question, category, question_type, doc_id, expected_answer, ground_truth_keywords, difficulty, gold_chunk_ids
- eval_focus: "text" | "ocr_image" (OCR/스캔 이미지 청크 근거면 ocr_image, 일반 본문·표 청크면 text)

[엄격 규칙 — 위반 시 무효]
1. gold_chunk_ids: answerable 질문은 위 청크 목록의 chunk_id만, 최소 1개. unanswerable은 반드시 [] (빈 배열).
2. expected_answer의 핵심 수치·고유명사는 gold_chunk_ids가 가리키는 청크 text에 실제로 있어야 함 (unanswerable 제외).
3. ground_truth_keywords는 해당 청크 text에 그대로 등장하는 단어·숫자만 (2~5개).
4. comparison은 A·B 모두가 청크에 있을 때만. 없으면 fact/summary로 바꾸거나 unanswerable.
5. unanswerable은 "이 문서 청크 어디에도 없는" 내용만. 타 기관·타 사업·외부 정책 등.
6. doc_id는 항상 "{doc_id}".

[평가 관점 — question_type]
- fact, requirement_detail: 단일 사실·요구사항
- summary: 여러 청크 종합
- follow_up: 같은 문서 안 선행 맥락 1문장 전제
- comparison: 동일 문서 내 두 항목 대조 (근거 둘 다 있을 때만)
- unanswerable: 제공 청크에 없음 → expected_answer는 "문서에서 확인되지 않음", gold_chunk_ids는 []

[OCR/이미지 성능 검증 질문]
- source=ocr 또는 chunk_type이 ocr_로 시작하는 청크가 있으면 eval_focus=ocr_image 질문을 최소 {ocr_min}개 포함.
- 질문은 이미지(스캔)에만 나오는 정보를 묻게 하세요. 예:
  - "별지 적정 사업기간 산정서에서 종합 검토 결과 적정 사업기간은?"
  - "영향평가 검토결과서에 기재된 사업기간(일)은?"
  - "검토항목별 추정 사업기간이 5개월로 적힌 항목은?"
- OCR 청크의 image_stem·표 행(검토항목/검토의견/추정 사업기간)을 활용하세요.
- 일반 HWP 본문 청크와 OCR 청크에 같은 내용이 있으면, OCR 전용 질문은 OCR 청크의 고유 표현(이미지 파일명·OCR 표 행)을 쓰세요.

category 예: 기능 요구사항, 보안, 일정, 예산, 입찰·계약, 부록·양식, OCR/이미지.
expected_answer: 정답 전문이 아닌 검수용 핵심 요지(짧게).
difficulty: easy(단일 팩트), medium(요약·세부), hard(종합·후속·unanswerable).

[생성 비율 가이드 — OCR 최소 개수 충족 후 나머지]
- OCR/이미지(eval_focus=ocr_image): 최소 {ocr_min}개
- fact·requirement_detail: 약 35%
- summary: 약 25%
- follow_up: 약 15%
- unanswerable: 약 10%

질문은 한 번의 RAG 호출로 답 가능한 분량으로 작성하세요.
"""


def generate_eval_questions(
    input_dir: Path,
    output_path: Path,
    pattern: str = "*_chunks.jsonl",
    *,
    recursive: bool = True,
    extra_chunk_paths: list[Path] | None = None,
    max_docs: int = 5,
    max_chunks_per_doc: int = 8,
    max_chars_per_chunk: int = 1200,
    questions_per_doc: int = 3,
    overwrite: bool = False,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"이미 파일이 있습니다: {output_path}")
    chunks = load_chunk_files(
        input_dir,
        pattern,
        recursive=recursive,
        extra_chunk_paths=extra_chunk_paths,
    )
    if not chunks:
        scope = f"{input_dir}/**/{pattern}" if recursive else f"{input_dir}/{pattern}"
        raise RuntimeError(
            f"청크 JSONL을 찾지 못했습니다: {scope} "
            f"(OCR handoff: {default_ocr_chunks_path(input_dir)})"
        )
    rows = build_generation_inputs(
        chunks,
        max_docs=max_docs,
        max_chunks_per_doc=max_chunks_per_doc,
        max_chars_per_chunk=max_chars_per_chunk,
        questions_per_doc=questions_per_doc,
    )
    write_jsonl(output_path, rows)
    print(f"wrote_generation_inputs: {output_path}")


def parse_question_response(content: str) -> list[dict[str, Any]]:
    data = json.loads(content)
    if isinstance(data, dict):
        for key in ("questions", "items", "data"):
            value = data.get(key)
            if isinstance(value, list):
                data = value
                break
    if not isinstance(data, list):
        raise ValueError("OpenAI 응답이 JSON 배열이 아닙니다.")
    rows: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        keywords = item.get("ground_truth_keywords")
        if not isinstance(keywords, list):
            keywords = []
        gold_chunk_ids = item.get("gold_chunk_ids")
        if not isinstance(gold_chunk_ids, list):
            gold_chunk_ids = []
        eval_focus = str(item.get("eval_focus", "")).strip().lower()
        if eval_focus not in {"text", "ocr_image"}:
            eval_focus = ""
        rows.append(
            {
                "question": question,
                "category": str(item.get("category", "")).strip(),
                "question_type": str(item.get("question_type", "")).strip(),
                "doc_id": str(item.get("doc_id", "")).strip(),
                "expected_answer": str(item.get("expected_answer", "")).strip(),
                "ground_truth_keywords": [str(keyword) for keyword in keywords],
                "gold_chunk_ids": [str(chunk_id) for chunk_id in gold_chunk_ids],
                "difficulty": str(item.get("difficulty", "")).strip(),
                **({"eval_focus": eval_focus} if eval_focus else {}),
            }
        )
    return rows


def call_openai_for_questions(prompt: str, model: str) -> list[dict[str, Any]]:
    client = OpenAI()
    request: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    if supports_chat_temperature(model):
        request["temperature"] = 0.2
    response = client.chat.completions.create(**request)
    content = response.choices[0].message.content or "[]"
    return parse_question_response(content)


def generate_questions_with_openai(
    input_path: Path,
    output_path: Path,
    *,
    model: str,
    overwrite: bool = False,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"이미 파일이 있습니다: {output_path}")
    rows = read_jsonl(input_path)
    if not rows:
        raise RuntimeError(f"질문 생성 입력 파일이 비어있습니다: {input_path}")

    from tqdm.auto import tqdm

    output_rows: list[dict[str, Any]] = []
    progress = tqdm(rows, desc="Generating eval questions", unit="doc")
    for row in progress:
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        chunks_for_row = row.get("chunks")
        if not isinstance(chunks_for_row, list):
            chunks_for_row = []
        chunk_by_id: dict[str, dict[str, Any]] = {}
        for chunk in chunks_for_row:
            if isinstance(chunk, dict) and chunk.get("chunk_id"):
                chunk_by_id[str(chunk["chunk_id"])] = chunk
        valid_chunk_ids = set(chunk_by_id)
        doc_id = str(row.get("doc_id") or "").strip()
        if doc_id:
            progress.set_postfix_str(doc_id[:40] + ("…" if len(doc_id) > 40 else ""), refresh=False)
        questions = call_openai_for_questions(prompt, model)
        for question in questions:
            if not question.get("doc_id"):
                question["doc_id"] = row.get("doc_id", "")
            question_type = str(question.get("question_type") or "").strip().lower()
            is_unanswerable = question_type == "unanswerable"
            gold_ids = [
                chunk_id
                for chunk_id in question.get("gold_chunk_ids", [])
                if chunk_id in valid_chunk_ids
            ]
            if is_unanswerable:
                if gold_ids:
                    # 모델이 잘못 청크를 붙인 unanswerable은 제외 (답 가능 질문으로 오염 방지)
                    continue
                question["gold_chunk_ids"] = []
            elif not gold_ids:
                continue
            else:
                question["gold_chunk_ids"] = gold_ids
            if not question.get("eval_focus"):
                if is_unanswerable:
                    question["eval_focus"] = "text"
                elif any(str(chunk_by_id[cid].get("source") or "") == "ocr" for cid in gold_ids):
                    question["eval_focus"] = "ocr_image"
                else:
                    question["eval_focus"] = "text"
            output_rows.append(question)

    write_jsonl(output_path, output_rows)
    print(f"wrote_eval_questions: {output_path}")
    print(f"questions: {len(output_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="data/v2",
        help="청크 JSONL 루트 (하위 폴더까지 검색, run-pipeline 산출 구조)",
    )
    parser.add_argument("--pattern", default="*_chunks.jsonl")
    parser.add_argument(
        "--extra-chunk-file",
        action="append",
        default=[],
        metavar="PATH",
        help="추가 청크 JSONL (여러 번 지정 가능)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="하위 폴더까지 glob (기본: 켜짐)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="input-dir 바로 아래 파일만",
    )
    parser.add_argument("--output", default="data/v2/eval_question_generation_inputs.jsonl")
    parser.add_argument("--generation-input", default="data/v2/eval_question_generation_inputs.jsonl")
    parser.add_argument("--eval-output", default="data/v2/eval_questions.jsonl")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=5,
        help="질문 생성에 쓸 문서 수 상한 (0이면 발견한 문서 전부)",
    )
    parser.add_argument("--max-chunks-per-doc", type=int, default=8)
    parser.add_argument("--max-chars-per-chunk", type=int, default=1200)
    parser.add_argument("--questions-per-doc", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--call-openai", action="store_true")
    args = parser.parse_args()

    load_dotenv()

    if args.call_openai:
        generate_questions_with_openai(
            input_path=Path(args.generation_input),
            output_path=Path(args.eval_output),
            model=args.model,
            overwrite=args.overwrite,
        )
        return

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    extra_paths = [Path(path) for path in args.extra_chunk_file]
    chunk_files = find_chunk_files(input_dir, args.pattern, recursive=args.recursive)
    ocr_path = default_ocr_chunks_path(input_dir)
    if ocr_path.is_file() and ocr_path not in chunk_files:
        chunk_files = sorted({*chunk_files, ocr_path})
    chunks = load_chunk_files(
        input_dir,
        args.pattern,
        recursive=args.recursive,
        extra_chunk_paths=extra_paths,
    )
    grouped = group_chunks_by_doc(chunks)
    print(f"recursive: {args.recursive}")
    print(f"chunk_files: {len(chunk_files)}")
    for path in chunk_files:
        print(f"- {path}")
    print(f"chunks: {len(chunks)}")
    print(f"docs: {len(grouped)}")
    for doc_id in sorted(grouped):
        items = grouped[doc_id]
        table_like = sum(
            1
            for row in items
            if "table" in str(row.get("chunk_type") or chunk_metadata(row).get("chunk_type") or "").lower()
        )
        print(f"  doc {doc_id}: chunks={len(items)} (table-ish={table_like})")

    if args.dry_run:
        return

    generate_eval_questions(
        input_dir=input_dir,
        output_path=output_path,
        pattern=args.pattern,
        recursive=args.recursive,
        extra_chunk_paths=extra_paths,
        max_docs=args.max_docs,
        max_chunks_per_doc=args.max_chunks_per_doc,
        max_chars_per_chunk=args.max_chars_per_chunk,
        questions_per_doc=args.questions_per_doc,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
