from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.jsonl import read_jsonl, write_jsonl


def find_chunk_files(input_dir: Path, pattern: str = "*_chunks.jsonl") -> list[Path]:
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def load_chunk_files(input_dir: Path, pattern: str = "*_chunks.jsonl") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in find_chunk_files(input_dir, pattern):
        for row in read_jsonl(path):
            item = dict(row)
            item["_source_chunk_file"] = str(path)
            rows.append(item)
    return rows


def chunk_text(row: dict[str, Any], max_chars: int) -> str:
    text = str(row.get("chunk_text") or row.get("text") or "").strip()
    return text[:max_chars]


def chunk_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def resolve_doc_id(row: dict[str, Any]) -> str:
    metadata = chunk_metadata(row)
    return str(
        row.get("doc_id")
        or metadata.get("doc_id")
        or metadata.get("file_name")
        or metadata.get("source_file")
        or Path(str(row.get("_source_chunk_file", "document"))).stem
    )


def group_chunks_by_doc(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in chunks:
        grouped.setdefault(resolve_doc_id(row), []).append(row)
    return grouped


def sample_chunks(rows: list[dict[str, Any]], max_chunks: int) -> list[dict[str, Any]]:
    if len(rows) <= max_chunks:
        return rows
    if max_chunks <= 1:
        return [rows[0]]
    step = (len(rows) - 1) / (max_chunks - 1)
    indices = [round(i * step) for i in range(max_chunks)]
    return [rows[index] for index in indices]


def build_generation_inputs(
    chunks: list[dict[str, Any]],
    *,
    max_docs: int,
    max_chunks_per_doc: int,
    max_chars_per_chunk: int,
    questions_per_doc: int,
) -> list[dict[str, Any]]:
    grouped = group_chunks_by_doc(chunks)
    rows: list[dict[str, Any]] = []
    for doc_id in sorted(grouped)[:max_docs]:
        sampled = sample_chunks(grouped[doc_id], max_chunks_per_doc)
        context_chunks = []
        source_files = sorted({str(row.get("_source_chunk_file", "")) for row in sampled if row.get("_source_chunk_file")})
        for row in sampled:
            metadata = chunk_metadata(row)
            context_chunks.append(
                {
                    "chunk_id": str(row.get("chunk_id") or row.get("id") or ""),
                    "chunk_type": str(row.get("chunk_type") or metadata.get("chunk_type") or ""),
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
    context = json.dumps(chunks, ensure_ascii=False, indent=2)
    return f"""당신은 RAG 성능평가용 질문셋 생성자입니다.
    아래 문서 청크만 근거로 평가 질문을 생성하세요. (다른 문서·외부 지식은 사용하지 마세요.)

    문서 ID:
    {doc_id}

    문서 청크:
    {context}

    총 {questions_per_doc}개의 질문을 생성하세요.
    출력은 반드시 {{"questions": [...]}} 형태의 JSON 객체로만 작성하세요.
    각 항목은 question, category, question_type, doc_id, expected_answer, ground_truth_keywords, difficulty 필드를 포함해야 합니다.

    [평가 관점 — question_type과 맞춤]
    - 단일 문서에서 요청 내용을 정확히 뽑아 답하는지 → fact, requirement_detail
    - 여러 청크/항목을 종합·정리하는지 → summary
    - 앞선 주제에 이어 묻는 후속 맥락인지 → follow_up (같은 문서 안에서 앞 질문 주제를 전제로 한 문장)
    - 문서에 없는 내용은 모른다고 하는지 → unanswerable
    - (선택) 같은 문서 안 서로 다른 과업·기능·조항을 대조 → comparison

    category는 청크 내용에 맞게 자유롭게 정하세요. 예: 기능 요구사항, 보안, 운영 현황, 예산, 일정, 입찰·계약, 성능·품질, 부록·양식.
    ground_truth_keywords는 위 청크 텍스트에 실제로 등장하는 단어·숫자만 넣으세요.
    expected_answer는 정답 전문이 아니라 검수용 핵심 요지입니다. 청크에 없으면 "문서에서 확인되지 않음".
    difficulty: easy(단일 팩트), medium(요약·세부), hard(종합·후속·unanswerable).

    [질문 말투·구조 — 아래는 스타일 참고용. 문장을 그대로 복사하지 말 것]
    청크·metadata에 있는 발주기관, 사업·시스템명, 과업명, 기술 키워드, 부서명, 조항 제목 등으로 {{}} 자리를 채워 같은 뉘앙스로 새로 작성하세요.
    발주기관·사업명이 없으면 청크에 나온 고유 표현(과제명, 시스템, 표 제목, 절 이름 등)으로 대체하세요. 없는 기관·사업명을 지어내지 마세요.

    스타일 참고 (실제 과제 예시와 같은 말투):
    - "{{발주기관}}이 발주한 {{사업·시스템}} 관련 사업 요구사항을 정리해 줘." → summary
    - "{{세부 요구 항목}}에 대해서 더 자세히 알려 줘." → follow_up 또는 requirement_detail
    - "{{사업·시스템}} 요구에서 {{기술·기능}}에 대한 요구사항이 있나?" → fact
    - "{{과업·사업}}이 왜 추진되는지 목적을 알려 줘." → fact 또는 summary
    - "{{항목 A}}랑 {{항목 B}}를 비교해 줄래." → comparison (둘 다 이 문서 청크에 있을 때)
    - "{{항목}}에 대한 요구가 있는지 찾아보고, 문서를 기반으로 정확하게 알려 줘." → fact
    - "{{다른 기관·타 사업}}이 발주한 유사 사업이 이 문서에도 나와 있나?" → unanswerable (이 문서에 없으면)

    [생성 비율 — 모두 이 문서 청크만 근거]
    1. 단일 문서 팩트·세부 (약 40%): fact, requirement_detail
    2. 요약·리스트화 (약 30%): summary
    3. 후속·맥락 질문 (약 20%): follow_up
    4. 환각 방지 (약 10%): unanswerable

    질문은 한 번의 RAG 호출로 답하기 적당한 분량으로 작성하세요(과도한 다단계 조사·전체 문서 전수 검색을 요구하지 않음).
    """


def generate_eval_questions(
    input_dir: Path,
    output_path: Path,
    pattern: str = "*_chunks.jsonl",
    max_docs: int = 5,
    max_chunks_per_doc: int = 8,
    max_chars_per_chunk: int = 1200,
    questions_per_doc: int = 3,
    overwrite: bool = False,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"이미 파일이 있습니다: {output_path}")
    chunks = load_chunk_files(input_dir, pattern)
    if not chunks:
        raise RuntimeError(f"청크 JSONL 파일을 찾지 못했습니다: {input_dir / pattern}")
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
        rows.append(
            {
                "question": question,
                "category": str(item.get("category", "")).strip(),
                "question_type": str(item.get("question_type", "")).strip(),
                "doc_id": str(item.get("doc_id", "")).strip(),
                "expected_answer": str(item.get("expected_answer", "")).strip(),
                "ground_truth_keywords": [str(keyword) for keyword in keywords],
                "difficulty": str(item.get("difficulty", "")).strip(),
            }
        )
    return rows


def call_openai_for_questions(prompt: str, model: str) -> list[dict[str, Any]]:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
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

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        questions = call_openai_for_questions(prompt, model)
        for question in questions:
            if not question.get("doc_id"):
                question["doc_id"] = row.get("doc_id", "")
            output_rows.append(question)

    write_jsonl(output_path, output_rows)
    print(f"wrote_eval_questions: {output_path}")
    print(f"questions: {len(output_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/v2")
    parser.add_argument("--pattern", default="*_chunks.jsonl")
    parser.add_argument("--output", default="data/v2/eval_question_generation_inputs.jsonl")
    parser.add_argument("--generation-input", default="data/v2/eval_question_generation_inputs.jsonl")
    parser.add_argument("--eval-output", default="data/v2/eval_questions.jsonl")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-docs", type=int, default=5)
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

    chunk_files = find_chunk_files(input_dir, args.pattern)
    chunks = load_chunk_files(input_dir, args.pattern)
    print(f"chunk_files: {len(chunk_files)}")
    for path in chunk_files:
        print(f"- {path}")
    print(f"chunks: {len(chunks)}")
    print(f"docs: {len(group_chunks_by_doc(chunks))}")

    if args.dry_run:
        return

    generate_eval_questions(
        input_dir=input_dir,
        output_path=output_path,
        pattern=args.pattern,
        max_docs=args.max_docs,
        max_chunks_per_doc=args.max_chunks_per_doc,
        max_chars_per_chunk=args.max_chars_per_chunk,
        questions_per_doc=args.questions_per_doc,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
