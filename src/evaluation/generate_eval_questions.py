from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.engine.question_taxonomy import (
    COVER_FORM_CATEGORY,
    is_cover_form_metadata_question,
)
from src.models.openai_client import supports_chat_temperature
from src.utils.jsonl import read_jsonl, write_jsonl


_GENERIC_DOC_PARENT_NAMES = frozenset({"data", "v2", "raw", "processed", "outputs", "ocr_rag"})
DEFAULT_OCR_CHUNKS_REL = Path("ocr_rag/ocr_input_chunks.jsonl")

# unanswerable 질문이 문서 주제(예산·기간·요구사항 등)와 겹치면 RAG가 관련 청크를 가져와 거절 평가가 깨짐.
UNANSWERABLE_FORBIDDEN_PHRASES: tuple[str, ...] = (
    "총 사업비",
    "총 계약",
    "총액",
    "총 사업",
    "사업비",
    "계약금",
    "소요예산",
    "입찰보증",
    "사업기간",
    "개발기간",
    "용역기간",
    "계약 기간",
    "주관기관",
    "발주기관",
    "수요기관",
    "이 사업",
    "본 사업",
    "해당 사업",
    "본 제안",
    "이 문서",
    "본 문서",
    "제안요청서",
    "별지 제",
    "요구사항 명칭",
    "요구사항 번호",
)

UNANSWERABLE_DOC_SCOPED_PATTERN = re.compile(
    r"(이|본|해당)\s*(사업|문서|제안|계약|용역|공고|과업|사업의)",
    re.I,
)

UNANSWERABLE_REQUIREMENT_ID_PATTERN = re.compile(
    r"\b(?:SFR|PMR|DAR|ECR|SER|TER|MPR|PER|QMR)-\s*\d+",
    re.I,
)

_UNANSWERABLE_QUESTION_TOKEN_PATTERN = re.compile(r"[가-힣]{3,}|[A-Za-z]{4,}")

_UNANSWERABLE_TOKEN_STOPWORDS = frozenset(
    {
        "무엇",
        "어떻게",
        "얼마",
        "있는",
        "있나",
        "있습",
        "되어",
        "대한",
        "관련",
        "문서",
        "확인",
        "기재",
        "명시",
        "제공",
        "해당",
        "경우",
        "여부",
        "내용",
        "항목",
        "질문",
        "답변",
        "가능",
    }
)


def corpus_text_blob(chunks: list[dict[str, Any]]) -> str:
    return "\n".join(str(chunk.get("text") or "") for chunk in chunks if isinstance(chunk, dict)).casefold()


def unanswerable_filter_reason(
    question: str,
    chunks: list[dict[str, Any]],
) -> str | None:
    """unanswerable 후보가 문서 주제와 겹치면 제외 사유를 반환. 통과 시 None."""
    q = str(question or "").strip()
    if not q:
        return "empty_question"
    q_cf = q.casefold()

    for phrase in UNANSWERABLE_FORBIDDEN_PHRASES:
        if phrase.casefold() in q_cf:
            return f"forbidden_phrase:{phrase}"

    if UNANSWERABLE_DOC_SCOPED_PATTERN.search(q):
        return "doc_scoped_wording"

    if UNANSWERABLE_REQUIREMENT_ID_PATTERN.search(q):
        return "requirement_id_in_question"

    blob = corpus_text_blob(chunks)
    if not blob:
        return None

    tokens = [
        t
        for t in _UNANSWERABLE_QUESTION_TOKEN_PATTERN.findall(q)
        if t.casefold() not in _UNANSWERABLE_TOKEN_STOPWORDS
    ]
    if len(tokens) >= 2:
        hits = sum(1 for t in tokens if t.casefold() in blob)
        if hits >= 2 or (len(tokens) <= 3 and hits == len(tokens)):
            return "corpus_keyword_overlap"

    return None


# 「중 하나」「모두 나열」 등은 expected 한 줄 채점과 맞지 않아 fact 단일 정답으로 유도한다.
OPEN_ENDED_QUESTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"중\s*하나", re.I),
    re.compile(r"하나(?:를|만)?\s*(?:적|쓰|골라|고르|말)", re.I),
    re.compile(r"모두\s*나열", re.I),
    re.compile(r"전부\s*나열", re.I),
    re.compile(r"(?:세|3)\s*가지(?:를)?\s*적", re.I),
    re.compile(r"몇\s*가지(?:를)?\s*적", re.I),
    re.compile(r"(?:예시|항목).{0,12}중\s*(?:하나|일부)", re.I),
)

_REQUIREMENT_ID_IN_QUESTION = re.compile(
    r"((?:SFR|PMR|DAR|ECR|SER|TER|MPR|PER|QMR|COR|SOR)-\s*\d+(?:-\d+)?)",
    re.I,
)


def open_ended_question_filter_reason(question: str) -> str | None:
    """열거형·다중 정답 질문이면 제외 사유 반환. 통과 시 None."""
    q = str(question or "").strip()
    if not q:
        return "empty_question"
    for pattern in OPEN_ENDED_QUESTION_PATTERNS:
        if pattern.search(q):
            return "open_ended_wording"
    return None


# img_001, OCR ID, chunk_id 등은 사용자 질문이 아니라 파이프라인 내부 식별자.
OCR_INTERNAL_ID_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"img_\d+", re.I),
    re.compile(r"image_stem", re.I),
    re.compile(r"ocr\s*id\s*[:=]", re.I),
    re.compile(r"ocr_chunk_", re.I),
    re.compile(r"slim_chunk_", re.I),
)


def ocr_internal_id_filter_reason(question: str) -> str | None:
    """OCR/청크 내부 식별자가 질문에 노출되면 제외 사유 반환. 통과 시 None."""
    q = str(question or "").strip()
    if not q:
        return "empty_question"
    for pattern in OCR_INTERNAL_ID_PATTERNS:
        if pattern.search(q):
            return "ocr_internal_id"
    return None


def rewrite_ocr_internal_id_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    """img_* 등 내부 ID를 질문에서 제거하고 표/별지 표현으로 치환."""
    question = str(row.get("question") or "").strip()
    if not ocr_internal_id_filter_reason(question):
        return row

    cleaned = question
    cleaned = re.sub(r"OCR\s*이미지\s*\(\s*img_\d+\s*\)", "표", cleaned, flags=re.I)
    cleaned = re.sub(r"이미지\s*\(\s*img_\d+\s*\)", "표", cleaned, flags=re.I)
    cleaned = re.sub(r"이미지\s*\(\s*OCR\s*\)", "표", cleaned, flags=re.I)
    cleaned = re.sub(r"이미지\s+img_\d+", "표", cleaned, flags=re.I)
    cleaned = re.sub(r"이미지\s+청크\s*\(\s*img_\d+\s*\)", "표", cleaned, flags=re.I)
    cleaned = re.sub(r"img_\d+", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,·")

    if not cleaned or ocr_internal_id_filter_reason(cleaned):
        return row

    out = dict(row)
    out["question"] = cleaned
    return out


def _first_expected_item(expected_answer: str) -> str:
    text = str(expected_answer or "").strip()
    if not text:
        return ""
    for sep in (";", " / ", " · ", " 및 "):
        if sep in text:
            text = text.split(sep, 1)[0]
    for sep in (",", "，"):
        if sep in text:
            text = text.split(sep, 1)[0]
    return text.strip(" .)")


def _count_cor_ids(keywords: list[str], expected_answer: str) -> int:
    cor_ids = {
        token
        for token in keywords
        if re.fullmatch(r"COR-\d+", str(token).strip(), flags=re.I)
    }
    if cor_ids:
        return len(cor_ids)
    found = _REQUIREMENT_ID_IN_QUESTION.findall(expected_answer)
    cor_ids = {item.upper().replace(" ", "") for item in found if item.upper().startswith("COR-")}
    return len(cor_ids)


def rewrite_open_ended_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    """기존 eval 행의 열거형 질문을 단일 정답 fact 형태로 변환."""
    question = str(row.get("question") or "").strip()
    if not open_ended_question_filter_reason(question):
        return row

    out = dict(row)
    expected = str(row.get("expected_answer") or "").strip()
    keywords = [str(k) for k in (row.get("ground_truth_keywords") or []) if str(k).strip()]
    gold_ids = [str(g) for g in (row.get("gold_chunk_ids") or []) if str(g).strip()]
    q_cf = question.casefold()

    if "모두 나열" in q_cf or "전부 나열" in q_cf:
        if "cor" in q_cf or any(k.upper().startswith("COR-") for k in keywords):
            count = _count_cor_ids(keywords, expected) or 4
            out.update(
                {
                    "question": "제약사항(COR) 고유번호는 해당 목록에 총 몇 개가 명시되어 있는가?",
                    "question_type": "fact",
                    "expected_answer": f"{count}개",
                    "ground_truth_keywords": keywords[:5] if keywords else [],
                    "gold_chunk_ids": gold_ids[:1],
                    "difficulty": "easy",
                }
            )
            return out

    if re.search(r"중\s*하나", question, re.I):
        item = _first_expected_item(expected)
        req_match = _REQUIREMENT_ID_IN_QUESTION.search(question)
        req_id = req_match.group(1) if req_match else "해당 요구사항"
        if item:
            out.update(
                {
                    "question": f"{req_id} 산출정보에 '{item}'가 명시되어 있는가?",
                    "question_type": "fact",
                    "expected_answer": f"예 ({item})",
                    "ground_truth_keywords": [w for w in [req_id.replace(" ", ""), item] if w],
                    "gold_chunk_ids": gold_ids[:1],
                    "difficulty": "easy",
                }
            )
            return out

    if re.search(r"(?:세|3)\s*가지(?:를)?\s*적", question, re.I):
        item = keywords[0] if keywords else _first_expected_item(expected)
        if item:
            out.update(
                {
                    "question": f"문서 산출정보(예시)에 '{item}'이 명시되어 있는가?",
                    "question_type": "fact",
                    "expected_answer": f"예 ({item})",
                    "ground_truth_keywords": [item],
                    "gold_chunk_ids": gold_ids[:1],
                    "difficulty": "easy",
                }
            )
            return out

    return row


SINGLE_ANSWER_GENERATION_RULES = """[단일 정답 — answerable 질문 공통]
- expected_answer는 **하나의 짧은 정답**(단어·숫자·한 문장)만 적으세요. 채점은 이 한 줄과 비교합니다.
- 금지 표현 (이런 질문은 만들지 말 것):
  - "중 하나", "하나만 적으세요", "예시 중", "일부", "몇 가지를 적"
  - "모두 나열", "전부 나열", "전체 ID 나열"
- 대신 fact / requirement_detail 로 **구체 항목 하나**만 묻세요.
- 좋은 예:
  - "DAR-004 산출정보에 '데이터 주제영역 정의서'가 명시되어 있는가?" → expected: "예 (데이터 주제영역 정의서)"
  - "COR-005의 제약 내용은?" → expected: (해당 COR-005 문장 요지)
  - "제약사항(COR) 고유번호는 목록에 총 몇 개인가?" → expected: "4개"
- summary 타입이라도 **열거·개수·단일 팩트**가 아니면 만들지 마세요.
  summary는 2~3개 bullet 요지를 expected에 적되, 질문은 "주요 목적을 요약"처럼 **고정 서술**만 허용합니다.
- comparison은 A·B 각각 **한 필드**만 묻거나, 차이/공통 **한 가지**만 묻세요."""


def normalize_eval_row_category(row: dict[str, Any]) -> dict[str, Any]:
    if not is_cover_form_metadata_question(str(row.get("question") or "")):
        return row
    out = dict(row)
    out["category"] = COVER_FORM_CATEGORY
    return out


COVER_FORM_CATEGORY_RULES = f"""[category — 표지·양식 메타 ({COVER_FORM_CATEGORY})]
- 질문이 아래 **표지/양식 메타**를 묻으면 category는 반드시 "{COVER_FORM_CATEGORY}" 로 지정하세요.
  (입찰·계약, 연락처 등 다른 category 사용 금지)
- 해당 주제: 담당자·사업책임자 **성명**, **연락처/전화번호**, **이메일**, **작성 연월·연도**, 표지/제안요청서(표지)/목차 상단 연월
- 좋은 예:
  - "제안서 표지의 작성 연월은?" → category: {COVER_FORM_CATEGORY}
  - "담당자 이메일은?" → category: {COVER_FORM_CATEGORY}
  - "사업책임자 성명은?" → category: {COVER_FORM_CATEGORY}
- 요구사항 ID·기능·예산·일정 본문은 {COVER_FORM_CATEGORY}가 아닙니다."""


UNANSWERABLE_GENERATION_RULES = """[unanswerable 전용 — 다른 모든 지침보다 우선]
- 질문 주제는 아래 "문서 청크"와 **완전히 무관**한 외부 사실만 다루세요.
- 이 문서·이 사업·본 제안·요구사항 번호(SFR/PMR 등)·총 사업비/계약금/사업기간/주관기관처럼
  RFP 본문에 자주 나오는 표현으로 질문하지 마세요. (검색에 걸려 거절 평가가 무의미해짐)
- 좋은 예 (주제가 문서와 분리됨):
  - "국토교통부 스마트시티 관련 국가 예산 규모는?" (본 HWP와 무관한 타 부처)
  - "AWS GovCloud의 FedRAMP 인증 범위는?" (본문에 없는 외부 클라우드)
  - "EU MDR 전체 인증 절차 단계는?" (본문에 조문 전문이 없을 때)
- 나쁜 예 (금지 → fact로 만들거나 unanswerable로 두지 말 것):
  - "이 사업의 총 사업비는?"
  - "요구사항 SFR-005의 명칭은?"
  - "주관기관은 어디인가?"
- expected_answer: "문서에서 확인되지 않음" (또는 동의어), gold_chunk_ids: []
"""


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
    # OCR 청크가 있으면 "검증용"으로만 소량 포함한다.
    # - 질문이 2개 이상이면 OCR 질문 1개만 요구 (나머지는 텍스트/표 기반)
    # - 질문이 1개면 OCR만 강제하지 않는다 (텍스트/표가 있으면 그쪽도 가능)
    # - OCR handoff 파일이 없거나 샘플에 이미지 청크가 없으면 이미지 질문 0개 (강제 없음)
    ocr_min = 1 if has_ocr and questions_per_doc >= 2 else 0
    ocr_max = 1 if has_ocr and questions_per_doc >= 2 else 0

    context = json.dumps(chunks, ensure_ascii=False, indent=2)
    chunk_context_note = ""
    if has_ocr:
        chunk_context_note = (
            f"\n\n[이미지 청크 안내] 이 문서에는 PaddleOCR 등으로 추출한 이미지 청크가 "
            f"{ocr_chunk_count}개 포함되어 있습니다(source=ocr, chunk_type=ocr_*). "
            f"이미지(OCR) 성능 확인용 질문은 {ocr_min}~{ocr_max}개만 포함하세요. "
            f"나머지 질문은 반드시 텍스트/표 청크(source!=ocr)에서 답할 수 있게 만드세요."
        )
    else:
        chunk_context_note = (
            "\n\n[이미지 청크 없음] 이 문서 컨텍스트에는 OCR/스캔 이미지 청크(source=ocr, "
            "chunk_type=ocr_*)가 없습니다. eval_focus=ocr_image 질문을 만들지 마세요. "
            "모든 질문은 eval_focus=text이며, gold_chunk_ids도 텍스트/표 청크만 가리켜야 합니다."
        )

    if has_ocr:
        eval_focus_field = (
            '- eval_focus: "text" | "ocr_image" '
            "(OCR/스캔 이미지 청크 근거면 ocr_image, 일반 본문·표 청크면 text)"
        )
        ocr_rules = f"""[이미지(OCR) 성능 검증 질문]
- eval_focus=ocr_image 질문을 {ocr_min}~{ocr_max}개만 포함.
- 질문은 이미지(스캔)에만 나오는 정보를 묻게 하세요. 예:
  - "별지 검토결과서에서 기재된 추정 사업기간은?"
  - "검토항목 중 '① 기능점수(FP) 기반 SW사업 적정 개발기간 산정표'의 추정 사업기간은?"
  - "검토항목 중 '추정 사업기간'이 5개월로 적힌 항목은?"
- 질문 본문에 img_001, img_004, image_stem, OCR ID, chunk_id, ocr_chunk_* 등
  **내부 식별자를 절대 쓰지 마세요.** (금지)
  사용자는 파일명을 모릅니다. **별지·양식명, 표 제목, 검토항목, 열 이름**으로만 물으세요.
- OCR 청크 text의 표/별지 **제목·검토항목·열 이름**(검토항목/검토의견/추정 사업기간)을 활용하세요.
  image_stem은 gold_chunk_ids 매칭용 메타일 뿐, 질문 문장에 노출하지 마세요.
- 텍스트/표 청크에 이미 동일 내용이 있으면, 그 질문은 eval_focus=text로 작성하고 gold_chunk_ids도 텍스트/표 청크를 가리키게 하세요.
"""
        ratio_image = f"- 이미지(eval_focus=ocr_image): {ocr_min}~{ocr_max}개\n"
    else:
        eval_focus_field = (
            '- eval_focus: 반드시 "text"만 사용 '
            "(이 문서 컨텍스트에 OCR/이미지 청크가 없음 — ocr_image 금지)"
        )
        ocr_rules = """[이미지(OCR) 청크 없음 — 예외]
- 제공된 청크 목록에 source=ocr 또는 chunk_type=ocr_* 가 없습니다.
- eval_focus=ocr_image 질문을 0개로 두세요 (생성하지 마세요).
- 이미지·스캔·별지 OCR 등을 묻는 질문도 만들지 마세요.
"""
        ratio_image = ""

    unanswerable_rules = UNANSWERABLE_GENERATION_RULES
    single_answer_rules = SINGLE_ANSWER_GENERATION_RULES
    cover_form_rules = COVER_FORM_CATEGORY_RULES

    return f"""당신은 RAG 성능평가용 질문셋 생성자입니다.
아래 문서 청크만 근거로 평가 질문을 생성하세요. (다른 문서·외부 지식·추측 금지)

문서 ID (doc_id는 반드시 이 값과 동일):
{doc_id}

문서 청크:
{context}
{chunk_context_note}

총 {questions_per_doc}개의 질문을 생성하세요.
출력은 반드시 {{"questions": [...]}} 형태의 JSON 객체로만 작성하세요.

각 항목 필수 필드:
- question, category, question_type, doc_id, expected_answer, ground_truth_keywords, difficulty, gold_chunk_ids
- {eval_focus_field}

[엄격 규칙 — 위반 시 무효]
1. gold_chunk_ids: answerable 질문은 위 청크 목록의 chunk_id만, 최소 1개. unanswerable은 반드시 [] (빈 배열).
2. expected_answer의 핵심 수치·고유명사는 gold_chunk_ids가 가리키는 청크 text에 실제로 있어야 함 (unanswerable 제외).
3. ground_truth_keywords는 해당 청크 text에 그대로 등장하는 단어·숫자만 (2~5개).
4. comparison은 A·B 모두가 청크에 있을 때만. 없으면 fact/summary로 바꾸거나 unanswerable.
5. unanswerable은 "이 문서 청크 어디에도 없는" **외부 주제**만. 타 기관·타 사업·미언급 법령·타 제품 등.
6. doc_id는 항상 "{doc_id}".

{single_answer_rules}

{cover_form_rules}

{unanswerable_rules}

[평가 관점 — question_type]
- fact, requirement_detail: 단일 사실·요구사항
- summary: 여러 청크 종합
- follow_up: 같은 문서 안 선행 맥락 1문장 전제
- comparison: 동일 문서 내 두 항목 대조 (근거 둘 다 있을 때만)
- unanswerable: 제공 청크와 주제가 분리됨 → expected_answer는 "문서에서 확인되지 않음", gold_chunk_ids는 []

{ocr_rules}
category 예: 기능 요구사항, 보안, 일정, 예산, 입찰·계약, 부록·양식{", 이미지" if has_ocr else ""}.
expected_answer: 정답 전문이 아닌 검수용 핵심 요지(짧게).
difficulty: easy(단일 팩트), medium(요약·세부), hard(종합·후속·unanswerable).

[생성 비율 가이드{"" if has_ocr else " — 텍스트/표만"}]
{ratio_image}- fact·requirement_detail: 약 35%
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
    dropped_unanswerable = 0
    dropped_open_ended = 0
    dropped_ocr_internal_id = 0
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
        has_ocr_in_context = any(
            str(chunk.get("source") or "") == "ocr"
            for chunk in chunks_for_row
            if isinstance(chunk, dict)
        )
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
                reject = unanswerable_filter_reason(
                    str(question.get("question") or ""),
                    chunks_for_row,
                )
                if reject:
                    dropped_unanswerable += 1
                    continue
                question["gold_chunk_ids"] = []
            elif not gold_ids:
                continue
            else:
                question["gold_chunk_ids"] = gold_ids
                reject_open = open_ended_question_filter_reason(str(question.get("question") or ""))
                if reject_open:
                    dropped_open_ended += 1
                    continue
                reject_ocr_id = ocr_internal_id_filter_reason(str(question.get("question") or ""))
                if reject_ocr_id:
                    dropped_ocr_internal_id += 1
                    continue
            if not question.get("eval_focus"):
                if is_unanswerable:
                    question["eval_focus"] = "text"
                elif has_ocr_in_context and any(
                    str(chunk_by_id[cid].get("source") or "") == "ocr" for cid in gold_ids
                ):
                    question["eval_focus"] = "ocr_image"
                else:
                    question["eval_focus"] = "text"
            elif not has_ocr_in_context and question.get("eval_focus") == "ocr_image":
                question["eval_focus"] = "text"
            question = normalize_eval_row_category(question)
            output_rows.append(question)

    write_jsonl(output_path, output_rows)
    print(f"wrote_eval_questions: {output_path}")
    print(f"questions: {len(output_rows)}")
    if dropped_unanswerable:
        print(f"dropped_unanswerable (topic overlap filter): {dropped_unanswerable}")
    if dropped_open_ended:
        print(f"dropped_open_ended (single-answer filter): {dropped_open_ended}")
    if dropped_ocr_internal_id:
        print(f"dropped_ocr_internal_id (img_* / chunk id filter): {dropped_ocr_internal_id}")


def rewrite_eval_questions_jsonl(path: Path) -> tuple[int, int, int, int]:
    """JSONL eval 질문셋 정규화(열거형→단일 정답, OCR img_* 제거, 표지·양식 category)."""
    rows = read_jsonl(path)
    if not rows:
        return 0, 0, 0, 0
    open_ended_rewritten = 0
    ocr_id_rewritten = 0
    category_rewritten = 0
    output: list[dict[str, Any]] = []
    for row in rows:
        updated = rewrite_open_ended_eval_row(row)
        if updated != row:
            open_ended_rewritten += 1
        before_ocr = updated.get("question")
        updated = rewrite_ocr_internal_id_eval_row(updated)
        if updated.get("question") != before_ocr:
            ocr_id_rewritten += 1
        before_category = updated.get("category")
        updated = normalize_eval_row_category(updated)
        if updated.get("category") != before_category:
            category_rewritten += 1
        output.append(updated)
    write_jsonl(path, output)
    return len(output), open_ended_rewritten, ocr_id_rewritten, category_rewritten


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
    parser.add_argument(
        "--rewrite-open-ended",
        action="store_true",
        help="eval JSONL 정규화(열거형→단일 정답, img_* 제거, 표지·양식 category) (--eval-output 경로 사용)",
    )
    args = parser.parse_args()

    load_dotenv()

    if args.rewrite_open_ended:
        target = Path(args.eval_output)
        total, open_ended_rewritten, ocr_id_rewritten, category_rewritten = rewrite_eval_questions_jsonl(
            target
        )
        print(f"rewrote_eval_questions: {target}")
        print(
            f"rows: {total}, open_ended_rewritten: {open_ended_rewritten}, "
            f"ocr_id_rewritten: {ocr_id_rewritten}, category_rewritten: {category_rewritten}"
        )
        return

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
