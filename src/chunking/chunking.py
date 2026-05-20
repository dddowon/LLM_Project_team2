#!/usr/bin/env python3
"""Create slim RAG chunks from an already parsed HWP prechunk JSONL.

Standalone chunking-only script. It does not import other local project .py files.

Input:
  - prechunk JSONL produced by hwp_parse_prechunk_only.py

Output:
  - one slim chunk JSONL only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("eda/hwp_prechunk_all.jsonl")
DEFAULT_OUTPUT = Path("eda/hwp_text_chunks_slim.jsonl")

ROMAN_MAJOR_RE = re.compile(r"^(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+|I{1,3}|IV|V|VI{0,3}|IX|X)[.)]?\s+")
NUMBER_DOT_RE = re.compile(r"^\d{1,2}(?:\.\d{1,2})*\.\s*")
NUMBER_PAREN_RE = re.compile(r"^\d{1,2}\)\s*")
KOREAN_DOT_RE = re.compile(r"^[가-힣]\.\s*")
KOREAN_PAREN_RE = re.compile(r"^[가-힣]\)\s*")
CIRCLED_RE = re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]\s*")
BULLET_RE = re.compile(r"^[□○◦ㅇㆍ\-]\s*")

FIELD_LIKE_RE = re.compile(
    r"(발주자명|대표자|대표\s*자|성명|주소|전화|팩스|이메일|사업자등록|"
    r"주민등록|생년월일|날인|서명|인감|담당자|신청인|업체명|회사명|"
    r"기관명|소속|직위|연락처|작성자|제출자)"
)
FORM_PLACEHOLDER_RE = re.compile(r"(○{2,}|OOO|%|점|금\s*원)")
APPENDIX_RE = re.compile(r"(붙임|별지|서식|양식|첨부)", re.I)
VALUE_LINE_RE = re.compile(r"[:：]\s*\S{2,}")
SENTENCE_LIKE_RE = re.compile(r"(다|함|것|경우|여부|있는지|한다|하겠습니다|하여야\s*한다)[.)]?$")
NUMBERED_SENTENCE_CUE_RE = re.compile(
    r"(경우|경우에는|여부|있는지|란에|표기|기재|작성|제출|누설|위반|처벌|"
    r"제기하지|않겠|않음|하여야|하도록|되도록|따라|의거|위한|대한|관한|"
    r"발생|제시|명시|포함|기준으로|평가하|활용|첨부)"
)
CONTINUATION_BODY_RE = re.compile(r"^\s*[②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]\s*")
LEGAL_CONTINUATION_HEADING_RE = re.compile(r"(확정|변경|계약금액|계약기간|심의|시행령|법률|조정)")
REQUIREMENT_HEADING_RE = re.compile(
    r"(요구사항|Requirement|SFR|SER|DAR|DIR|SIR|PER|QUR|COR|PSR|PMR|MPR|MHR|ECR|CSR|SDR)",
    re.I,
)
CODE_COLON_RE = re.compile(r"\([A-Z]{2,5}\s*:\s*[A-Za-z][A-Za-z\s/.-]{2,}\)|[:：]\s*[A-Z]{2,5}\s*$")
INLINE_VALUE_FIELD_RE = re.compile(
    r"(사업\s*기간|사업\s*금액|사업\s*예산|소요\s*예산|계약\s*방법|입찰\s*방법|"
    r"입찰\s*보증\s*금액|입찰보증금액|보증\s*금액|보\s*증\s*액|"
    r"계약방법|입찰방식|입찰방법|제출\s*장소|제출장소|공고\s*및\s*접수기간|접수\s*기간|접수기간|"
    r"제안서\s*접수|제안서\s*심사|계약협상일시|계약협상|발주기관명|발주자명|대\s*표\s*자|"
    r"주소|계좌|보험|기간|준공일|착수일|계약명|사업명|사\s*업\s*명|용역명|용\s*역\s*명|"
    r"사업예산|금액|예산|전화|팩스|이메일)\s*[:：]"
)
INLINE_VALUE_PAYLOAD_RE = re.compile(
    r"[:：]\s*[^()]{0,80}(?:\d{2,}|원|개월|일간|은행|계좌|대표자|소재지|착수|계약|제한경쟁|협상)"
)
COLON_VALUE_PAYLOAD_RE = re.compile(
    r"[:：].{0,140}(?:원|개월|일간|은행|계좌|대표자|소재지|착수|계약|제한경쟁|협상|"
    r"종합평가|등록|제출|점|%)"
)
SUBITEM_SENTENCE_CUE_RE = re.compile(
    r"(공개하지|않는다|하여야|작성하여야|기재하여야|포함시켜야|유의하여|"
    r"거부할 수|요청하지 않는 한|적용할 수|가능하다|필요|있음|없음|"
    r"누설|위반|비밀|외부에 공개|자료 활용|제출된|발주기관이|발견될|"
    r"입찰참가 제외|낙찰취소|계약해지|경우에는|법령|고시|지침|"
    r"법률시행령|제\d+조|품질 확보|사업 완료|지출금액|위약금|기성대금)"
)

TEXT_BOUNDARY_RE = re.compile(
    r"\n\s*(?=(?:\d{1,2}(?:\.\d{1,2})*[.)]\s|[가-힣][.)]\s|[①-⑳]|[○●□■※ㆍ\-]))"
)
SPACE_RE = re.compile(r"[ \t\u00a0]+")


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


def default_summary_output(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_summary.csv")


def default_sample_output(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_sample.jsonl")


def stable_hash(value: str, *, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def clean_text_block(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def path_text(path_items: list[str] | tuple[str, ...]) -> str:
    return " > ".join(str(item).strip() for item in path_items if str(item).strip())


def section_type(path_items: list[str]) -> str:
    text = " ".join(path_items)
    if re.search(r"사업\s*개요|사업안내|목표|추진\s*배경", text):
        return "overview"
    if re.search(r"과업|요구사항|제안\s*요청|수행\s*범위", text):
        return "requirements"
    if re.search(r"보안|개인정보|접근권한|암호화", text):
        return "security"
    if re.search(r"평가|배점|협상", text):
        return "evaluation"
    if re.search(r"입찰|계약|공모|제출", text):
        return "bid_contract"
    if re.search(r"붙임|서식|양식", text):
        return "appendix_form"
    return "body"


def record_section_path(record: dict[str, Any]) -> tuple[str, ...]:
    value = record.get("section_path") or []
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    return (text,) if text else tuple()


def classify_heading_pattern(heading: str) -> str:
    heading = re.sub(r"\s+", " ", heading).strip()
    if ROMAN_MAJOR_RE.search(heading):
        return "roman_major"
    if NUMBER_DOT_RE.search(heading):
        return "number_dot"
    if NUMBER_PAREN_RE.search(heading):
        return "number_paren"
    if KOREAN_DOT_RE.search(heading):
        return "korean_dot"
    if KOREAN_PAREN_RE.search(heading):
        return "korean_paren"
    if CIRCLED_RE.search(heading):
        return "circled_number"
    if BULLET_RE.search(heading):
        return "bullet_marker"
    if heading:
        return "bare_title"
    return "empty"


def is_noise_heading(heading: str, full_path: tuple[str, ...], body_text: str = "") -> bool:
    compact = re.sub(r"\s+", " ", heading).strip()
    full_text = path_text(full_path)
    raw_requirement_heading = bool(REQUIREMENT_HEADING_RE.search(compact))
    requirement_sentence_like = (
        len(compact) > 45
        and (
            SENTENCE_LIKE_RE.search(compact)
            or SUBITEM_SENTENCE_CUE_RE.search(compact)
            or NUMBERED_SENTENCE_CUE_RE.search(compact)
        )
    )
    is_requirement_heading = raw_requirement_heading and not requirement_sentence_like
    is_code_colon = bool(CODE_COLON_RE.search(compact))
    pattern = classify_heading_pattern(compact)
    if not compact:
        return True
    if len(compact) > 105 and not is_requirement_heading:
        return True
    if pattern in {"number_paren", "korean_dot", "korean_paren"} and len(compact) > 70 and not is_requirement_heading:
        return True
    if pattern == "number_dot" and not is_requirement_heading:
        if (
            len(compact) > 25
            and CONTINUATION_BODY_RE.search(body_text)
            and LEGAL_CONTINUATION_HEADING_RE.search(compact)
        ):
            return True
        if len(compact) > 30 and (SENTENCE_LIKE_RE.search(compact) or SUBITEM_SENTENCE_CUE_RE.search(compact)):
            return True
        if len(compact) > 70:
            return True
        if len(compact) > 35 and NUMBERED_SENTENCE_CUE_RE.search(compact):
            return True
        if re.search(r"\d{1,2}\.\s+.+\d{1,2}\.\s+", compact):
            return True
    if pattern in {"circled_number", "bullet_marker"} and len(compact) > 60 and not is_requirement_heading:
        return True
    if re.search(r"[:：]\s*$", compact):
        return True
    if FIELD_LIKE_RE.search(compact) and not is_requirement_heading:
        return True
    if (
        (":" in compact or "：" in compact)
        and not is_requirement_heading
        and not is_code_colon
        and pattern in {"number_dot", "number_paren", "korean_dot", "korean_paren", "circled_number"}
        and len(compact) > 25
    ):
        return True
    if FORM_PLACEHOLDER_RE.search(compact) and (":" in compact or "：" in compact or APPENDIX_RE.search(full_text)):
        return True
    if INLINE_VALUE_FIELD_RE.search(compact) and not is_requirement_heading:
        return True
    if (
        VALUE_LINE_RE.search(compact)
        and not ROMAN_MAJOR_RE.search(compact)
        and not is_code_colon
        and (
            INLINE_VALUE_FIELD_RE.search(compact)
            or INLINE_VALUE_PAYLOAD_RE.search(compact)
            or COLON_VALUE_PAYLOAD_RE.search(compact)
            or FORM_PLACEHOLDER_RE.search(compact)
            or FIELD_LIKE_RE.search(compact)
        )
    ):
        return True
    if len(compact) > 45 and SENTENCE_LIKE_RE.search(compact) and not is_requirement_heading:
        return True
    if (
        pattern in {"number_paren", "korean_dot", "korean_paren"}
        and len(compact) > 30
        and not is_requirement_heading
        and (
            SENTENCE_LIKE_RE.search(compact)
            or SUBITEM_SENTENCE_CUE_RE.search(compact)
            or re.search(r"(으로|하며|하여|통해|대한|위한|중심의|따른|되도록|하도록)", compact)
        )
    ):
        return True
    if (
        APPENDIX_RE.search(full_text)
        and not is_requirement_heading
        and (
            len(compact) > 60
            or FIELD_LIKE_RE.search(compact)
            or (pattern == "number_dot" and len(compact) > 28 and NUMBERED_SENTENCE_CUE_RE.search(compact))
        )
    ):
        return True
    return False


def unique_nodes_by_document(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    nodes_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record_index, record in enumerate(records):
        file_name = str(record.get("file_name") or "").strip()
        path_items = record_section_path(record)
        if not file_name or not path_items:
            continue
        for depth in range(1, len(path_items) + 1):
            prefix = path_items[:depth]
            key = (file_name, prefix)
            if key in seen:
                continue
            seen.add(key)
            heading = prefix[-1]
            nodes_by_doc[file_name].append(
                {
                    "file_name": file_name,
                    "depth": depth,
                    "section_path": path_text(prefix),
                    "heading": heading,
                    "pattern": classify_heading_pattern(heading),
                    "is_noise": is_noise_heading(heading, prefix, str(record.get("text") or "")),
                    "first_record_index": record_index,
                }
            )
    return nodes_by_doc


def dominant_patterns(pattern_counts: Counter[str], *, min_share: float = 0.15) -> list[str]:
    total = sum(pattern_counts.values())
    if total == 0:
        return []
    selected: list[str] = []
    for pattern, count in pattern_counts.most_common():
        if count >= 2 or count / total >= min_share:
            selected.append(pattern)
    return selected


def allowed_patterns_by_doc(nodes_by_doc: dict[str, list[dict[str, Any]]]) -> dict[str, dict[int, set[str]]]:
    result: dict[str, dict[int, set[str]]] = {}
    for file_name, nodes in nodes_by_doc.items():
        depth_counts: dict[int, Counter[str]] = defaultdict(Counter)
        for node in nodes:
            if node.get("is_noise"):
                continue
            depth_counts[int(node["depth"])][str(node["pattern"])] += 1
        result[file_name] = {
            depth: set(dominant_patterns(counts) or [counts.most_common(1)[0][0]])
            for depth, counts in depth_counts.items()
            if counts
        }
    return result


def clean_section_path(
    raw_path: list[str],
    *,
    file_name: str,
    body_text: str = "",
    allowed_patterns: dict[str, dict[int, set[str]]],
    strict_patterns: bool,
) -> tuple[list[str], list[dict[str, str]]]:
    cleaned: list[str] = []
    dropped: list[dict[str, str]] = []
    doc_allowed = allowed_patterns.get(file_name, {})
    for index, heading in enumerate(raw_path, start=1):
        heading = str(heading).strip()
        if not heading:
            continue
        prefix = tuple(raw_path[:index])
        pattern = classify_heading_pattern(heading)
        reasons: list[str] = []
        heading_body_text = body_text if index == len(raw_path) else ""
        if is_noise_heading(heading, prefix, heading_body_text):
            reasons.append("noise_rule")
        allowed_at_depth = doc_allowed.get(index)
        if strict_patterns and allowed_at_depth and pattern not in allowed_at_depth:
            reasons.append(f"pattern_outlier:{pattern}")
        if reasons:
            dropped.append({"heading": heading, "reason": ",".join(reasons), "pattern": pattern})
        else:
            cleaned.append(heading)
    return cleaned, dropped


def heading_line_candidate(line: str) -> str:
    line = clean_text_block(line)
    if not line or len(line) > 110:
        return ""
    pattern = classify_heading_pattern(line)
    if pattern in {"roman_major", "number_dot", "number_paren", "korean_dot", "korean_paren", "circled_number", "bare_title"}:
        return pattern
    return ""


def choose_inline_heading_depth(
    *,
    pattern: str,
    base_depth: int,
    current_path: list[str],
    doc_allowed: dict[int, set[str]],
) -> int | None:
    current_depth = len(current_path)
    current_pattern = classify_heading_pattern(current_path[-1]) if current_path else ""
    if current_depth >= base_depth + 1:
        child_allowed = doc_allowed.get(current_depth + 1, set())
        same_allowed = doc_allowed.get(current_depth, set())
        if pattern in child_allowed and pattern != current_pattern:
            return current_depth + 1
        if pattern in same_allowed:
            return current_depth
        if pattern in child_allowed:
            return current_depth + 1
    for depth in range(base_depth + 1, max(doc_allowed.keys(), default=base_depth) + 1):
        if pattern in doc_allowed.get(depth, set()):
            return depth
    return None


def split_record_by_inline_headings(
    *,
    file_name: str,
    base_path: list[str],
    text: str,
    allowed_patterns: dict[str, dict[int, set[str]]],
    enable: bool,
) -> list[dict[str, Any]]:
    if not enable or not text.strip():
        return [{"cleaned_section_path": base_path, "text": text}]
    doc_allowed = allowed_patterns.get(file_name, {})
    if not doc_allowed:
        return [{"cleaned_section_path": base_path, "text": text}]

    base_depth = len(base_path)
    current_path = list(base_path)
    buffer: list[str] = []
    segments: list[dict[str, Any]] = []
    lines = text.splitlines()

    def flush() -> None:
        body = clean_text_block("\n".join(buffer))
        if body:
            segments.append({"cleaned_section_path": list(current_path), "text": body})
        buffer.clear()

    def next_non_empty_pattern(start_index: int) -> str:
        for candidate in lines[start_index + 1 :]:
            candidate = candidate.strip()
            if candidate:
                return heading_line_candidate(candidate)
        return ""

    for line_index, raw_line in enumerate(lines):
        line = raw_line.strip()
        pattern = heading_line_candidate(line)
        target_depth: int | None = None
        if pattern:
            prospective_prefix = tuple([*current_path[: max(0, len(current_path))], line])
            next_pattern = next_non_empty_pattern(line_index)
            looks_like_plain_numbered_list = pattern == "number_paren" and next_pattern == "number_paren"
            if not looks_like_plain_numbered_list and not is_noise_heading(line, prospective_prefix):
                target_depth = choose_inline_heading_depth(
                    pattern=pattern,
                    base_depth=base_depth,
                    current_path=current_path,
                    doc_allowed=doc_allowed,
                )
        if target_depth is not None and target_depth > base_depth:
            flush()
            keep_len = max(base_depth, target_depth - 1)
            current_path = [*current_path[:keep_len], line]
            continue
        buffer.append(raw_line)

    flush()
    if not segments:
        segments.append({"cleaned_section_path": base_path, "text": text})
    return segments


def build_clean_prechunk_records(
    records: list[dict[str, Any]],
    *,
    strict_patterns: bool,
    include_cover: bool,
    split_inline_headings: bool,
) -> list[dict[str, Any]]:
    nodes_by_doc = unique_nodes_by_document(records)
    allowed_patterns = allowed_patterns_by_doc(nodes_by_doc)
    output: list[dict[str, Any]] = []
    emitted_dropped_heading_keys: set[tuple[str, tuple[str, ...], str]] = set()
    last_cleaned_path_by_file: dict[str, list[str]] = {}

    for record_index, record in enumerate(records):
        content_type = str(record.get("content_type") or "")
        if content_type not in {"section_text", "cover_text"}:
            continue
        if content_type == "cover_text" and not include_cover:
            continue

        file_name = str(record.get("file_name") or "")
        raw_path = list(record_section_path(record))
        text = clean_text_block(record.get("text") or "")
        if content_type == "cover_text":
            cleaned_path: list[str] = []
            dropped: list[dict[str, str]] = []
        else:
            cleaned_path, dropped = clean_section_path(
                raw_path,
                file_name=file_name,
                body_text=text,
                allowed_patterns=allowed_patterns,
                strict_patterns=strict_patterns,
            )

        if (
            content_type == "section_text"
            and raw_path
            and dropped
            and text
            and re.match(r"^\s*[②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]\s*", text)
            and any(item.get("heading") == raw_path[-1] for item in dropped)
        ):
            previous_path = last_cleaned_path_by_file.get(file_name)
            if previous_path and previous_path[: len(cleaned_path)] == cleaned_path and len(previous_path) > len(cleaned_path):
                cleaned_path = list(previous_path)

        dropped_heading_lines: list[str] = []
        for item in dropped:
            dropped_key = (file_name, tuple(cleaned_path), item["heading"])
            if dropped_key in emitted_dropped_heading_keys:
                continue
            emitted_dropped_heading_keys.add(dropped_key)
            dropped_heading_lines.append(item["heading"])
        if dropped_heading_lines:
            text = clean_text_block("\n".join(dropped_heading_lines) + "\n" + text)
        if not text:
            continue

        segments = (
            [{"cleaned_section_path": cleaned_path, "text": text}]
            if content_type == "cover_text"
            else split_record_by_inline_headings(
                file_name=file_name,
                base_path=cleaned_path,
                text=text,
                allowed_patterns=allowed_patterns,
                enable=split_inline_headings,
            )
        )

        for segment_index, segment in enumerate(segments, start=1):
            segment_path = list(segment["cleaned_section_path"])
            segment_text = clean_text_block(segment["text"])
            if not segment_text:
                continue
            output.append(
                {
                    "file_name": file_name,
                    "content_type": content_type,
                    "raw_section_path": raw_path,
                    "cleaned_section_path": segment_path,
                    "section_path": segment_path,
                    "section_type": "cover" if content_type == "cover_text" else section_type(segment_path),
                    "heading": segment_path[-1] if segment_path else "",
                    "text": segment_text,
                    "source_record_index": record_index,
                    "source_segment_index": segment_index,
                }
            )
            if segment_path:
                last_cleaned_path_by_file[file_name] = list(segment_path)
    return output


def flush_group(group: list[dict[str, Any]], grouped: list[dict[str, Any]]) -> None:
    if not group:
        return
    first = group[0]
    texts: list[str] = []
    source_indices: list[int] = []
    for record in group:
        text = clean_text_block(record.get("text") or "")
        if text:
            texts.append(text)
        source_indices.append(int(record.get("source_record_index", -1)))
    text = clean_text_block("\n\n".join(texts))
    if text:
        grouped.append(
            {
                "file_name": first["file_name"],
                "content_type": first["content_type"],
                "cleaned_section_path": first["cleaned_section_path"],
                "section_type": first["section_type"],
                "heading": first["heading"],
                "source_record_indices": source_indices,
                "text": text,
            }
        )


def group_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_key: tuple[Any, ...] | None = None
    for record in records:
        key = (record.get("file_name"), record.get("content_type"), tuple(record.get("cleaned_section_path") or []))
        if current_key is None or key == current_key:
            current.append(record)
            current_key = key
        else:
            flush_group(current, grouped)
            current = [record]
            current_key = key
    flush_group(current, grouped)
    return grouped


def split_long_unit(unit: str, chunk_size: int) -> list[str]:
    unit = unit.strip()
    if len(unit) <= chunk_size:
        return [unit] if unit else []
    parts: list[str] = []
    start = 0
    while start < len(unit):
        parts.append(unit[start : start + chunk_size].strip())
        start += chunk_size
    return [part for part in parts if part]


def normalize_inline(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u00a0", " ").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", text)
    return SPACE_RE.sub(" ", text).strip()


def compact_key(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def text_units(text: str) -> list[str]:
    units: list[str] = []
    for paragraph in re.split(r"\n{2,}", clean_text_block(text)):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        parts = [part.strip() for part in TEXT_BOUNDARY_RE.split(paragraph) if part.strip()]
        units.extend(parts or [paragraph])
    return units


def split_oversized_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    text = clean_text_block(text)
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end), text.rfind("다. ", start, end))
            if boundary > start + int(chunk_size * 0.55):
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def split_section_text(text: str, *, chunk_size: int, overlap: int, min_chars: int) -> list[str]:
    units = text_units(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        joined = clean_text_block("\n\n".join(current))
        current = []
        current_len = 0
        if not joined:
            return
        if len(joined) > chunk_size:
            chunks.extend(split_oversized_text(joined, chunk_size=chunk_size, overlap=overlap))
        else:
            chunks.append(joined)

    for unit in units:
        unit_len = len(unit)
        separator_len = 2 if current else 0
        if current and current_len + separator_len + unit_len > chunk_size:
            flush()
        if unit_len > chunk_size:
            flush()
            chunks.extend(split_oversized_text(unit, chunk_size=chunk_size, overlap=overlap))
        else:
            current.append(unit)
            current_len += separator_len + unit_len
    flush()

    if len(chunks) >= 2 and len(chunks[-1]) < min_chars:
        chunks[-2] = clean_text_block(chunks[-2] + "\n\n" + chunks[-1])
        chunks.pop()

    overlapped: list[str] = []
    for index, chunk in enumerate(chunks):
        if index == 0 or overlap <= 0:
            overlapped.append(chunk)
            continue
        tail = chunks[index - 1][-overlap:].strip()
        if tail:
            overlapped.append(clean_text_block(tail + "\n\n" + chunk))
        else:
            overlapped.append(chunk)
    return [chunk for chunk in overlapped if chunk]


def is_low_value_text(body: str, *, content_type: str) -> bool:
    compact = compact_key(body)
    if not compact:
        return True
    if compact in {"목차", "차례", "tableofcontents"}:
        return True
    if content_type == "cover_text" and len(compact) <= 30 and ("목차" in compact or "차례" in compact):
        return True
    return False


def base_metadata(record: dict[str, Any]) -> dict[str, Any]:
    section_path = record.get("section_path") or []
    metadata = {
        "file_name": record.get("file_name", ""),
        "source_record_index": record.get("_source_record_index"),
        "source_content_type": record.get("content_type", ""),
        "section_path": section_path,
        "section_path_text": path_text(section_path),
        "section_type": record.get("section_type", ""),
        "heading": record.get("heading", ""),
    }
    return {key: value for key, value in metadata.items() if value not in ("", None, [])}


def chunk_type_label(chunk_type: str) -> str:
    if chunk_type == "section_text":
        return "본문"
    if chunk_type == "cover_text":
        return "표지"
    if chunk_type.startswith("table_"):
        return "표"
    return chunk_type


def rag_chunk_context_prefix(
    metadata: dict[str, Any], *, chunk_type: str, extra_lines: list[str] | None = None
) -> str:
    lines = [f"문서명: {metadata.get('file_name', '')}"]
    section = metadata.get("section_path_text")
    if section:
        lines.append(f"섹션경로: {section}")
    heading = metadata.get("heading")
    if heading and heading != section:
        lines.append(f"제목: {heading}")
    lines.append(f"자료유형: {chunk_type_label(chunk_type)}")
    if extra_lines:
        lines.extend(line for line in extra_lines if normalize_inline(line))
    return "\n".join(lines)


def make_chunk(
    chunk_no: int,
    *,
    chunk_type: str,
    body: str,
    metadata: dict[str, Any],
    extra_prefix_lines: list[str] | None = None,
) -> dict[str, Any]:
    clean_body = clean_text_block(body)
    prefix = rag_chunk_context_prefix(metadata, chunk_type=chunk_type, extra_lines=extra_prefix_lines)
    chunk_text = clean_text_block(prefix + "\n\n" + clean_body)
    content_hash = stable_hash(chunk_text)
    metadata = dict(metadata)
    metadata.update(
        {
            "chunk_type": chunk_type,
            "body_chars": len(clean_body),
            "chunk_chars": len(chunk_text),
            "content_hash": content_hash,
        }
    )
    return {
        "chunk_id": f"chunk_{chunk_no:08d}_{content_hash}",
        "chunk_type": chunk_type,
        "chunk_text": chunk_text,
        "metadata": metadata,
    }


def cells_to_lines(cells: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    seen_values: set[str] = set()
    for key, value in cells.items():
        clean_key = normalize_inline(key)
        clean_value = normalize_inline(value)
        if not clean_value or clean_value == clean_key:
            continue

        value_key = compact_key(clean_value)
        if clean_key.startswith("col_") and value_key in seen_values:
            continue
        seen_values.add(value_key)

        if clean_key.startswith("col_"):
            lines.append(clean_value)
        else:
            lines.append(f"{clean_key}: {clean_value}")
    return lines


def table_row_text(row: dict[str, Any]) -> str:
    cells = row.get("cells")
    if isinstance(cells, dict) and cells:
        lines = cells_to_lines(cells)
        if lines:
            return "\n".join(lines)
    return clean_text_block(row.get("text", ""))


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        body = table_row_text(row)
        key = compact_key(body)
        if not key or key in seen:
            continue
        seen.add(key)
        copied = dict(row)
        copied["_rag_row_text"] = body
        deduped.append(copied)
    return deduped


def row_group_label(rows: list[dict[str, Any]]) -> str:
    indices = [str(row.get("row_index")) for row in rows if row.get("row_index") is not None]
    if not indices:
        return ""
    if len(indices) == 1:
        return indices[0]
    return f"{indices[0]}-{indices[-1]}"


def group_table_rows(
    rows: list[dict[str, Any]], *, table_chunk_size: int, max_rows_per_chunk: int
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: list[tuple[str, list[dict[str, Any]]]] = []
    current_rows: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_len = 0

    def row_part(row: dict[str, Any]) -> str:
        label = row.get("row_index")
        prefix = f"행 {label}\n" if label is not None else ""
        return clean_text_block(prefix + str(row["_rag_row_text"]))

    def flush() -> None:
        nonlocal current_rows, current_parts, current_len
        if current_parts:
            grouped.append(("\n\n".join(current_parts), current_rows))
        current_rows = []
        current_parts = []
        current_len = 0

    for row in rows:
        part = row_part(row)
        if not part:
            continue
        part_len = len(part)
        if current_parts and (
            current_len + 2 + part_len > table_chunk_size or len(current_rows) >= max_rows_per_chunk
        ):
            flush()
        if part_len > table_chunk_size:
            flush()
            for piece in split_oversized_text(part, chunk_size=table_chunk_size, overlap=80):
                single = dict(row)
                single["_rag_row_text"] = piece
                grouped.append((piece, [single]))
            continue
        current_rows.append(row)
        current_parts.append(part)
        current_len += (2 if current_parts else 0) + part_len
    flush()
    return grouped


def table_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = base_metadata(record)
    table = record.get("table") or {}
    metadata.update(
        {
            "table_id": record.get("table_id", ""),
            "table_type": record.get("table_type", ""),
            "table_shape": table.get("shape"),
            "table_rows": table.get("rows"),
            "table_cols": table.get("cols"),
            "table_cell_count": table.get("cell_count"),
        }
    )
    if table.get("nested_table_count") is not None:
        metadata["nested_table_count"] = table.get("nested_table_count")
    return {key: value for key, value in metadata.items() if value not in ("", None, [])}


def chunks_from_table(
    record: dict[str, Any],
    *,
    next_chunk_no: int,
    table_chunk_size: int,
    max_rows_per_chunk: int,
    pending_context_note: str = "",
) -> tuple[list[dict[str, Any]], int]:
    table = record.get("table") or {}
    metadata = table_metadata(record)
    table_type = str(metadata.get("table_type", "table"))
    prefix_lines: list[str] = []
    if pending_context_note:
        prefix_lines.append(f"직전본문: {pending_context_note}")
        prefix_lines.append("표위치: 위 직전본문 다음에 이 표가 이어짐")

    chunks: list[dict[str, Any]] = []

    rows_data = table.get("rows_data")
    if isinstance(rows_data, list) and rows_data:
        rows = dedupe_rows(rows_data)
        for body, grouped_rows in group_table_rows(
            rows, table_chunk_size=table_chunk_size, max_rows_per_chunk=max_rows_per_chunk
        ):
            row_indices = [row.get("row_index") for row in grouped_rows if row.get("row_index") is not None]
            row_metadata = dict(metadata)
            row_metadata.update(
                {
                    "row_indices": row_indices,
                    "row_range": row_group_label(grouped_rows),
                    "deduped_table_rows": len(rows),
                }
            )
            chunks.append(
                make_chunk(
                    next_chunk_no,
                    chunk_type=f"table_rows/{table_type}",
                    body=body,
                    metadata=row_metadata,
                    extra_prefix_lines=prefix_lines,
                )
            )
            next_chunk_no += 1
        return chunks, next_chunk_no

    row_groups = table.get("row_groups")
    if isinstance(row_groups, list) and row_groups:
        for group in row_groups:
            body = clean_text_block(group.get("text", ""))
            if not body:
                continue
            group_metadata = dict(metadata)
            group_metadata["row_range"] = group.get("row_range", "")
            for piece_index, piece in enumerate(
                split_oversized_text(body, chunk_size=table_chunk_size, overlap=80), start=1
            ):
                piece_metadata = dict(group_metadata)
                if piece_index > 1:
                    piece_metadata["split_part"] = piece_index
                chunks.append(
                    make_chunk(
                        next_chunk_no,
                        chunk_type=f"table_group/{table_type}",
                        body=piece,
                        metadata=piece_metadata,
                        extra_prefix_lines=prefix_lines,
                    )
                )
                next_chunk_no += 1
        return chunks, next_chunk_no

    summary_lines = table.get("summary_lines")
    if isinstance(summary_lines, list) and summary_lines:
        current: list[str] = []
        current_len = 0
        group_index = 1

        def flush_summary() -> None:
            nonlocal current, current_len, group_index, next_chunk_no
            if not current:
                return
            body = "\n".join(current)
            summary_metadata = dict(metadata)
            summary_metadata["summary_group_index"] = group_index
            chunks.append(
                make_chunk(
                    next_chunk_no,
                    chunk_type=f"table_summary/{table_type}",
                    body=body,
                    metadata=summary_metadata,
                    extra_prefix_lines=prefix_lines,
                )
            )
            next_chunk_no += 1
            group_index += 1
            current = []
            current_len = 0

        for line in summary_lines:
            clean_line = clean_text_block(line)
            if not clean_line:
                continue
            if current and current_len + 1 + len(clean_line) > table_chunk_size:
                flush_summary()
            if len(clean_line) > table_chunk_size:
                flush_summary()
                for piece_index, piece in enumerate(
                    split_oversized_text(clean_line, chunk_size=table_chunk_size, overlap=80), start=1
                ):
                    summary_metadata = dict(metadata)
                    summary_metadata["summary_group_index"] = group_index
                    summary_metadata["split_part"] = piece_index
                    chunks.append(
                        make_chunk(
                            next_chunk_no,
                            chunk_type=f"table_summary/{table_type}",
                            body=piece,
                            metadata=summary_metadata,
                            extra_prefix_lines=prefix_lines,
                        )
                    )
                    next_chunk_no += 1
                group_index += 1
                continue
            current.append(clean_line)
            current_len += len(clean_line) + 1
        flush_summary()
        return chunks, next_chunk_no

    markdown = table.get("markdown")
    if isinstance(markdown, str) and markdown.strip():
        body = clean_text_block(markdown)
        for piece_index, piece in enumerate(
            split_oversized_text(body, chunk_size=table_chunk_size, overlap=80), start=1
        ):
            md_metadata = dict(metadata)
            if piece_index > 1:
                md_metadata["split_part"] = piece_index
            chunks.append(
                make_chunk(
                    next_chunk_no,
                    chunk_type=f"table_markdown/{table_type}",
                    body=piece,
                    metadata=md_metadata,
                    extra_prefix_lines=prefix_lines,
                )
            )
            next_chunk_no += 1
        return chunks, next_chunk_no

    return chunks, next_chunk_no


def overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return text.strip()
    tail = text[-overlap:].strip()
    for separator in ("\n\n", "\n", ". ", "。 "):
        position = tail.find(separator)
        if position >= 0 and position + len(separator) < len(tail):
            return tail[position + len(separator) :].strip()
    return tail


def split_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    text = clean_text_block(text)
    if not text:
        return []

    units: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if len(block) <= chunk_size:
            units.append(block)
        else:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(lines) > 1:
                units.extend(lines)
            else:
                units.extend(split_long_unit(block, chunk_size))

    chunks: list[str] = []
    current = ""
    for unit in units:
        split_units = split_long_unit(unit, chunk_size) if len(unit) > chunk_size else [unit]
        for piece in split_units:
            if not current:
                current = piece
            elif len(current) + 2 + len(piece) <= chunk_size:
                current = current + "\n\n" + piece
            else:
                chunks.append(current.strip())
                if overlap > 0 and len(current) > overlap:
                    current = clean_text_block(overlap_tail(current, overlap) + "\n\n" + piece)
                else:
                    current = piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def context_prefix(record: dict[str, Any], *, chunk_type: str) -> str:
    lines: list[str] = []
    section = path_text(record.get("cleaned_section_path") or [])
    if section:
        lines.append(f"섹션경로: {section}")
    lines.append("자료유형: 표지" if chunk_type == "cover_text" else "자료유형: 본문")
    return "\n".join(lines)


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("file_name", "section_path_text", "section_type", "heading", "part_index"):
        value = metadata.get(key)
        if value not in (None, "", [], {}):
            compact[key] = value
    return compact


def make_chunks(
    grouped_records: list[dict[str, Any]],
    *,
    chunk_size: int,
    overlap: int,
    include_doc_name_in_text: bool,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for record in grouped_records:
        chunk_type = str(record.get("content_type") or "section_text")
        parts = split_text(str(record.get("text") or ""), chunk_size=chunk_size, overlap=overlap)
        for part_index, body in enumerate(parts, start=1):
            prefix_lines = []
            if include_doc_name_in_text:
                prefix_lines.append(f"문서명: {record['file_name']}")
            prefix = context_prefix(record, chunk_type=chunk_type)
            if prefix:
                prefix_lines.append(prefix)
            chunk_text = clean_text_block("\n".join(prefix_lines) + "\n\n" + body)
            content_hash = stable_hash(chunk_text)
            chunk_no = len(chunks) + 1
            section_path = record.get("cleaned_section_path") or []
            metadata = compact_metadata(
                {
                    "file_name": record["file_name"],
                    "section_path_text": path_text(section_path),
                    "section_type": record.get("section_type", ""),
                    "heading": record.get("heading", ""),
                    "part_index": part_index,
                }
            )
            chunks.append(
                {
                    "chunk_id": f"text_chunk_{chunk_no:08d}_{content_hash}",
                    "chunk_type": chunk_type,
                    "chunk_text": chunk_text,
                    "metadata": metadata,
                }
            )
    return chunks


def flush_text_buffer(
    text_records: list[dict[str, Any]],
    *,
    next_chunk_no: int,
    text_chunk_size: int,
    text_overlap: int,
    min_text_chars: int,
    short_context_chars: int,
    as_table_context: bool,
    following_table: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int, str]:
    if not text_records:
        return [], next_chunk_no, ""

    body = clean_text_block("\n\n".join(str(record.get("text", "")) for record in text_records))
    if not body:
        return [], next_chunk_no, ""

    first = text_records[0]
    if is_low_value_text(body, content_type=str(first.get("content_type", ""))):
        return [], next_chunk_no, ""

    if as_table_context and len(body) <= short_context_chars:
        return [], next_chunk_no, body

    metadata = base_metadata(first)
    metadata["source_record_indices"] = [record.get("_source_record_index") for record in text_records]
    metadata["merged_record_count"] = len(text_records)
    chunk_type = "cover_text" if first.get("content_type") == "cover_text" else "section_text"

    chunks: list[dict[str, Any]] = []
    for part_index, part in enumerate(
        split_section_text(body, chunk_size=text_chunk_size, overlap=text_overlap, min_chars=min_text_chars), start=1
    ):
        if len(part) < min_text_chars and chunks:
            previous = chunks[-1]
            previous["chunk_text"] = clean_text_block(previous["chunk_text"] + "\n\n" + part)
            previous["metadata"]["body_chars"] += len(part)
            previous["metadata"]["chunk_chars"] = len(previous["chunk_text"])
            continue
        part_metadata = dict(metadata)
        part_metadata["part_index"] = part_index
        chunks.append(make_chunk(next_chunk_no, chunk_type=chunk_type, body=part, metadata=part_metadata))
        next_chunk_no += 1
    if chunks and following_table:
        marker = table_following_marker(following_table)
        if marker:
            chunks[-1]["chunk_text"] = clean_text_block(chunks[-1]["chunk_text"] + "\n\n" + marker)
            chunks[-1]["metadata"]["next_table_id"] = following_table.get("table_id", "")
            chunks[-1]["metadata"]["next_table_type"] = following_table.get("table_type", "")
            chunks[-1]["metadata"]["next_table_shape"] = (following_table.get("table") or {}).get("shape", "")
    return chunks, next_chunk_no, ""


def table_following_marker(record: dict[str, Any]) -> str:
    table = record.get("table") or {}
    table_id = record.get("table_id", "")
    table_type = record.get("table_type", "")
    table_shape_value = table.get("shape", "")
    size = ""
    if table.get("rows") and table.get("cols"):
        size = f", {table.get('rows')}행x{table.get('cols')}열"
    if not table_id:
        return ""
    parts = [f"[다음 표: {table_id}"]
    if table_type:
        parts.append(f", 유형={table_type}")
    if table_shape_value:
        parts.append(f", 형태={table_shape_value}")
    if size:
        parts.append(size)
    parts.append("]")
    return "".join(parts)


def same_text_bucket(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("file_name") == right.get("file_name")
        and left.get("content_type") == right.get("content_type")
        and (left.get("section_path") or []) == (right.get("section_path") or [])
    )


def build_rag_chunks(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    text_buffer: list[dict[str, Any]] = []
    next_chunk_no = 1

    def extend_text_buffer(record: dict[str, Any]) -> None:
        nonlocal text_buffer
        if text_buffer and not same_text_bucket(text_buffer[-1], record):
            flush_to_chunks(as_table_context=False)
        text_buffer.append(record)

    def flush_to_chunks(*, as_table_context: bool, following_table: dict[str, Any] | None = None) -> str:
        nonlocal text_buffer, next_chunk_no, chunks
        emitted, next_chunk_no, context_note = flush_text_buffer(
            text_buffer,
            next_chunk_no=next_chunk_no,
            text_chunk_size=args.text_chunk_size,
            text_overlap=args.text_overlap,
            min_text_chars=args.min_text_chars,
            short_context_chars=args.short_context_chars,
            as_table_context=as_table_context,
            following_table=following_table,
        )
        chunks.extend(emitted)
        text_buffer = []
        return context_note

    for record in records:
        content_type = record.get("content_type")
        if content_type == "section_text" or (content_type == "cover_text" and args.include_cover):
            extend_text_buffer(record)
            continue

        if content_type == "table":
            context_note = flush_to_chunks(as_table_context=True, following_table=record)
            table_chunks, next_chunk_no = chunks_from_table(
                record,
                next_chunk_no=next_chunk_no,
                table_chunk_size=args.table_chunk_size,
                max_rows_per_chunk=args.max_table_rows,
                pending_context_note=context_note,
            )
            chunks.extend(table_chunks)
            continue

        if content_type == "toc" and args.include_toc:
            context_note = flush_to_chunks(as_table_context=False)
            toc_record = dict(record)
            toc_record["content_type"] = "section_text"
            toc_record["text"] = clean_text_block(record.get("text", ""))
            toc_chunks, next_chunk_no, _ = flush_text_buffer(
                [toc_record],
                next_chunk_no=next_chunk_no,
                text_chunk_size=args.text_chunk_size,
                text_overlap=0,
                min_text_chars=args.min_text_chars,
                short_context_chars=args.short_context_chars,
                as_table_context=False,
            )
            if context_note:
                for chunk in toc_chunks:
                    chunk["chunk_text"] = clean_text_block(f"직전본문: {context_note}\n\n{chunk['chunk_text']}")
            chunks.extend(toc_chunks)
            continue

        flush_to_chunks(as_table_context=False)

    flush_to_chunks(as_table_context=False)
    return chunks


def compact_output_metadata(chunks: list[dict[str, Any]]) -> None:
    """Keep only metadata that is useful for retrieval filters and citations."""
    common_keys = ["file_name", "section_path", "section_type", "heading"]
    table_keys = ["table_type", "table_shape", "table_id", "row_range"]
    link_keys = ["next_table_id", "next_table_type", "next_table_shape"]
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        compact: dict[str, Any] = {}
        for key in common_keys:
            value = metadata.get(key)
            if value not in ("", None, []):
                compact[key] = value
        if str(chunk.get("chunk_type", "")).startswith("table_"):
            for key in table_keys:
                value = metadata.get(key)
                if value not in ("", None, []):
                    compact[key] = value
        else:
            for key in link_keys:
                value = metadata.get(key)
                if value not in ("", None, []):
                    compact[key] = value
        chunk["metadata"] = compact


def write_summary_csv(path: Path, chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_type = Counter(str(chunk.get("chunk_type", "")) for chunk in chunks)
    by_file = Counter(str(chunk.get("metadata", {}).get("file_name", "")) for chunk in chunks)
    lengths = [len(str(chunk.get("chunk_text", ""))) for chunk in chunks]
    rows = [
        {"metric": "total_chunks", "value": len(chunks)},
        {"metric": "files", "value": len(by_file)},
        {"metric": "short_chunks_lt_200", "value": sum(length < 200 for length in lengths)},
        {"metric": "long_chunks_gt_1500", "value": sum(length > 1500 for length in lengths)},
        {"metric": "max_chunk_chars", "value": max(lengths) if lengths else 0},
        {
            "metric": "avg_chunk_chars",
            "value": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        },
    ]
    rows.extend(
        {"metric": f"chunk_type:{chunk_type}", "value": count}
        for chunk_type, count in by_type.most_common()
    )

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_sample_jsonl(path: Path, chunks: list[dict[str, Any]], *, sample_size: int) -> None:
    if sample_size <= 0:
        return
    write_jsonl(path, chunks[:sample_size])


def chunk_prechunk_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    chunk_size: int,
    overlap: int,
    strict_patterns: bool,
    split_inline_headings: bool,
    include_cover: bool,
    include_doc_name_in_text: bool,
) -> dict[str, Any]:
    prechunk_records = read_jsonl(input_path)
    clean_records = build_clean_prechunk_records(
        prechunk_records,
        strict_patterns=strict_patterns,
        include_cover=include_cover,
        split_inline_headings=split_inline_headings,
    )
    grouped_records = group_records(clean_records)
    chunks = make_chunks(
        grouped_records,
        chunk_size=chunk_size,
        overlap=overlap,
        include_doc_name_in_text=include_doc_name_in_text,
    )
    write_jsonl(output_path, chunks)
    return {
        "input_records": len(prechunk_records),
        "text_records": len(clean_records),
        "grouped_sections": len(grouped_records),
        "chunks": len(chunks),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one slim RAG chunk JSONL from parsed HWP prechunk JSONL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="parsed prechunk JSONL")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output slim chunk JSONL")
    parser.add_argument("--chunk-size", type=int, default=900, help="text body chunk size before prefix")
    parser.add_argument("--overlap", type=int, default=150, help="character overlap")
    parser.add_argument("--strict-patterns", action="store_true", help="drop headings whose pattern is not dominant")
    parser.add_argument("--no-inline-heading-split", action="store_true", help="do not split text body by detected inline heading lines")
    parser.add_argument("--no-cover", action="store_true", help="exclude cover_text records")
    parser.add_argument("--include-doc-name-in-text", action="store_true", help="prepend document name to chunk_text")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    result = chunk_prechunk_jsonl(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        strict_patterns=args.strict_patterns,
        split_inline_headings=not args.no_inline_heading_split,
        include_cover=not args.no_cover,
        include_doc_name_in_text=args.include_doc_name_in_text,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
