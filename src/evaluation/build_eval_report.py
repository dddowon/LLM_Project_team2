from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.evaluation.answer import is_answerable_question_type, is_refusal_answer
from src.utils.jsonl import read_jsonl

QUESTION_TYPE_LABELS = {
    "fact": "사실 확인",
    "summary": "요약",
    "comparison": "비교",
    "follow_up": "후속 질문",
    "requirement_detail": "요구사항 상세",
    "unanswerable": "문서 외 질문",
}

QUESTION_TYPE_ORDER = list(QUESTION_TYPE_LABELS.values()) + ["미분류"]

FAILURE_REASON_LABELS = {
    "wrong_doc": "잘못된 문서 검색 (Wrong Doc)",
    "wrong_refusal": "무단 거절 (Wrong Refusal)",
    "wrong_answer": "오답 (Wrong Answer)",
    "low_retrieval": "검색 부족 (Low Retrieval)",
    "should_refuse": "거절 필요·답변함 (Should Refuse)",
}

STAGE_SUMMARY_TITLES = {
    "retrieval": "1. 검색 엔진 성능 (Retrieval Stage)",
    "generation": "2. 생성 모델 검증 (LLM Judge Stage)",
    "answer": "3. 최종 답변 및 태스크 달성도 (Task Success Stage)",
}

CHART_SECTION_TITLES = {
    "retrieval": "📈 지표별 시각화 분석 — 1. 검색 엔진 평가",
    "generation": "📈 지표별 시각화 분석 — 2. 생성 모델 검증",
    "answer": "📈 지표별 시각화 분석 — 3. 최종 답변 및 태스크 달성도",
}

FAILURE_PRIORITY = {
    "wrong_doc": 0,
    "wrong_refusal": 1,
    "hallucination": 2,
    "wrong_answer": 3,
    "low_retrieval": 4,
    "should_refuse": 5,
    "legacy": 6,
}

LOW_SCORE_THRESHOLD = 4
HALLUCINATION_CORRECTNESS_MAX = 2

# 화면·CSV 표기용: 한글 설명 + JSONL 필드명 (예시 리포트 기준)
METRIC_TITLES: dict[str, str] = {
    "doc_hit": "대상 문서 적중률",
    "retrieval_keyword_hit": "키워드 매칭률",
    "context_precision": "검색 문맥 정밀도",
    "f_score": "답변 충실도",
    "r_score": "질문 적합성",
    "s_score": "정보 종합력",
    "correctness_score": "기대 답변 유사도",
    "task_success": "태스크 최종 성공률",
    "wrong_refusal": "무단 거절 오류율",
    "total_latency_ms": "평균 추론 시간",
}

FAILURE_METRIC_TITLES: dict[str, str] = {
    **METRIC_TITLES,
    "task_success": "태스크 최종 성공 여부",
    "wrong_refusal": "무단 거절 여부",
}

RETRIEVAL_SUMMARY_KEYS = ("doc_hit", "retrieval_keyword_hit", "context_precision")
GENERATION_SUMMARY_KEYS = ("f_score", "r_score", "s_score")
ANSWER_SUMMARY_KEYS = ("correctness_score", "task_success", "wrong_refusal")

REPORT_STYLES = """
body { font-family: 'Malgun Gothic', Arial, sans-serif; margin: 0; background: #f4f7f9; color: #222; line-height: 1.45; }
.report-wrap { max-width: 1200px; margin: 0 auto; padding: 28px 20px 48px; }
h1 { text-align: center; color: #333; font-weight: 700; margin: 0 0 12px; font-size: 1.65rem; }
h2 { border-left: 5px solid #4f7cff; padding-left: 12px; margin: 0; font-size: 1.05rem; color: #111; font-weight: 700; }
h3 { margin: 0 0 10px; font-size: 0.95rem; color: #444; }
.report-meta { text-align: center; color: #666; font-size: 13px; margin-bottom: 20px; }
.report-nav { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin: 0 0 28px; padding: 0; list-style: none; }
.report-nav a { display: inline-block; padding: 6px 14px; border-radius: 999px; background: #fff; border: 1px solid #d8e0ea; color: #335; font-size: 13px; text-decoration: none; }
.report-nav a:hover { border-color: #4f7cff; color: #4f7cff; }
.report-section { margin-bottom: 36px; scroll-margin-top: 16px; }
.stage-title { font-size: 13px; color: #4f7cff; margin: 20px 0 8px; font-weight: 700; letter-spacing: 0.02em; }
.cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 8px; }
.card { border-radius: 10px; padding: 14px 12px; background: #fff; box-shadow: 0 1px 6px rgba(0,0,0,0.05); text-align: center; border: 1px solid #eef2f5; font-size: 12px; color: #555; }
.card-latency { max-width: 260px; }
.metric { font-size: 1.45rem; font-weight: 700; color: #4f7cff; margin-top: 4px; }
.chart-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 12px; }
.chart-box { background: #fff; padding: 14px; border-radius: 10px; box-shadow: 0 1px 6px rgba(0,0,0,0.05); border: 1px solid #eef2f5; }
.chart-box h3 { font-size: 13px; margin-bottom: 8px; }
.note, .muted { color: #666; font-size: 13px; }
.muted { color: #888; }
.table-scroll { overflow-x: auto; margin-top: 12px; border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,0.04); -webkit-overflow-scrolling: touch; }
table { border-collapse: collapse; width: 100%; background: #fff; font-size: 13px; }
table.summary-table th, table.summary-table td { border: 1px solid #eee; padding: 10px 8px; text-align: center; }
table.summary-table th { background: #f8f9fa; color: #444; font-weight: 600; white-space: nowrap; }
table.summary-table tbody tr:nth-child(even) { background: #fafbfc; }
table.failure-detail { font-size: 12px; min-width: 1100px; }
table.failure-detail th, table.failure-detail td { border: 1px solid #eee; padding: 8px; vertical-align: top; }
table.failure-detail thead th { background: #f8f9fa; text-align: center; font-weight: 600; color: #444; }
table.failure-detail tr.group-head th { font-size: 11px; padding: 6px 4px; color: #fff; border-color: transparent; }
table.failure-detail th.group-retrieval { background: #5b7fd6; }
table.failure-detail th.group-generation { background: #6a9b6e; }
table.failure-detail th.group-answer { background: #c9873a; }
table.failure-detail th.group-text { background: #6c757d; }
table.failure-detail .sticky-col { position: sticky; left: 0; z-index: 2; background: #fff; min-width: 100px; box-shadow: 2px 0 4px rgba(0,0,0,0.04); }
table.failure-detail thead .sticky-col { background: #f8f9fa; z-index: 3; }
table.failure-detail .metric-col { text-align: center; white-space: nowrap; font-weight: 600; min-width: 52px; }
table.failure-detail .text-col { min-width: 180px; max-width: 280px; text-align: left; }
table.failure-detail .failure-text { max-height: 10rem; overflow: auto; line-height: 1.5; font-size: 12px; word-break: break-word; }
table.failure-detail .refusal-answer { background: #fff8e6; border-left: 3px solid #e6a700; padding-left: 8px; }
table.failure-detail .hallucination-answer { background: #fff0f0; border-left: 3px solid #d9534f; padding-left: 8px; }
table.failure-detail tbody tr.data-row:nth-child(4n+1) { background: #fcfdff; }
.tag { display: inline-block; margin: 2px 4px 2px 0; padding: 2px 6px; border-radius: 4px; font-size: 11px; background: #eef2ff; color: #334; }
.tag-refusal { background: #fff3cd; color: #856404; font-weight: 600; }
.tag-hallucination { background: #f8d7da; color: #721c24; font-weight: 600; }
table.failure-detail tr.failure-evidence-row td { border-top: 1px dashed #dde3ea; background: #fafbfd; font-size: 12px; }
.evidence-section-title { font-size: 11px; font-weight: 700; color: #4f7cff; margin: 8px 0 4px; }
details.fold-section { background: #fff; border: 1px solid #e8edf2; border-radius: 10px; margin-top: 16px; box-shadow: 0 1px 6px rgba(0,0,0,0.04); }
details.fold-section > summary { cursor: pointer; padding: 14px 16px; list-style: none; user-select: none; }
details.fold-section > summary::-webkit-details-marker { display: none; }
details.fold-section > summary::before { content: "▸ "; color: #4f7cff; font-weight: 700; }
details.fold-section[open] > summary::before { content: "▾ "; }
details.fold-section > summary h2 { display: inline; border: none; padding: 0; font-size: 1.05rem; }
details.fold-section .fold-body { padding: 0 16px 16px; }
details.fold-legend { margin-top: 32px; }
details.fold-legend .metrics-legend { border: none; box-shadow: none; margin: 0; padding: 0 4px 8px; }
.metrics-legend h2 { margin-top: 0; border-left: 5px solid #4f7cff; padding-left: 12px; font-size: 1.05rem; }
.legend-stage { margin: 16px 0 6px; font-size: 13px; color: #4f7cff; font-weight: 700; border-bottom: 1px dashed #eef2f5; padding-bottom: 4px; }
.metrics-legend dt { font-weight: 700; margin-top: 10px; font-size: 13px; color: #333; }
.metrics-legend dd { margin: 4px 0 0; color: #555; font-size: 13px; line-height: 1.55; text-align: left; }
.legend-footnote { margin-top: 14px; color: #666; font-size: 13px; border-top: 1px solid #eee; padding-top: 10px; }
.metrics-legend code { background: #f0f2f5; padding: 1px 5px; border-radius: 3px; font-size: 12px; color: #d93737; font-family: Consolas, Monaco, monospace; }
table.failure-detail th .metric-field { display: block; font-size: 10px; color: #888; font-weight: normal; margin-top: 2px; }
@media (max-width: 900px) { .cards { grid-template-columns: 1fr 1fr; } }
@media (max-width: 560px) { .cards { grid-template-columns: 1fr; } .report-wrap { padding: 16px 12px 32px; } }
"""


def metric_label(field: str) -> str:
    title = METRIC_TITLES.get(field, field)
    return f"{title} ({field})"


def row_question_type_key(row: dict[str, Any]) -> str:
    return str(row.get("question_type") or "").strip().lower()


def row_is_answerable(row: dict[str, Any]) -> bool:
    return is_answerable_question_type(row_question_type_key(row))


def row_wrong_refusal(row: dict[str, Any]) -> bool | None:
    """JSONL에 wrong_refusal이 없으면 답변 가능 + 거절 패턴으로 추정."""
    if row.get("wrong_refusal") is not None:
        return bool(row.get("wrong_refusal"))
    if not row_is_answerable(row):
        return None
    return row_is_refusal(row)


def fmt_wrong_refusal_flag(row: dict[str, Any]) -> str:
    value = row_wrong_refusal(row)
    if value is None:
        return "해당 없음"
    return "1" if value else "0"


def metric_th(field: str, *, failure_table: bool = False) -> str:
    titles = FAILURE_METRIC_TITLES if failure_table else METRIC_TITLES
    title = titles.get(field, field)
    return (
        f"<th>{html.escape(title)}"
        f'<span class="metric-field">({html.escape(field)})</span></th>'
    )


def as_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: float | None, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if value is not None else "-"


def fmt_percent(value: float | None) -> str:
    return f"{round(value * 100)}%" if value is not None else "-"


def fmt_seconds(ms_value: float | None) -> str:
    return f"{ms_value / 1000:.2f}초" if ms_value is not None else "-"


def fmt_task_success(value: Any) -> str:
    if value is None:
        return "-"
    return "✓" if value else "✗"


def fmt_metric_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "1" if value else "0"
    num = as_number(value)
    if num is not None:
        return str(int(num)) if num == int(num) else fmt(num)
    text = str(value).strip()
    if not text or text.lower() == "none":
        return "-"
    return text


def row_category(row: dict[str, Any]) -> str:
    category = str(row.get("category") or "").strip()
    return category or "미분류"


def row_question_type(row: dict[str, Any]) -> str:
    question_type = str(row.get("question_type") or "").strip()
    if not question_type:
        return "미분류"
    return QUESTION_TYPE_LABELS.get(question_type, question_type)


def row_is_refusal(row: dict[str, Any]) -> bool:
    if row.get("is_refusal") is not None:
        return bool(row.get("is_refusal"))
    return is_refusal_answer(str(row.get("answer") or ""))


def score_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [v for v in (as_number(row.get(key)) for row in rows) if v is not None]


def bool_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [1.0 if row.get(key) else 0.0 for row in rows if row.get(key) is not None]
    return mean(values)


def _aggregate_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in items if row_is_answerable(row)]
    wrong_refusal_values = [
        1.0 if row_wrong_refusal(row) else 0.0
        for row in answerable
        if row_wrong_refusal(row) is not None
    ]
    return {
        "mean_doc_hit": mean(score_values(items, "doc_hit")),
        "mean_retrieval_keyword_hit": mean(score_values(items, "retrieval_keyword_hit")),
        "mean_context_precision": mean(score_values(items, "context_precision")),
        "mean_f_score": mean(score_values(items, "f_score")),
        "mean_r_score": mean(score_values(items, "r_score")),
        "mean_s_score": mean(score_values(items, "s_score")),
        "mean_correctness_score": mean(score_values(items, "correctness_score")),
        "mean_task_success": bool_rate(items, "task_success"),
        "mean_wrong_refusal_answerable": mean(wrong_refusal_values) if wrong_refusal_values else None,
        "n_answerable": len(answerable),
        "mean_total_latency_ms": mean(score_values(items, "total_latency_ms")),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"n": len(rows), **_aggregate_metrics(rows)}


def _question_type_sort_key(label: str) -> tuple[int, str]:
    if label in QUESTION_TYPE_ORDER:
        return (QUESTION_TYPE_ORDER.index(label), label)
    return (len(QUESTION_TYPE_ORDER), label)


def summarize_by_question_type(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row_question_type(row)].append(row)
    summary_rows: list[dict[str, Any]] = []
    for label, items in sorted(grouped.items(), key=lambda kv: _question_type_sort_key(kv[0])):
        qt_key = row_question_type_key(items[0]) if items else ""
        summary_rows.append(
            {
                "label": label,
                "question_type_key": qt_key,
                "n": len(items),
                **_aggregate_metrics(items),
            }
        )
    return summary_rows


def fmt_wrong_refusal_for_type_row(row: dict[str, Any]) -> str:
    if row.get("question_type_key") == "unanswerable":
        return "해당 없음"
    return fmt_percent(row.get("mean_wrong_refusal_answerable"))


def _score_below(row: dict[str, Any], key: str) -> bool:
    value = as_number(row.get(key))
    return value is not None and value < LOW_SCORE_THRESHOLD


def _binary_metric_is_zero(row: dict[str, Any], key: str) -> bool:
    return as_number(row.get(key)) == 0.0


def is_hallucination_candidate(row: dict[str, Any]) -> bool:
    if row_is_refusal(row):
        return False
    if _score_below(row, "f_score"):
        return True
    correctness = as_number(row.get("correctness_score"))
    return correctness is not None and correctness <= HALLUCINATION_CORRECTNESS_MAX


def failure_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    reason = str(row.get("failure_reason") or "").strip()
    if reason:
        tags.append(FAILURE_REASON_LABELS.get(reason, reason))
    if row_is_refusal(row) and not any("거절" in tag for tag in tags):
        if row_wrong_refusal(row) or reason == "wrong_refusal":
            tags.append("무단 거절")
        elif reason != "should_refuse":
            tags.append("거절 응답")
    if is_hallucination_candidate(row):
        tags.append("환각 의심")
    return tags or ["기타 저점수"]


def failure_primary_kind(row: dict[str, Any]) -> str:
    reason = str(row.get("failure_reason") or "").strip()
    if reason:
        return reason
    if is_hallucination_candidate(row):
        return "hallucination"
    if row_wrong_refusal(row):
        return "wrong_refusal"
    return "legacy"


def should_include_in_failures(row: dict[str, Any]) -> bool:
    if row.get("failure_reason") or row.get("wrong_refusal"):
        return True
    if is_hallucination_candidate(row):
        return True
    if row.get("task_success") is False:
        return True
    if row_is_refusal(row) and row.get("task_success") is not True:
        return True
    if any(_score_below(row, key) for key in ("f_score", "r_score", "s_score", "correctness_score")):
        return True
    return any(
        _binary_metric_is_zero(row, key)
        for key in ("doc_hit", "retrieval_keyword_hit", "context_precision")
    )


def failure_sort_key(row: dict[str, Any]) -> tuple[int, float, float, float]:
    doc_hit = as_number(row.get("doc_hit"))
    f_score = as_number(row.get("f_score"))
    correctness = as_number(row.get("correctness_score"))
    return (
        FAILURE_PRIORITY.get(failure_primary_kind(row), 99),
        doc_hit if doc_hit is not None else 1.0,
        f_score if f_score is not None else 5.0,
        correctness if correctness is not None else 5.0,
    )


def count_failure_candidates(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if should_include_in_failures(row))


def failure_rows(rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not should_include_in_failures(row):
            continue
        key = (str(row.get("question") or ""), str(row.get("doc_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        enriched = dict(row)
        enriched["_failure_tags"] = failure_tags(row)
        candidates.append(enriched)
    return sorted(candidates, key=failure_sort_key)[:top_n]


def failure_reason_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        reason = str(row.get("failure_reason") or "").strip()
        if reason:
            counts[FAILURE_REASON_LABELS.get(reason, reason)] += 1
        elif row_wrong_refusal(row):
            counts[FAILURE_REASON_LABELS["wrong_refusal"]] += 1
        elif is_hallucination_candidate(row):
            counts["환각 의심(라벨 없음)"] += 1
    return counts


def source_chunk_id_list(row: dict[str, Any]) -> list[str]:
    sources = row.get("sources")
    if not isinstance(sources, list):
        return []
    out: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        cid = str(source.get("chunk_id", "")).strip()
        if cid and cid not in out:
            out.append(cid)
    return out


def source_chunk_ids_csv_field(row: dict[str, Any]) -> str:
    return ", ".join(source_chunk_id_list(row))


def source_chunk_ids_html(row: dict[str, Any]) -> str:
    ids = source_chunk_id_list(row)
    if not ids:
        return '<span class="muted">(매칭된 검색 근거 문맥 데이터 없음)</span>'
    items = "".join(f"<li><code>{html.escape(cid)}</code></li>" for cid in ids)
    return f'<ul class="chunk-id-list">{items}</ul>'


def source_file_names_short(row: dict[str, Any], *, max_chars: int = 120) -> str:
    sources = row.get("sources")
    if not isinstance(sources, list):
        return ""
    seen: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        meta = source.get("metadata")
        name = ""
        if isinstance(meta, dict):
            name = str(meta.get("file_name") or meta.get("source_file") or "").strip()
        if name and name not in seen:
            seen.append(name)
    joined = " | ".join(seen)
    return joined if len(joined) <= max_chars else joined[: max_chars - 1] + "…"


def cell_text_html(text: Any, *, max_chars: int = 1200) -> str:
    raw = "" if text is None else str(text)
    if len(raw) > max_chars:
        raw = raw[: max_chars - 1] + "…"
    return html.escape(raw).replace("\n", "<br/>")


def answer_cell_html(row: dict[str, Any]) -> str:
    css_classes = ["failure-text"]
    if row_is_refusal(row):
        css_classes.append("refusal-answer")
    if is_hallucination_candidate(row):
        css_classes.append("hallucination-answer")
    return f'<div class="{" ".join(css_classes)}">{cell_text_html(row.get("answer"), max_chars=1200)}</div>'


def failure_tags_html(row: dict[str, Any]) -> str:
    tags = row.get("_failure_tags") or failure_tags(row)
    parts: list[str] = []
    for tag in tags:
        cls = "tag"
        if tag == "환각 의심":
            cls += " tag-hallucination"
        elif "거절" in tag:
            cls += " tag-refusal"
        parts.append(f'<span class="{cls}">{html.escape(tag)}</span>')
    return " ".join(parts)


def failure_table_html(failures: list[dict[str, Any]]) -> str:
    if not failures:
        return '<p class="muted">표시할 오답 후보가 없습니다.</p>'

    metric_fields = (
        "doc_hit",
        "retrieval_keyword_hit",
        "context_precision",
        "f_score",
        "r_score",
        "s_score",
        "correctness_score",
        "task_success",
        "wrong_refusal",
    )
    ncols = 3 + len(metric_fields) + 3
    group_head = (
        "<tr class=\"group-head\">"
        '<th rowspan="2" class="sticky-col">유형</th>'
        '<th rowspan="2">질문 성격</th>'
        '<th colspan="3" class="group-retrieval">① 검색</th>'
        '<th colspan="3" class="group-generation">② 생성</th>'
        '<th colspan="3" class="group-answer">③ 답</th>'
        '<th colspan="3" class="group-text">본문</th>'
        "</tr>"
    )
    metric_head = (
        "<tr>"
        + "".join(metric_th(f, failure_table=True) for f in metric_fields)
        + "<th>질문</th><th>기대 답변</th><th>모델 출력</th>"
        "</tr>"
    )
    body_rows: list[str] = []

    for row in failures:
        cells = [
            f'<td class="sticky-col">{failure_tags_html(row)}</td>',
            f"<td>{html.escape(row_question_type(row))}</td>",
            f'<td class="metric-col">{html.escape(fmt_metric_value(row.get("doc_hit")))}</td>',
            f'<td class="metric-col">{html.escape(fmt_metric_value(row.get("retrieval_keyword_hit")))}</td>',
            f'<td class="metric-col">{html.escape(fmt_metric_value(row.get("context_precision")))}</td>',
            f'<td class="metric-col">{html.escape(fmt_metric_value(row.get("f_score")))}</td>',
            f'<td class="metric-col">{html.escape(fmt_metric_value(row.get("r_score")))}</td>',
            f'<td class="metric-col">{html.escape(fmt_metric_value(row.get("s_score")))}</td>',
            f'<td class="metric-col">{html.escape(fmt_metric_value(row.get("correctness_score")))}</td>',
            f'<td class="metric-col">{html.escape(fmt_task_success(row.get("task_success")))}</td>',
            f'<td class="metric-col">{html.escape(fmt_wrong_refusal_flag(row))}</td>',
            f'<td class="text-col"><div class="failure-text">{cell_text_html(row.get("question"), max_chars=600)}</div></td>',
            f'<td class="text-col"><div class="failure-text">{cell_text_html(row.get("expected_answer"), max_chars=1200)}</div></td>',
            f'<td class="text-col">{answer_cell_html(row)}</td>',
        ]
        body_rows.append('<tr class="data-row">' + "".join(cells) + "</tr>")
        evidence = (
            '<div class="failure-evidence-wrap">'
            '<div class="evidence-section-title">조회된 참조 근거 문서</div>'
            f'<div>{html.escape(source_file_names_short(row, max_chars=500))}</div>'
            '<div class="evidence-section-title">조회된 참조 청크 단락 ID</div>'
            f"{source_chunk_ids_html(row)}"
            "</div>"
        )
        body_rows.append(
            f'<tr class="failure-evidence-row"><td colspan="{ncols}">{evidence}</td></tr>'
        )

    return (
        '<div class="table-scroll">'
        '<table class="failure-detail"><thead>'
        + group_head
        + metric_head
        + "</thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table></div>"
    )


def failure_reason_table_html(counts: Counter[str]) -> str:
    if not counts:
        return '<p class="muted">failure_reason 집계 없음 (구버전 JSONL일 수 있음)</p>'
    rows = [[label, f"{count}건"] for label, count in counts.most_common()]
    return table_html(["실패 유형 정의", "발생 건수"], rows)


def write_failures_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "실패 유형",
        "질문 성격",
        "RFP 항목(category)",
        metric_label("doc_hit"),
        metric_label("retrieval_keyword_hit"),
        metric_label("context_precision"),
        metric_label("f_score"),
        metric_label("r_score"),
        metric_label("s_score"),
        metric_label("correctness_score"),
        metric_label("task_success"),
        FAILURE_METRIC_TITLES["wrong_refusal"] + " (wrong_refusal)",
        "거절 응답 (is_refusal)",
        "환각 의심",
        metric_label("total_latency_ms"),
        "질문",
        "기대 답변",
        "모델 답변",
        "근거 문서",
        "근거 청크 ID",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "실패 유형": " | ".join(row.get("_failure_tags") or failure_tags(row)),
                    "질문 성격": row_question_type(row),
                    "RFP 항목(category)": row_category(row),
                    metric_label("doc_hit"): row.get("doc_hit"),
                    metric_label("retrieval_keyword_hit"): row.get("retrieval_keyword_hit"),
                    metric_label("context_precision"): row.get("context_precision"),
                    metric_label("f_score"): row.get("f_score"),
                    metric_label("r_score"): row.get("r_score"),
                    metric_label("s_score"): row.get("s_score"),
                    metric_label("correctness_score"): row.get("correctness_score"),
                    metric_label("task_success"): row.get("task_success"),
                    FAILURE_METRIC_TITLES["wrong_refusal"]
                    + " (wrong_refusal)": fmt_wrong_refusal_flag(row),
                    "거절 응답 (is_refusal)": row_is_refusal(row),
                    "환각 의심": is_hallucination_candidate(row),
                    metric_label("total_latency_ms"): fmt_seconds(
                        as_number(row.get("total_latency_ms"))
                    ),
                    "질문": row.get("question", ""),
                    "기대 답변": row.get("expected_answer", ""),
                    "모델 답변": row.get("answer", ""),
                    "근거 문서": source_file_names_short(row),
                    "근거 청크 ID": source_chunk_ids_csv_field(row),
                }
            )


def bar_svg(items: list[tuple[str, float | None]], *, max_value: float) -> str:
    bar_height, gap, label_width, value_width, width = 26, 10, 210, 60, 760
    height = max(40, len(items) * (bar_height + gap) + 10)
    bar_width = width - label_width - value_width - 30
    use_percent = max_value <= 1.0
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    for index, (label, value) in enumerate(items):
        y = 8 + index * (bar_height + gap)
        ratio = 0.0 if value is None else max(0.0, min(1.0, value / max_value))
        current_width = round(bar_width * ratio, 1)
        if value is None:
            display = "-"
        elif use_percent:
            display = fmt_percent(value)
        else:
            display = fmt(value)
        parts.append(
            f'<text x="0" y="{y + 18}" style="font-size:12px;">{html.escape(label[:40])}</text>'
        )
        parts.append(
            f'<rect x="{label_width}" y="{y}" width="{bar_width}" height="{bar_height}" '
            f'rx="5" fill="#f0f2f5"/>'
        )
        parts.append(
            f'<rect x="{label_width}" y="{y}" width="{current_width}" height="{bar_height}" '
            f'rx="5" fill="#4f7cff"/>'
        )
        parts.append(
            f'<text x="{label_width + bar_width + 12}" y="{y + 18}" '
            f'style="font-size:12px;">{display}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def chart_for_metric(by_type: list[dict[str, Any]], metric_key: str, *, max_value: float) -> str:
    return bar_svg([(row["label"], row.get(metric_key)) for row in by_type], max_value=max_value)


def table_html(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-scroll">'
        f'<table class="summary-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
        "</div>"
    )


def report_nav_html() -> str:
    links = (
        ("#summary", "요약"),
        ("#legend", "지표 정의"),
        ("#by-type", "유형별 통계"),
        ("#charts", "시각화"),
        ("#failures", "오답 노트"),
    )
    items = "".join(
        f'<li><a href="{html.escape(href)}">{html.escape(label)}</a></li>'
        for href, label in links
    )
    return f'<nav aria-label="리포트 목차"><ul class="report-nav">{items}</ul></nav>'


def question_type_table_html(by_type: list[dict[str, Any]]) -> str:
    return table_html(
        [
            "질문 성격",
            "샘플 수 (n)",
            metric_label("doc_hit"),
            metric_label("retrieval_keyword_hit"),
            metric_label("context_precision"),
            metric_label("f_score"),
            metric_label("r_score"),
            metric_label("s_score"),
            metric_label("correctness_score"),
            metric_label("task_success"),
            metric_label("wrong_refusal"),
            metric_label("total_latency_ms"),
        ],
        [
            [
                row["label"],
                f"{row['n']}개",
                fmt_percent(row["mean_doc_hit"]),
                fmt_percent(row["mean_retrieval_keyword_hit"]),
                fmt_percent(row["mean_context_precision"]),
                fmt(row["mean_f_score"]),
                fmt(row["mean_r_score"]),
                fmt(row["mean_s_score"]),
                fmt(row["mean_correctness_score"]),
                fmt_percent(row["mean_task_success"]),
                fmt_wrong_refusal_for_type_row(row),
                fmt_seconds(row["mean_total_latency_ms"]),
            ]
            for row in by_type
        ],
    )


def summary_cards_html(summary: dict[str, Any]) -> str:
    def card(field: str, value: str) -> str:
        label = html.escape(metric_label(field))
        return f'<div class="card"><div>{label}</div><div class="metric">{value}</div></div>'

    retrieval = "".join(
        card(
            key,
            fmt_percent(summary[f"mean_{key}"])
            if key != "total_latency_ms"
            else fmt_seconds(summary["mean_total_latency_ms"]),
        )
        for key in RETRIEVAL_SUMMARY_KEYS
    )
    generation = "".join(card(key, fmt(summary[f"mean_{key}"])) for key in GENERATION_SUMMARY_KEYS)
    answer_cards: list[str] = []
    for key in ANSWER_SUMMARY_KEYS:
        if key == "wrong_refusal":
            value = fmt_percent(summary.get("mean_wrong_refusal_answerable"))
        elif key == "task_success":
            value = fmt_percent(summary.get("mean_task_success"))
        else:
            value = fmt(summary.get(f"mean_{key}"))
        answer_cards.append(card(key, value))
    answer = "".join(answer_cards)
    n_answerable = int(summary.get("n_answerable") or 0)
    answer_stage_note = (
        f'<p class="note">'
        f"<strong>태스크 최종 성공률</strong>: 전체 {summary['n']}문항 중 질문 유형별 pass 비율. "
        f"<strong>무단 거절 오류율</strong>: 답변 가능 {n_answerable}문항만 집계(문서 외는 표에서 「해당 없음」). "
        f"두 %는 서로 반대가 아니며, 성공률이 낮아도 거절 오류율이 0%일 수 있습니다."
        f"</p>"
    )

    return f"""
    <section id="summary" class="report-section">
    <div class="stage-title">{STAGE_SUMMARY_TITLES["retrieval"]}</div>
    <div class="cards">{retrieval}</div>
    <div class="stage-title">{STAGE_SUMMARY_TITLES["generation"]}</div>
    <div class="cards">{generation}</div>
    <div class="stage-title">{STAGE_SUMMARY_TITLES["answer"]}</div>
    {answer_stage_note}
    <div class="cards">{answer}</div>
    <div class="cards" style="margin-top: 12px;">
        <div class="card card-latency"><div>{html.escape(metric_label("total_latency_ms"))}</div>
        <div class="metric">{fmt_seconds(summary["mean_total_latency_ms"])}</div></div>
    </div>
    </section>
    """


def stage_charts_html(
    by_type: list[dict[str, Any]],
    section_title: str,
    specs: list[tuple[str, str, float]],
    *,
    open_by_default: bool = False,
) -> str:
    boxes = "".join(
        f'<div class="chart-box"><h3>{html.escape(title)}</h3>'
        f"{chart_for_metric(by_type, key, max_value=max_val)}</div>"
        for title, key, max_val in specs
    )
    open_attr = " open" if open_by_default else ""
    return (
        f'<details class="fold-section chart-stage"{open_attr}>'
        f"<summary><h2>{html.escape(section_title)}</h2></summary>"
        f'<div class="fold-body"><div class="chart-container">{boxes}</div></div>'
        "</details>"
    )


def metrics_legend_fold_html() -> str:
    return (
        '<details id="legend" class="fold-section fold-legend" open>'
        "<summary><h2>📘 RAG 파이프라인 지표 정의서</h2></summary>"
        f'<div class="fold-body">{metrics_legend_body_html()}</div>'
        "</details>"
    )


def metrics_legend_body_html() -> str:
    return """<div class="metrics-legend">
    <p class="note">본 수치는 <code>src/evaluation</code>의 RAG 평가 하네스 시스템을 통해 산출됩니다.
    LLM 판정 점수(f/r/s)는 절대적 정답률이 아닌, 동일 스케일 환경 하의
    <strong>상대적 성능 비교 지표</strong>로 활용할 때 유용합니다.</p>

    <h3 class="legend-stage">1. 검색 엔진 평가 (Retrieval Stage)</h3>
    <dl>
    <dt>doc_hit · 대상 문서 적중률</dt>
    <dd>사용자가 던진 질문에 매핑된 실제 정답 문서 ID(<code>doc_id</code>)가 검색 엔진을 거쳐
    최상위 top-k 청크 결과에 포함되었는지를 판정합니다. (포함 1, 미포함 0)</dd>
    <dt>retrieval_keyword_hit · 키워드 매칭률</dt>
    <dd>평가 가이드라인에 지정된 정답 필수 핵심어(<code>ground_truth_keywords</code>)가
    검색되어 올라온 전체 청크 텍스트 내에 단 하나라도 매칭되어 포함되었는지를 검증합니다.
    (포함 1, 미포함 0)</dd>
    <dt>context_precision · 검색 문맥 정밀도</dt>
    <dd>불러온 top-k 청크 중에서 정답 핵심 키워드를 실제로 포함하고 있는 '유효 청크'의 밀도와 비율을
    계산합니다. (키워드가 포함된 청크 수 ÷ k, 범위: 0~1)</dd>
    </dl>

    <h3 class="legend-stage">2. 생성 모델 평가 (LLM Judge Stage · 0~5점 척도)</h3>
    <dl>
    <dt>f_score · 답변 충실도 (Faithfulness)</dt>
    <dd>생성된 답변이 모델의 자체 지식이나 환각 없이, 오직 <strong>'검색 엔진이 찾아다 준 근거 문맥'</strong>에만
    철저히 기반하여 작성되었는지 검증합니다. 4점 미만일 경우 오답 노트에 <strong>환각 의심</strong> 라벨이 부여됩니다.</dd>
    <dt>r_score · 질문 적합성 (Relevance)</dt>
    <dd>모델이 딴소리를 하거나 우회하지 않고, 사용자가 원래 물어본 질문의 핵심 의도에 얼마나 정면으로
    알맞은 답변을 생성했는지를 LLM 판정관이 평가합니다.</dd>
    <dt>s_score · 정보 종합력 (Synthesis)</dt>
    <dd>여러 군데로 흩어져서 검색된 다수의 근거 문서 청크들을 하나의 유기적인 논리로 엮어,
    사용자가 읽기 쉽게 종합 가공했는지를 평가합니다.</dd>
    </dl>

    <h3 class="legend-stage">3. 최종 답변 및 태스크 달성도 (Task Success Stage)</h3>
    <dl>
    <dt>correctness_score · 기대 답변 유사도</dt>
    <dd>평가 데이터셋의 정답(<code>expected_answer</code>)과 비교하여, 모델이 생성한 답변의 핵심 정보가
    얼마나 정확히 일치하는지 판정합니다. (0~5점 척도)</dd>
    <dt>task_success · 태스크 최종 성공률</dt>
    <dd>RAG 시스템이 사용자의 요청을 올바르게 처리하여 <strong>'합격(Pass)'한 비율</strong>입니다.
    질문 유형에 따라 판정 기준이 다릅니다.<br>
    • <strong>답변 가능 질문</strong>: 답변을 회피/거절하지 않고, 기대 답변 유사도가 4점 이상(≥4)인 경우 합격<br>
    • <strong>답변 불가능 질문</strong> (<code>unanswerable</code>): 억지로 오답을 지어내지 않고,
    가이드라인대로 올바르게 답변을 거절한 경우 합격</dd>
    <dt>wrong_refusal · 무단 거절 오류율</dt>
    <dd>근거 문서에 정답이 존재하여 <strong>'답변이 가능한 질문'들 중에서, 모델이 내용을 찾을 수 없다며
    무단으로 답변을 거부한 비율</strong>입니다.<br>
    • <strong>주의</strong>: 답변 불가능한 질문은 계산에서 제외(해당 없음)됩니다.<br>
    • <strong>지표 해석</strong>: 시스템의 실패 유형에는 무단 거절 외에도 '검색 실패', '오답(환각)' 등이 따로 존재하므로,
    <u>[성공률 + 무단 거절률]의 합이 반드시 100%가 되지는 않습니다.</u></dd>
    </dl>

    <p class="legend-footnote">💡 <strong>오답 노트 시각화 가이드:</strong> 시스템 오동작 유형 중
    <strong>무단 거절</strong> 케이스는 <span class="tag tag-refusal">노란색 배경</span>,
    <strong>환각 의심</strong> 케이스는 <span class="tag tag-hallucination">붉은색 배경</span>으로 표기됩니다.
    상세 목록 건수는 <code>--top-n</code>으로 조절합니다.</p>
    </div>"""


def render_html(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    failure_pool_count: int,
) -> str:
    summary = summarize(rows)
    by_type = summarize_by_question_type(rows)

    def chart_max_value(field: str) -> float:
        if field in ("f_score", "r_score", "s_score", "correctness_score"):
            return 5.0
        return 1.0

    def chart_specs(fields: tuple[str, ...]) -> list[tuple[str, str, float]]:
        specs: list[tuple[str, str, float]] = []
        for field in fields:
            agg_key = (
                "mean_wrong_refusal_answerable"
                if field == "wrong_refusal"
                else f"mean_{field}"
            )
            specs.append((metric_label(field), agg_key, chart_max_value(field)))
        return specs

    retrieval_charts = stage_charts_html(
        by_type,
        CHART_SECTION_TITLES["retrieval"],
        chart_specs(RETRIEVAL_SUMMARY_KEYS),
        open_by_default=True,
    )
    generation_charts = stage_charts_html(
        by_type, CHART_SECTION_TITLES["generation"], chart_specs(GENERATION_SUMMARY_KEYS)
    )
    answer_charts = stage_charts_html(
        by_type,
        CHART_SECTION_TITLES["answer"],
        chart_specs(("correctness_score", "task_success", "wrong_refusal")),
    )

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG 시스템 성능 평가 리포트</title>
<style>{REPORT_STYLES}</style>
</head>
<body>
<div class="report-wrap">
<h1>📊 RAG 시스템 성능 평가 리포트</h1>
<p class="report-meta">
  전체 <strong>{summary["n"]}</strong>문항 · 오답 노트 <strong>{len(failures)}</strong>건
  (후보 {failure_pool_count}건) · 상단 목차로 이동
</p>
{report_nav_html()}
{summary_cards_html(summary)}
{metrics_legend_fold_html()}
<section id="by-type" class="report-section">
<h2>📋 질문 성격별 세부 지표 통계</h2>
<p class="note">질문 유형별 평균 지표입니다. 무단 거절 오류율은 답변 가능 유형만 집계합니다.</p>
{question_type_table_html(by_type)}
</section>
<section id="charts" class="report-section">
<p class="note">유형별 막대 차트입니다. 섹션 제목을 클릭하면 접거나 펼칠 수 있습니다.</p>
{retrieval_charts}
{generation_charts}
{answer_charts}
</section>
<section id="failures" class="report-section">
<h2>📝 디버깅 오답 노트 (Error Analysis)</h2>
<p class="note">무단 거절·환각 의심 등 실패 케이스입니다. 표는 가로 스크롤이 가능합니다.</p>
<h3>실패 유형 분포</h3>
{failure_reason_table_html(failure_reason_counts(rows))}
{failure_table_html(failures)}
</section>
</div>
</body>
</html>"""


def build_report(input_path: Path, html_output: Path, failures_output: Path, top_n: int) -> None:
    rows = read_jsonl(input_path)
    if not rows:
        print("데이터가 없습니다.")
        return

    pool_count = count_failure_candidates(rows)
    failures = failure_rows(rows, top_n)
    html_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.write_text(
        render_html(rows, failures, failure_pool_count=pool_count),
        encoding="utf-8",
    )
    write_failures_csv(failures_output, failures)
    print(f"리포트 생성 완료: {html_output}")
    print(f"오답 CSV: {failures_output} ({len(failures)}건 / 후보 {pool_count}건)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/eval_harness_results.jsonl")
    parser.add_argument("--html-output", default="outputs/eval_report.html")
    parser.add_argument("--failures-output", default="outputs/eval_failures.csv")
    parser.add_argument("--top-n", type=int, default=20, help="오답 노트에 표시할 최대 건수")
    args = parser.parse_args()
    build_report(
        input_path=Path(args.input),
        html_output=Path(args.html_output),
        failures_output=Path(args.failures_output),
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
