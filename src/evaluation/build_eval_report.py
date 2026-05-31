from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.evaluation.answer import (
    classify_answer_refusal,
    classify_failure_reason,
    is_answerable_question_type,
    score_pass,
)
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

EVAL_FOCUS_LABELS = {
    "text": "본문·표 (HWP)",
    "ocr_image": "OCR/스캔 이미지",
}
EVAL_FOCUS_ORDER = list(EVAL_FOCUS_LABELS.values()) + ["미분류"]

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
    "recall_at_5": "정답 근거 검색률",
    "mrr": "정답 근거 순위",
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

RETRIEVAL_SUMMARY_KEYS = (
    "doc_hit",
    "retrieval_keyword_hit",
    "context_precision",
    "recall_at_5",
    "mrr",
)
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
.tag-partial { background: #e8f4fd; color: #0c5460; font-weight: 600; }
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


def row_eval_focus_key(row: dict[str, Any]) -> str:
    focus = str(row.get("eval_focus") or "").strip().lower()
    if focus in EVAL_FOCUS_LABELS:
        return focus
    return ""


def row_eval_focus(row: dict[str, Any]) -> str:
    key = row_eval_focus_key(row)
    if not key:
        return "미분류"
    return EVAL_FOCUS_LABELS[key]


def row_is_answerable(row: dict[str, Any]) -> bool:
    return is_answerable_question_type(row_question_type_key(row))


def row_refusal_kind(row: dict[str, Any]) -> str:
    kind = str(row.get("refusal_kind") or "").strip().lower()
    if kind in ("none", "full", "partial"):
        return kind
    return classify_answer_refusal(str(row.get("answer") or ""))


def row_is_full_refusal(row: dict[str, Any]) -> bool:
    if row.get("is_refusal") is not None and row.get("refusal_kind"):
        return bool(row.get("is_refusal"))
    return row_refusal_kind(row) == "full"


def row_partial_limitation(row: dict[str, Any]) -> bool:
    if row.get("partial_limitation") is not None:
        return bool(row.get("partial_limitation"))
    return row_refusal_kind(row) == "partial"


def row_wrong_refusal(row: dict[str, Any]) -> bool | None:
    """무단 거절 = 답변 가능 질문의 전면 거절만 (부분 범위 제한 제외)."""
    if not row_is_answerable(row):
        return None
    if row_partial_limitation(row):
        return False
    if row.get("wrong_refusal") is not None:
        return bool(row.get("wrong_refusal"))
    return row_is_full_refusal(row)


def row_appropriate_refusal_success(row: dict[str, Any]) -> bool:
    """문서 외 질문에서 적절히 거절해 task_success 합격인 경우."""
    return (
        not row_is_answerable(row)
        and row.get("task_success") is True
        and row_is_full_refusal(row)
    )


def summarize_task_outcomes(rows: list[dict[str, Any]]) -> dict[str, int]:
    passed = sum(1 for row in rows if row.get("task_success") is True)
    failed = sum(1 for row in rows if row.get("task_success") is False)
    appropriate = sum(1 for row in rows if row_appropriate_refusal_success(row))
    return {
        "passed": passed,
        "failed": failed,
        "appropriate_refusal": appropriate,
        "unknown": len(rows) - passed - failed,
    }


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
    """하위 호환: 전면 거절 여부."""
    return row_is_full_refusal(row)


def enrich_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """구버전 JSONL도 리포트에서 전면/부분 거절·task_success를 일관되게 재계산."""
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        kind = classify_answer_refusal(str(item.get("answer") or ""))
        full_refusal = kind == "full"
        partial = kind == "partial"
        item["refusal_kind"] = kind
        item["partial_limitation"] = partial
        item["is_refusal"] = full_refusal

        answerable = row_is_answerable(item)
        correctness_score = item.get("correctness_score")
        correctness_pass = item.get("correctness_pass")
        if correctness_pass is None and isinstance(correctness_score, int):
            correctness_pass = score_pass(int(correctness_score))

        if answerable:
            item["wrong_refusal"] = full_refusal
            if correctness_pass is None:
                item["task_success"] = not full_refusal
            else:
                item["task_success"] = (not full_refusal) and correctness_pass
            item["appropriate_refusal"] = False
        else:
            item["wrong_refusal"] = False
            item["task_success"] = full_refusal
            item["appropriate_refusal"] = full_refusal

        item["failure_reason"] = classify_failure_reason(
            question_type=str(item.get("question_type") or ""),
            doc_hit=as_number(item.get("doc_hit")),
            keyword_hit=as_number(item.get("retrieval_keyword_hit")),
            is_refusal=full_refusal,
            correctness_pass=correctness_pass if isinstance(correctness_pass, bool) else None,
            has_doc_id=bool(str(item.get("doc_id") or "").strip()),
        )
        enriched.append(item)
    return enriched


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
        "mean_recall_at_5": mean(score_values(items, "recall_at_5")),
        "mean_mrr": mean(score_values(items, "mrr")),
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


def _eval_focus_sort_key(label: str) -> tuple[int, str]:
    if label in EVAL_FOCUS_ORDER:
        return (EVAL_FOCUS_ORDER.index(label), label)
    return (len(EVAL_FOCUS_ORDER), label)


def summarize_by_eval_focus(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row_eval_focus(row)].append(row)
    summary_rows: list[dict[str, Any]] = []
    for label, items in sorted(grouped.items(), key=lambda kv: _eval_focus_sort_key(kv[0])):
        focus_key = row_eval_focus_key(items[0]) if items else ""
        summary_rows.append(
            {
                "label": label,
                "eval_focus_key": focus_key,
                "n": len(items),
                **_aggregate_metrics(items),
            }
        )
    return summary_rows


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


def row_correctness_pass(row: dict[str, Any]) -> bool | None:
    if row.get("correctness_pass") is not None:
        return bool(row.get("correctness_pass"))
    score = row.get("correctness_score")
    if isinstance(score, int):
        return score_pass(score)
    return None


def is_hallucination_candidate(row: dict[str, Any]) -> bool:
    if row_is_full_refusal(row):
        return False
    if row_correctness_pass(row) is True:
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
    if row_wrong_refusal(row) or reason == "wrong_refusal":
        if not any("무단 거절" in tag for tag in tags):
            tags.append("무단 거절")
    elif row_is_full_refusal(row) and reason != "should_refuse" and not any(
        "거절" in tag for tag in tags
    ):
        tags.append("거절 응답")
    if row_partial_limitation(row) and not any("부분 범위" in tag for tag in tags):
        tags.append("부분 범위 제한")
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
    """조치가 필요한 실패만 오답 풀에 포함 (합격·적절 거절·검색만 나쁜 합격 제외)."""
    if row_partial_limitation(row) and not (
        row.get("failure_reason") or row_wrong_refusal(row) or row.get("task_success") is False
    ):
        return False
    if row_appropriate_refusal_success(row):
        return False
    if row.get("task_success") is True:
        return False
    if row.get("failure_reason") or row_wrong_refusal(row):
        return True
    if is_hallucination_candidate(row):
        return True
    if row.get("task_success") is False:
        return True
    if row_is_full_refusal(row):
        return True
    if any(_score_below(row, key) for key in ("f_score", "r_score", "s_score", "correctness_score")):
        return True
    return any(
        _binary_metric_is_zero(row, key)
        for key in ("doc_hit", "retrieval_keyword_hit", "context_precision", "recall_at_5", "mrr")
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
    if row_is_full_refusal(row):
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
        elif tag == "부분 범위 제한":
            cls += " tag-partial"
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
        "recall_at_5",
        "mrr",
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
        '<th colspan="5" class="group-retrieval">① 검색</th>'
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
            f'<td class="metric-col">{html.escape(fmt_metric_value(row.get("recall_at_5")))}</td>',
            f'<td class="metric-col">{html.escape(fmt_metric_value(row.get("mrr")))}</td>',
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


PARTIAL_LIMITATION_CSV_LABEL = "부분 범위 제한 (partial_limitation)"


def row_dedupe_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("question") or ""), str(row.get("doc_id") or ""))


def eval_csv_fieldnames() -> list[str]:
    return [
        "결과 분류",
        "CSV 비고",
        "질문 성격",
        "RFP 항목(category)",
        metric_label("doc_hit"),
        metric_label("retrieval_keyword_hit"),
        metric_label("context_precision"),
        metric_label("recall_at_5"),
        metric_label("mrr"),
        metric_label("f_score"),
        metric_label("r_score"),
        metric_label("s_score"),
        metric_label("correctness_score"),
        metric_label("task_success"),
        FAILURE_METRIC_TITLES["wrong_refusal"] + " (wrong_refusal)",
        "전면 거절 (is_refusal)",
        PARTIAL_LIMITATION_CSV_LABEL,
        "환각 의심",
        metric_label("total_latency_ms"),
        "질문",
        "기대 답변",
        "모델 답변",
        "근거 문서",
        "근거 청크 ID",
    ]


def eval_row_to_csv_dict(row: dict[str, Any], *, outcome: str) -> dict[str, str | bool]:
    return {
        "결과 분류": outcome,
        "CSV 비고": str(row.get("_csv_note") or ""),
        "질문 성격": row_question_type(row),
        "RFP 항목(category)": row_category(row),
        metric_label("doc_hit"): fmt_metric_value(row.get("doc_hit")),
        metric_label("retrieval_keyword_hit"): fmt_metric_value(row.get("retrieval_keyword_hit")),
        metric_label("context_precision"): fmt_metric_value(row.get("context_precision")),
        metric_label("recall_at_5"): fmt_metric_value(row.get("recall_at_5")),
        metric_label("mrr"): fmt_metric_value(row.get("mrr")),
        metric_label("f_score"): fmt_metric_value(row.get("f_score")),
        metric_label("r_score"): fmt_metric_value(row.get("r_score")),
        metric_label("s_score"): fmt_metric_value(row.get("s_score")),
        metric_label("correctness_score"): fmt_metric_value(row.get("correctness_score")),
        metric_label("task_success"): fmt_task_success(row.get("task_success")),
        FAILURE_METRIC_TITLES["wrong_refusal"] + " (wrong_refusal)": fmt_wrong_refusal_flag(row),
        "전면 거절 (is_refusal)": row_is_full_refusal(row),
        PARTIAL_LIMITATION_CSV_LABEL: row_partial_limitation(row),
        "환각 의심": is_hallucination_candidate(row),
        metric_label("total_latency_ms"): fmt_seconds(as_number(row.get("total_latency_ms"))),
        "질문": str(row.get("question") or ""),
        "기대 답변": str(row.get("expected_answer") or ""),
        "모델 답변": str(row.get("answer") or ""),
        "근거 문서": source_file_names_short(row),
        "근거 청크 ID": source_chunk_ids_csv_field(row),
    }


def write_eval_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = eval_csv_fieldnames()
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            outcome = str(row.get("_csv_outcome") or "")
            writer.writerow(eval_row_to_csv_dict(row, outcome=outcome))


def partial_limitation_rows_for_csv(
    all_rows: list[dict[str, Any]],
    html_failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen = {row_dedupe_key(row) for row in html_failures}
    extra: list[dict[str, Any]] = []
    for row in all_rows:
        if not row_partial_limitation(row):
            continue
        key = row_dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)
        enriched = dict(row)
        enriched["_csv_outcome"] = "부분 범위 제한"
        enriched["_csv_note"] = "HTML 오답 노트 제외·CSV 전용"
        extra.append(enriched)
    return extra


def success_rows_for_csv(
    all_rows: list[dict[str, Any]],
    html_failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """HTML 오답 노트에 없고 task_success 합격인 문항."""
    html_keys = {row_dedupe_key(row) for row in html_failures}
    successes: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in all_rows:
        key = row_dedupe_key(row)
        if key in html_keys or key in seen:
            continue
        if row.get("task_success") is not True:
            continue
        seen.add(key)
        enriched = dict(row)
        parts = ["태스크 성공"]
        if row_partial_limitation(row):
            parts.append("부분 범위 제한")
        enriched["_csv_outcome"] = " · ".join(parts)
        enriched["_csv_note"] = "HTML 오답 노트 제외·합격"
        successes.append(enriched)
    return successes


def write_failures_csv(
    path: Path,
    html_failures: list[dict[str, Any]],
    *,
    all_rows: list[dict[str, Any]] | None = None,
) -> None:
    partial_only = (
        partial_limitation_rows_for_csv(all_rows, html_failures) if all_rows is not None else []
    )
    csv_rows: list[dict[str, Any]] = []
    for row in html_failures:
        enriched = dict(row)
        enriched["_csv_outcome"] = " | ".join(row.get("_failure_tags") or failure_tags(row))
        enriched.setdefault("_csv_note", "")
        csv_rows.append(enriched)
    csv_rows.extend(partial_only)
    write_eval_rows_csv(path, csv_rows)


def write_successes_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_eval_rows_csv(path, rows)


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


def _metrics_breakdown_table_html(by_group: list[dict[str, Any]]) -> str:
    return table_html(
        [
            "구분",
            "샘플 수 (n)",
            metric_label("doc_hit"),
            metric_label("retrieval_keyword_hit"),
            metric_label("context_precision"),
            metric_label("recall_at_5"),
            metric_label("mrr"),
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
                fmt_percent(row["mean_recall_at_5"]),
                fmt(row["mean_mrr"]),
                fmt(row["mean_f_score"]),
                fmt(row["mean_r_score"]),
                fmt(row["mean_s_score"]),
                fmt(row["mean_correctness_score"]),
                fmt_percent(row["mean_task_success"]),
                fmt_wrong_refusal_for_type_row(row),
                fmt_seconds(row["mean_total_latency_ms"]),
            ]
            for row in by_group
        ],
    )


def report_nav_html() -> str:
    links = (
        ("#summary", "요약"),
        ("#outcomes", "태스크 결과"),
        ("#legend", "지표 정의"),
        ("#by-type", "유형별 통계"),
        ("#by-focus", "근거 유형"),
        ("#charts", "시각화"),
        ("#failures", "오답 노트"),
    )
    items = "".join(
        f'<li><a href="{html.escape(href)}">{html.escape(label)}</a></li>'
        for href, label in links
    )
    return f'<nav aria-label="리포트 목차"><ul class="report-nav">{items}</ul></nav>'


def question_type_table_html(by_type: list[dict[str, Any]]) -> str:
    return _metrics_breakdown_table_html(by_type)


def eval_focus_table_html(by_focus: list[dict[str, Any]]) -> str:
    return _metrics_breakdown_table_html(by_focus)


def summary_cards_html(summary: dict[str, Any]) -> str:
    def card(field: str, value: str) -> str:
        label = html.escape(metric_label(field))
        return f'<div class="card"><div>{label}</div><div class="metric">{value}</div></div>'

    retrieval = "".join(
        card(
            key,
            fmt(summary[f"mean_{key}"])
            if key == "mrr"
            else (
                fmt_percent(summary[f"mean_{key}"])
                if key != "total_latency_ms"
                else fmt_seconds(summary["mean_total_latency_ms"])
            ),
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


def outcome_summary_html(
    outcomes: dict[str, int],
    *,
    pool_count: int,
    displayed_failures: int,
    total: int,
) -> str:
    return f"""
    <section id="outcomes" class="report-section">
    <h2>✅ 태스크 결과 요약</h2>
    <div class="cards">
        <div class="card"><div>태스크 합격 (task_success ✓)</div>
        <div class="metric">{outcomes["passed"]}건</div></div>
        <div class="card"><div>태스크 불합격 (task_success ✗)</div>
        <div class="metric">{outcomes["failed"]}건</div></div>
        <div class="card"><div>적절 거절 · 문서 외</div>
        <div class="metric">{outcomes["appropriate_refusal"]}건</div></div>
        <div class="card"><div>조치 대상 오답 후보</div>
        <div class="metric">{pool_count}건</div></div>
    </div>
    <p class="note">
    <strong>태스크 합격/불합격</strong>은 최종 답변 품질 기준입니다.
    <strong>적절 거절</strong>은 문서 외(<code>unanswerable</code>) 질문에서 올바르게 거절한 합격 사례입니다.
    <strong>조치 대상 오답 후보</strong>는 합격·적절 거절·「검색만 나쁜데 답은 맞음」을 제외한 디버깅 풀입니다.
    HTML 오답 노트는 후보 {pool_count}건 중 우선순위 상위 <strong>{displayed_failures}건</strong>만 표시합니다.
    CSV(<code>eval_failures.csv</code>) 건수가 harness의 <code>task_success=✗</code>와 다를 수 있습니다.
    </p>
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
    <dd>답변 가능 질문에서 <strong>본문 없이 전면 거절</strong>한 비율입니다.
    일부만 「문서에 없음」이라고 적은 <strong>부분 범위 제한</strong> 답변은 포함하지 않습니다.<br>
    • <strong>주의</strong>: 답변 불가능 질문은 계산에서 제외(해당 없음)됩니다.<br>
    • <strong>doc_hit=0</strong>이어도 문서 외 질문에서 적절히 거절하면 합격이며, 오답 풀에 넣지 않습니다.<br>
    • <strong>부분 범위 제한</strong>·<strong>합격 사례</strong>는 HTML 오답 노트에는 넣지 않고
    <code>eval_failures.csv</code> / <code>eval_successes.csv</code>로 확인합니다.</dd>
    </dl>

    <p class="legend-footnote">💡 <strong>오답 노트·CSV 구분:</strong>
    HTML 오답 노트와 <code>eval_failures.csv</code>는 <strong>조치가 필요한 실패 후보</strong>만 모읍니다.
    harness 전체 불합격 건수와 CSV 행 수가 다를 수 있습니다(적절 거절·검색 저하 합격 제외).
    <strong>무단 거절(전면)</strong> <span class="tag tag-refusal">노란색</span>,
    동일 행에 부분 한계가 있으면 <span class="tag tag-partial">부분 범위 제한</span> 태그,
    <strong>환각 의심</strong> <span class="tag tag-hallucination">붉은색</span>.
    HTML 목록 건수는 <code>--top-n</code>, 부분 범위 제한 전용 행은 CSV에 추가됩니다.</p>
    </div>"""


def render_html(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    failure_pool_count: int,
) -> str:
    summary = summarize(rows)
    outcomes = summarize_task_outcomes(rows)
    by_type = summarize_by_question_type(rows)
    by_focus = summarize_by_eval_focus(rows)
    pool_rows = [row for row in rows if should_include_in_failures(row)]

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
  전체 <strong>{summary["n"]}</strong>문항 ·
  합격 <strong>{outcomes["passed"]}</strong> /
  불합격 <strong>{outcomes["failed"]}</strong> ·
  적절 거절 <strong>{outcomes["appropriate_refusal"]}</strong> ·
  오답 노트 <strong>{len(failures)}</strong>건 (후보 {failure_pool_count}건)
</p>
{report_nav_html()}
{summary_cards_html(summary)}
{outcome_summary_html(
    outcomes,
    pool_count=failure_pool_count,
    displayed_failures=len(failures),
    total=summary["n"],
)}
{metrics_legend_fold_html()}
<section id="by-type" class="report-section">
<h2>📋 질문 성격별 세부 지표 통계</h2>
<p class="note">질문 유형별 평균 지표입니다. 무단 거절 오류율은 답변 가능 유형만 집계합니다.</p>
{question_type_table_html(by_type)}
</section>
<section id="by-focus" class="report-section">
<h2>🖼️ 근거 유형별 세부 지표 (eval_focus)</h2>
<p class="note">질문셋의 <code>eval_focus</code> 필드 기준입니다.
<code>ocr_image</code>는 PaddleOCR 등 이미지 청크 근거 질문,
<code>text</code>는 HWP 본문·표 청크 근거 질문입니다. 미분류는 필드가 비어 있거나 알 수 없는 값입니다.</p>
{eval_focus_table_html(by_focus)}
</section>
<section id="charts" class="report-section">
<p class="note">유형별 막대 차트입니다. 섹션 제목을 클릭하면 접거나 펼칠 수 있습니다.</p>
{retrieval_charts}
{generation_charts}
{answer_charts}
</section>
<section id="failures" class="report-section">
<h2>📝 디버깅 오답 노트 (Error Analysis)</h2>
<p class="note">무단 거절·오답·환각·거절 누락 등 <strong>조치가 필요한</strong> 실패만 표시합니다.
문서 외 적절 거절·태스크 합격(검색만 저조)은 제외됩니다.
합격·부분 범위 제한은 <code>eval_successes.csv</code> /
<code>eval_failures.csv</code>를 참고하세요. 표는 가로 스크롤이 가능합니다.</p>
<h3>실패 유형 분포 (조치 대상 후보 {failure_pool_count}건)</h3>
{failure_reason_table_html(failure_reason_counts(pool_rows))}
{failure_table_html(failures)}
</section>
</div>
</body>
</html>"""


def build_report(
    input_path: Path,
    html_output: Path,
    failures_output: Path,
    top_n: int,
    *,
    successes_output: Path | None = None,
) -> None:
    rows = enrich_eval_rows(read_jsonl(input_path))
    if not rows:
        print("데이터가 없습니다.")
        return

    pool_count = count_failure_candidates(rows)
    outcomes = summarize_task_outcomes(rows)
    failures = failure_rows(rows, top_n)
    html_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.write_text(
        render_html(rows, failures, failure_pool_count=pool_count),
        encoding="utf-8",
    )
    partial_only = partial_limitation_rows_for_csv(rows, failures)
    write_failures_csv(failures_output, failures, all_rows=rows)

    successes_path = successes_output or failures_output.with_name("eval_successes.csv")
    successes = success_rows_for_csv(rows, failures)
    write_successes_csv(successes_path, successes)

    print(f"리포트 생성 완료: {html_output}")
    print(
        f"태스크 결과: 합격 {outcomes['passed']} · 불합격 {outcomes['failed']} · "
        f"적절 거절 {outcomes['appropriate_refusal']}"
    )
    print(
        f"오답 CSV: {failures_output} "
        f"(HTML {len(failures)}건 + 부분범위 CSV전용 {len(partial_only)}건 / 후보 {pool_count}건)"
    )
    print(f"합격 CSV: {successes_path} ({len(successes)}건)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/eval_harness_results.jsonl")
    parser.add_argument("--html-output", default="outputs/eval_report.html")
    parser.add_argument("--failures-output", default="outputs/eval_failures.csv")
    parser.add_argument(
        "--successes-output",
        default="outputs/eval_successes.csv",
        help="HTML 오답 노트에 없는 합격(task_success) 사례 CSV",
    )
    parser.add_argument("--top-n", type=int, default=20, help="오답 노트에 표시할 최대 건수")
    args = parser.parse_args()
    build_report(
        input_path=Path(args.input),
        html_output=Path(args.html_output),
        failures_output=Path(args.failures_output),
        top_n=args.top_n,
        successes_output=Path(args.successes_output),
    )


if __name__ == "__main__":
    main()
