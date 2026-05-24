from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.utils.jsonl import read_jsonl


QUESTION_TYPE_LABELS = {
    "fact": "사실 확인",
    "summary": "요약",
    "comparison": "비교",
    "follow_up": "후속 질문",
    "requirement_detail": "요구사항 상세",
    "unanswerable": "문서 외 질문",
}


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


def metrics_legend_html() -> str:
    """
    리포트 상단에 표시할 지표 정의(코드: llm_judge, retrieval, harness와 동일한 의미).
    
    Parameters:
    - 없음: 매개변수 없이 고정된 HTML 문자열 템플릿을 반환합니다.
    """
    return """<div class="metrics-legend">
    <h2>📘 지표 설명</h2>
    <p class="legend-intro">아래 수치들은 <strong>src/evaluation</strong>의 RAG 평가 하네스(Harness)에서 계산됩니다. LLM 판정 점수(f/r/s)는 절대적인 정답률을 의미하기보다, 동일한 기준 아래 여러 실험(실행 간) 성능을 <strong>상대 비교할 때 유용합니다.</strong></p>

    <dl>
    <dt>전체 질문 수</dt>
    <dd>이번 성능 평가에 포함된 총 질문(데이터셋의 행)의 개수입니다.</dd>

    <dt>응답 시간 (total_latency_ms)</dt>
    <dd>해당 질문 한 건에 대해 문서 검색, 답변 생성, LLM 판정 등이 끝날 때까지 걸린 총 시간(밀리초, ms)입니다. 리포트 상단 카드에는 직관성을 위해 초 단위로 환산하여 표시합니다.</dd>

    <dt>답변 정확도 (f_score, Faithfulness)</dt>
    <dd>판정 모델이 바라본 답변의 <strong>충실도(환각 여부)</strong>입니다. 검색으로 가져온 근거 텍스트에 기반하여, 거짓 정보(환각) 없이 답변이 올바르게 작성되었는지를 0~5점 사이의 정수로 채점합니다.</dd>

    <dt>질문 이해도 (r_score, Relevance)</dt>
    <dd>생성된 답변이 사용자의 질문 의도에 얼마나 직접적이고 적절하게 대응하는지(<strong>적합성</strong>)를 따져 0~5점 사이의 정수로 채점합니다.</dd>

    <dt>종합 능력 (s_score, Synthesis)</dt>
    <dd>분산된 여러 근거 데이터를 하나로 묶어, 사용자의 질문 의도에 맞게 논리적으로 잘 <strong>구조화(종합)</strong>했는지를 0~5점 사이의 정수로 채점합니다.</dd>

    <dt>검색 성공률 (Hit Rate, retrieval_keyword_hit)</dt>
    <dd>질문셋에 정의된 정답 키워드(<code>ground_truth_keywords</code>) 중 <strong>단 하나라도</strong> 이번에 검색된 모든 청크 텍스트를 이어 붙인 문자열 안에 부분 문자열로 포함되어 있으면 1, 없으면 0으로 판정합니다. 질문별 결과는 0 또는 1이며, 리포트 상단 카드 및 표에 표기되는 퍼센트(%)는 이 결과들의 <strong>평균값</strong>입니다. 정답 키워드가 비어 있는 경우는 의미 있는 히트(Hit)로 보기 어렵습니다.</dd>

    <dt>문맥 정밀도 (context_precision)</dt>
    <dd>모델이 가져온 총 <em>k</em>개의 청크 중, 정답 키워드가 실제 포함된 청크의 비율을 뜻합니다. 즉, <strong>(키워드가 포함된 청크 수 &divide; <em>k</em>)</strong> 값으로 0~1 사이의 범위를 가집니다. 검색 결과 안에서 &ldquo;관련 있어 보이는&rdquo; 알짜배기 청크의 비율에 가깝지만, 단어의 표현이 다르면 같은 의미를 지니고 있더라도 0으로 계산될 수 있어 기준이 다소 엄격합니다.</dd>
    </dl>

    <p class="legend-footnote"><strong>질문 유형</strong>은 평가 JSONL의 <code>category</code>(예: 기능 요구사항, 예산)이고, <strong>질문 성격</strong>은 <code>question_type</code>을 한글 라벨로 바꾼 값입니다.</p>
    </div>"""


def row_category(row: dict[str, Any]) -> str:
    category = str(row.get("category") or "").strip()
    return category or "미분류"


def row_question_type(row: dict[str, Any]) -> str:
    question_type = str(row.get("question_type") or "").strip()
    if not question_type:
        return "미분류"
    return QUESTION_TYPE_LABELS.get(question_type, question_type)


def score_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = [as_number(row.get(key)) for row in rows]
    return [value for value in values if value is not None]


def bool_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [1.0 if row.get(key) else 0.0 for row in rows if row.get(key) is not None]
    return mean(values)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "mean_doc_hit": mean(score_values(rows, "doc_hit")),
        "mean_f_score": mean(score_values(rows, "f_score")),
        "mean_r_score": mean(score_values(rows, "r_score")),
        "mean_s_score": mean(score_values(rows, "s_score")),
        "mean_correctness_score": mean(score_values(rows, "correctness_score")),
        "mean_retrieval_keyword_hit": mean(score_values(rows, "retrieval_keyword_hit")),
        "mean_context_precision": mean(score_values(rows, "context_precision")),
        "mean_task_success": bool_rate(rows, "task_success"),
        "mean_wrong_refusal": bool_rate(rows, "wrong_refusal"),
        "mean_total_latency_ms": mean(score_values(rows, "total_latency_ms")),
    }


def summarize_by_field(rows: list[dict[str, Any]], field_name: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = row_question_type(row) if field_name == "question_type" else row_category(row)
        grouped[label].append(row)

    summary_rows: list[dict[str, Any]] = []
    for label, items in sorted(grouped.items()):
        summary_rows.append(
            {
                "label": label,
                "n": len(items),
                "mean_f_score": mean(score_values(items, "f_score")),
                "mean_r_score": mean(score_values(items, "r_score")),
                "mean_s_score": mean(score_values(items, "s_score")),
                "mean_retrieval_keyword_hit": mean(score_values(items, "retrieval_keyword_hit")),
                "mean_context_precision": mean(score_values(items, "context_precision")),
                "mean_total_latency_ms": mean(score_values(items, "total_latency_ms")),
            }
        )
    return summary_rows


def source_chunk_id_list(row: dict[str, Any]) -> list[str]:
    """검색 sources에 포함된 chunk_id를 순서대로(중복 제거)."""
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
    """CSV·엑셀용: 전체 ID를 한 줄로."""
    return ", ".join(source_chunk_id_list(row))


def source_chunk_ids_html(row: dict[str, Any]) -> str:
    """오답 노트 HTML: ID가 잘리지 않도록 목록으로 표시."""
    ids = source_chunk_id_list(row)
    if not ids:
        return '<span class="muted">(검색 근거 없음)</span>'
    items = "".join(f"<li><code>{html.escape(cid)}</code></li>" for cid in ids)
    return f'<ul class="chunk-id-list">{items}</ul>'


def source_file_names_short(row: dict[str, Any], *, max_chars: int = 120) -> str:
    """검색 근거에 붙은 metadata.file_name 등(어느 HWP/문서인지)."""
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
    if len(joined) <= max_chars:
        return joined
    return joined[: max_chars - 1] + "…"


def cell_text_html(text: Any, *, max_chars: int = 1200) -> str:
    """테이블 셀용: 이스케이프 후 줄바꿈을 <br>로 (스크롤은 CSS로)."""
    raw = "" if text is None else str(text)
    if len(raw) > max_chars:
        raw = raw[: max_chars - 1] + "…"
    return html.escape(raw).replace("\n", "<br/>")


def failure_table_html(failures: list[dict[str, Any]]) -> str:
    """모델 답변은 본문 행만 두고, 근거 문서·청크는 바로 아래 행에 블록으로 분리한다."""
    headers = [
        "질문 유형",
        "질문 성격",
        "f",
        "r",
        "s",
        "검색",
        "문맥",
        "질문",
        "기대 답변",
        "모델 답변",
    ]
    ncols = len(headers)
    head = "".join(f'<th class="col-h-{index}">{html.escape(h)}</th>' for index, h in enumerate(headers))
    body_rows: list[str] = []
    for row in failures:
        main_cells = [
            html.escape(row_category(row)),
            html.escape(row_question_type(row)),
            html.escape(str(row.get("f_score", "-"))),
            html.escape(str(row.get("r_score", "-"))),
            html.escape(str(row.get("s_score", "-"))),
            html.escape(str(row.get("retrieval_keyword_hit", "-"))),
            html.escape(str(row.get("context_precision", "-"))),
            f'<div class="failure-text">{cell_text_html(row.get("question"), max_chars=600)}</div>',
            f'<div class="failure-text">{cell_text_html(row.get("expected_answer"), max_chars=1200)}</div>',
            f'<div class="failure-text">{cell_text_html(row.get("answer"), max_chars=1200)}</div>',
        ]
        body_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in main_cells) + "</tr>")
        evidence_inner = (
            '<div class="failure-evidence-wrap">'
            '<div class="evidence-section-title">근거 문서</div>'
            f'<div class="failure-sources failure-sources-doc">{html.escape(source_file_names_short(row, max_chars=500))}</div>'
            '<div class="evidence-section-title">근거 청크 ID</div>'
            '<div class="failure-sources failure-sources-ids">'
            f"{source_chunk_ids_html(row)}"
            "</div>"
            "</div>"
        )
        body_rows.append(
            f'<tr class="failure-evidence-row"><td colspan="{ncols}">{evidence_inner}</td></tr>'
        )
    return (
        '<table class="failure-detail"><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )


FAILURE_PRIORITY = {
    "wrong_doc": 0,
    "wrong_refusal": 1,
    "wrong_answer": 2,
    "low_retrieval": 3,
    "should_refuse": 4,
}


def failure_sort_key(row: dict[str, Any]) -> tuple[int, float, float]:
    reason = str(row.get("failure_reason") or "")
    priority = FAILURE_PRIORITY.get(reason, 99)
    doc_hit = as_number(row.get("doc_hit"))
    correctness = as_number(row.get("correctness_score"))
    return (
        priority,
        doc_hit if doc_hit is not None else 1.0,
        correctness if correctness is not None else 5.0,
    )


def failure_rows(rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row.get("failure_reason")]
    legacy: list[dict[str, Any]] = []
    for row in rows:
        if row.get("failure_reason"):
            continue
        f_score = as_number(row.get("f_score"))
        r_score = as_number(row.get("r_score"))
        s_score = as_number(row.get("s_score"))
        hit = as_number(row.get("retrieval_keyword_hit"))
        precision = as_number(row.get("context_precision"))
        if (
            (f_score is not None and f_score < 4)
            or (r_score is not None and r_score < 4)
            or (s_score is not None and s_score < 4)
            or hit == 0.0
            or precision == 0.0
        ):
            legacy.append(row)
    merged = candidates + legacy
    return sorted(merged, key=failure_sort_key)[:top_n]


def write_failures_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "질문 유형",
        "질문 성격",
        "답변 정확도",
        "질문 이해도",
        "종합 능력",
        "검색 성공",
        "문맥 정밀도",
        "응답 시간",
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
                    "질문 유형": row_category(row),
                    "질문 성격": row_question_type(row),
                    "답변 정확도": row.get("f_score"),
                    "질문 이해도": row.get("r_score"),
                    "종합 능력": row.get("s_score"),
                    "검색 성공": row.get("retrieval_keyword_hit"),
                    "문맥 정밀도": row.get("context_precision"),
                    "응답 시간": fmt_seconds(as_number(row.get("total_latency_ms"))),
                    "질문": row.get("question", ""),
                    "기대 답변": row.get("expected_answer", ""),
                    "모델 답변": row.get("answer", ""),
                    "근거 문서": source_file_names_short(row),
                    "근거 청크 ID": source_chunk_ids_csv_field(row),
                }
            )


def bar_svg(items: list[tuple[str, float | None]], *, max_value: float, width: int = 760) -> str:
    bar_height, gap, label_width, value_width = 26, 10, 210, 60
    height = max(40, len(items) * (bar_height + gap) + 10)
    bar_width = width - label_width - value_width - 30
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']

    for index, (label, value) in enumerate(items):
        y = 8 + index * (bar_height + gap)
        ratio = 0.0 if value is None else max(0.0, min(1.0, value / max_value))
        current_width = round(bar_width * ratio, 1)
        display = fmt(value)
        parts.append(
            f'<text x="0" y="{y + 18}" class="svg-label" style="font-size:12px;">'
            f"{html.escape(label[:40])}</text>"
        )
        parts.append(
            f'<rect x="{label_width}" y="{y}" width="{bar_width}" height="{bar_height}" '
            'rx="5" style="fill:#f0f2f5;"/>'
        )
        parts.append(
            f'<rect x="{label_width}" y="{y}" width="{current_width}" height="{bar_height}" '
            'rx="5" style="fill:#4f7cff;"/>'
        )
        parts.append(
            f'<text x="{label_width + bar_width + 12}" y="{y + 18}" '
            f'class="svg-value" style="font-size:12px;">{display}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def table_html(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def render_html(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> str:
    summary = summarize(rows)
    by_category = summarize_by_field(rows, "category")
    by_question_type = summarize_by_field(rows, "question_type")

    category_chart = bar_svg(
        [(row["label"], row["mean_f_score"]) for row in by_category],
        max_value=5.0,
    )
    relevance_chart = bar_svg(
        [(row["label"], row["mean_r_score"]) for row in by_category],
        max_value=5.0,
    )
    synthesis_chart = bar_svg(
        [(row["label"], row["mean_s_score"]) for row in by_category],
        max_value=5.0,
    )
    hit_chart = bar_svg(
        [(row["label"], row["mean_retrieval_keyword_hit"]) for row in by_category],
        max_value=1.0,
    )
    precision_chart = bar_svg(
        [(row["label"], row["mean_context_precision"]) for row in by_category],
        max_value=1.0,
    )
    type_f_chart = bar_svg(
        [(row["label"], row["mean_f_score"]) for row in by_question_type],
        max_value=5.0,
    )
    type_relevance_chart = bar_svg(
        [(row["label"], row["mean_r_score"]) for row in by_question_type],
        max_value=5.0,
    )
    type_synthesis_chart = bar_svg(
        [(row["label"], row["mean_s_score"]) for row in by_question_type],
        max_value=5.0,
    )
    type_hit_chart = bar_svg(
        [(row["label"], row["mean_retrieval_keyword_hit"]) for row in by_question_type],
        max_value=1.0,
    )
    type_precision_chart = bar_svg(
        [(row["label"], row["mean_context_precision"]) for row in by_question_type],
        max_value=1.0,
    )

    category_table = table_html(
        [
            "질문 유형",
            "질문 수",
            "답변 정확도(f_score)",
            "질문 이해도(r_score)",
            "종합 능력(s_score)",
            "검색 성공률(Hit Rate)",
            "문맥 정밀도(context_precision)",
            "평균 응답 시간(total_latency_ms)",
        ],
        [
            [
                row["label"],
                f"{row['n']}개",
                fmt(row["mean_f_score"]),
                fmt(row["mean_r_score"]),
                fmt(row["mean_s_score"]),
                fmt_percent(row["mean_retrieval_keyword_hit"]),
                fmt_percent(row["mean_context_precision"]),
                fmt_seconds(row["mean_total_latency_ms"]),
            ]
            for row in by_category
        ],
    )
    question_type_table = table_html(
        [
            "질문 성격",
            "질문 수",
            "답변 정확도(f_score)",
            "질문 이해도(r_score)",
            "종합 능력(s_score)",
            "검색 성공률(Hit Rate)",
            "문맥 정밀도(context_precision)",
            "평균 응답 시간(total_latency_ms)",
        ],
        [
            [
                row["label"],
                f"{row['n']}개",
                fmt(row["mean_f_score"]),
                fmt(row["mean_r_score"]),
                fmt(row["mean_s_score"]),
                fmt_percent(row["mean_retrieval_keyword_hit"]),
                fmt_percent(row["mean_context_precision"]),
                fmt_seconds(row["mean_total_latency_ms"]),
            ]
            for row in by_question_type
        ],
    )
    failure_table = failure_table_html(failures)

    return f"""<!doctype html>
    <html lang="ko">
    <head>
    <meta charset="utf-8">
    <title>RAG 성능 평가 리포트</title>
    <style>
    body {{ font-family: 'Malgun Gothic', Arial, sans-serif; margin: 32px; background-color: #f4f7f9; color: #222; }}
    h1 {{ text-align: center; color: #333; }}
    h2 {{ border-left: 5px solid #4f7cff; padding-left: 12px; margin-top: 40px; font-size: 18px; }}
    h3 {{ margin-top: 0; font-size: 15px; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 20px; }}
    .card {{ border-radius: 12px; padding: 20px; background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.05); text-align: center; }}
    .metric {{ font-size: 26px; font-weight: 700; color: #4f7cff; margin-top: 8px; }}
    .chart-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
    .chart-box {{ background: #fff; padding: 15px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
    .note {{ margin-top: 10px; color: #666; font-size: 13px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 15px; background: #fff; border-radius: 8px; overflow: hidden; }}
    th, td {{ border: 1px solid #eee; padding: 12px; text-align: center; }}
    th {{ background: #f8f9fa; }}
    table.failure-detail {{ font-size: 13px; width: 100%; table-layout: fixed; }}
    table.failure-detail th, table.failure-detail td {{ text-align: left; vertical-align: top; padding: 8px 10px; }}
    table.failure-detail th:nth-child(-n+7),
    table.failure-detail td:nth-child(-n+7) {{ text-align: center; white-space: nowrap; width: 3.2rem; }}
    table.failure-detail td:nth-child(1), table.failure-detail th:nth-child(1) {{ width: 6.5rem; }}
    table.failure-detail td:nth-child(2), table.failure-detail th:nth-child(2) {{ width: 6.5rem; }}
    table.failure-detail .failure-text {{
        max-width: 100%; max-height: 12rem; overflow: auto; line-height: 1.45; font-size: 12px; text-align: left;
        word-break: break-word; hyphens: auto;
    }}
    table.failure-detail .failure-sources {{
        font-size: 11px; color: #555; line-height: 1.3; word-break: break-word; overflow: auto;
    }}
    table.failure-detail .failure-sources-doc {{ max-height: 5rem; }}
    table.failure-detail .failure-sources-ids {{ max-height: none; }}
    table.failure-detail ul.chunk-id-list {{
        margin: 4px 0 0 1.1rem; padding: 0 0 0 0.4rem; font-size: 12px; line-height: 1.55;
    }}
    table.failure-detail ul.chunk-id-list code {{ font-size: 11px; }}
    table.failure-detail .muted {{ color: #888; font-size: 12px; }}
    table.failure-detail tr.failure-evidence-row td {{
        border-top: 1px dashed #dde3ea;
        background: #fafbfd;
        padding: 6px 10px 12px 10px;
    }}
    table.failure-detail .failure-evidence-wrap {{ text-align: left; }}
    table.failure-detail .evidence-section-title {{
        font-size: 11px; font-weight: 700; color: #4f7cff; margin: 10px 0 4px 0; letter-spacing: 0.02em;
    }}
    table.failure-detail .failure-evidence-wrap .evidence-section-title:first-child {{ margin-top: 2px; }}
    .metrics-legend {{ background: #fff; padding: 22px 26px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-top: 28px; max-width: 960px; margin-left: auto; margin-right: auto; }}
    .metrics-legend h2 {{ margin-top: 0; border-left: 5px solid #4f7cff; padding-left: 12px; font-size: 18px; }}
    .legend-intro {{ color: #444; font-size: 14px; line-height: 1.6; margin: 12px 0 8px 0; }}
    .metrics-legend dl {{ margin: 16px 0 0 0; }}
    .metrics-legend dt {{ font-weight: 700; color: #2c3e50; margin-top: 16px; font-size: 14px; }}
    .metrics-legend dt:first-of-type {{ margin-top: 0; }}
    .metrics-legend dd {{ margin: 6px 0 0 0; color: #444; font-size: 14px; line-height: 1.55; padding-left: 0; text-align: left; }}
    .legend-footnote {{ margin: 18px 0 0 0; color: #666; font-size: 13px; line-height: 1.5; }}
    .metrics-legend code {{ background: #f0f2f5; padding: 1px 6px; border-radius: 4px; font-size: 12px; }}
    </style>
    </head>
    <body>
    <h1>📊 RAG 성능 평가 리포트</h1>
    <div class="cards">
        <div class="card"><div>전체 질문 수</div><div class="metric">{summary["n"]}개</div></div>
        <div class="card"><div>평균 답변 정확도(f_score)</div><div class="metric">{fmt(summary["mean_f_score"])}</div></div>
        <div class="card"><div>평균 질문 이해도(r_score)</div><div class="metric">{fmt(summary["mean_r_score"])}</div></div>
        <div class="card"><div>평균 종합 능력(s_score)</div><div class="metric">{fmt(summary["mean_s_score"])}</div></div>
        <div class="card"><div>평균 검색 성공률(Hit Rate)</div><div class="metric">{fmt_percent(summary["mean_retrieval_keyword_hit"])}</div></div>
        <div class="card"><div>평균 응답 시간(total_latency_ms)</div><div class="metric">{fmt_seconds(summary["mean_total_latency_ms"])}</div></div>
    </div>
    {metrics_legend_html()}
    <h2>📈 유형별 지표 상세 분석</h2>
    <div class="chart-container">
        <div class="chart-box"><h3>🎯 답변의 정확도(f_score)</h3>{category_chart}</div>
        <div class="chart-box"><h3>💡 질문 이해도(r_score)</h3>{relevance_chart}</div>
        <div class="chart-box"><h3>🚀 종합 처리 능력(s_score)</h3>{synthesis_chart}</div>
        <div class="chart-box"><h3>🔍 정보 검색 성능(Hit Rate / retrieval_keyword_hit)</h3>{hit_chart}</div>
        <div class="chart-box"><h3>📌 문맥 정밀도(context_precision)</h3>{precision_chart}</div>
    </div>
    <h2>📋 질문 유형별 데이터 요약</h2>
    {category_table}
    <h2>📈 질문 성격별 지표 상세 분석</h2>
    <div class="chart-container">
        <div class="chart-box"><h3>🎯 답변의 정확도(f_score)</h3>{type_f_chart}</div>
        <div class="chart-box"><h3>💡 질문 이해도(r_score)</h3>{type_relevance_chart}</div>
        <div class="chart-box"><h3>🚀 종합 처리 능력(s_score)</h3>{type_synthesis_chart}</div>
        <div class="chart-box"><h3>🔍 정보 검색 성능(Hit Rate / retrieval_keyword_hit)</h3>{type_hit_chart}</div>
        <div class="chart-box"><h3>📌 문맥 정밀도(context_precision)</h3>{type_precision_chart}</div>
    </div>
    <h2>📋 질문 성격별 데이터 요약</h2>
    {question_type_table}
    <h2>📝 오답 노트 / 실패 사례</h2>
    <div class="note">답변 품질 점수가 낮거나 검색 성공률·문맥 정밀도가 0인 케이스를 우선 표시합니다.</div>
    <div class="note"><strong>모델 답변</strong>은 입력 JSONL의 <code>answer</code> 필드를 그대로 보여 줍니다. 과거 프롬프트가 답변 끝에 <code>source_id</code> 목록을 붙이도록 했다면 그 형태가 남아 있을 수 있습니다. 프롬프트를 바꾼 뒤에도 <strong>같은 JSONL로 리포트만 다시 만들면</strong> <code>answer</code>는 변하지 않습니다. <strong>평가 하네스를 다시 실행</strong>해 새 결과 파일을 생성해야 수정된 프롬프트가 반영됩니다. 오답 노트 아래에는 <strong>근거 문서</strong>·<strong>근거 청크 ID</strong>(<code>sources</code>)만 둡니다.</div>
    {failure_table}
    </body>
    </html>"""


def build_report(input_path: Path, html_output: Path, failures_output: Path, top_n: int) -> None:
    rows = read_jsonl(input_path)
    if not rows:
        print("데이터가 없습니다.")
        return

    failures = failure_rows(rows, top_n)
    html_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.write_text(render_html(rows, failures), encoding="utf-8")
    write_failures_csv(failures_output, failures)
    print(f"리포트 생성 완료: {html_output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/eval_harness_results.jsonl")
    parser.add_argument("--html-output", default="outputs/eval_report.html")
    parser.add_argument("--failures-output", default="outputs/eval_failures.csv")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()
    build_report(
        input_path=Path(args.input),
        html_output=Path(args.html_output),
        failures_output=Path(args.failures_output),
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
