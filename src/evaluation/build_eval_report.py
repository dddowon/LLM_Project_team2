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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "mean_f_score": mean(score_values(rows, "f_score")),
        "mean_r_score": mean(score_values(rows, "r_score")),
        "mean_s_score": mean(score_values(rows, "s_score")),
        "mean_retrieval_keyword_hit": mean(score_values(rows, "retrieval_keyword_hit")),
        "mean_context_precision": mean(score_values(rows, "context_precision")),
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


def source_ids(row: dict[str, Any]) -> str:
    sources = row.get("sources")
    if not isinstance(sources, list):
        return ""
    ids = []
    for source in sources[:5]:
        if isinstance(source, dict):
            ids.append(str(source.get("chunk_id", "")))
    return ", ".join(source_id for source_id in ids if source_id)


def failure_score(row: dict[str, Any]) -> float:
    f_score = as_number(row.get("f_score"))
    r_score = as_number(row.get("r_score"))
    s_score = as_number(row.get("s_score"))
    hit = as_number(row.get("retrieval_keyword_hit"))
    precision = as_number(row.get("context_precision"))

    score = 0.0
    score += f_score if f_score is not None else 5.0
    score += r_score if r_score is not None else 5.0
    score += s_score if s_score is not None else 5.0
    if hit == 0.0:
        score -= 2.0
    if precision == 0.0:
        score -= 1.0
    return score


def failure_rows(rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
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
            candidates.append(row)
    return sorted(candidates, key=failure_score)[:top_n]


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
        "근거 청크",
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
                    "근거 청크": source_ids(row),
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
    failure_table = table_html(
        [
            "질문 유형",
            "질문 성격",
            "답변 정확도(f_score)",
            "질문 이해도(r_score)",
            "종합 능력(s_score)",
            "검색 성공(Hit Rate)",
            "문맥 정밀도(context_precision)",
            "질문",
            "근거 청크",
        ],
        [
            [
                row_category(row),
                row_question_type(row),
                row.get("f_score", "-"),
                row.get("r_score", "-"),
                row.get("s_score", "-"),
                row.get("retrieval_keyword_hit", "-"),
                row.get("context_precision", "-"),
                str(row.get("question", ""))[:140],
                source_ids(row),
            ]
            for row in failures
        ],
    )

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
