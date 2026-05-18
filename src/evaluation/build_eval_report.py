from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.utils.jsonl import read_jsonl


def as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def row_category(row: dict[str, Any]) -> str:
    category = str(row.get("category") or "").strip()
    return category or "uncategorized"


def score_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = as_number(row.get(key))
        if value is not None:
            values.append(value)
    return values


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    f_scores = score_values(rows, "f_score")
    r_scores = score_values(rows, "r_score")
    hits = score_values(rows, "retrieval_keyword_hit")
    categories = Counter(row_category(row) for row in rows)
    return {
        "n": len(rows),
        "mean_f_score": mean(f_scores),
        "mean_r_score": mean(r_scores),
        "mean_retrieval_keyword_hit": mean(hits),
        "category_count": dict(sorted(categories.items())),
    }


def summarize_by_category(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row_category(row)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for category, items in sorted(grouped.items()):
        summary_rows.append(
            {
                "category": category,
                "n": len(items),
                "mean_f_score": mean(score_values(items, "f_score")),
                "mean_r_score": mean(score_values(items, "r_score")),
                "mean_retrieval_keyword_hit": mean(score_values(items, "retrieval_keyword_hit")),
            }
        )
    return summary_rows


def failure_score(row: dict[str, Any]) -> float:
    f_score = as_number(row.get("f_score"))
    r_score = as_number(row.get("r_score"))
    hit = as_number(row.get("retrieval_keyword_hit"))
    total = 0.0
    total += f_score if f_score is not None else 5.0
    total += r_score if r_score is not None else 5.0
    if hit == 0.0:
        total -= 2.0
    return total


def failure_rows(rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        f_score = as_number(row.get("f_score"))
        r_score = as_number(row.get("r_score"))
        hit = as_number(row.get("retrieval_keyword_hit"))
        if (f_score is not None and f_score < 4) or (r_score is not None and r_score < 4) or hit == 0.0:
            candidates.append(row)
    return sorted(candidates, key=failure_score)[:top_n]


def source_ids(row: dict[str, Any]) -> str:
    sources = row.get("sources")
    if not isinstance(sources, list):
        return ""
    ids = []
    for source in sources[:5]:
        if isinstance(source, dict):
            ids.append(str(source.get("chunk_id", "")))
    return ", ".join(source_id for source_id in ids if source_id)


def write_failures_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "f_score",
        "r_score",
        "retrieval_keyword_hit",
        "question",
        "expected_answer",
        "answer",
        "source_ids",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "category": row_category(row),
                    "f_score": row.get("f_score"),
                    "r_score": row.get("r_score"),
                    "retrieval_keyword_hit": row.get("retrieval_keyword_hit"),
                    "question": row.get("question", ""),
                    "expected_answer": row.get("expected_answer", ""),
                    "answer": row.get("answer", ""),
                    "source_ids": source_ids(row),
                }
            )


def bar_svg(items: list[tuple[str, float | None]], *, max_value: float, width: int = 760) -> str:
    bar_height = 26
    gap = 10
    label_width = 210
    value_width = 60
    height = max(40, len(items) * (bar_height + gap) + 10)
    bar_width = width - label_width - value_width - 30
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">']
    for index, (label, value) in enumerate(items):
        y = 8 + index * (bar_height + gap)
        safe_label = html.escape(label[:40])
        display = "-" if value is None else f"{value:.2f}"
        ratio = 0.0 if value is None else max(0.0, min(1.0, value / max_value))
        current_width = round(bar_width * ratio, 1)
        parts.append(f'<text x="0" y="{y + 18}" class="svg-label">{safe_label}</text>')
        parts.append(f'<rect x="{label_width}" y="{y}" width="{bar_width}" height="{bar_height}" rx="5" class="bar-bg"/>')
        parts.append(f'<rect x="{label_width}" y="{y}" width="{current_width}" height="{bar_height}" rx="5" class="bar"/>')
        parts.append(f'<text x="{label_width + bar_width + 12}" y="{y + 18}" class="svg-value">{display}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def histogram_svg(values: list[float], *, max_score: int = 5, width: int = 760) -> str:
    counts = Counter(int(round(value)) for value in values)
    max_count = max(counts.values(), default=1)
    items = [(str(score), counts.get(score, 0) / max_count) for score in range(max_score + 1)]
    return bar_svg([(label, value) for label, value in items], max_value=1.0, width=width)


def table_html(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def render_html(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> str:
    summary = summarize(rows)
    by_category = summarize_by_category(rows)
    category_chart = bar_svg(
        [(row["category"], row["mean_f_score"]) for row in by_category],
        max_value=5.0,
    )
    relevance_chart = bar_svg(
        [(row["category"], row["mean_r_score"]) for row in by_category],
        max_value=5.0,
    )
    hit_chart = bar_svg(
        [(row["category"], row["mean_retrieval_keyword_hit"]) for row in by_category],
        max_value=1.0,
    )
    category_table = table_html(
        ["category", "n", "mean_f_score", "mean_r_score", "mean_retrieval_keyword_hit"],
        [
            [
                row["category"],
                row["n"],
                fmt(row["mean_f_score"]),
                fmt(row["mean_r_score"]),
                fmt(row["mean_retrieval_keyword_hit"]),
            ]
            for row in by_category
        ],
    )
    failure_table = table_html(
        ["category", "f_score", "r_score", "hit", "question", "source_ids"],
        [
            [
                row_category(row),
                row.get("f_score", "-"),
                row.get("r_score", "-"),
                row.get("retrieval_keyword_hit", "-"),
                str(row.get("question", ""))[:140],
                source_ids(row),
            ]
            for row in failures
        ],
    )
    raw_summary = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    f_hist = histogram_svg(score_values(rows, "f_score"))
    r_hist = histogram_svg(score_values(rows, "r_score"))

    return f"""<!doctype html>
    <html lang="ko">
    <head>
    <meta charset="utf-8">
    <title>RAG 평가 리포트</title>
    <style>
    body {{ font-family: Arial, "Malgun Gothic", sans-serif; margin: 32px; color: #222; }}
    h1, h2 {{ margin-top: 28px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 14px; background: #fafafa; }}
    .metric {{ font-size: 26px; font-weight: 700; margin-top: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f2f2f2; }}
    pre {{ background: #f7f7f7; padding: 12px; overflow-x: auto; }}
    .bar-bg {{ fill: #eee; }}
    .bar {{ fill: #4f7cff; }}
    .svg-label, .svg-value {{ font-size: 13px; fill: #222; }}
    </style>
    </head>
    <body>
    <h1>RAG 평가 리포트</h1>
    <div class="cards">
    <div class="card"><div>전체 질문 수</div><div class="metric">{summary["n"]}</div></div>
    <div class="card"><div>평균 답변 정확도</div><div class="metric">{fmt(summary["mean_f_score"])}</div></div>
    <div class="card"><div>평균 질문 이해도</div><div class="metric">{fmt(summary["mean_r_score"])}</div></div>
    <div class="card"><div>평균 문서 검색 성공률</div><div class="metric">{fmt(summary["mean_retrieval_keyword_hit"])}</div></div>
    </div>
    <h2>질문 유형별 요약</h2>
    {category_table}
    <h2>질문 유형별 답변의 정확도</h2>
    {category_chart}
    <h2>질문 유형별 질문 이해도</h2>
    {relevance_chart}
    <h2>질문 유형별 정보 검색 성능</h2>
    {hit_chart}
    <h2>답변 정확도 점수 분포</h2>
    {f_hist}
    <h2>질문 이해도 점수 분포</h2>
    {r_hist}
    <h2>낮은 점수 또는 검색 실패 사례</h2>
    {failure_table}
    <h2>원본 요약 데이터</h2>
    <pre>{raw_summary}</pre>
    </body>
    </html>
    """


def build_report(
    input_path: Path,
    html_output: Path,
    failures_output: Path,
    *,
    top_n: int,
) -> None:
    rows = read_jsonl(input_path)
    if not rows:
        raise RuntimeError(f"평가 결과 JSONL이 비어있습니다: {input_path}")
    failures = failure_rows(rows, top_n)
    html_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.write_text(render_html(rows, failures), encoding="utf-8")
    write_failures_csv(failures_output, failures)
    print(f"wrote_report: {html_output}")
    print(f"wrote_failures: {failures_output}")
    print(f"rows: {len(rows)}")
    print(f"failures: {len(failures)}")


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
