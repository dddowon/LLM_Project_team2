"""평가 JSONL → 리포트 HTML/오답 CSV 파이프라인 검증."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from src.evaluation.answer import evaluate_answer_metrics
from src.evaluation.build_eval_report import (
    build_report,
    count_failure_candidates,
    failure_rows,
    failure_tags,
    render_html,
)
from src.evaluation.harness_metrics import evaluate_row_metrics
from src.evaluation.retrieval import evaluate_retrieval_metrics

FIXTURE = Path(__file__).resolve().parents[1] / "outputs" / "_eval_report_example.jsonl"

HARNESS_METRIC_KEYS = frozenset(
    {
        "doc_hit",
        "retrieval_keyword_hit",
        "context_precision",
        "task_success",
        "wrong_refusal",
        "is_refusal",
    }
)


def _load_fixture_rows() -> list[dict]:
    rows: list[dict] = []
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def test_harness_metrics_shape_on_synthetic_row() -> None:
    row = {
        "question": "예산은?",
        "expected_answer": "12억",
        "question_type": "fact",
        "doc_id": "예산안",
        "ground_truth_keywords": ["12억"],
    }
    sources = [
        {
            "chunk_id": "c1",
            "text": "총 사업비 12억 원",
            "metadata": {"file_name": "예산안.hwp"},
        }
    ]
    retrieval = evaluate_retrieval_metrics(row, sources)
    assert retrieval["doc_hit"] == 1.0

    metrics = evaluate_row_metrics(
        row,
        answer="총 사업비는 12억 원입니다.",
        sources=sources,
        judge_model="gpt-5-mini",
        run_llm_judge=False,
        run_correctness_judge=False,
    )
    for key in HARNESS_METRIC_KEYS:
        assert key in metrics, f"missing {key}"
    assert metrics["wrong_refusal"] is False
    assert metrics["task_success"] is True


def test_wrong_refusal_detected_without_field() -> None:
    row = {
      "question": "예산?",
      "expected_answer": "12억",
      "question_type": "fact",
      "answer": "제공된 문서에서 확인되지 않습니다.",
      "is_refusal": True,
  }
    out = evaluate_answer_metrics(
        row,
        row["answer"],
        judge_model="gpt-5-mini",
        run_correctness_judge=False,
        doc_hit=1.0,
        keyword_hit=0.0,
    )
    assert out["wrong_refusal"] is True
    assert out["failure_reason"] == "wrong_refusal"


def test_build_report_from_fixture() -> None:
    rows = _load_fixture_rows()
    assert len(rows) == 9

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        html_out = tmp_path / "report.html"
        csv_out = tmp_path / "failures.csv"
        build_report(
            input_path=FIXTURE,
            html_output=html_out,
            failures_output=csv_out,
            top_n=50,
        )
        html = html_out.read_text(encoding="utf-8")
        assert html_out.exists() and csv_out.exists()

        summary_pos = html.index('id="summary"')
        legend_pos = html.index('id="legend"')
        by_type_pos = html.index('id="by-type"')
        failures_pos = html.index('id="failures"')
        assert summary_pos < legend_pos < by_type_pos < failures_pos

        assert "무단 거절 오류율 (wrong_refusal)" in html
        assert "group-retrieval" in html
        assert ">None<" not in html

        failures = failure_rows(rows, top_n=50)
        assert len(failures) == count_failure_candidates(rows)
        assert len(failures) >= 5

        with csv_out.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            csv_rows = list(reader)
        assert len(csv_rows) == len(failures)
        assert "무단 거절 여부 (wrong_refusal)" in fieldnames


def test_failure_tags_cover_fixture_reasons() -> None:
    rows = _load_fixture_rows()
    reasons = {str(r.get("failure_reason") or "") for r in rows if r.get("failure_reason")}
    tagged: set[str] = set()
    for row in rows:
        if row.get("failure_reason") or row.get("task_success") is False:
            tagged.update(failure_tags(row))
    assert "wrong_refusal" in reasons
    assert any("무단 거절" in t or "Wrong Refusal" in t for t in tagged)


def test_render_html_failure_pool_count() -> None:
    rows = _load_fixture_rows()
    failures = failure_rows(rows, 50)
    html = render_html(rows, failures, failure_pool_count=count_failure_candidates(rows))
    assert "전체 <strong>9</strong>문항" in html
