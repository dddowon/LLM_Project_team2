"""평가 JSONL → 리포트 HTML/오답 CSV 파이프라인 검증."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from src.evaluation.answer import (
    classify_answer_refusal,
    classify_failure_reason,
    evaluate_answer_metrics,
    is_full_refusal_answer,
)
from src.evaluation.build_eval_report import (
    build_report,
    count_failure_candidates,
    enrich_eval_rows,
    failure_rows,
    failure_tags,
    render_html,
    row_appropriate_refusal_success,
    should_include_in_failures,
    success_rows_for_csv,
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


def test_harness_merge_keeps_generation_scores_when_row_metrics_skips_judge() -> None:
    """구 merge 버그: base에 f_score=None이 있으면 generation 점수가 덮어쓰이지 않음."""
    base = {
        "doc_hit": 1.0,
        "f_score": None,
        "r_score": None,
        "s_score": None,
        "task_success": True,
    }
    generation = {
        "f_score": 4,
        "r_score": 5,
        "s_score": 4,
        "faithfulness_given_answer": 4,
    }

    buggy = {
        **base,
        **{k: v for k, v in generation.items() if k not in base},
    }
    assert buggy.get("f_score") is None

    fixed = {**base, **generation}
    assert fixed["f_score"] == 4


def test_row_metrics_omits_generation_keys_when_judge_off() -> None:
    from src.evaluation.harness_metrics import evaluate_row_metrics

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
    metrics = evaluate_row_metrics(
        row,
        answer="총 사업비는 12억 원입니다.",
        sources=sources,
        judge_model="gpt-5-mini",
        run_llm_judge=False,
        run_correctness_judge=False,
    )
    assert "f_score" not in metrics
    assert "r_score" not in metrics
    assert "s_score" not in metrics


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


def test_partial_limitation_is_not_full_refusal() -> None:
    answer = (
        "네트워크 스위치 요구(ECR-04): CPU Load Balancing, IPv6 지원, "
        "Static/Dynamic 라우팅 프로토콜 지원 등 문서에 명시되어 있습니다.\n"
        "제출 방식 등 추가 항목: 문서에서 확인되지 않습니다."
    )
    assert classify_answer_refusal(answer) == "partial"
    assert not is_full_refusal_answer(answer)


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
    assert out["refusal_kind"] == "full"
    assert out["wrong_refusal"] is True
    assert out["partial_limitation"] is False
    assert out["failure_reason"] == "wrong_refusal"


def test_unanswerable_doc_hit_zero_not_wrong_doc() -> None:
    reason = classify_failure_reason(
        question_type="unanswerable",
        doc_hit=0.0,
        keyword_hit=0.0,
        is_refusal=True,
        correctness_pass=None,
        has_doc_id=True,
    )
    assert reason is None


def test_appropriate_refusal_excluded_from_failure_pool() -> None:
    row = {
        "question": "다른 사업 예산은?",
        "doc_id": "doc_budget",
        "question_type": "unanswerable",
        "answer": "제공된 문서에서 확인되지 않습니다.",
        "doc_hit": 0.0,
        "retrieval_keyword_hit": 0.0,
        "task_success": True,
    }
    enriched = enrich_eval_rows([row])[0]
    assert enriched["failure_reason"] is None
    assert row_appropriate_refusal_success(enriched)
    assert not should_include_in_failures(enriched)


def test_task_success_with_low_retrieval_only_excluded_from_pool() -> None:
    rows = _load_fixture_rows()
    enriched = enrich_eval_rows(rows)
    low_retrieval_pass = next(
        r for r in enriched if r.get("failure_reason") == "low_retrieval" and r.get("task_success")
    )
    assert not should_include_in_failures(low_retrieval_pass)


def test_build_report_from_fixture() -> None:
    rows = _load_fixture_rows()
    assert len(rows) == 9

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        html_out = tmp_path / "report.html"
        csv_out = tmp_path / "failures.csv"
        successes_out = tmp_path / "successes.csv"
        build_report(
            input_path=FIXTURE,
            html_output=html_out,
            failures_output=csv_out,
            top_n=50,
            successes_output=successes_out,
        )
        html = html_out.read_text(encoding="utf-8")
        assert html_out.exists() and csv_out.exists() and successes_out.exists()

        summary_pos = html.index('id="summary"')
        outcomes_pos = html.index('id="outcomes"')
        legend_pos = html.index('id="legend"')
        by_type_pos = html.index('id="by-type"')
        failures_pos = html.index('id="failures"')
        assert summary_pos < outcomes_pos < legend_pos < by_type_pos < failures_pos

        assert "무단 거절 오류율 (wrong_refusal)" in html
        assert "조치 대상 오답 후보" in html
        assert "group-retrieval" in html
        assert ">None<" not in html

        enriched = enrich_eval_rows(rows)
        failures = failure_rows(enriched, top_n=50)
        assert len(failures) == count_failure_candidates(enriched)
        assert len(failures) == 4

        with csv_out.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            csv_rows = list(reader)
        assert len(csv_rows) >= len(failures)
        assert "무단 거절 여부 (wrong_refusal)" in fieldnames
        assert "부분 범위 제한 (partial_limitation)" in fieldnames
        assert "결과 분류" in fieldnames

        expected_successes = success_rows_for_csv(enriched, failures)
        with successes_out.open(encoding="utf-8-sig", newline="") as f:
            success_reader = csv.DictReader(f)
            success_rows = list(success_reader)
        assert len(success_rows) == len(expected_successes)
        assert all(r.get("결과 분류", "").startswith("태스크 성공") for r in success_rows)


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
    rows = enrich_eval_rows(_load_fixture_rows())
    failures = failure_rows(rows, 50)
    html = render_html(rows, failures, failure_pool_count=count_failure_candidates(rows))
    assert "전체 <strong>9</strong>문항" in html
    assert "합격 <strong>5</strong>" in html
    assert "불합격 <strong>4</strong>" in html
