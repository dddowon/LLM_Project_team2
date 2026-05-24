from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree

from src.config import load_config
from src.engine.rag import RagEngine
from src.engine.vector_store import ChromaVectorStore
from src.evaluation.generation import evaluate_generation_metrics
from src.evaluation.harness_metrics import evaluate_row_metrics, summarize_harness_results
from src.utils.jsonl import read_jsonl, write_jsonl


def _feedback_scores_safe(client: Client | None, scores: dict[str, float]) -> None:
    if client is None:
        return
    run_tree = get_current_run_tree()
    if run_tree is None:
        return
    run_id = run_tree.id
    for key, score in scores.items():
        if score is None:
            continue
        try:
            client.create_feedback(run_id=run_id, key=key, score=float(score))
        except Exception:
            continue


@traceable(name="llm_judge", run_type="llm", tags=["rag-harness", "judge"])
def _traced_generation_metrics(
    query: str,
    contexts: list[Any],
    answer: str,
    *,
    judge_model: str,
    run_llm_judge: bool,
    client: Client | None,
) -> dict[str, Any]:
    out = evaluate_generation_metrics(
        query,
        contexts,
        answer,
        judge_model=judge_model,
        run_llm_judge=run_llm_judge,
    )
    if run_llm_judge and isinstance(out.get("f_score"), int):
        _feedback_scores_safe(
            client,
            {
                "faithfulness_0_1": int(out["f_score"]) / 5.0,
                "relevance_0_1": int(out["r_score"]) / 5.0,
                "synthesis_0_1": int(out["s_score"]) / 5.0,
            },
        )
    return out


@traceable(name="rag_eval_row", run_type="chain", tags=["rag-harness"])
def _eval_one_row(
    engine: RagEngine,
    row: dict[str, Any],
    *,
    judge_model: str,
    run_llm_judge: bool,
    run_correctness_judge: bool,
    langsmith_client: Client | None,
) -> dict[str, Any]:
    question = row["question"]
    doc_id = str(row.get("doc_id") or "").strip() or None

    start = time.perf_counter()
    result = engine.answer(question, include_source_text=True, doc_id=doc_id)
    total_latency_ms = round((time.perf_counter() - start) * 1000, 2)
    contexts: list[Any] = list(result.get("sources") or [])
    answer = str(result.get("answer", ""))

    retrieval_and_answer = evaluate_row_metrics(
        row,
        answer=answer,
        sources=contexts,
        judge_model=judge_model,
        run_llm_judge=False,
        run_correctness_judge=run_correctness_judge,
    )
    generation = _traced_generation_metrics(
        question,
        contexts,
        answer,
        judge_model=judge_model,
        run_llm_judge=run_llm_judge,
        client=langsmith_client,
    )

    merged: dict[str, Any] = {
        **row,
        "answer": answer,
        "sources": contexts,
        "resolved_doc_id": result.get("resolved_doc_id"),
        "total_latency_ms": total_latency_ms,
        **retrieval_and_answer,
        **{k: v for k, v in generation.items() if k not in retrieval_and_answer},
    }
    if generation.get("judge_error"):
        merged["judge_error"] = generation["judge_error"]

    feedback: dict[str, float] = {"total_latency_ms": total_latency_ms}
    if merged.get("doc_hit") is not None:
        feedback["doc_hit"] = float(merged["doc_hit"])
    if merged.get("retrieval_keyword_hit") is not None:
        feedback["retrieval_keyword_hit"] = float(merged["retrieval_keyword_hit"])
    if merged.get("context_precision") is not None:
        feedback["context_precision"] = float(merged["context_precision"])
    if merged.get("task_success") is not None:
        feedback["task_success"] = 1.0 if merged.get("task_success") else 0.0
    if run_llm_judge and isinstance(merged.get("f_score"), int):
        feedback["faithfulness_0_1"] = int(merged["f_score"]) / 5.0
        feedback["relevance_0_1"] = int(merged["r_score"]) / 5.0
        feedback["synthesis_0_1"] = int(merged["s_score"]) / 5.0
    if isinstance(merged.get("correctness_score"), int):
        feedback["correctness_0_1"] = int(merged["correctness_score"]) / 5.0
    _feedback_scores_safe(langsmith_client, feedback)
    return merged


@traceable(name="rag_eval_batch", run_type="chain", tags=["rag-harness", "batch"])
def _eval_batch_traced(
    engine: RagEngine,
    rows: list[dict[str, Any]],
    *,
    judge_model: str,
    run_llm_judge: bool,
    run_correctness_judge: bool,
    langsmith_client: Client | None,
) -> list[dict[str, Any]]:
    from tqdm.auto import tqdm

    outputs: list[dict[str, Any]] = []
    progress = tqdm(rows, desc="Evaluating harness", unit="q")
    for i, row in enumerate(progress):
        doc_id = str(row.get("doc_id") or "").strip()
        if doc_id:
            progress.set_postfix_str(doc_id[:40] + ("…" if len(doc_id) > 40 else ""), refresh=False)
        outputs.append(
            _eval_one_row(
                engine,
                row,
                judge_model=judge_model,
                run_llm_judge=run_llm_judge,
                run_correctness_judge=run_correctness_judge,
                langsmith_client=langsmith_client,
                langsmith_extra={"metadata": {"row_index": i}},
            )
        )
    return outputs


def run_eval_harness(
    config_path: str | Path,
    *,
    evaluation_set: Path | None = None,
    output_path: Path | None = None,
    judge_model: str = "gpt-5-mini",
    run_llm_judge: bool = True,
    run_correctness_judge: bool = True,
    langsmith_feedback: bool = True,
) -> tuple[Path, dict[str, Any]]:
    config = load_config(config_path)
    eval_path = evaluation_set or config.paths.evaluation_set
    out_path = output_path or Path("outputs/eval_harness_results.jsonl")

    rows = read_jsonl(eval_path)
    if not rows:
        raise RuntimeError(f"평가 질문셋이 없습니다: {eval_path}")

    store = ChromaVectorStore.load(config.paths.index_dir)
    engine = RagEngine(config, store)

    ls_client: Client | None = None
    if langsmith_feedback and (
        os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    ):
        try:
            ls_client = Client()
        except Exception:
            ls_client = None

    results = _eval_batch_traced(
        engine,
        rows,
        judge_model=judge_model,
        run_llm_judge=run_llm_judge,
        run_correctness_judge=run_correctness_judge,
        langsmith_client=ls_client,
        langsmith_extra={"metadata": {"evaluation_set": str(eval_path)}},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, results)

    summary = summarize_harness_results(results)
    _feedback_scores_safe(
        ls_client,
        {
            k: float(v)
            for k, v in summary.items()
            if v is not None and k != "n" and isinstance(v, (int, float))
        },
    )
    return out_path, summary
