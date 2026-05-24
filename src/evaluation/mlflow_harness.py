from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow

from src.config import AppConfig, load_config
from src.engine.rag import RagEngine
from src.engine.vector_store import ChromaVectorStore
from src.evaluation.harness_metrics import evaluate_row_metrics, summarize_harness_results
from src.utils.jsonl import read_jsonl, write_jsonl


DEFAULT_EXPERIMENT = "bidmate-rag-eval"


def _eval_one_row(
    engine: RagEngine,
    row: dict[str, Any],
    *,
    judge_model: str,
    run_llm_judge: bool,
    run_correctness_judge: bool,
) -> dict[str, Any]:
    question = row["question"]
    doc_id = str(row.get("doc_id") or "").strip() or None

    start = time.perf_counter()
    result = engine.answer(question, include_source_text=True, doc_id=doc_id)
    total_latency_ms = round((time.perf_counter() - start) * 1000, 2)
    contexts: list[Any] = list(result.get("sources") or [])
    answer = str(result.get("answer", ""))

    metrics = evaluate_row_metrics(
        row,
        answer=answer,
        sources=contexts,
        judge_model=judge_model,
        run_llm_judge=run_llm_judge,
        run_correctness_judge=run_correctness_judge,
    )

    return {
        **row,
        "answer": answer,
        "sources": contexts,
        "resolved_doc_id": result.get("resolved_doc_id"),
        "total_latency_ms": total_latency_ms,
        **metrics,
    }


def _log_config_params(config: AppConfig, config_path: Path) -> None:
    mlflow.log_param("config_path", str(config_path))
    mlflow.log_param("evaluation_set", str(config.paths.evaluation_set))
    mlflow.log_param("index_dir", str(config.paths.index_dir))
    mlflow.log_param("embedding_model", config.openai.embedding_model)
    mlflow.log_param("generation_model", config.openai.generation_model)
    mlflow.log_param("top_k", config.retrieval.top_k)
    mlflow.log_param("score_threshold", config.retrieval.score_threshold)
    mlflow.log_param("chunk_size", config.chunking.chunk_size)
    mlflow.log_param("chunk_overlap", config.chunking.chunk_overlap)


def _log_summary_metrics(summary: dict[str, Any]) -> None:
    for key, value in summary.items():
        if key == "n" or value is None:
            continue
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, float(value))
    if summary.get("n") is not None:
        mlflow.log_metric("n", float(summary["n"]))


def _log_artifacts(
    *,
    out_path: Path,
    config_path: Path,
    eval_path: Path,
) -> None:
    mlflow.log_artifact(str(out_path))
    if config_path.exists():
        mlflow.log_artifact(str(config_path))
    if eval_path.exists():
        mlflow.log_artifact(str(eval_path))


def run_eval_harness_mlflow(
    config_path: str | Path,
    *,
    evaluation_set: Path | None = None,
    output_path: Path | None = None,
    judge_model: str = "gpt-5-mini",
    run_llm_judge: bool = True,
    run_correctness_judge: bool = True,
    tracking_uri: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    run_name: str | None = None,
) -> tuple[Path, dict[str, Any], str]:
    config_path = Path(config_path)
    config = load_config(config_path)
    eval_path = evaluation_set or config.paths.evaluation_set
    out_path = output_path or Path("outputs/eval_harness_results.jsonl")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    mlflow.set_experiment(experiment_name)
    resolved_run_name = run_name or datetime.now(timezone.utc).strftime("eval_%Y%m%d_%H%M%S")

    rows = read_jsonl(eval_path)
    if not rows:
        raise RuntimeError(f"평가 질문셋이 없습니다: {eval_path}")

    store = ChromaVectorStore.load(config.paths.index_dir)
    engine = RagEngine(config, store)

    results: list[dict[str, Any]] = []
    with mlflow.start_run(run_name=resolved_run_name):
        _log_config_params(config, config_path)
        mlflow.log_param("judge_model", judge_model)
        mlflow.log_param("run_llm_judge", run_llm_judge)
        mlflow.log_param("run_correctness_judge", run_correctness_judge)

        from tqdm.auto import tqdm

        progress = tqdm(rows, desc="Evaluating harness", unit="q")
        for row in progress:
            doc_id = str(row.get("doc_id") or "").strip()
            if doc_id:
                progress.set_postfix_str(doc_id[:40] + ("…" if len(doc_id) > 40 else ""), refresh=False)
            results.append(
                _eval_one_row(
                    engine,
                    row,
                    judge_model=judge_model,
                    run_llm_judge=run_llm_judge,
                    run_correctness_judge=run_correctness_judge,
                )
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_path, results)

        summary = summarize_harness_results(results)
        _log_summary_metrics(summary)
        _log_artifacts(out_path=out_path, config_path=config_path, eval_path=eval_path)

        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else ""

    return out_path, summary, run_id
