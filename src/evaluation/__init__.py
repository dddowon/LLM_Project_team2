from src.evaluation.harness_metrics import summarize_harness_results
from src.evaluation.langsmith_harness import run_eval_harness

__all__ = ["run_eval_harness", "summarize_harness_results"]


def __getattr__(name: str):
    if name == "run_eval_harness_mlflow":
        from src.evaluation.mlflow_harness import run_eval_harness_mlflow

        return run_eval_harness_mlflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
