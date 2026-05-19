from src.evaluation.harness_metrics import summarize_harness_results
from src.evaluation.mlflow_harness import run_eval_harness_mlflow

__all__ = ["run_eval_harness_mlflow", "summarize_harness_results"]


def __getattr__(name: str):
    if name == "run_eval_harness":
        from src.evaluation.langsmith_harness import run_eval_harness

        return run_eval_harness
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
