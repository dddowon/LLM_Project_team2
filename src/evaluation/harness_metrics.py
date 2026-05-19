from __future__ import annotations

from typing import Any


def summarize_harness_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    hits = [r["retrieval_keyword_hit"] for r in results if r.get("retrieval_keyword_hit") is not None]
    precisions = [r["context_precision"] for r in results if r.get("context_precision") is not None]
    fs = [r["f_score"] for r in results if isinstance(r.get("f_score"), int)]
    rs = [r["r_score"] for r in results if isinstance(r.get("r_score"), int)]
    ss = [r["s_score"] for r in results if isinstance(r.get("s_score"), int)]
    latencies = [
        r["total_latency_ms"]
        for r in results
        if isinstance(r.get("total_latency_ms"), (int, float))
    ]
    return {
        "n": len(results),
        "mean_retrieval_keyword_hit": (sum(hits) / len(hits)) if hits else None,
        "mean_context_precision": (sum(precisions) / len(precisions)) if precisions else None,
        "mean_f_score": (sum(fs) / len(fs)) if fs else None,
        "mean_r_score": (sum(rs) / len(rs)) if rs else None,
        "mean_s_score": (sum(ss) / len(ss)) if ss else None,
        "mean_total_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
    }
