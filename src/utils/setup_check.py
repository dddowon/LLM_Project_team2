from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


REQUIRED_MODULES = {
    "chromadb": "chromadb",
    "numpy": "numpy",
    "openai": "openai",
    "pandas": "pandas",
    "pydantic": "pydantic",
    "pypdf": "pypdf",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "tqdm": "tqdm",
}


def _module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _format_status(result: CheckResult) -> str:
    status = "OK" if result.ok else ("WARN" if not result.required else "FAIL")
    return f"[{status}] {result.name}: {result.detail}"


def _check_python_version() -> CheckResult:
    version = ".".join(map(str, sys.version_info[:3]))
    ok = sys.version_info >= (3, 10)
    return CheckResult(
        name="Python version",
        ok=ok,
        detail=f"{version} detected, requires >= 3.10",
    )


def _check_modules() -> list[CheckResult]:
    results = []
    for module_name, package_name in REQUIRED_MODULES.items():
        ok = _module_exists(module_name)
        detail = "installed" if ok else f"missing, install with `pip install -e .` ({package_name})"
        results.append(CheckResult(name=f"Python package `{package_name}`", ok=ok, detail=detail))
    return results


def _check_paths(config_path: str) -> list[CheckResult]:
    results: list[CheckResult] = []
    path = Path(config_path)
    if not path.exists():
        return [
            CheckResult(
                name="Config file",
                ok=False,
                detail=f"{config_path} not found",
            )
        ]

    results.append(CheckResult(name="Config file", ok=True, detail=str(path)))
    try:
        from src.config import load_config

        config = load_config(path)
    except Exception as exc:
        return [
            *results,
            CheckResult(
                name="Config parse",
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
            ),
        ]

    results.append(
        CheckResult(
            name="Vector DB provider",
            ok=config.vector_store.provider.lower() == "chroma",
            detail=(
                f"{config.vector_store.provider} / {config.vector_store.index_type} / "
                f"{config.vector_store.distance_metric}"
            ),
        )
    )

    raw_dir = config.paths.raw_data_dir
    metadata_csv = config.paths.metadata_csv
    index_dir = config.paths.index_dir
    eval_set = config.paths.evaluation_set

    results.append(
        CheckResult(
            name="Raw data directory",
            ok=raw_dir.exists(),
            detail=str(raw_dir) if raw_dir.exists() else f"{raw_dir} not found yet",
            required=False,
        )
    )
    results.append(
        CheckResult(
            name="Metadata CSV",
            ok=metadata_csv.exists(),
            detail=str(metadata_csv) if metadata_csv.exists() else f"{metadata_csv} not found yet",
            required=False,
        )
    )
    results.append(
        CheckResult(
            name="Chroma index directory",
            ok=(index_dir / "chroma.sqlite3").exists() and (index_dir / "chunks.json").exists(),
            detail=str(index_dir) if index_dir.exists() else f"{index_dir} will be created by ingest",
            required=False,
        )
    )
    results.append(
        CheckResult(
            name="Evaluation questions",
            ok=eval_set.exists(),
            detail=str(eval_set) if eval_set.exists() else f"{eval_set} not found yet",
            required=False,
        )
    )
    return results


def _check_environment() -> list[CheckResult]:
    if _module_exists("dotenv"):
        from dotenv import load_dotenv

        load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")
    pycache_prefix = os.getenv("PYTHONPYCACHEPREFIX", "")

    return [
        CheckResult(
            name="OPENAI_API_KEY",
            ok=bool(api_key),
            detail="set" if api_key else "missing, create `.env` from `.env.example`",
        ),
        CheckResult(
            name="PYTHONPYCACHEPREFIX",
            ok=bool(pycache_prefix),
            detail=pycache_prefix if pycache_prefix else "not set, caches may appear beside source files",
            required=False,
        ),
    ]


def _check_optional_tools() -> list[CheckResult]:
    hwp5txt = shutil.which("hwp5txt")
    return [
        CheckResult(
            name="HWP extractor `hwp5txt`",
            ok=bool(hwp5txt),
            detail=hwp5txt or "not installed, needed only when ingesting .hwp files",
            required=False,
        )
    ]


def _check_openai_connection() -> CheckResult:
    try:
        from openai import OpenAI

        client = OpenAI()
        models = client.models.list()
        first_model = models.data[0].id if models.data else "no models returned"
    except Exception as exc:
        return CheckResult(
            name="OpenAI API connection",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
        )
    return CheckResult(name="OpenAI API connection", ok=True, detail=f"connected, sample={first_model}")


def run_setup_check(config_path: str, check_openai: bool = False) -> int:
    results: list[CheckResult] = []
    results.append(_check_python_version())
    results.extend(_check_modules())
    results.extend(_check_environment())
    results.extend(_check_paths(config_path))
    results.extend(_check_optional_tools())

    if check_openai:
        results.append(_check_openai_connection())

    print("Setup check")
    print("===========")
    for result in results:
        print(_format_status(result))

    failed_required = [result for result in results if result.required and not result.ok]
    if failed_required:
        print("\nRequired checks failed. Fix the FAIL items above and run this command again.")
        return 1

    print("\nRequired checks passed. WARN items can be handled when that data or feature is needed.")
    return 0
