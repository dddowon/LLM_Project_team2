from __future__ import annotations

import importlib
import importlib.metadata as metadata
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


REQUIRED_PACKAGES = (
    "paddleocr",
    "paddlex",
)

OPTIONAL_IMPORTS = (
    "cv2",
    "paddle",
)

ENGINE_CLASS_CHECKS = (
    ("PaddleOCR", True, "pp_ocrv5"),
    ("PPStructureV3", True, "pp_structurev3"),
    ("TableRecognitionPipelineV2", True, "table_recognition_v2"),
    # Some version lines may not expose PaddleOCRVL; keep optional and report clearly.
    ("PaddleOCRVL", False, "paddleocr_vl (native)"),
)


def _format_status(result: CheckResult) -> str:
    status = "OK" if result.ok else ("WARN" if not result.required else "FAIL")
    return f"[{status}] {result.name}: {result.detail}"


def _check_package_installed(package_name: str) -> CheckResult:
    try:
        version = metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return CheckResult(
            name=f"Python package `{package_name}`",
            ok=False,
            detail="missing",
        )
    return CheckResult(
        name=f"Python package `{package_name}`",
        ok=True,
        detail=f"installed (version={version})",
    )


def _check_paddle_runtime_package() -> CheckResult:
    cpu_version = None
    gpu_version = None
    try:
        cpu_version = metadata.version("paddlepaddle")
    except metadata.PackageNotFoundError:
        pass
    try:
        gpu_version = metadata.version("paddlepaddle-gpu")
    except metadata.PackageNotFoundError:
        pass

    if cpu_version and gpu_version:
        return CheckResult(
            name="Paddle runtime package",
            ok=True,
            detail=f"both installed (paddlepaddle={cpu_version}, paddlepaddle-gpu={gpu_version})",
        )
    if gpu_version:
        return CheckResult(
            name="Paddle runtime package",
            ok=True,
            detail=f"paddlepaddle-gpu installed (version={gpu_version})",
        )
    if cpu_version:
        return CheckResult(
            name="Paddle runtime package",
            ok=True,
            detail=f"paddlepaddle installed (version={cpu_version})",
        )
    return CheckResult(
        name="Paddle runtime package",
        ok=False,
        detail="missing (`paddlepaddle` or `paddlepaddle-gpu` required)",
    )


def _check_import(module_name: str, required: bool = True) -> CheckResult:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return CheckResult(
            name=f"Import `{module_name}`",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            required=required,
        )
    return CheckResult(
        name=f"Import `{module_name}`",
        ok=True,
        detail="ok",
        required=required,
    )


def _check_paddleocr_v3_apis() -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        paddleocr_module = importlib.import_module("paddleocr")
    except Exception as exc:
        return [
            CheckResult(
                name="Import module `paddleocr`",
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
            )
        ]

    for class_name, required, engine_name in ENGINE_CLASS_CHECKS:
        has_class = hasattr(paddleocr_module, class_name)
        detail = (
            f"available (engine={engine_name})"
            if has_class
            else f"not found (engine={engine_name})"
        )
        results.append(
            CheckResult(
                name=f"Engine class `paddleocr.{class_name}`",
                ok=has_class,
                detail=detail,
                required=required,
            )
        )

    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        return [
            CheckResult(
                name="Import `paddleocr.PaddleOCR`",
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
            )
        ]

    results.append(CheckResult(name="Import `paddleocr.PaddleOCR`", ok=True, detail="ok"))

    for attr_name in ("predict", "ocr"):
        has_attr = hasattr(PaddleOCR, attr_name)
        results.append(
            CheckResult(
                name=f"PaddleOCR API `{attr_name}`",
                ok=has_attr,
                detail="available" if has_attr else "not found in current version",
            )
        )

    has_paddleocr_vl = hasattr(paddleocr_module, "PaddleOCRVL")
    if has_paddleocr_vl:
        compatibility_detail = "native `paddleocr_vl` path is available."
        compatibility_ok = True
    else:
        compatibility_detail = (
            "`PaddleOCRVL` missing; `paddleocr_vl` engine path is unavailable in current environment."
        )
        compatibility_ok = False
    results.append(
        CheckResult(
            name="Engine compatibility `paddleocr_vl`",
            ok=compatibility_ok,
            detail=compatibility_detail,
            required=False,
        )
    )

    return results


def _check_paddlex_model_cache() -> CheckResult:
    cache_dir = Path.home() / ".paddlex" / "official_models"
    if not cache_dir.exists():
        return CheckResult(
            name="PaddleX model cache",
            ok=False,
            detail=f"not found ({cache_dir})",
            required=False,
        )
    model_dirs = sorted([path.name for path in cache_dir.iterdir() if path.is_dir()])
    vl_dirs = [name for name in model_dirs if "vl" in name.lower()]
    if vl_dirs:
        preview = ", ".join(vl_dirs[:5])
        extra = "" if len(vl_dirs) <= 5 else f" ... (+{len(vl_dirs) - 5} more)"
        detail = f"VL-like cached models found: {preview}{extra}"
        return CheckResult(name="PaddleX model cache", ok=True, detail=detail, required=False)
    return CheckResult(
        name="PaddleX model cache",
        ok=False,
        detail=f"no VL-like cached models (total_cached={len(model_dirs)})",
        required=False,
    )


def run_ocr3_setup_check() -> int:
    results: list[CheckResult] = []
    for pkg in REQUIRED_PACKAGES:
        results.append(_check_package_installed(pkg))
    results.append(_check_paddle_runtime_package())

    for module_name in OPTIONAL_IMPORTS:
        results.append(_check_import(module_name, required=False))

    results.extend(_check_paddleocr_v3_apis())
    results.append(_check_paddlex_model_cache())

    print("OCR 3.x setup check")
    print("===================")
    for result in results:
        print(_format_status(result))

    failed_required = [result for result in results if result.required and not result.ok]
    if failed_required:
        print("\nRequired checks failed. Fix FAIL items and rerun.")
        return 1

    print("\nRequired checks passed.")
    print(
        "Note: engine imports and cache state are preflight checks. Final validation still requires real inference"
        " smoke tests for each engine."
    )
    return 0
