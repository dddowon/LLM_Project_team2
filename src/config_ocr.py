from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class OCRConfig(BaseModel):
    # [Design Intent] Pin config assumptions to the validated runtime stack.
    paddleocr_version: str = "3.0.3"
    paddlepaddle_gpu_version: str = "3.0.0"
    engine: str = "paddleocr_vl"  # pp_ocrv5 | pp_ocrv5_transformers | pp_structurev3 | table_recognition_v2 | paddleocr_vl
    device: str = "gpu:0"  # cpu | gpu:0
    lang: str = "korean"
    # Minimum OCR line confidence to keep in pred_text/pred_structure build.
    score_threshold: float = 0.0
    # Minimum value similarity for GT-vs-pred structured matching.
    structure_match_threshold: float = 0.65
    batch_size: int = 1


class OCRPathConfig(BaseModel):
    # [Design Intent] Keep OCR I/O paths in one config source for reproducible runs.
    images_root: str = "data/v2/ocr_images/v4_table_filtered_260531"
    gt_root: str = "data/v2/ocr_outputs/incoming_gt"
    output_root: str = "data/v2/ocr_outputs"


class OCRAppConfig(BaseModel):
    ocr: OCRConfig = OCRConfig()
    paths: OCRPathConfig = OCRPathConfig()


def load_ocr_config(path: str | Path = "configs/ocr_default.yaml") -> OCRAppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    return OCRAppConfig.model_validate(data)
