from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from paddleocr import PaddleOCR


def run_single(image_path: Path, lang: str = "korean") -> list[dict]:
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, lang=lang)
    result = ocr.predict(str(image_path))

    rows: list[dict] = []
    for r in result:
        rec_texts = r.get("rec_texts", [])
        rec_scores = r.get("rec_scores", [])
        rec_polys = r.get("rec_polys", [])
        for idx, text in enumerate(rec_texts):
            rows.append(
                {
                    "text": str(text),
                    "score": float(rec_scores[idx]) if idx < len(rec_scores) else None,
                    "poly": rec_polys[idx].tolist() if idx < len(rec_polys) and hasattr(rec_polys[idx], "tolist") else (rec_polys[idx] if idx < len(rec_polys) else None),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaddleOCR on a single image and save normalized JSON")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lang", default="korean")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    start = time.perf_counter()
    rows = run_single(image_path, lang=args.lang)
    latency_ms = (time.perf_counter() - start) * 1000.0
    payload = {
        "image_path": str(image_path),
        "model": "paddleocr",
        "lang": args.lang,
        "status": "success",
        "latency_ms": round(latency_ms, 2),
        "ocr_lines": rows,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ocr_lines: {len(rows)}")
    print(f"latency_ms: {payload['latency_ms']}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
