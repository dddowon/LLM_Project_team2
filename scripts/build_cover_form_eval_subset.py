"""Build eval subset for cover-form (표지·양식 meta) prompt A/B tests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.question_taxonomy import should_apply_cover_form_answer_hint


def build_cover_form_subset(input_path: Path, output_path: Path) -> tuple[int, int]:
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept = [
        row
        for row in rows
        if should_apply_cover_form_answer_hint(
            str(row.get("question") or ""),
            category=str(row.get("category") or "") or None,
        )
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in kept) + ("\n" if kept else ""),
        encoding="utf-8",
    )
    return len(rows), len(kept)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/v2/eval_questions_failures.jsonl",
        help="Source eval JSONL",
    )
    parser.add_argument(
        "--output",
        default="data/v2/eval_questions_cover_form.jsonl",
        help="Cover-form only subset",
    )
    args = parser.parse_args()
    total, kept = build_cover_form_subset(Path(args.input), Path(args.output))
    print(f"wrote {args.output}: {kept}/{total} rows")


if __name__ == "__main__":
    main()
