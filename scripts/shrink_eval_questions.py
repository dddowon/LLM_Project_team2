"""Shrink eval_questions.jsonl to N questions per doc with diverse question_type."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ALL_TYPES = [
    "fact",
    "summary",
    "comparison",
    "follow_up",
    "requirement_detail",
    "unanswerable",
]


def pick_one(candidates: list[dict], *, used_questions: set[str]) -> dict | None:
    for row in sorted(candidates, key=lambda r: (r.get("difficulty") or "", str(r.get("question") or ""))):
        key = str(row.get("question") or "")
        if key and key not in used_questions:
            return row
    return None


def shrink_rows(rows: list[dict], *, per_doc: int = 5) -> list[dict]:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        doc_id = str(row.get("doc_id") or "").strip()
        if doc_id:
            by_doc[doc_id].append(row)

    out: list[dict] = []
    for doc_index, doc_id in enumerate(sorted(by_doc)):
        pool = by_doc[doc_id]
        by_type: dict[str, list[dict]] = defaultdict(list)
        for row in pool:
            qtype = str(row.get("question_type") or "").strip()
            if qtype:
                by_type[qtype].append(row)

        omit = ALL_TYPES[doc_index % len(ALL_TYPES)]
        preferred = [t for t in ALL_TYPES if t != omit]
        used_questions: set[str] = set()
        used_types: set[str] = set()
        selected: list[dict] = []

        for qtype in preferred:
            if len(selected) >= per_doc:
                break
            if qtype not in by_type:
                continue
            row = pick_one(by_type[qtype], used_questions=used_questions)
            if row is None:
                continue
            used_questions.add(str(row.get("question") or ""))
            used_types.add(qtype)
            selected.append(row)

        if len(selected) < per_doc:
            for qtype in ALL_TYPES:
                if len(selected) >= per_doc:
                    break
                if qtype in used_types or qtype not in by_type:
                    continue
                row = pick_one(by_type[qtype], used_questions=used_questions)
                if row is None:
                    continue
                used_questions.add(str(row.get("question") or ""))
                used_types.add(qtype)
                selected.append(row)

        if len(selected) < per_doc:
            remaining = [
                r
                for r in pool
                if str(r.get("question") or "") not in used_questions
            ]
            for row in sorted(remaining, key=lambda r: str(r.get("question") or "")):
                if len(selected) >= per_doc:
                    break
                key = str(row.get("question") or "")
                if key in used_questions:
                    continue
                used_questions.add(key)
                selected.append(row)

        out.extend(selected[:per_doc])

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--per-doc", type=int, default=5)
    parser.add_argument("--backup", action="store_true", default=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if args.backup and output_path == input_path:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak1000")
        if not backup_path.exists():
            backup_path.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")

    shrunk = shrink_rows(rows, per_doc=args.per_doc)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in shrunk) + "\n",
        encoding="utf-8",
    )

    from collections import Counter

    docs = len({r.get("doc_id") for r in shrunk})
    types = Counter(r.get("question_type") for r in shrunk)
    print(f"wrote {len(shrunk)} rows, {docs} docs -> {output_path}")
    print("question_type counts:", dict(types))


if __name__ == "__main__":
    main()
