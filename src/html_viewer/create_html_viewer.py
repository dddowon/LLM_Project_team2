#!/usr/bin/env python3
"""Render chunk JSONL into a searchable review HTML page.

Expected chunk rows match the output shape from the team parsing/chunking
pipeline:

    {
        "chunk_id": "...",
        "chunk_type": "...",
        "chunk_text": "...",
        "metadata": {...}
    }
    python src/render_chunk_jsonl_review_html.py \
  --input /mnt/e/codeit/LLM_Project_team2/data/v2/BioIN_chunking/chunks.jsonl \
  --output /mnt/e/codeit/LLM_Project_team2/data/BioIN_chunking_chunks_review.html \
  --title "BioIN Chunk Review"
    
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


IMPORTANT_METADATA_KEYS = [
    "section_type",
    "heading",
    "part_index",
    "source_content_type",
    "table_id",
    "table_type",
    "table_shape",
    "table_rows",
    "table_cols",
    "table_cell_count",
    "nested_table_count",
    "row_range",
    "row_indices",
    "deduped_table_rows",
    "summary_group_index",
    "split_part",
    "body_chars",
    "chunk_chars",
    "content_hash",
    "source_record_index",
    "source_record_indices",
    "merged_record_count",
    "next_table_id",
    "next_table_type",
    "next_table_shape",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row at {path}:{line_no}")
            rows.append(row)
    return rows


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def compact_space(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def text_of(row: dict[str, Any]) -> str:
    chunk_text = row.get("chunk_text")
    if chunk_text is not None:
        return str(chunk_text)
    return str(row.get("text") or "")


def metadata_of(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def extract_prefix_value(text: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def doc_name_of(row: dict[str, Any]) -> str:
    metadata = metadata_of(row)
    return str(
        metadata.get("file_name")
        or metadata.get("source_file")
        or row.get("doc_id")
        or extract_prefix_value(text_of(row), "문서명")
        or ""
    )


def section_path_of(row: dict[str, Any]) -> str:
    metadata = metadata_of(row)
    section_text = metadata.get("section_path_text")
    if section_text:
        return str(section_text)
    section_path = metadata.get("section_path")
    if isinstance(section_path, list):
        return " > ".join(str(item) for item in section_path if str(item).strip())
    if section_path:
        return str(section_path)
    return extract_prefix_value(text_of(row), "섹션경로")


def chunk_type_of(row: dict[str, Any]) -> str:
    metadata = metadata_of(row)
    return str(row.get("chunk_type") or metadata.get("chunk_type") or "unknown")


def section_type_of(row: dict[str, Any]) -> str:
    return str(metadata_of(row).get("section_type") or "unknown")


def preview_of(text: str, limit: int = 260) -> str:
    preview = compact_space(text)
    if len(preview) <= limit:
        return preview
    return preview[:limit] + "..."


def jsonish(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def metadata_cards(metadata: dict[str, Any]) -> str:
    html_items: list[str] = []
    seen: set[str] = set()

    for key in IMPORTANT_METADATA_KEYS:
        value = metadata.get(key)
        if value in ("", None, []):
            continue
        html_items.append(metadata_item(key, value))
        seen.add(key)

    for key in sorted(metadata):
        if key in seen or key == "file_name":
            continue
        value = metadata.get(key)
        if value in ("", None, []):
            continue
        html_items.append(metadata_item(key, value))

    return "".join(html_items)


def metadata_item(key: str, value: Any) -> str:
    return f"<div><b>{esc(key)}</b><span>{esc(jsonish(value))}</span></div>"


def search_blob(row: dict[str, Any]) -> str:
    metadata = metadata_of(row)
    values = [
        row.get("chunk_id", ""),
        chunk_type_of(row),
        section_type_of(row),
        doc_name_of(row),
        section_path_of(row),
        text_of(row),
        json.dumps(metadata, ensure_ascii=False),
    ]
    return " ".join(str(value) for value in values).lower()


def render_card(row: dict[str, Any], index: int) -> str:
    metadata = metadata_of(row)
    chunk_type = chunk_type_of(row)
    section_type = section_type_of(row)
    chunk_text = text_of(row)
    is_table = chunk_type.startswith("table_") or chunk_type.startswith("table/")
    source_label = "표" if is_table else "본문"
    css_class = "table-card" if is_table else "text-card"

    return f"""
    <article class="card {css_class}"
      data-type="{esc(chunk_type)}"
      data-section="{esc(section_type)}"
      data-text="{esc(search_blob(row))}">
      <div class="card-top">
        <div class="title-area">
          <div class="chunk-line">#{index:,} · {esc(row.get("chunk_id") or "")}</div>
          <h2>{esc(section_path_of(row) or "(section path 없음)")}</h2>
          <p class="doc">{esc(doc_name_of(row))}</p>
        </div>
        <div class="badges">
          <span>{esc(source_label)}</span>
          <span>{esc(chunk_type)}</span>
          <span>{esc(section_type)}</span>
          <span>{len(chunk_text):,}자</span>
        </div>
      </div>
      <div class="preview"><b>미리보기</b> {esc(preview_of(chunk_text))}</div>
      <div class="meta-grid">{metadata_cards(metadata)}</div>
      <pre>{esc(chunk_text)}</pre>
    </article>
    """


def stat_chips(values: Counter[str]) -> str:
    return "".join(
        f"<span>{esc(key)} <b>{count:,}</b></span>"
        for key, count in values.most_common()
    )


def select_options(values: Counter[str], label: str) -> str:
    options = [f'<option value="">{esc(label)}</option>']
    options.extend(
        f'<option value="{esc(key)}">{esc(key)} ({count:,})</option>'
        for key, count in values.most_common()
    )
    return "".join(options)


def doc_stat_chips(values: Counter[str]) -> str:
    chips: list[str] = []
    for key, count in values.most_common():
        short_key = key if len(key) <= 58 else key[:58] + "..."
        chips.append(f'<span title="{esc(key)}">{esc(short_key)} <b>{count:,}</b></span>')
    return "".join(chips)


def render_html(rows: list[dict[str, Any]], *, input_path: Path, title: str) -> str:
    chunk_types = Counter(chunk_type_of(row) for row in rows)
    section_types = Counter(section_type_of(row) for row in rows)
    documents = Counter(doc_name_of(row) or "(document 없음)" for row in rows)
    table_count = sum(
        1
        for row in rows
        if chunk_type_of(row).startswith("table_") or chunk_type_of(row).startswith("table/")
    )
    text_count = len(rows) - table_count
    cards = "\n".join(render_card(row, index) for index, row in enumerate(rows, start=1))

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      --bg: #f6f1ec;
      --paper: #fffdfa;
      --ink: #1f2328;
      --muted: #69707d;
      --line: #ded5cc;
      --blue: #345c9c;
      --green: #2f6b4f;
      --brown: #956019;
      --chip: #eef2ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, "Malgun Gothic", sans-serif;
      line-height: 1.55;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(246, 241, 236, .97);
      border-bottom: 1px solid var(--line);
      padding: 18px 28px 16px;
      backdrop-filter: blur(8px);
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      letter-spacing: 0;
    }}
    .sub {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 14px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(260px, 1fr) 230px 210px 160px;
      gap: 10px;
      margin-top: 14px;
    }}
    input, select, button {{
      height: 42px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      color: var(--ink);
      padding: 0 12px;
      font-size: 14px;
    }}
    button {{
      cursor: pointer;
      font-weight: 700;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      padding: 16px 28px 12px;
    }}
    .summary div {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px 14px;
    }}
    .summary b {{
      display: block;
      font-size: 23px;
    }}
    .summary span {{
      color: var(--muted);
      font-size: 13px;
    }}
    .stats {{
      padding: 0 28px 10px;
      display: grid;
      gap: 10px;
    }}
    .stat-row {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }}
    .stat-row strong {{
      min-width: 92px;
    }}
    .stat-row span {{
      display: inline-flex;
      gap: 4px;
      align-items: center;
      border: 1px solid var(--line);
      background: white;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 13px;
    }}
    main {{
      padding: 4px 28px 40px;
      display: grid;
      gap: 16px;
    }}
    .card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-left: 5px solid var(--green);
      border-radius: 10px;
      padding: 18px;
      box-shadow: 0 1px 2px rgba(0, 0, 0, .04);
    }}
    .table-card {{
      border-left-color: var(--brown);
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 18px;
    }}
    .chunk-line {{
      color: var(--blue);
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 5px;
    }}
    h2 {{
      margin: 0;
      font-size: 18px;
      line-height: 1.35;
      letter-spacing: 0;
    }}
    .doc {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 13px;
      word-break: break-all;
    }}
    .badges {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 6px;
      min-width: 270px;
    }}
    .badges span {{
      background: var(--chip);
      color: #263f75;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 700;
    }}
    .table-card .badges span {{
      background: #fff2d9;
      color: #744908;
    }}
    .preview {{
      margin: 14px 0 12px;
      padding: 10px 12px;
      background: #f5f2ee;
      border: 1px solid #ece2d8;
      border-radius: 8px;
      color: #374151;
    }}
    .preview b {{
      margin-right: 6px;
      color: #111827;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 8px;
      margin: 12px 0;
    }}
    .meta-grid div {{
      min-width: 0;
      background: white;
      border: 1px solid #eadfd5;
      border-radius: 8px;
      padding: 8px;
    }}
    .meta-grid b {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 2px;
    }}
    .meta-grid span {{
      word-break: break-word;
    }}
    pre {{
      margin: 0;
      padding: 14px;
      border-radius: 8px;
      background: #11161d;
      color: #f4f6f8;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: Consolas, "D2Coding", monospace;
      font-size: 13px;
      line-height: 1.55;
    }}
    .hidden {{
      display: none;
    }}
    @media (max-width: 900px) {{
      header, .summary, .stats, main {{
        padding-left: 16px;
        padding-right: 16px;
      }}
      .controls {{
        grid-template-columns: 1fr;
      }}
      .card-top {{
        display: block;
      }}
      .badges {{
        justify-content: flex-start;
        min-width: 0;
        margin-top: 10px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{esc(title)}</h1>
    <div class="sub" id="status">{len(rows):,} / {len(rows):,} chunks · source: {esc(input_path)}</div>
    <div class="controls">
      <input id="search" placeholder="문서명, 섹션, chunk_text 검색">
      <select id="type">{select_options(chunk_types, "모든 chunk_type")}</select>
      <select id="section">{select_options(section_types, "모든 section_type")}</select>
      <button id="reset">초기화</button>
    </div>
  </header>
  <section class="summary">
    <div><b>{len(rows):,}</b><span>총 chunk</span></div>
    <div><b>{text_count:,}</b><span>본문/표지 chunk</span></div>
    <div><b>{table_count:,}</b><span>표 chunk</span></div>
    <div><b>{len(chunk_types):,}</b><span>chunk_type 종류</span></div>
  </section>
  <section class="stats">
    <div class="stat-row"><strong>chunk_type</strong>{stat_chips(chunk_types)}</div>
    <div class="stat-row"><strong>section_type</strong>{stat_chips(section_types)}</div>
    <div class="stat-row"><strong>document</strong>{doc_stat_chips(documents)}</div>
  </section>
  <main id="cards">
    {cards}
  </main>
  <script>
    const search = document.getElementById("search");
    const type = document.getElementById("type");
    const section = document.getElementById("section");
    const reset = document.getElementById("reset");
    const status = document.getElementById("status");
    const cards = Array.from(document.querySelectorAll(".card"));
    const total = cards.length;

    function applyFilters() {{
      const query = search.value.trim().toLowerCase();
      const typeValue = type.value;
      const sectionValue = section.value;
      let shown = 0;
      for (const card of cards) {{
        const keep =
          (!query || card.dataset.text.includes(query)) &&
          (!typeValue || card.dataset.type === typeValue) &&
          (!sectionValue || card.dataset.section === sectionValue);
        card.classList.toggle("hidden", !keep);
        if (keep) shown += 1;
      }}
      status.textContent = `${{shown.toLocaleString()}} / ${{total.toLocaleString()}} chunks · source: {esc(input_path)}`;
    }}

    [search, type, section].forEach((element) => element.addEventListener("input", applyFilters));
    reset.addEventListener("click", () => {{
      search.value = "";
      type.value = "";
      section.value = "";
      applyFilters();
    }});
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Chunk JSONL input path")
    parser.add_argument(
        "--output",
        type=Path,
        help="HTML output path. Defaults to <input_stem>_review.html next to the input.",
    )
    parser.add_argument("--title", default="Chunk JSONL Review", help="HTML page title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    if not rows:
        raise RuntimeError(f"No rows found in {args.input}")

    output = args.output or args.input.with_name(f"{args.input.stem}_review.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(rows, input_path=args.input, title=args.title), encoding="utf-8")

    print(f"input: {args.input}")
    print(f"output: {output}")
    print(f"chunks: {len(rows):,}")


if __name__ == "__main__":
    main()
