#!/usr/bin/env python3
"""Create an HTML review page for mixed HWP/PDF slim chunks and raw tables."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
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
            if isinstance(row, dict):
                rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def compact(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def metadata_of(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def chunk_text_of(row: dict[str, Any]) -> str:
    return str(row.get("chunk_text") or row.get("text") or "")


def chunk_type_of(row: dict[str, Any]) -> str:
    metadata = metadata_of(row)
    return str(row.get("chunk_type") or metadata.get("chunk_type") or "")


def doc_name_of(row: dict[str, Any]) -> str:
    metadata = metadata_of(row)
    return str(metadata.get("file_name") or row.get("file_name") or row.get("doc_id") or "")


def section_path_text(value: Any) -> str:
    if isinstance(value, list):
        return " > ".join(str(item) for item in value if str(item).strip())
    return str(value or "")


def section_of(row: dict[str, Any]) -> str:
    metadata = metadata_of(row)
    return str(
        metadata.get("section_path_text")
        or section_path_text(metadata.get("section_path"))
        or row.get("section_path_text")
        or section_path_text(row.get("section_path"))
        or ""
    )


def section_type_of(row: dict[str, Any]) -> str:
    return str(metadata_of(row).get("section_type") or row.get("section_type") or "")


def table_id_of(row: dict[str, Any]) -> str:
    metadata = metadata_of(row)
    return str(metadata.get("table_id") or row.get("table_id") or "")


def table_type_of(row: dict[str, Any]) -> str:
    metadata = metadata_of(row)
    return str(metadata.get("table_type") or row.get("table_type") or "")


def table_key(file_name: str, table_id: str) -> tuple[str, str]:
    return (file_name, table_id)


def build_table_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        file_name = str(row.get("file_name") or "")
        table_id = str(row.get("table_id") or "")
        if file_name and table_id:
            output[table_key(file_name, table_id)] = row
    return output


def metadata_html(metadata: dict[str, Any]) -> str:
    keep_order = [
        "section_type",
        "heading",
        "table_id",
        "table_type",
        "table_shape",
        "row_range",
        "next_table_id",
        "next_table_type",
        "global_index",
    ]
    items: list[str] = []
    seen: set[str] = set()
    for key in keep_order:
        value = metadata.get(key)
        if value not in ("", None, [], {}):
            items.append(f"<div><b>{esc(key)}</b><span>{esc(json_value(value))}</span></div>")
            seen.add(key)
    for key in sorted(metadata):
        if key in seen or key == "file_name":
            continue
        value = metadata.get(key)
        if value not in ("", None, [], {}):
            items.append(f"<div><b>{esc(key)}</b><span>{esc(json_value(value))}</span></div>")
    return "".join(items)


def json_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render_grid(grid: Any, *, max_rows: int = 80, max_cols: int = 16) -> str:
    if not isinstance(grid, list) or not grid:
        return ""
    rows = [row for row in grid if isinstance(row, list)]
    if not rows:
        return ""
    col_count = min(max((len(row) for row in rows), default=0), max_cols)
    rendered_rows: list[str] = []
    for row in rows[:max_rows]:
        cells = [f"<td>{esc(row[idx] if idx < len(row) else '')}</td>" for idx in range(col_count)]
        rendered_rows.append("<tr>" + "".join(cells) + "</tr>")
    truncated = ""
    if len(rows) > max_rows:
        truncated = f"<p class=\"muted\">table rows truncated: {max_rows}/{len(rows)}</p>"
    return f"{truncated}<div class=\"table-wrap\"><table>{''.join(rendered_rows)}</table></div>"


def render_structured_list(title: str, value: Any) -> str:
    if not isinstance(value, list) or not value:
        return ""
    lines = [f"<h4>{esc(title)}</h4>"]
    for item in value[:60]:
        if isinstance(item, dict):
            lines.append(f"<pre class=\"mini\">{esc(json.dumps(item, ensure_ascii=False, indent=2))}</pre>")
        else:
            lines.append(f"<pre class=\"mini\">{esc(item)}</pre>")
    if len(value) > 60:
        lines.append(f"<p class=\"muted\">{title} truncated: 60/{len(value)}</p>")
    return "".join(lines)


def render_raw_table(raw: dict[str, Any] | None) -> str:
    if not raw:
        return ""
    grid = raw.get("table_grid")
    parts = [
        "<section class=\"raw-table\">",
        "<h3>Raw Table</h3>",
        "<div class=\"raw-meta\">",
        f"<span>table_id <b>{esc(raw.get('table_id'))}</b></span>",
        f"<span>table_type <b>{esc(raw.get('table_type'))}</b></span>",
        f"<span>shape <b>{esc(raw.get('table_shape'))}</b></span>",
        f"<span>size <b>{esc(raw.get('rows'))} x {esc(raw.get('cols'))}</b></span>",
        "</div>",
        render_grid(grid),
        render_structured_list("rows_data", raw.get("rows_data")),
        render_structured_list("row_groups", raw.get("row_groups")),
        render_structured_list("summary_lines", raw.get("summary_lines")),
        "</section>",
    ]
    return "".join(parts)


def search_blob(row: dict[str, Any]) -> str:
    metadata = metadata_of(row)
    values = [
        row.get("chunk_id", ""),
        chunk_type_of(row),
        section_type_of(row),
        table_type_of(row),
        doc_name_of(row),
        section_of(row),
        chunk_text_of(row),
        json.dumps(metadata, ensure_ascii=False),
    ]
    return " ".join(str(value) for value in values).lower()


def render_card(
    row: dict[str, Any],
    index: int,
    *,
    raw_table: dict[str, Any] | None,
) -> str:
    metadata = metadata_of(row)
    chunk_type = chunk_type_of(row)
    is_table = chunk_type.startswith("table_") or chunk_type.startswith("table/")
    text = chunk_text_of(row)
    doc = doc_name_of(row)
    section = section_of(row)
    section_type = section_type_of(row)
    table_type = table_type_of(row)
    table_id = table_id_of(row)
    raw_table_html = render_raw_table(raw_table) if is_table else ""
    badges = [
        chunk_type or "unknown",
        section_type or "unknown",
    ]
    if table_type:
        badges.append(table_type)
    if table_id:
        badges.append(table_id)
    badge_html = "".join(f"<span>{esc(badge)}</span>" for badge in badges)
    return f"""
<article class="card {'table-card' if is_table else 'text-card'}"
  data-search="{esc(search_blob(row))}"
  data-chunk-type="{esc(chunk_type)}"
  data-section-type="{esc(section_type)}"
  data-doc="{esc(doc)}"
  data-is-table="{'1' if is_table else '0'}">
  <div class="card-head">
    <div>
      <p class="index">#{index:,} · {esc(row.get('chunk_id') or '')}</p>
      <h2>{esc(section or '(no section path)')}</h2>
      <p class="doc">{esc(doc)}</p>
    </div>
    <div class="badges">{badge_html}<span>{len(text):,} chars</span></div>
  </div>
  <div class="meta-grid">{metadata_html(metadata)}</div>
  <pre class="chunk-text">{esc(text)}</pre>
  {raw_table_html}
</article>
"""


def option_html(counter: Counter[str], label: str) -> str:
    values = [f'<option value="">{esc(label)}</option>']
    for value, count in counter.most_common():
        values.append(f'<option value="{esc(value)}">{esc(value)} ({count:,})</option>')
    return "".join(values)


def chip_html(counter: Counter[str], limit: int = 30) -> str:
    chips: list[str] = []
    for value, count in counter.most_common(limit):
        label = value if len(value) <= 80 else value[:80] + "..."
        chips.append(f"<span title=\"{esc(value)}\">{esc(label)} <b>{count:,}</b></span>")
    return "".join(chips)


def render_html(
    chunks: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    *,
    chunks_path: Path,
    tables_path: Path,
    title: str,
) -> str:
    table_index = build_table_index(tables)
    chunk_types = Counter(chunk_type_of(row) for row in chunks)
    section_types = Counter(section_type_of(row) for row in chunks)
    docs = Counter(doc_name_of(row) for row in chunks)
    table_types = Counter(table_type_of(row) for row in chunks if table_type_of(row))
    table_chunk_count = sum(1 for row in chunks if str(chunk_type_of(row)).startswith("table_"))
    cards: list[str] = []
    for index, row in enumerate(chunks, start=1):
        raw = table_index.get(table_key(doc_name_of(row), table_id_of(row)))
        cards.append(render_card(row, index, raw_table=raw))
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --text: #1d2433;
  --muted: #667085;
  --line: #d9dee8;
  --blue: #2f5acf;
  --green: #11795b;
  --amber: #a86400;
  --soft-blue: #edf2ff;
  --soft-green: #eaf7f1;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans KR", Arial, sans-serif;
}}
header {{
  position: sticky;
  top: 0;
  z-index: 3;
  background: rgba(246, 247, 249, 0.96);
  border-bottom: 1px solid var(--line);
  padding: 18px 24px 14px;
  backdrop-filter: blur(8px);
}}
h1 {{ margin: 0 0 10px; font-size: 24px; }}
.sub {{ color: var(--muted); font-size: 13px; line-height: 1.45; }}
.stats, .filters, .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
.stats span, .chips span {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 6px 10px;
  font-size: 12px;
}}
.filters input, .filters select {{
  height: 36px;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: white;
  padding: 0 10px;
  min-width: 180px;
  font-size: 13px;
}}
.filters input {{ min-width: min(520px, 100%); flex: 1; }}
main {{ padding: 20px 24px 64px; }}
.card {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-left: 5px solid var(--blue);
  border-radius: 8px;
  margin: 0 0 16px;
  padding: 16px;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}}
.table-card {{ border-left-color: var(--green); }}
.card-head {{ display: flex; gap: 16px; justify-content: space-between; align-items: flex-start; }}
.index {{ margin: 0 0 6px; color: var(--muted); font-size: 12px; }}
h2 {{ margin: 0; font-size: 17px; line-height: 1.35; }}
.doc {{ margin: 6px 0 0; color: var(--muted); font-size: 13px; word-break: break-all; }}
.badges {{ display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; min-width: 220px; }}
.badges span {{
  background: var(--soft-blue);
  color: #183a91;
  border-radius: 999px;
  padding: 5px 8px;
  font-size: 12px;
  white-space: nowrap;
}}
.table-card .badges span {{ background: var(--soft-green); color: #0b5b44; }}
.meta-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 8px;
  margin-top: 14px;
}}
.meta-grid div {{
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  min-width: 0;
}}
.meta-grid b {{ display: block; color: var(--muted); font-size: 11px; margin-bottom: 4px; }}
.meta-grid span {{ font-size: 13px; word-break: break-word; }}
pre {{
  white-space: pre-wrap;
  word-break: break-word;
  overflow-wrap: anywhere;
  font-family: "D2Coding", "Cascadia Mono", "Consolas", monospace;
}}
.chunk-text {{
  margin: 14px 0 0;
  background: #fbfcff;
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 14px;
  font-size: 13px;
  line-height: 1.6;
}}
.raw-table {{
  margin-top: 14px;
  border: 1px solid #b9d7cb;
  background: #fbfffd;
  border-radius: 7px;
  padding: 12px;
}}
.raw-table h3 {{ margin: 0 0 10px; font-size: 15px; color: #0b5b44; }}
.raw-table h4 {{ margin: 12px 0 6px; font-size: 13px; }}
.raw-meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }}
.raw-meta span {{
  background: #eaf7f1;
  border: 1px solid #c6e5d7;
  border-radius: 999px;
  padding: 5px 8px;
  font-size: 12px;
}}
.table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; background: white; }}
table {{ border-collapse: collapse; min-width: 100%; font-size: 12px; }}
td {{ border: 1px solid var(--line); padding: 7px; vertical-align: top; white-space: pre-wrap; }}
.mini {{ margin: 6px 0; padding: 8px; background: white; border: 1px solid var(--line); border-radius: 6px; font-size: 12px; }}
.muted {{ color: var(--muted); font-size: 12px; }}
.hidden {{ display: none; }}
@media (max-width: 760px) {{
  header, main {{ padding-left: 12px; padding-right: 12px; }}
  .card-head {{ display: block; }}
  .badges {{ justify-content: flex-start; margin-top: 10px; min-width: 0; }}
}}
</style>
</head>
<body>
<header>
  <h1>{esc(title)}</h1>
  <div class="sub">
    chunks: {esc(chunks_path)}<br>
    raw tables: {esc(tables_path)}
  </div>
  <div class="stats">
    <span>chunks <b>{len(chunks):,}</b></span>
    <span>text chunks <b>{len(chunks) - table_chunk_count:,}</b></span>
    <span>table chunks <b>{table_chunk_count:,}</b></span>
    <span>raw tables <b>{len(tables):,}</b></span>
    <span>documents <b>{len(docs):,}</b></span>
    <span id="visibleCount">visible <b>{len(chunks):,}</b></span>
  </div>
  <div class="filters">
    <input id="q" placeholder="검색: 문서명, 섹션, chunk_id, 본문, 표 내용">
    <select id="docFilter">{option_html(docs, "문서 전체")}</select>
    <select id="typeFilter">{option_html(chunk_types, "chunk type 전체")}</select>
    <select id="sectionFilter">{option_html(section_types, "section type 전체")}</select>
    <select id="tableOnly">
      <option value="">본문+표 전체</option>
      <option value="1">표 chunk만</option>
      <option value="0">본문 chunk만</option>
    </select>
  </div>
  <div class="chips">{chip_html(table_types) or '<span>table type 없음</span>'}</div>
</header>
<main id="cards">
{''.join(cards)}
</main>
<script>
const q = document.getElementById('q');
const docFilter = document.getElementById('docFilter');
const typeFilter = document.getElementById('typeFilter');
const sectionFilter = document.getElementById('sectionFilter');
const tableOnly = document.getElementById('tableOnly');
const visibleCount = document.getElementById('visibleCount');
const cards = Array.from(document.querySelectorAll('.card'));

function applyFilters() {{
  const query = q.value.trim().toLowerCase();
  const doc = docFilter.value;
  const type = typeFilter.value;
  const section = sectionFilter.value;
  const table = tableOnly.value;
  let visible = 0;
  for (const card of cards) {{
    const okQuery = !query || card.dataset.search.includes(query);
    const okDoc = !doc || card.dataset.doc === doc;
    const okType = !type || card.dataset.chunkType === type;
    const okSection = !section || card.dataset.sectionType === section;
    const okTable = !table || card.dataset.isTable === table;
    const ok = okQuery && okDoc && okType && okSection && okTable;
    card.classList.toggle('hidden', !ok);
    if (ok) visible += 1;
  }}
  visibleCount.innerHTML = `visible <b>${{visible.toLocaleString()}}</b>`;
}}
[q, docFilter, typeFilter, sectionFilter, tableOnly].forEach(el => el.addEventListener('input', applyFilters));
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create mixed chunk/table review HTML.")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Mixed HWP/PDF Chunk Review")
    parser.add_argument("--limit", type=int, default=0, help="Debug only: render first N chunks")
    args = parser.parse_args()

    chunks = read_jsonl(args.chunks, limit=args.limit)
    tables = read_jsonl(args.tables)
    html_text = render_html(chunks, tables, chunks_path=args.chunks, tables_path=args.tables, title=args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(f"chunks: {len(chunks)}")
    print(f"tables: {len(tables)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
