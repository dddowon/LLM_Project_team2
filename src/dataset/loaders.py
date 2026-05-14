from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from src.dataset.schema import Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".hwp"}


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[page {page_no}]\n{text}")
    return "\n\n".join(pages)


def load_hwp(path: Path) -> str:
    """Load HWP through the optional `hwp5txt` command when available."""
    try:
        result = subprocess.run(
            ["hwp5txt", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "HWP 파일 처리를 위해 `pyhwp`의 `hwp5txt` 명령이 필요합니다. "
            "VM에서 `pip install pyhwp` 후 다시 실행해 주세요."
        ) from exc
    return result.stdout


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".hwp":
        return load_hwp(path)
    if suffix in {".txt", ".md"}:
        return load_text(path)
    raise ValueError(f"지원하지 않는 파일 형식입니다: {path}")


def read_metadata(metadata_csv: Path) -> pd.DataFrame:
    if not metadata_csv.exists():
        return pd.DataFrame()
    return pd.read_csv(metadata_csv)


def _find_file_column(metadata: pd.DataFrame) -> str | None:
    candidates = ["file_path", "filepath", "path", "file_name", "filename", "파일명", "문서명"]
    for column in candidates:
        if column in metadata.columns:
            return column
    return None


def _row_to_metadata(row: pd.Series) -> dict[str, str]:
    values = {}
    for key, value in row.to_dict().items():
        if pd.isna(value):
            continue
        values[str(key)] = str(value)
    return values


def load_documents(raw_data_dir: Path, metadata_csv: Path) -> list[Document]:
    metadata = read_metadata(metadata_csv)
    file_column = _find_file_column(metadata) if not metadata.empty else None
    documents: list[Document] = []

    if file_column:
        rows = metadata.to_dict(orient="records")
        for i, row in enumerate(rows):
            file_value = str(row[file_column])
            path = Path(file_value)
            if not path.is_absolute():
                path = raw_data_dir / path
            if not path.exists():
                matches = list(raw_data_dir.rglob(Path(file_value).name))
                if not matches:
                    continue
                path = matches[0]
            text = load_document_text(path)
            documents.append(
                Document(
                    doc_id=str(row.get("doc_id") or row.get("id") or path.stem or i),
                    path=str(path),
                    text=text,
                    metadata={k: str(v) for k, v in row.items() if pd.notna(v)},
                )
            )
        return documents

    for i, path in enumerate(sorted(raw_data_dir.rglob("*"))):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = load_document_text(path)
        documents.append(
            Document(
                doc_id=path.stem or str(i),
                path=str(path),
                text=text,
                metadata={"file_name": path.name},
            )
        )
    return documents
