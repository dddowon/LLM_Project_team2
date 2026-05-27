from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zlib

try:
    import olefile
except ImportError:  # pragma: no cover - optional dependency
    olefile = None

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - optional dependency
    fitz = None

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
DEFAULT_PDF_MIN_WIDTH = 100
DEFAULT_PDF_MIN_HEIGHT = 40
DEFAULT_PDF_MIN_AREA = 10_000
DEFAULT_PDF_MIN_BYTES = 1_000


def detect_ext(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8"):
        return ".jpg"
    if data.startswith(b"\x89PNG"):
        return ".png"
    if data.startswith(b"BM"):
        return ".bmp"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return ".gif"
    return None


def try_decompress(data: bytes) -> bytes | None:
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS, 15 + 32):
        try:
            return zlib.decompress(data, wbits)
        except zlib.error:
            continue
    return None


def extract_images_from_hwp(hwp_path: Path, output_dir: Path) -> list[Path]:
    if olefile is None:  # pragma: no cover - import-time dependency
        raise ImportError("olefile is required for HWP extraction. Install with `pip install olefile`.")
    if not olefile.isOleFile(str(hwp_path)):
        return []

    saved: list[Path] = []
    # Keep original filename (including extension) as doc_key folder name
    # to match pipelines that keep .pdf/.hwp suffixes in doc_id.
    save_dir = output_dir / hwp_path.name
    save_dir.mkdir(parents=True, exist_ok=True)

    with olefile.OleFileIO(str(hwp_path)) as ole:
        image_count = 0
        for stream in ole.listdir():
            stream_name = "/".join(stream)
            if not stream_name.startswith("BinData/"):
                continue

            try:
                data = ole.openstream(stream).read()
                ext = detect_ext(data)

                if ext is None:
                    decompressed = try_decompress(data)
                    if decompressed is not None:
                        decomp_ext = detect_ext(decompressed)
                        if decomp_ext is not None:
                            data = decompressed
                            ext = decomp_ext

                if ext is None:
                    ext = Path(stream_name).suffix.lower()

                if ext not in ALLOWED_EXTS:
                    continue

                image_count += 1
                save_path = save_dir / f"img_{image_count:03d}{ext}"
                save_path.write_bytes(data)
                saved.append(save_path)
            except Exception:
                continue

    return saved


def _normalize_pdf_ext(ext: str) -> str:
    normalized = ext.lower().strip().lstrip(".")
    if not normalized:
        return ".png"
    if normalized == "jpeg":
        return ".jpg"
    return f".{normalized}"


def _should_keep_pdf_image(
    image_data: dict,
    image_bytes: bytes,
    seen_hashes: set[str],
    *,
    min_width: int,
    min_height: int,
    min_area: int,
    min_bytes: int,
) -> bool:
    width = int(image_data.get("width", 0) or 0)
    height = int(image_data.get("height", 0) or 0)
    area = width * height
    byte_size = len(image_bytes)
    image_hash = hashlib.md5(image_bytes).hexdigest()

    if image_hash in seen_hashes:
        return False
    if width < min_width or height < min_height:
        return False
    if area < min_area:
        return False
    if byte_size < min_bytes:
        return False

    seen_hashes.add(image_hash)
    return True


def extract_images_from_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    min_width: int = DEFAULT_PDF_MIN_WIDTH,
    min_height: int = DEFAULT_PDF_MIN_HEIGHT,
    min_area: int = DEFAULT_PDF_MIN_AREA,
    min_bytes: int = DEFAULT_PDF_MIN_BYTES,
) -> list[Path]:
    if fitz is None:  # pragma: no cover - import-time dependency
        raise ImportError(
            "PyMuPDF is required for PDF extraction. Install with `pip install pymupdf`."
        )

    saved: list[Path] = []
    # Keep original filename (including extension) as doc_key folder name
    # to match pipelines that keep .pdf suffixes in doc_id.
    save_dir = output_dir / pdf_path.name
    save_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    image_count = 0

    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            for image_info in page.get_images(full=True):
                xref = image_info[0]
                try:
                    image_data = doc.extract_image(xref)
                    image_bytes = image_data.get("image")
                    if not isinstance(image_bytes, (bytes, bytearray)):
                        continue
                    image_bytes = bytes(image_bytes)
                    if not _should_keep_pdf_image(
                        image_data,
                        image_bytes,
                        seen_hashes,
                        min_width=min_width,
                        min_height=min_height,
                        min_area=min_area,
                        min_bytes=min_bytes,
                    ):
                        continue

                    ext = _normalize_pdf_ext(str(image_data.get("ext", "png")))
                    image_count += 1
                    save_path = save_dir / f"img_{image_count:03d}{ext}"
                    save_path.write_bytes(image_bytes)
                    saved.append(save_path)
                except Exception:
                    continue

    return saved


def _iter_source_files(
    input_dir: Path,
    *,
    include_hwp: bool,
    include_pdf: bool,
    recursive: bool,
) -> list[Path]:
    patterns: list[str] = []
    if include_hwp:
        patterns.append("*.hwp")
    if include_pdf:
        patterns.append("*.pdf")

    iterator_method = input_dir.rglob if recursive else input_dir.glob
    found: list[Path] = []
    for pattern in patterns:
        found.extend([path for path in iterator_method(pattern) if path.is_file()])
    return sorted(found, key=lambda p: (p.suffix.lower(), p.name))


def extract_images_in_dir(
    input_dir: Path,
    output_dir: Path,
    *,
    limit: int = 0,
    include_hwp: bool = True,
    include_pdf: bool = True,
    recursive: bool = False,
    pdf_min_width: int = DEFAULT_PDF_MIN_WIDTH,
    pdf_min_height: int = DEFAULT_PDF_MIN_HEIGHT,
    pdf_min_area: int = DEFAULT_PDF_MIN_AREA,
    pdf_min_bytes: int = DEFAULT_PDF_MIN_BYTES,
) -> list[Path]:
    source_files = _iter_source_files(
        input_dir=input_dir,
        include_hwp=include_hwp,
        include_pdf=include_pdf,
        recursive=recursive,
    )
    if limit > 0:
        source_files = source_files[:limit]

    all_saved: list[Path] = []
    for source_path in source_files:
        suffix = source_path.suffix.lower()
        if suffix == ".hwp":
            saved = extract_images_from_hwp(source_path, output_dir)
        elif suffix == ".pdf":
            saved = extract_images_from_pdf(
                source_path,
                output_dir,
                min_width=pdf_min_width,
                min_height=pdf_min_height,
                min_area=pdf_min_area,
                min_bytes=pdf_min_bytes,
            )
        else:
            continue
        all_saved.extend(saved)
    return all_saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract embedded images from HWP/PDF files")
    parser.add_argument("--input-dir", required=True, help="Directory containing .hwp/.pdf files")
    parser.add_argument("--output-dir", required=True, help="Directory to save extracted images")
    parser.add_argument("--limit", type=int, default=0, help="Process first N files only; 0=all")
    parser.add_argument("--hwp-only", action="store_true", help="Extract only from HWP files")
    parser.add_argument("--pdf-only", action="store_true", help="Extract only from PDF files")
    parser.add_argument("--recursive", action="store_true", help="Recursively search input-dir")
    parser.add_argument("--pdf-min-width", type=int, default=DEFAULT_PDF_MIN_WIDTH)
    parser.add_argument("--pdf-min-height", type=int, default=DEFAULT_PDF_MIN_HEIGHT)
    parser.add_argument("--pdf-min-area", type=int, default=DEFAULT_PDF_MIN_AREA)
    parser.add_argument("--pdf-min-bytes", type=int, default=DEFAULT_PDF_MIN_BYTES)
    args = parser.parse_args()

    include_hwp = not args.pdf_only
    include_pdf = not args.hwp_only

    saved = extract_images_in_dir(
        Path(args.input_dir),
        Path(args.output_dir),
        limit=args.limit,
        include_hwp=include_hwp,
        include_pdf=include_pdf,
        recursive=args.recursive,
        pdf_min_width=args.pdf_min_width,
        pdf_min_height=args.pdf_min_height,
        pdf_min_area=args.pdf_min_area,
        pdf_min_bytes=args.pdf_min_bytes,
    )
    print(f"saved_images: {len(saved)}")


if __name__ == "__main__":
    main()
