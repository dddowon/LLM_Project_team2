from __future__ import annotations

import argparse
from pathlib import Path
import zlib

import olefile

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def detect_ext(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8"):
        return ".jpg"
    if data.startswith(b"\x89PNG"):
        return ".png"
    if data.startswith(b"BM"):
        return ".bmp"
    if data.startswith(b"GIF8"):
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
    if not olefile.isOleFile(str(hwp_path)):
        return []

    saved: list[Path] = []
    save_dir = output_dir / hwp_path.stem
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


def extract_images_in_dir(input_dir: Path, output_dir: Path, limit: int = 0) -> list[Path]:
    hwp_files = sorted(input_dir.glob("*.hwp"))
    if limit > 0:
        hwp_files = hwp_files[:limit]

    all_saved: list[Path] = []
    for hwp_path in hwp_files:
        saved = extract_images_from_hwp(hwp_path, output_dir)
        all_saved.extend(saved)
    return all_saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract embedded images from HWP files")
    parser.add_argument("--input-dir", required=True, help="Directory containing .hwp files")
    parser.add_argument("--output-dir", required=True, help="Directory to save extracted images")
    parser.add_argument("--limit", type=int, default=0, help="Process first N files only; 0=all")
    args = parser.parse_args()

    saved = extract_images_in_dir(Path(args.input_dir), Path(args.output_dir), limit=args.limit)
    print(f"saved_images: {len(saved)}")


if __name__ == "__main__":
    main()
