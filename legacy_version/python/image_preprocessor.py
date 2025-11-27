"""Image preprocessing helpers for attachment uploads.

- Enforces max dimension while preserving aspect ratio.
- Aggressively converts everything to JPEG (quality≈70) to shrink payloads.
- Compresses and returns base64 payloads suitable for API calls.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional

from PIL import Image, ImageOps

ONE_MB = 1 * 1024 * 1024
SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".jfif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".heic",
    ".heif",
}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def preprocess_image(
    input_path: Path | str,
    *,
    output_dir: Optional[Path | str] = None,
    max_side: int = 1600,
    size_threshold: int = ONE_MB,
    jpeg_quality: float = 0.7,
) -> Dict[str, object]:
    src = Path(input_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"输入文件不存在: {src}")

    output_dir = Path(output_dir) if output_dir else src.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    original_size = src.stat().st_size
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img)
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        target_format = "JPEG"
        mime = "image/jpeg"
        save_kwargs = {
            "quality": max(1, min(int(jpeg_quality * 100), 95)),
            "optimize": True,
        }
        img = img.convert("RGB")

        img.save(buffer, format=target_format, **save_kwargs)
        processed_bytes = buffer.getvalue()
        width, height = img.size

    output_name = src.stem + "_frontend.jpg"
    output_path = (Path(output_dir) / output_name).resolve()
    output_path.write_bytes(processed_bytes)

    base64_payload = base64.b64encode(processed_bytes).decode("ascii")
    compressed_size = len(processed_bytes)
    data = {
        "source": str(src),
        "output": str(output_path),
        "source_size": original_size,
        "output_size": compressed_size,
        "ratio": round(compressed_size / original_size, 3) if original_size else 0,
        "exceeded_threshold": compressed_size > size_threshold,
        "backend_compression_suggested": compressed_size > size_threshold,
        "format": target_format,
        "mime": mime,
        "width": width,
        "height": height,
        "base64": base64_payload,
    }
    return data


__all__ = ["preprocess_image", "is_image_file", "ONE_MB", "SUPPORTED_IMAGE_EXTENSIONS"]
