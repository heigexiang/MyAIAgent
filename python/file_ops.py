"""Shared utilities for file read/write helpers and diff previews."""
from __future__ import annotations

import difflib
from pathlib import Path
from typing import Optional, Sequence


def _ensure_text(value: str) -> str:
    if value is None:
        return ""
    return value.replace("\r\n", "\n")


def read_text_for_diff(path: Path) -> Optional[str]:
    try:
        data = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except UnicodeDecodeError:
        data = path.read_text(encoding="utf-8", errors="replace")
    return _ensure_text(data)


def build_unified_diff(rel_path: str, original: Optional[str], updated: str) -> str:
    old_lines: Sequence[str] = list(_ensure_text(original or "").splitlines(keepends=True))
    new_lines: Sequence[str] = list(_ensure_text(updated or "").splitlines(keepends=True))
    diff_iter = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
    )
    return "".join(diff_iter)
