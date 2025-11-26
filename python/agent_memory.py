"""Lightweight persistent memory manager for chat interactions."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_MEMORY_DIR = Path(__file__).resolve().parent.parent / "agent_data"
DEFAULT_MEMORY_FILE = DEFAULT_MEMORY_DIR / "agent_memory.json"


class AgentMemory:
    """Persists recent user/assistant exchanges with size limits."""

    def __init__(
        self,
        *,
        storage_path: Optional[Path | str] = None,
        max_entries: int = 100,
    ) -> None:
        base_path = Path(storage_path) if storage_path else DEFAULT_MEMORY_FILE
        self.path = base_path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries if max_entries > 0 else 100
        self._lock = threading.Lock()
        self._entries: List[Dict[str, Any]] = self._load()
        self._ensure_file()

    # ------------------------------------------------------------------
    def add_interaction(self, user: str, assistant: str) -> None:
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user": self._clip(user),
            "assistant": self._clip(assistant),
        }
        with self._lock:
            self._entries.append(record)
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries :]
            self._save_locked()

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self._save_locked()

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._entries)

    def __len__(self) -> int:  # pragma: no cover - trivial
        with self._lock:
            return len(self._entries)

    def count(self, reload: bool = False) -> int:
        with self._lock:
            if reload:
                self._entries = self._load()
            return len(self._entries)

    def reload(self) -> None:
        with self._lock:
            self._entries = self._load()

    # ------------------------------------------------------------------
    def _clip(self, text: str, limit: int = 2000) -> str:
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "...(截断)"

    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            return []
        return []

    def _ensure_file(self) -> None:
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _save_locked(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._entries, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


__all__ = ["AgentMemory", "DEFAULT_MEMORY_FILE"]
