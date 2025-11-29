"""Utility wrapper to interact with python/agent.py via pure Python functions.

Usage:
    from python.agent_runtime import AgentRuntime

    agent = AgentRuntime()
    agent.start()
    response = agent.send_chat("列目录")
    print(response.text)
    agent.stop()

The wrapper hides the subprocess/stdio handling used by the VS Code extension,
so you can drive the agent logic entirely from Python code (scripts, notebooks,
unit tests, etc.).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Dict, List, Optional

from python.file_ops import build_unified_diff, read_text_for_diff

AGENT_SCRIPT = Path(__file__).resolve().parent / "agent.py"


@dataclass
class FileOperationResult:
    path: Path
    action: str
    applied: bool
    diff: Optional[str] = None
    error: Optional[str] = None


@dataclass
class AgentResponse:
    """Structured response from agent.py."""

    text: str
    operations: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def apply_operations(self, workspace_root: Path) -> List[FileOperationResult]:
        """Apply write operations to the given workspace_root.

        Returns the list of operation results including patch previews.
        """
        results: List[FileOperationResult] = []
        for op in self.operations:
            action = op.get("action")
            if action != "writeFile":
                continue
            rel = op.get("path")
            content = op.get("content", "")
            if not rel:
                continue
            target = workspace_root / rel
            result = FileOperationResult(path=target, action=action, applied=False)
            original = read_text_for_diff(target)
            result.diff = build_unified_diff(rel, original, content or "")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                result.applied = True
            except Exception as exc:
                result.error = str(exc)
            results.append(result)
        return results


class AgentRuntime:
    """Manage a background agent.py process and expose simple call APIs."""

    def __init__(self,
                 workspace_root: Optional[os.PathLike[str]] = None,
                 python_executable: Optional[str] = None,
                 script_path: Optional[os.PathLike[str]] = None,
                 startup_timeout: float = 5.0,
                 response_timeout: float = 30.0) -> None:
        self.workspace_root = Path(workspace_root or os.getcwd())
        self.python_executable = python_executable or sys.executable
        self.script_path = Path(script_path) if script_path else AGENT_SCRIPT
        self.startup_timeout = startup_timeout
        self.response_timeout = response_timeout
        self._proc: Optional[subprocess.Popen[str]] = None
        self._stdout_queue: "Queue[str]" = Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._last_request_id = 0

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._proc and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            [self.python_executable, str(self.script_path)],
            cwd=str(self.workspace_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._stdout_pump, daemon=True)
        self._reader_thread.start()
        self._wait_for_ready()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None

    # ------------------------------------------------------------------
    def send_chat(self, text: str) -> AgentResponse:
        if not self._proc or self._proc.poll() is not None:
            raise RuntimeError("Agent is not running. Call start() first.")
        self._last_request_id = int(time.time() * 1000)
        payload = {"id": self._last_request_id, "type": "chat", "text": text}
        line = json.dumps(payload, ensure_ascii=False)
        assert self._proc.stdin is not None
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()
        raw = self._wait_for_response()
        text_resp = raw.get("text", "")
        operations = raw.get("operations") or []
        return AgentResponse(text=text_resp, operations=operations, raw=raw)

    # ------------------------------------------------------------------
    def _stdout_pump(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            clean = line.strip()
            if clean:
                self._stdout_queue.put(clean)

    def _wait_for_ready(self) -> None:
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            try:
                line = self._stdout_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") == "ready":
                return
        raise TimeoutError("Agent did not report ready state in time")

    def _wait_for_response(self) -> Dict[str, Any]:
        deadline = time.time() + self.response_timeout
        while time.time() < deadline:
            try:
                line = self._stdout_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") == "response":
                return data
        raise TimeoutError("Timed out waiting for agent response")


# ----------------------------------------------------------------------
def demo() -> None:
    agent = AgentRuntime()
    agent.start()
    try:
        for prompt in ["列目录", "创建示例", "读取 examples/hello.txt"]:
            resp = agent.send_chat(prompt)
            results = resp.apply_operations(agent.workspace_root)
            print("PROMPT:", prompt)
            print("TEXT:", resp.text)
            if results:
                print("OPERATIONS:")
                for result in results:
                    status = "ok" if result.applied and not result.error else f"error: {result.error}"
                    print(f" - {result.action} -> {result.path} ({status})")
            print("---")
    finally:
        agent.stop()


if __name__ == "__main__":
    demo()
