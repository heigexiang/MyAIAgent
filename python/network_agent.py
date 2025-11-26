"""Simple Python agent that talks to a remote chat-completion style API.

The goal is to offer the same core workflow as the VS Code extension but fully
in Python so you can integrate it into CLI tools, notebooks, or backend tasks.

Environment variables (all optional, can also pass via constructor):
    MODEL_ENDPOINT   - Chat completion endpoint (default OpenAI-compatible URL)
    MODEL_API_KEY    - API key/token
    MODEL_MODEL      - Model name (default gpt-4o-mini)
    MODEL_AUTH_TYPE  - "Bearer" or "X-API-Key" (default Bearer)

Usage:
    from python.network_agent import NetworkAgent

    agent = NetworkAgent()
    reply, ops = agent.send_message("请列目录", attachments=["README.md"])
    print(reply)
    for op in ops:
        print(op)
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from python.agent_memory import AgentMemory
from python.image_preprocessor import is_image_file, preprocess_image

MODEL_PROFILES = {
    "gpt-4.1": {
        "endpoint": "https://aihubmix.com/v1/responses",
        "schema": "responses",
        "supports_temperature": True,
        "supports_reasoning": False,
        "supports_web_search": False,
    },
    "gpt-5": {
        "endpoint": "https://aihubmix.com/v1/responses",
        "schema": "responses",
        "supports_temperature": False,
        "supports_reasoning": True,
        "supports_web_search": True,
    },
}

# Cherry Studio 默认等待时间为 10 分钟（参见 packages/shared/config/constant.ts 中 defaultTimeout）
DEFAULT_TIMEOUT_SECONDS = 10 * 60.0
WEB_SEARCH_MIN_TIMEOUT = DEFAULT_TIMEOUT_SECONDS
TEXT_ATTACHMENT_MAX_CHARS = 20000
TEXT_ATTACHMENT_MAX_BYTES = 512 * 1024
TEXT_FILE_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".tex",
    ".log",
    ".csv",
    ".tsv",
    ".json",
    ".yml",
    ".yaml",
    ".xml",
    ".html",
    ".htm",
    ".py",
    ".js",
    ".ts",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
}

DEFAULT_HISTORY_NAME = "default"
DEFAULT_HISTORY_DIR = Path(__file__).resolve().parent.parent / "agent_data" / "histories"
HISTORY_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")

DEFAULT_MODEL = "gpt-4.1"
DEFAULT_ENDPOINT = MODEL_PROFILES[DEFAULT_MODEL]["endpoint"]
DEFAULT_SYSTEM_PROMPT = (
    "你运行在 Cherry Studio 中。若用户请求创建/修改/生成文件，请仅在回答末尾输出一行标记 ###OPERATIONS "
    "后跟一个 JSON 数组，数组元素格式 {\"action\":\"writeFile\",\"path\":\"相对路径\",\"content\":\"文件内容\"}。"
    "其它解释放在前面。若无文件操作则不要输出 ###OPERATIONS。禁止在 JSON 中出现注释。"
)


class NetworkAgent:
    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        auth_type: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_history: int = 20,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        memory_manager: Optional[AgentMemory] = None,
        memory_storage_path: Optional[os.PathLike[str] | str] = None,
        memory_max_entries: int = 100,
        history_name: Optional[str] = None,
        history_storage_dir: Optional[os.PathLike[str] | str] = None,
    ) -> None:
        self.endpoint = endpoint or os.getenv("MODEL_ENDPOINT") or DEFAULT_ENDPOINT
        self.api_key = api_key or os.getenv("MODEL_API_KEY")
        self.model = model or os.getenv("MODEL_MODEL") or DEFAULT_MODEL
        self.auth_type = (auth_type or os.getenv("MODEL_AUTH_TYPE") or "Bearer").strip()
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.temperature = temperature
        self.max_history = max_history
        self.timeout = timeout
        self.conversation: List[Dict[str, Any]] = []
        self.reasoning_effort: Optional[str] = "medium"
        self.web_search_enabled = False
        self.streaming_enabled = False
        self.last_preview: Dict[str, Any] = {"request": None, "response": None}
        self.memory_error: Optional[str] = None
        self.history_dir = Path(history_storage_dir) if history_storage_dir else DEFAULT_HISTORY_DIR
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.current_history_name = self._normalize_history_name(history_name or DEFAULT_HISTORY_NAME)
        self.memory = self._init_memory(memory_manager, memory_storage_path, memory_max_entries)
        self._load_history_into_conversation()
        self._bootstrap_memory_history()

    # ------------------------------------------------------------------
    def set_reasoning_effort(self, effort: Optional[str]) -> None:
        """Configure the reasoning effort parameter sent to the API."""
        if effort is None:
            self.reasoning_effort = None
            return
        normalized = effort.strip().lower()
        allowed = {"minimal", "low", "medium", "high"}
        if normalized not in allowed:
            raise ValueError(f"无效的 reasoning effort: {effort}")
        self.reasoning_effort = normalized

    def set_web_search_enabled(self, enabled: bool) -> None:
        self.web_search_enabled = bool(enabled)

    def set_streaming_enabled(self, enabled: bool) -> None:
        self.streaming_enabled = bool(enabled)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.conversation.clear()
        self.last_preview = {"request": None, "response": None}
        self._save_history()

    def clear_memory(self) -> None:
        self.reset()
        if self.memory is not None:
            self.memory.clear()
        self.memory_error = None

    @property
    def history_name(self) -> str:
        return self.current_history_name

    def list_histories(self) -> List[str]:
        files = [DEFAULT_HISTORY_NAME]
        for path in self.history_dir.glob("*.json"):
            name = path.stem
            if name not in files:
                files.append(name)
        unique = sorted(set(files))
        return unique

    def switch_history(self, name: str) -> List[Dict[str, Any]]:
        normalized = self._normalize_history_name(name)
        path = self._history_path(normalized)
        if not path.exists():
            raise ValueError(f"历史 '{name}' 不存在")
        self.current_history_name = normalized
        self.conversation = self._load_history(normalized)
        self.last_preview = {"request": None, "response": None}
        return list(self.conversation)

    def create_history(self, name: str) -> List[Dict[str, Any]]:
        normalized = self._normalize_history_name(name)
        path = self._history_path(normalized)
        if path.exists():
            raise ValueError(f"历史 '{name}' 已存在")
        self.current_history_name = normalized
        self.conversation = []
        self._save_history()
        return []

    def export_conversation(self) -> List[Dict[str, str]]:
        snapshot: List[Dict[str, str]] = []
        for message in self.conversation:
            snapshot.append(
                {
                    "role": message.get("role") or "user",
                    "text": self._content_to_text(message.get("content")),
                }
            )
        return snapshot

    def memory_count(self, reload: bool = False) -> int:
        if self.memory is None:
            return 0
        return self.memory.count(reload=reload)

    def memory_snapshot(self) -> List[Dict[str, Any]]:
        if self.memory is None:
            return []
        return self.memory.snapshot()

    @property
    def memory_path(self) -> Optional[str]:
        if self.memory is not None:
            return str(self.memory.path)
        return None

    def memory_status(self) -> str:
        if self.memory is not None:
            return "active"
        if self.memory_error:
            return f"disabled: {self.memory_error}"
        return "disabled"

    # ------------------------------------------------------------------
    def send_message(
        self,
        text: str,
        *,
        attachments: Optional[Sequence[Any]] = None,
        temperature: Optional[float] = None,
        stream_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if not text and not attachments:
            raise ValueError("请输入内容或提供附件")
        user_content = self._prepare_user_content(text, attachments)
        self.conversation.append({"role": "user", "content": user_content})
        self._prune_history()
        profile = self._get_model_profile()
        try:
            payload = self._build_payload(profile, temperature)
            request_timeout = self.timeout
            if profile.get("supports_web_search") and self.web_search_enabled:
                request_timeout = max(request_timeout, WEB_SEARCH_MIN_TIMEOUT)
            use_stream = self.streaming_enabled and profile.get("schema") == "responses"
            raw = self._post_json(
                payload,
                profile["endpoint"],
                timeout=request_timeout,
                stream=use_stream,
                stream_callback=stream_callback,
            )
            reply = self._extract_reply(raw, profile)
        except Exception:
            self._save_history()
            raise
        self.conversation.append({"role": "assistant", "content": reply})
        if self.memory is not None:
            user_snapshot = self._content_to_text(user_content)
            self.memory.add_interaction(user_snapshot, reply)
        self._save_history()
        operations = parse_operations(reply)
        return reply, operations

    # ------------------------------------------------------------------
    def build_request_preview(
        self,
        text: str,
        *,
        attachments: Optional[Sequence[Any]] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return the JSON payload that *would* be sent for the next request."""
        if not text and not attachments:
            raise ValueError("请输入内容或提供附件")
        user_content = self._prepare_user_content(text, attachments)
        simulated_history = self.conversation + [{"role": "user", "content": user_content}]
        profile = self._get_model_profile()
        body = self._build_payload(profile, temperature, conversation_override=simulated_history)
        headers = self._build_headers()
        return {
            "endpoint": profile["endpoint"],
            "headers": mask_headers(headers),
            "body": body,
        }

    def _build_input(self) -> List[Dict[str, Any]]:
        return self._build_input_from(self.conversation)

    def _build_input_from(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        base: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": self.system_prompt,
                    }
                ],
            }
        ]
        cap = self.max_history * 2
        history = conversation[-cap:]
        for message in history:
            role = message.get("role") or "user"
            content = message.get("content")
            blocks = self._normalize_message_content(role, content)
            if not blocks:
                continue
            base.append({"role": role, "content": blocks})
        return base

    def _prepare_user_content(
        self,
        text: str,
        attachments: Optional[Sequence[Any]] = None,
    ) -> Any:
        text = (text or "").strip()
        attachment_blocks = self._build_attachment_blocks(attachments)
        if not attachment_blocks:
            return text
        blocks: List[Dict[str, Any]] = []
        if text:
            blocks.append({"type": "input_text", "text": text})
        blocks.extend(attachment_blocks)
        if not blocks:
            return ""
        if len(blocks) == 1 and blocks[0].get("type") == "input_text":
            return blocks[0]["text"]
        return blocks

    def _build_payload(
        self,
        profile: Dict[str, Any],
        temperature_value: Optional[float],
        *,
        conversation_override: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        conversation = conversation_override if conversation_override is not None else self.conversation
        schema = profile.get("schema", "responses")
        payload: Dict[str, Any] = {"model": self.model}
        if schema == "chat":
            payload["messages"] = self._build_chat_messages_from(conversation)
        else:
            payload["input"] = self._build_input_from(conversation)
        temp_value = (
            temperature_value
            if isinstance(temperature_value, (int, float))
            else self.temperature
        )
        if profile.get("supports_temperature", True) and isinstance(temp_value, (int, float)):
            payload["temperature"] = temp_value
        if profile.get("supports_reasoning") and self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if profile.get("supports_web_search") and self.web_search_enabled:
            payload["tools"] = [{"type": "web_search_preview"}]
        if self.streaming_enabled and schema == "responses":
            payload["stream"] = True
        return payload

    def _build_chat_messages_from(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.system_prompt,
                        }
                    ],
                }
            )
        cap = self.max_history * 2
        for message in conversation[-cap:]:
            role = message.get("role") or "user"
            content = message.get("content")
            blocks = self._normalize_message_content(role, content)
            chat_blocks = self._convert_blocks_to_chat(blocks)
            if chat_blocks:
                messages.append({"role": role, "content": chat_blocks})
        return messages

    def _normalize_message_content(self, role: str, content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, list):
            return content
        text = (content or "").strip()
        if not text:
            return []
        key = "output_text" if role == "assistant" else "input_text"
        return [{"type": key, "text": text}]

    def _convert_blocks_to_chat(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chat_blocks: List[Dict[str, Any]] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in {"input_text", "output_text", "text"}:
                text = block.get("text")
                if text:
                    chat_blocks.append({"type": "text", "text": text})
            elif btype == "input_image":
                image_url = block.get("image_url")
                if isinstance(image_url, str):
                    chat_blocks.append({"type": "image_url", "image_url": {"url": image_url}})
                elif isinstance(image_url, dict):
                    chat_blocks.append({"type": "image_url", "image_url": image_url})
        return chat_blocks

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            fragments: List[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype in {"input_text", "output_text", "text"}:
                    text = block.get("text")
                    if text:
                        fragments.append(text)
                elif btype in {"input_image", "image_url"}:
                    fragments.append("[图片附件]")
                elif btype == "input_audio":
                    fragments.append("[音频附件]")
                elif btype == "input_file":
                    fragments.append("[文件附件]")
            if fragments:
                return "\n".join(fragments)
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)

    def _build_attachment_blocks(self, attachments: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        if not attachments:
            return blocks
        for raw in attachments:
            normalized = self._normalize_attachment(raw)
            if not normalized:
                continue
            if normalized.get("kind") == "image" and normalized.get("base64"):
                blocks.extend(self._build_image_blocks(normalized))
            else:
                path_value = normalized.get("path")
                if path_value:
                    block = self._build_file_block(Path(path_value))
                    if block:
                        blocks.append(block)
        return blocks

    def _build_file_block(self, path: Path) -> Optional[Dict[str, Any]]:
        text_payload = self._try_inline_text(path)
        if text_payload is not None:
            return {"type": "input_text", "text": text_payload}
        formatted = self._format_attachment(path)
        if not formatted:
            return None
        return {"type": "input_text", "text": formatted}

    def _build_image_blocks(self, info: Dict[str, Any]) -> List[Dict[str, Any]]:
        mime = info.get("mime") or "image/png"
        data_url = info.get("data_url")
        base64_payload = info.get("base64")
        if not base64_payload:
            return []
        if not data_url:
            data_url = f"data:{mime};base64,{base64_payload}"
        block: Dict[str, Any] = {"type": "input_image", "image_url": data_url}
        return [block]

    def _normalize_attachment(self, raw: Any) -> Optional[Dict[str, Any]]:
        if raw is None:
            return None
        if isinstance(raw, dict):
            info = dict(raw)
            path_value = info.get("path")
            if path_value:
                info["path"] = Path(path_value).expanduser().resolve().as_posix()
            if info.get("kind") == "image" and not info.get("base64") and info.get("path"):
                meta = preprocess_image(Path(info["path"]))
                info.update(self._image_info_from_meta(Path(info["path"]), meta))
            return info
        try:
            path = Path(raw).expanduser().resolve()
        except OSError:
            return None
        if path.exists() and is_image_file(path):
            meta = preprocess_image(path)
            return self._image_info_from_meta(path, meta)
        return {"kind": "file", "path": path.as_posix()}

    def _image_info_from_meta(self, path: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "kind": "image",
            "path": path.as_posix(),
            "mime": meta.get("mime", "image/png"),
            "base64": meta.get("base64", ""),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "source_size": meta.get("source_size"),
            "output_size": meta.get("output_size"),
            "ratio": meta.get("ratio"),
            "source_path": Path(meta.get("source", path)).as_posix() if meta.get("source") else path.as_posix(),
            "processed_path": Path(meta.get("output", path)).as_posix() if meta.get("output") else path.as_posix(),
        }

    def _get_model_profile(self) -> Dict[str, Any]:
        profile = MODEL_PROFILES.get(self.model)
        if profile:
            return dict(profile)
        return {
            "endpoint": self.endpoint,
            "schema": "responses",
            "supports_temperature": True,
        }

    def _prune_history(self) -> None:
        cap = self.max_history * 2
        if len(self.conversation) > cap:
            self.conversation = self.conversation[-cap:]

    def _post_json(
        self,
        body: Dict[str, Any],
        endpoint: str,
        *,
        timeout: Optional[float] = None,
        stream: bool = False,
        stream_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("未配置 API Key。请设置 MODEL_API_KEY 环境变量或在构造函数传入 api_key。")
        data = json.dumps(body).encode("utf-8")
        headers = self._build_headers()
        target = endpoint or self.endpoint
        request = urllib.request.Request(target, data=data, headers=headers)
        start = time.time()
        effective_timeout = timeout if timeout is not None else self.timeout
        try:
            if stream:
                result = self._post_streaming_request(
                    request,
                    timeout=effective_timeout,
                    callback=stream_callback,
                )
            else:
                with urllib.request.urlopen(request, timeout=effective_timeout) as resp:
                    resp_bytes = resp.read()
                    text = resp_bytes.decode("utf-8", errors="replace")
                    result = json.loads(text)
        except urllib.error.HTTPError as err:
            payload = err.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {err.code}: {payload}") from err
        except urllib.error.URLError as err:
            raise RuntimeError(f"请求失败: {err.reason}") from err
        finally:
            elapsed = time.time() - start
            self.last_preview["request"] = {
                "endpoint": target,
                "headers": mask_headers(headers),
                "body": body,
                "elapsed": elapsed,
            }
        self.last_preview["response"] = result
        return result

    def _post_streaming_request(
        self,
        request: urllib.request.Request,
        *,
        timeout: float,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        accumulator = {
            "text_parts": [],
            "output_items": [],
            "final_response": None,
            "usage": None,
        }
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            buffer: List[str] = []
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    if buffer:
                        payload = "".join(buffer).strip()
                        buffer.clear()
                        self._handle_stream_payload(payload, accumulator, callback)
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data_line = line[5:].lstrip()
                    if data_line == "[DONE]":
                        break
                    buffer.append(data_line)
                else:
                    buffer.append(line)
            if buffer:
                payload = "".join(buffer).strip()
                self._handle_stream_payload(payload, accumulator, callback)
        final_resp = accumulator["final_response"]
        if final_resp is None:
            final_resp = {}
        text_value = "".join(accumulator["text_parts"])
        if text_value and not final_resp.get("output_text"):
            final_resp["output_text"] = [text_value]
        if accumulator["output_items"] and not final_resp.get("output"):
            final_resp["output"] = accumulator["output_items"]
        if accumulator["usage"] and not final_resp.get("usage"):
            final_resp["usage"] = accumulator["usage"]
        if text_value and not final_resp.get("text"):
            final_resp["text"] = text_value
        return final_resp

    def _handle_stream_payload(
        self,
        payload: str,
        accumulator: Dict[str, Any],
        callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        if not payload:
            return
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            return
        if callback:
            for event in self._chunk_to_events(chunk):
                callback(event)
        self._accumulate_stream_chunk(chunk, accumulator)

    def _chunk_to_events(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        chunk_type = chunk.get("type", "")
        if chunk_type == "response.reasoning_summary_text.delta":
            delta = chunk.get("delta")
            if delta:
                events.append({"type": "thinking", "text": delta})
        elif chunk_type == "response.output_text.delta":
            delta = chunk.get("delta")
            if delta:
                events.append({"type": "text", "text": delta})
        elif chunk_type == "response.output_item.added":
            item = chunk.get("item", {})
            status = item.get("type")
            if status:
                events.append({"type": "status", "status": status})
        elif chunk_type == "response.completed":
            events.append({"type": "complete"})
        return events

    def _accumulate_stream_chunk(self, chunk: Dict[str, Any], accumulator: Dict[str, Any]) -> None:
        if "output" in chunk:
            accumulator["output_items"].extend(chunk.get("output", []))
            for item in chunk.get("output", []):
                if item.get("type") == "message":
                    for block in item.get("content", []):
                        if block.get("type") in {"output_text", "text"}:
                            text = block.get("text")
                            if text:
                                accumulator["text_parts"].append(text)
        chunk_type = chunk.get("type")
        if chunk_type == "response.output_text.delta":
            delta = chunk.get("delta")
            if delta:
                accumulator["text_parts"].append(delta)
        elif chunk_type == "response.reasoning_summary_text.delta":
            delta = chunk.get("delta")
            if delta:
                accumulator.setdefault("thinking_parts", []).append(delta)
        elif chunk_type == "response.completed":
            accumulator["final_response"] = chunk.get("response")
            accumulator["usage"] = chunk.get("response", {}).get("usage")
        elif chunk_type == "response.error":
            message = chunk.get("error", {}).get("message") or chunk.get("message") or "流式响应失败"
            raise RuntimeError(message)

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if not self.api_key:
            return headers
        if self.auth_type.lower() == "bearer":
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.auth_type.lower() == "x-api-key":
            headers["X-API-Key"] = self.api_key
        else:
            headers[self.auth_type] = self.api_key
        return headers

    def _init_memory(
        self,
        memory_manager: Optional[AgentMemory],
        storage_path: Optional[os.PathLike[str] | str],
        max_entries: int,
    ) -> Optional[AgentMemory]:
        if memory_manager is not None:
            return memory_manager
        try:
            manager = AgentMemory(storage_path=storage_path, max_entries=max_entries)
            self.memory_error = None
            return manager
        except Exception as exc:  # pragma: no cover - defensive
            self.memory_error = str(exc)
            print(f"[NetworkAgent] Memory disabled: {exc}")
            return None

    def _bootstrap_memory_history(self) -> None:
        if self.memory is None:
            return
        if self.conversation:
            return
        records = self.memory.snapshot()
        if not records:
            return
        usable = records[-(self.max_history) :]
        for item in usable:
            user = item.get("user")
            assistant = item.get("assistant")
            if user:
                self.conversation.append({"role": "user", "content": user})
            if assistant:
                self.conversation.append({"role": "assistant", "content": assistant})
        self._prune_history()

    def _extract_reply(self, raw: Dict[str, Any], profile: Dict[str, Any]) -> str:
        schema = profile.get("schema", "responses")
        if schema == "chat":
            choices = raw.get("choices") or []
            if not choices:
                return "(无响应内容)"
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, list):
                texts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
                merged = "\n".join(t.strip() for t in texts if t)
                if merged:
                    return merged
            text = (message.get("content") or "").strip()
            return text or "(无响应内容)"
        texts = raw.get("output_text")
        if texts:
            merged = "\n".join(t.strip() for t in texts if t)
            if merged:
                return merged
        output_items = raw.get("output") or []
        for item in output_items:
            if item.get("type") != "message":
                continue
            for block in item.get("content", []):
                if block.get("type") in {"output_text", "text"}:
                    text = (block.get("text") or "").strip()
                    if text:
                        return text
        return raw.get("text", "(无响应内容)") or "(无响应内容)"

    def _format_attachment(self, path: Path) -> str:
        if not path.exists():
            return f"<FILE name=\"{path.name}\">(文件不存在)</FILE>"
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
            body = text if len(text) <= 4000 else text[:4000] + "\n...(截断)"
            return f"<FILE name=\"{path.name}\" bytes={len(data)} path=\"{path.as_posix()}\">\n{body}\n</FILE>"
        except UnicodeDecodeError:
            b64 = base64.b64encode(data).decode("ascii")
            preview = b64[:8000] + ("...(截断)" if len(b64) > 8000 else "")
            return f"<BINARY name=\"{path.name}\" bytes={len(data)} path=\"{path.as_posix()}\">BASE64:{preview}</BINARY>"

    def _try_inline_text(self, path: Path) -> Optional[str]:
        if not path.exists() or not path.is_file():
            return None
        suffix = path.suffix.lower()
        looks_like_text = suffix in TEXT_FILE_EXTENSIONS
        max_bytes = TEXT_ATTACHMENT_MAX_BYTES
        try:
            with path.open("rb") as fh:
                sample = fh.read(max_bytes + 4096)
        except OSError:
            return None
        if not sample:
            return f"{path.name}\n(空文件)"
        decoded = self._decode_text_bytes(sample)
        if decoded is None and not looks_like_text:
            return None
        text_content = decoded or sample.decode("utf-8", errors="ignore")
        text_content = text_content.strip()
        if not text_content:
            return None
        if len(text_content) > TEXT_ATTACHMENT_MAX_CHARS:
            text_content = text_content[:TEXT_ATTACHMENT_MAX_CHARS] + "\n...(截断)"
        return f"{path.name}\n{text_content}"

    def _decode_text_bytes(self, data: bytes) -> Optional[str]:
        encodings = ["utf-8", "utf-16", "utf-16-le", "utf-16-be", "utf-8-sig", "gb18030"]
        for encoding in encodings:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return None

    # ------------------------------------------------------------------
    def _load_history_into_conversation(self) -> None:
        self.conversation = self._load_history(self.current_history_name)

    def _load_history(self, name: str) -> List[Dict[str, Any]]:
        path = self._history_path(name)
        if not path.exists():
            self._ensure_history_file(name)
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            return []
        return []

    def _save_history(self) -> None:
        path = self._history_path(self.current_history_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.conversation, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _history_path(self, name: str) -> Path:
        return self.history_dir / f"{name}.json"

    def _ensure_history_file(self, name: str) -> None:
        path = self._history_path(name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[]", encoding="utf-8")

    def _normalize_history_name(self, name: str) -> str:
        candidate = (name or "").strip()
        if not candidate:
            raise ValueError("历史名称不能为空")
        if not HISTORY_NAME_PATTERN.match(candidate):
            raise ValueError("历史名称仅支持字母、数字、下划线和连字符，且长度不超过48字符")
        return candidate


# ----------------------------------------------------------------------
def parse_operations(text: str) -> List[Dict[str, Any]]:
    marker = "###OPERATIONS"
    idx = text.find(marker)
    if idx == -1:
        return []
    snippet = text[idx + len(marker):].strip()
    if snippet.startswith("```"):
        snippet = snippet[3:]
        fence_end = snippet.find("```")
        if fence_end != -1:
            snippet = snippet[:fence_end]
    snippet = snippet.lstrip("json").strip()
    try:
        data = json.loads(snippet)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("operations"), list):
            return data["operations"]
    except json.JSONDecodeError:
        return []
    return []


def mask_headers(headers: Dict[str, str]) -> Dict[str, str]:
    masked: Dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()
        if "authorization" in lower or "api" in lower:
            masked[key] = mask_secret(value)
        else:
            masked[key] = value
    return masked


def mask_secret(value: str) -> str:
    if not value:
        return value
    if len(value) <= 8:
        return "***"
    return value[:4] + "..." + value[-4:]


# ----------------------------------------------------------------------
def demo() -> None:
    agent = NetworkAgent()
    try:
        reply, ops = agent.send_message("你好，介绍一下这个仓库包含哪些主要文件？", attachments=[Path(__file__).parent.parent / "README.md"])
        print("ASSISTANT:\n", reply)
        if ops:
            print("OPERATIONS:")
            for op in ops:
                print(json.dumps(op, ensure_ascii=False))
    except RuntimeError as err:
        print("调用失败:", err)


if __name__ == "__main__":
    demo()
