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
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Match, Optional, Sequence, Tuple, Union

from openai import OpenAI

from python.agent_memory import AgentMemory
from python.image_preprocessor import is_image_file, preprocess_image

MODEL_PROFILES = {
    "gpt-4.1": {
        "endpoint": "https://aihubmix.com/v1",
        "schema": "responses",
        "supports_temperature": True,
        "supports_reasoning": False,
        "supports_web_search": False,
    },
    "gpt-5": {
        "endpoint": "https://aihubmix.com/v1",
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
PDF_MAX_BYTES = 32 * 1024 * 1024
PDF_MAX_PAGES = 100
PDF_SUPPORTED_MODELS = {
    "gpt-4.1",
    "gpt-5",
}
PDF_PAGE_PATTERN = re.compile(rb"/Type\s*/Page\b")
INCOMPLETE_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*$")
INCOMPLETE_URL_RE = re.compile(r"(https?://[^\s)]*)$")
INCOMPLETE_CITATION_RE = re.compile(r"\[\[[^\]]*$")
CITATION_REF_RE = re.compile(r"\[\[(\d+)\]\]")
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
    "后跟一个 JSON 数组，数组元素格式 {\"action\":\"writeFile\",\"path\":\"相对路径\",\"content\":\"文件内容\"}，"
    "必要时可额外附加 \"diff\" 字段放置 unified diff 预览。"
    "若需要读取工作区文件，请在 ###OPERATIONS 中追加 {\"action\":\"readFile\",\"path\":\"相对路径\"}，"
    "并等待后续响应提供文件内容，切勿臆造文件文本。"
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
        self._client: Optional[OpenAI] = None
        self._client_endpoint: Optional[str] = None

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
            raw = self._dispatch_via_sdk(
                payload,
                profile.get("endpoint") or self.endpoint,
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
        if self.streaming_enabled and stream_callback:
            try:
                stream_callback({"type": "final_text", "text": reply})
            except Exception:
                pass
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
        reasoning_effort = self._resolve_reasoning_effort()
        if profile.get("supports_reasoning") and reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        if profile.get("supports_web_search") and self.web_search_enabled:
            payload["tools"] = [{"type": "web_search"}]
        return payload

    def _resolve_reasoning_effort(self) -> Optional[str]:
        effort = self.reasoning_effort
        if effort is None:
            return None
        normalized = effort.strip().lower()
        allowed = {"minimal", "low", "medium", "high"}
        if normalized not in allowed:
            raise ValueError(f"无效的 reasoning effort: {effort}")
        conflict = (
            normalized == "minimal"
            and self.streaming_enabled
            and self.web_search_enabled
            and self._is_gpt5_series()
        )
        if conflict:
            return "low"
        return normalized

    def _is_gpt5_series(self) -> bool:
        model_name = (self.model or "").strip().lower()
        return model_name.startswith("gpt-5")

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
            fragments = [self._summarize_block(block) for block in content]
            fragments = [frag for frag in fragments if frag]
            if fragments:
                return "\n".join(fragments)
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)

    def _summarize_block(self, block: Any) -> str:
        if not isinstance(block, dict):
            return ""
        btype = block.get("type")
        if btype in {"input_text", "output_text", "text"}:
            text = block.get("text") or ""
            text = text.strip()
            if len(text) > 200:
                text = text[:200].rstrip() + "…"
            return text
        if btype in {"input_image", "image_url"}:
            return "[图片附件]"
        if btype == "input_audio":
            return "[音频附件]"
        if btype == "input_file":
            filename = block.get("filename") or block.get("name")
            if not filename:
                data = block.get("file_data") or block.get("data") or ""
                if isinstance(data, str) and len(data) > 30:
                    data = data[:30] + "…"
                filename = f"{data}"
            return f"[文件附件: {filename}]"
        return "[附件]"

    def _create_stream_normalizer(self) -> Optional["WebSearchStreamNormalizer"]:
        if not self.streaming_enabled:
            return None
        return WebSearchStreamNormalizer()

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
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pdf_block = self._build_pdf_block(path)
            if pdf_block is not None:
                return pdf_block
        text_payload = self._try_inline_text(path)
        if text_payload is not None:
            return {"type": "input_text", "text": text_payload}
        formatted = self._format_attachment(path)
        if not formatted:
            return None
        return {"type": "input_text", "text": formatted}

    def _build_pdf_block(self, path: Path) -> Optional[Dict[str, Any]]:
        model_key = (self.model or "").strip()
        if model_key not in PDF_SUPPORTED_MODELS:
            raise NotImplementedError(f"PDF attachments are not implemented for model '{model_key}' yet")
        return self._build_pdf_block_for_supported_gpt(path)

    def _build_pdf_block_for_supported_gpt(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            size = path.stat().st_size
        except OSError:
            return None
        if size > PDF_MAX_BYTES:
            return None
        try:
            data = path.read_bytes()
        except OSError:
            return None
        page_count = self._estimate_pdf_page_count(data)
        if page_count is None or page_count > PDF_MAX_PAGES:
            return None
        payload = base64.b64encode(data).decode("ascii")
        return {
            "type": "input_file",
            "filename": path.name,
            "file_data": f"data:application/pdf;base64,{payload}",
        }

    def _estimate_pdf_page_count(self, data: bytes) -> Optional[int]:
        if not data:
            return None
        try:
            matches = PDF_PAGE_PATTERN.findall(data)
        except re.error:
            return None
        return len(matches) or None

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

    def _dispatch_via_sdk(
        self,
        body: Dict[str, Any],
        endpoint: Optional[str],
        *,
        timeout: Optional[float],
        stream: bool,
        stream_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("未配置 API Key。请设置 MODEL_API_KEY 环境变量或在构造函数传入 api_key。")
        client = self._ensure_client(endpoint)
        payload = dict(body)
        headers = self._build_headers()
        target = (endpoint or self.endpoint or "").rstrip("/")
        request_record = {
            "endpoint": target,
            "headers": mask_headers(headers),
            "body": body,
        }
        self.last_preview["request"] = request_record
        start = time.time()
        effective_timeout = timeout if timeout is not None else self.timeout
        try:
            if stream:
                payload["stream"] = True
                result, raw_result = self._stream_via_sdk(
                    client,
                    payload,
                    timeout=effective_timeout,
                    callback=stream_callback,
                )
            else:
                payload.pop("stream", None)
                response = client.responses.create(timeout=effective_timeout, **payload)
                result = self._response_to_dict(response)
                raw_result = result
        finally:
            request_record["elapsed"] = time.time() - start
        self.last_preview["response"] = raw_result
        return result

    def _stream_via_sdk(
        self,
        client: OpenAI,
        body: Dict[str, Any],
        *,
        timeout: float,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        request_body = dict(body)
        request_body.pop("stream", None)
        accumulator = {
            "text_parts": [],
            "output_items": [],
            "final_response": None,
            "usage": None,
        }
        normalizer = self._create_stream_normalizer()
        final_response_obj: Optional[Any] = None
        with client.responses.stream(timeout=timeout, **request_body) as stream:
            for event in stream:
                chunk = self._event_to_dict(event)
                if not chunk:
                    continue
                self._handle_stream_payload(
                    chunk,
                    accumulator,
                    callback,
                    None,
                    normalizer,
                )
            final_response_obj = stream.get_final_response()
        if normalizer and callback:
            for text in normalizer.flush():
                if text:
                    callback({"type": "text", "text": text})
        raw_result = self._finalize_stream_result(accumulator)
        clean_result = (
            self._response_to_dict(final_response_obj)
            if final_response_obj is not None
            else raw_result
        )
        return clean_result, raw_result

    def _response_to_dict(self, response: Any) -> Dict[str, Any]:
        if response is None:
            return {}
        if isinstance(response, dict):
            return response
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "to_dict"):
            return response.to_dict()
        if hasattr(response, "__dict__"):
            raw = {k: v for k, v in response.__dict__.items() if not k.startswith("_")}
            if raw:
                return raw
        return json.loads(json.dumps(response, default=str))

    def _event_to_dict(self, event: Any) -> Dict[str, Any]:
        if event is None:
            return {}
        if isinstance(event, dict):
            return event
        if hasattr(event, "model_dump"):
            return event.model_dump()
        if hasattr(event, "to_dict"):
            return event.to_dict()
        data: Dict[str, Any] = {}
        event_type = getattr(event, "type", None)
        if event_type:
            data["type"] = event_type
        payload = getattr(event, "data", None)
        if isinstance(payload, dict):
            data.update(payload)
        elif payload is not None:
            data["data"] = payload
        if not data and hasattr(event, "__dict__"):
            data = {k: v for k, v in event.__dict__.items() if not k.startswith("_")}
        return data

    def _ensure_client(self, endpoint: Optional[str] = None) -> OpenAI:
        target = (endpoint or self.endpoint or DEFAULT_ENDPOINT).rstrip("/")
        if target.endswith("/responses"):
            target = target[: -len("/responses")]
        if not self._client or self._client_endpoint != target:
            self.endpoint = target
            self._client = self._create_client(base_url=target)
            self._client_endpoint = target
        return self._client

    def _create_client(self, *, base_url: Optional[str] = None) -> OpenAI:
        client_kwargs: Dict[str, Any] = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        custom_headers = self._custom_sdk_headers()
        if custom_headers:
            client_kwargs["default_headers"] = custom_headers
        return OpenAI(**client_kwargs)

    def _custom_sdk_headers(self) -> Dict[str, str]:
        if not self.api_key:
            return {}
        auth = (self.auth_type or "bearer").lower()
        if auth == "bearer":
            return {}
        if auth == "x-api-key":
            return {"X-API-Key": self.api_key}
        return {self.auth_type: self.api_key}

    def _handle_stream_payload(
        self,
        payload: Union[str, Dict[str, Any]],
        accumulator: Dict[str, Any],
        callback: Optional[Callable[[Dict[str, Any]], None]],
        event_name: Optional[str] = None,
        normalizer: Optional["WebSearchStreamNormalizer"] = None,
    ) -> None:
        if not payload:
            return
        if isinstance(payload, dict):
            chunk = payload
        else:
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                return
        if event_name and not chunk.get("type"):
            chunk["type"] = event_name
        if normalizer:
            normalizer.observe_chunk(chunk)
        if callback:
            for event in self._chunk_to_events(chunk, normalizer):
                callback(event)
        self._accumulate_stream_chunk(chunk, accumulator)

    def _chunk_to_events(
        self,
        chunk: Dict[str, Any],
        normalizer: Optional["WebSearchStreamNormalizer"] = None,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        chunk_type = chunk.get("type", "")
        if chunk_type == "response.reasoning_summary_text.delta":
            delta = chunk.get("delta")
            if delta:
                events.append({"type": "thinking", "text": delta})
        elif chunk_type == "response.output_text.delta":
            delta = chunk.get("delta")
            if delta:
                for text in self._normalize_stream_text(delta, normalizer):
                    if text:
                        events.append({"type": "text", "text": text})
        elif chunk_type == "response.output_item.added":
            item = chunk.get("item", {})
            status = item.get("type")
            if status:
                events.append({"type": "status", "status": status})
        elif chunk_type == "response.completed":
            if normalizer:
                for text in self._normalize_stream_text("", normalizer, flush=True):
                    if text:
                        events.append({"type": "text", "text": text})
            events.append({"type": "complete"})
        return events

    def _normalize_stream_text(
        self,
        delta: str,
        normalizer: Optional["WebSearchStreamNormalizer"],
        *,
        flush: bool = False,
    ) -> List[str]:
        if not normalizer:
            return [delta] if delta else []
        if flush:
            return normalizer.flush()
        return normalizer.feed(delta)

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

    def _finalize_stream_result(self, accumulator: Dict[str, Any]) -> Dict[str, Any]:
        base = accumulator.get("final_response")
        result = self._response_to_dict(base) if base is not None else {}
        if not isinstance(result, dict):
            result = {}
        text_value = "".join(accumulator.get("text_parts", []))
        if text_value and not result.get("output_text"):
            result["output_text"] = [text_value]
        output_items = accumulator.get("output_items") or []
        if output_items and not result.get("output"):
            result["output"] = output_items
        usage_value = accumulator.get("usage")
        if usage_value and not result.get("usage"):
            result["usage"] = usage_value
        if text_value and not result.get("text"):
            result["text"] = text_value
        return result

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if not self.api_key:
            return headers
        auth = (self.auth_type or "bearer").lower()
        if auth == "bearer":
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif auth == "x-api-key":
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


class WebSearchStreamNormalizer:
    def __init__(self, *, max_carry: int = 512) -> None:
        self._buffer: str = ""
        self._max_carry = max_carry
        self._search_results: List[Dict[str, Any]] = []
        self._seen_result_keys: set[Tuple[str, str]] = set()

    def feed(self, delta: str) -> List[str]:
        if not delta:
            return []
        self._buffer += delta
        return self._drain(flush=False)

    def flush(self) -> List[str]:
        outputs = self._drain(flush=True)
        if self._search_results:
            outputs.append(self._format_results_block())
            self._search_results.clear()
            self._seen_result_keys.clear()
        return outputs

    def observe_chunk(self, chunk: Dict[str, Any]) -> None:
        if not isinstance(chunk, dict):
            return
        self._ingest_search_results(chunk)

    # ------------------------------------------------------------------
    def _drain(self, *, flush: bool) -> List[str]:
        if not self._buffer:
            return []
        if flush:
            text = self._buffer
            self._buffer = ""
            return [self._apply_citations(text)] if text else []
        stable, carry = self._split_stable_prefix(self._buffer)
        if not stable:
            return []
        self._buffer = carry
        return [self._apply_citations(stable)]

    def _split_stable_prefix(self, text: str) -> Tuple[str, str]:
        carry_len = self._calculate_carry_length(text)
        if carry_len <= 0:
            return text, ""
        carry_len = min(carry_len, len(text), self._max_carry)
        if carry_len >= len(text):
            return "", text[-carry_len:]
        return text[:-carry_len], text[-carry_len:]

    def _calculate_carry_length(self, text: str) -> int:
        hold = 0
        for pattern in (INCOMPLETE_MARKDOWN_LINK_RE, INCOMPLETE_URL_RE, INCOMPLETE_CITATION_RE):
            match = pattern.search(text)
            if match:
                remaining = len(text) - match.start()
                hold = max(hold, remaining)
        return hold

    def _apply_citations(self, text: str) -> str:
        if not (text and self._search_results):
            return text

        def _replace(match: Match[str]) -> str:
            try:
                idx = int(match.group(1))
            except (TypeError, ValueError):
                return match.group(0)
            result = self._lookup_result(idx)
            if not result or not result.get("url"):
                return match.group(0)
            title = result.get("title") or f"参考 {idx}"
            return f"[{idx}. {title}]({result['url']})"

        return CITATION_REF_RE.sub(_replace, text)

    def _lookup_result(self, idx: int) -> Optional[Dict[str, Any]]:
        if idx <= 0:
            return None
        if idx <= len(self._search_results):
            return self._search_results[idx - 1]
        return None

    def _ingest_search_results(self, payload: Any) -> None:
        if isinstance(payload, dict):
            for key, value in payload.items():
                lowered = key.lower()
                if "web_search" in lowered and isinstance(value, (list, dict)):
                    self._ingest_search_results(value)
                elif lowered in {"results", "search_results", "web_search_results"}:
                    self._ingest_result_collection(value)
                elif isinstance(value, (dict, list)):
                    self._ingest_search_results(value)
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, (dict, list)):
                    self._ingest_search_results(item)

    def _ingest_result_collection(self, value: Any) -> None:
        if isinstance(value, dict):
            if any(k in value for k in ("url", "link")):
                self._add_result(value)
                return
            nested = value.get("results") or value.get("data")
            if nested is not None:
                self._ingest_result_collection(nested)
            return
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    self._add_result(item)

    def _add_result(self, entry: Dict[str, Any]) -> None:
        url = (entry.get("url") or entry.get("link") or "").strip()
        title = (entry.get("title") or entry.get("name") or url).strip()
        snippet = (entry.get("snippet") or entry.get("description") or entry.get("text") or "").strip()
        if not (url or title):
            return
        key = (title, url)
        if key in self._seen_result_keys:
            return
        self._seen_result_keys.add(key)
        normalized = {
            "title": title or "参考链接",
            "url": url,
            "snippet": snippet,
        }
        self._search_results.append(normalized)

    def _format_results_block(self) -> str:
        if not self._search_results:
            return ""
        lines = ["\n\n**Web search results**"]
        for idx, entry in enumerate(self._search_results, start=1):
            title = entry.get("title") or f"结果 {idx}"
            url = entry.get("url")
            snippet = entry.get("snippet")
            if url:
                line = f"{idx}. [{title}]({url})"
            else:
                line = f"{idx}. {title}"
            if snippet:
                line += f" - {snippet}"
            lines.append(line)
        return "\n".join(lines) + "\n"

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
