"""Simple desktop chat window for the NetworkAgent.

本文件现已连接到基于 OpenAI Python SDK 的 `python/network_agent.py`，
旧实现（直接发 HTTP 请求版）已完整归档到 `legacy_version/test.py` 及其同目录下的模块。

提供以下功能：
- 聊天记录展示（用户与助手分色）
- 输入框与发送按钮
- **发送前** 的 JSON 请求预览 + 最近响应预览
- 等待远端响应期间的动画提示
- 附件选择与图片压缩预览

运行前请将 `MODEL_API_KEY` 设置为你的远程模型密钥。
"""

from __future__ import annotations

import base64
import json
import os
import threading
import tkinter as tk
from io import BytesIO
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk
from typing import Any, Callable, Dict, List, Optional

from PIL import Image, ImageTk

from python.image_preprocessor import is_image_file, preprocess_image
from python.network_agent import NetworkAgent
from python.file_ops import build_unified_diff, read_text_for_diff

with open('api-key.txt', 'r', encoding='utf-8') as f:
    api_key = f.read().strip()

def _human_size(num: Optional[int]) -> str:
    if num is None:
        return "未知"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


MODEL_CHOICES = ["gpt-4.1", "gpt-5"]
REASONING_CHOICES = [
    ("极简 (minimal)", "minimal"),
    ("较低 (low)", "low"),
    ("默认 (medium)", "medium"),
    ("较高 (high)", "high"),
]


class ChatWindow:
    def __init__(self) -> None:
        self.workspace_root = Path(__file__).resolve().parent
        self._workspace_root_resolved = self.workspace_root.resolve()
        self.agent = NetworkAgent(api_key=api_key)
        self.root = tk.Tk()
        self.root.title("Cherry Studio Python Chat")
        self.root.geometry("840x640")
        self.model_var = tk.StringVar(value=self.agent.model)
        self.reasoning_var = tk.StringVar(value="medium")
        self.agent.set_reasoning_effort(self.reasoning_var.get())
        self.web_search_var = tk.BooleanVar(value=False)
        self.agent.set_web_search_enabled(self.web_search_var.get())
        self.streaming_var = tk.BooleanVar(value=False)
        self.agent.set_streaming_enabled(self.streaming_var.get())
        self.history_var = tk.StringVar(value=self.agent.history_name)

        self._build_menubar()

        container = tk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        self._main_canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self._main_canvas.yview)
        self._scrollable_body = tk.Frame(self._main_canvas)
        self._scrollable_body.bind(
            "<Configure>",
            lambda event: self._main_canvas.configure(scrollregion=self._main_canvas.bbox("all")),
        )
        self._canvas_window = self._main_canvas.create_window((0, 0), window=self._scrollable_body, anchor="nw")
        self._main_canvas.configure(yscrollcommand=scrollbar.set)

        self._main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.root.bind("<Configure>", self._sync_canvas_width)

        self.history = scrolledtext.ScrolledText(
            self._scrollable_body,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg="#111",
            fg="#f0f0f0",
        )
        self.history.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 4))

        input_frame = tk.Frame(self._scrollable_body)
        input_frame.pack(fill=tk.X, padx=10)

        self.input_box = tk.Text(input_frame, height=4, wrap=tk.WORD)
        self.input_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        button_frame = tk.Frame(input_frame)
        button_frame.pack(side=tk.LEFT, padx=(6, 0))

        self.add_attachment_button = tk.Button(button_frame, text="添加附件", width=16, command=self.add_attachments)
        self.add_attachment_button.pack(fill=tk.X)

        self.clear_attachments_button = tk.Button(button_frame, text="清空附件", width=16, command=self.clear_attachments)
        self.clear_attachments_button.pack(fill=tk.X, pady=(4, 8))

        self.send_button = tk.Button(button_frame, text="发送 (Ctrl+Enter)", width=16, command=self.send)
        self.send_button.pack(fill=tk.X)

        self.clear_memory_button = tk.Button(button_frame, text="清空记忆", width=16, command=self.clear_memory)
        self.clear_memory_button.pack(fill=tk.X, pady=(6, 0))

        self.spinner_label = tk.Label(button_frame, text="就绪", fg="#888")
        self.spinner_label.pack(pady=(6, 0))

        self.memory_label = tk.Label(button_frame, text="记忆条数: 0", fg="#aaa")
        self.memory_label.pack(pady=(6, 0))

        self.memory_path_label = tk.Label(button_frame, text="", fg="#777", wraplength=150, justify=tk.LEFT)
        self.memory_path_label.pack()

        self.model_label = tk.Label(button_frame, text=f"模型: {self.model_var.get()}", fg="#5dd39e")
        self.model_label.pack(pady=(6, 0))

        self.reasoning_label = tk.Label(button_frame, text=f"推理: {self.reasoning_var.get()}", fg="#f5a623")
        self.reasoning_label.pack(pady=(4, 0))

        self.web_search_check = tk.Checkbutton(
            button_frame,
            text="启用联网搜索",
            variable=self.web_search_var,
            command=self._toggle_web_search,
            anchor="w",
            justify=tk.LEFT,
        )
        self.web_search_check.pack(fill=tk.X, pady=(4, 0))

        self.streaming_check = tk.Checkbutton(
            button_frame,
            text="启用流式响应",
            variable=self.streaming_var,
            command=self._toggle_streaming,
            anchor="w",
            justify=tk.LEFT,
        )
        self.streaming_check.pack(fill=tk.X, pady=(2, 0))

        history_frame = ttk.LabelFrame(button_frame, text="对话历史")
        history_frame.pack(fill=tk.X, pady=(6, 0))
        self.history_combo = ttk.Combobox(history_frame, state="readonly", textvariable=self.history_var)
        self._refresh_history_selector()
        self.history_combo.pack(fill=tk.X, padx=2, pady=(2, 0))
        self.history_combo.bind("<<ComboboxSelected>>", self._on_history_selected)
        tk.Button(history_frame, text="新建历史", command=self._prompt_new_history).pack(fill=tk.X, padx=2, pady=(4, 0))
        tk.Button(history_frame, text="刷新列表", command=self._refresh_history_selector).pack(fill=tk.X, padx=2, pady=(2, 4))

        attachments_frame = ttk.LabelFrame(self._scrollable_body, text="附件")
        attachments_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=(4, 4))

        attachments_left = tk.Frame(attachments_frame)
        attachments_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.attachments_list = tk.Listbox(attachments_left, height=4)
        self.attachments_list.pack(fill=tk.BOTH, expand=True)
        self.attachments_list.bind("<<ListboxSelect>>", self._on_attachment_select)

        attachments_preview = tk.Frame(attachments_frame)
        attachments_preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        self.attachment_preview_frame = tk.Frame(attachments_preview, height=260, width=260, bg="#222")
        self.attachment_preview_frame.pack(fill=tk.BOTH, expand=True)
        self.attachment_preview_frame.pack_propagate(False)

        self.attachment_image_label = tk.Label(
            self.attachment_preview_frame,
            text="(无图片预览)",
            bg="#222",
            fg="#777",
            anchor=tk.CENTER,
        )
        self.attachment_image_label.pack(fill=tk.BOTH, expand=True)

        self.attachment_info_label = tk.Label(
            attachments_preview,
            text="(未选择附件)",
            justify=tk.LEFT,
            anchor="nw",
            wraplength=240,
        )
        self.attachment_info_label.pack(fill=tk.X, expand=False, pady=(6, 0))

        ops_frame = ttk.LabelFrame(self._scrollable_body, text="生成的文件操作")
        ops_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=(4, 4))
        self.operations_box = scrolledtext.ScrolledText(ops_frame, height=8, wrap=tk.NONE, state=tk.DISABLED)
        self.operations_box.pack(fill=tk.BOTH, expand=True)
        self.apply_operations_button = tk.Button(
            ops_frame,
            text="应用操作到磁盘",
            command=self.apply_pending_operations,
            state=tk.DISABLED,
        )
        self.apply_operations_button.pack(fill=tk.X, padx=4, pady=(4, 2))

        read_frame = ttk.LabelFrame(self._scrollable_body, text="读取文件")
        read_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=(4, 4))
        read_controls = tk.Frame(read_frame)
        read_controls.pack(fill=tk.X, padx=4, pady=(2, 2))
        tk.Label(read_controls, text="相对路径:").pack(side=tk.LEFT)
        self.read_path_var = tk.StringVar()
        self.read_path_entry = tk.Entry(read_controls, textvariable=self.read_path_var)
        self.read_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))
        tk.Button(read_controls, text="浏览", command=self.browse_read_file).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(read_controls, text="读取", command=self.read_file_content).pack(side=tk.LEFT)
        self.read_output_box = scrolledtext.ScrolledText(read_frame, height=8, wrap=tk.NONE, state=tk.DISABLED)
        self.read_output_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        stream_frame = ttk.LabelFrame(self._scrollable_body, text="实时流式状态")
        stream_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=(4, 4))
        self.streaming_status_var = tk.StringVar(value="状态: 流式未启用")
        self.stream_status_label = tk.Label(
            stream_frame,
            textvariable=self.streaming_status_var,
            anchor="w",
            justify=tk.LEFT,
            fg="#5dd39e",
        )
        self.stream_status_label.pack(fill=tk.X, padx=4, pady=(2, 0))
        self.stream_live_box = scrolledtext.ScrolledText(stream_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        self.stream_live_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 4))
        self._stream_thinking_buffer: List[str] = []
        self._stream_output_buffer: List[str] = []
        self._stream_rendered_text: str = ""
        self._stream_force_refresh = False
        self._pending_operations: List[Dict[str, Any]] = []
        self._auto_read_active = False
        self._auto_read_counter = 0
        self._auto_read_limit = 5

        preview_frame = ttk.LabelFrame(self._scrollable_body, text="待发送 JSON 预览")
        preview_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=(4, 4))

        self.request_preview = scrolledtext.ScrolledText(preview_frame, height=8, wrap=tk.NONE, state=tk.DISABLED)
        self.request_preview.pack(fill=tk.BOTH, expand=True)

        response_frame = ttk.LabelFrame(self._scrollable_body, text="最近响应 JSON")
        response_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=(4, 10))
        self.response_preview = scrolledtext.ScrolledText(response_frame, height=8, wrap=tk.NONE, state=tk.DISABLED)
        self.response_preview.pack(fill=tk.BOTH, expand=True)

        self.root.bind("<Control-Return>", lambda _event: self.send())
        self.input_box.bind("<KeyRelease>", self._on_input_change)
        self.attachments_data: List[Dict[str, Any]] = []

        self._spinner_running = False
        self._spinner_job: str | None = None
        self._spinner_frames = ["⠋", "⠙", "⠴", "⠦", "⠇", "⠋"]
        self._spinner_index = 0
        self._preview_job: str | None = None

        self._set_text(self.request_preview, "(输入内容以生成预览)")
        self._set_text(self.response_preview, "(暂无响应记录)")
        self._set_text(self.operations_box, "(暂无文件操作)")
        self._set_text(self.read_output_box, "(尚未读取任何文件)")
        self.restore_conversation_history()
        self.update_memory_label()
        self._reset_stream_display()

    # ------------------------------------------------------------------
    def run(self) -> None:
        if not self.agent.api_key:
            messagebox.showerror("缺少密钥", "请先设置 MODEL_API_KEY 环境变量或在 network_agent 中配置 api_key。")
        self.root.mainloop()

    # ------------------------------------------------------------------
    def send(self) -> None:
        text = self.input_box.get("1.0", tk.END).strip()
        attachments_payload = self._attachment_payloads()
        attachment_entries = list(self.attachments_data)
        if not text and not attachments_payload:
            messagebox.showinfo("提示", "请输入内容或添加附件。")
            return
        self.refresh_request_preview(text)
        display_text = self._compose_history_text(text, attachment_entries)
        self.append_history("user", display_text)
        self.input_box.delete("1.0", tk.END)
        self.send_button.config(state=tk.DISABLED)
        self.start_spinner()
        self._cancel_preview_job()
        self._set_text(self.request_preview, "(等待下一次输入以生成预览)")
        self._reset_stream_display()
        self._auto_read_counter = 0
        stream_callback = self._handle_stream_event if self.streaming_var.get() else None
        threading.Thread(
            target=self._worker_send,
            args=(text, attachments_payload, stream_callback),
            daemon=True,
        ).start()
        self.clear_attachments(silent=True)

    def _worker_send(
        self,
        text: str,
        attachments: List[Any],
        stream_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        try:
            reply, ops = self.agent.send_message(
                text,
                attachments=attachments,
                stream_callback=stream_callback,
            )
            error = None
        except Exception as exc:  # noqa: BLE001 - 展示真实错误
            reply = "(请求失败: {} )".format(exc)
            error = exc
        self.root.after(0, self._on_response, reply, error, ops if error is None else [])

    def _on_response(self, reply: str, error: Exception | None, operations: List[Dict[str, Any]]) -> None:
        self._process_agent_response(reply, error, operations, suppress_spinner=False, update_stream=True)

    def _process_agent_response(
        self,
        reply: str,
        error: Exception | None,
        operations: List[Dict[str, Any]],
        *,
        suppress_spinner: bool,
        update_stream: bool,
    ) -> None:
        if not suppress_spinner:
            self.stop_spinner()
            self.send_button.config(state=tk.NORMAL)
        if error:
            messagebox.showerror("调用失败", str(error))
            if self.streaming_var.get() and update_stream:
                self.streaming_status_var.set("状态: 调用失败")
                if not (self._stream_output_buffer or self._stream_thinking_buffer):
                    self._set_text(self.stream_live_box, f"(错误详情) {error}")
            self._pending_operations = []
            self._update_operations_preview([])
        else:
            if self.streaming_var.get() and update_stream:
                self.streaming_status_var.set("状态: 完成")
                self._update_stream_box()
            write_ops = [op for op in operations or [] if (op.get("action") == "writeFile")]
            read_ops = [op for op in operations or [] if (op.get("action") == "readFile")]
            self._pending_operations = write_ops
            self._update_operations_preview(self._pending_operations)
            if read_ops:
                self._handle_read_operations(read_ops)
        self.append_history("assistant", reply)
        self.refresh_response_preview()
        self.schedule_request_preview()
        self.update_memory_label()

    # ------------------------------------------------------------------
    def append_history(self, role: str, content: str) -> None:
        self.history.config(state=tk.NORMAL)
        tag = "user" if role == "user" else "assistant"
        prefix = "我" if role == "user" else "助手"
        self.history.insert(tk.END, f"[{prefix}]\n", tag)
        self.history.insert(tk.END, content + "\n\n")
        self.history.tag_config("user", foreground="#82d3ff")
        self.history.tag_config("assistant", foreground="#f6d365")
        self.history.config(state=tk.DISABLED)
        self.history.see(tk.END)

    def refresh_request_preview(self, text: str | None = None) -> None:
        self._preview_job = None
        pending = text if text is not None else self.input_box.get("1.0", tk.END).strip()
        attachments_payload = self._attachment_payloads()
        if not pending and not attachments_payload:
            self._set_text(self.request_preview, "(输入内容以生成预览)")
            return
        try:
            preview = self.agent.build_request_preview(pending, attachments=attachments_payload)
        except ValueError:
            self._set_text(self.request_preview, "(输入内容以生成预览)")
            return
        except Exception as exc:  # noqa: BLE001 - 展示真实错误
            self._set_text(self.request_preview, f"(预览失败: {exc})")
            return
        pretty = json.dumps(preview, ensure_ascii=False, indent=2)
        self._set_text(self.request_preview, pretty)

    def refresh_response_preview(self) -> None:
        preview = self.agent.last_preview
        self._set_text(
            self.response_preview,
            json.dumps(preview.get("response", {}), ensure_ascii=False, indent=2)
            if preview.get("response")
            else "(暂无响应记录)",
        )

    def _set_text(
        self,
        widget: scrolledtext.ScrolledText,
        value: str,
        keep_view: bool = False,
        follow_tail: bool = False,
    ) -> None:
        previous_state = str(widget.cget("state"))
        anchor_index = widget.index("@0,0") if keep_view else None
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, value)
        if follow_tail:
            widget.see(tk.END)
        elif keep_view and anchor_index:
            widget.see(anchor_index)
        else:
            widget.yview_moveto(0.0)
        widget.config(state=previous_state)

    def _insert_text(
        self,
        widget: scrolledtext.ScrolledText,
        value: str,
        follow_tail: bool = False,
    ) -> None:
        if not value:
            return
        previous_state = str(widget.cget("state"))
        anchor_index = widget.index("@0,0")
        widget.config(state=tk.NORMAL)
        widget.insert(tk.END, value)
        if follow_tail:
            widget.see(tk.END)
        else:
            widget.see(anchor_index)
        widget.config(state=previous_state)

    def _is_widget_near_bottom(self, widget: scrolledtext.ScrolledText, threshold: float = 0.985) -> bool:
        if not widget:
            return False
        try:
            _first, last = widget.yview()
        except tk.TclError:
            return False
        return last >= threshold

    def clear_memory(self) -> None:
        if not messagebox.askyesno("清空记忆", "将同时清空对话与持久化记忆，确定继续吗？"):
            return
        self.agent.clear_memory()
        self.history.config(state=tk.NORMAL)
        self.history.delete("1.0", tk.END)
        self.history.config(state=tk.DISABLED)
        self._set_text(self.request_preview, "(输入内容以生成预览)")
        self._set_text(self.response_preview, "(暂无响应记录)")
        self.update_memory_label()
        self.schedule_request_preview()

    def restore_conversation_history(self) -> None:
        self.history.config(state=tk.NORMAL)
        self.history.delete("1.0", tk.END)
        self.history.config(state=tk.DISABLED)
        for entry in self.agent.export_conversation():
            self.append_history(entry["role"], entry["text"] or "")

    def update_memory_label(self) -> None:
        status = self.agent.memory_status()
        if status != "active":
            self.memory_label.config(text="记忆条数: 0")
            self.memory_path_label.config(text=f"记忆状态: {status}")
            return
        count = self.agent.memory_count(reload=True)
        self.memory_label.config(text=f"记忆条数: {count}")
        path = self.agent.memory_path or "(未知路径)"
        self.memory_path_label.config(text=f"存储: {self._shorten_path(path)}")

    def _shorten_path(self, path: str, limit: int = 36) -> str:
        if len(path) <= limit:
            return path
        return "..." + path[-(limit - 3):]

    def _on_input_change(self, _event=None) -> None:  # noqa: ANN001 - Tkinter callback
        self.schedule_request_preview()

    def schedule_request_preview(self) -> None:
        self._cancel_preview_job()
        self._preview_job = self.root.after(400, self.refresh_request_preview)

    def _cancel_preview_job(self) -> None:
        if self._preview_job is not None:
            self.root.after_cancel(self._preview_job)
            self._preview_job = None

    def _sync_canvas_width(self, event: tk.Event) -> None:
        if event.widget is not self.root:
            return
        if getattr(self, "_main_canvas", None) is None:
            return
        self._main_canvas.itemconfig(self._canvas_window, width=self._main_canvas.winfo_width())

    def _build_menubar(self) -> None:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        model_menu = tk.Menu(menubar, tearoff=0)
        for name in MODEL_CHOICES:
            model_menu.add_command(label=name, command=lambda n=name: self._set_model(n))
        menubar.add_cascade(label="模型", menu=model_menu)

        reasoning_menu = tk.Menu(menubar, tearoff=0)
        for label, value in REASONING_CHOICES:
            reasoning_menu.add_command(label=label, command=lambda v=value: self._set_reasoning(v))
        menubar.add_cascade(label="思维链", menu=reasoning_menu)

        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_checkbutton(
            label="启用联网搜索",
            onvalue=True,
            offvalue=False,
            variable=self.web_search_var,
            command=self._toggle_web_search,
        )
        tools_menu.add_checkbutton(
            label="启用流式响应",
            onvalue=True,
            offvalue=False,
            variable=self.streaming_var,
            command=self._toggle_streaming,
        )
        menubar.add_cascade(label="工具", menu=tools_menu)

    def _refresh_history_selector(self) -> None:
        options = self.agent.list_histories()
        if hasattr(self, "history_combo"):
            self.history_combo["values"] = options
        current = self.history_var.get()
        if current not in options:
            fallback = self.agent.history_name if self.agent.history_name in options else (options[0] if options else "default")
            self.history_var.set(fallback)
        if hasattr(self, "history_combo"):
            self.history_combo.set(self.history_var.get())

    def _on_history_selected(self, _event=None) -> None:  # noqa: ANN001 - Tkinter callback
        name = self.history_var.get()
        if not name or name == self.agent.history_name:
            return
        try:
            self.agent.switch_history(name)
        except ValueError as exc:
            messagebox.showerror("切换历史失败", str(exc))
            self.history_var.set(self.agent.history_name)
            return
        self.restore_conversation_history()
        self.clear_attachments(silent=True)
        self._set_text(self.request_preview, "(输入内容以生成预览)")
        self._set_text(self.response_preview, "(暂无响应记录)")
        self.schedule_request_preview()

    def _prompt_new_history(self) -> None:
        name = simpledialog.askstring("新建历史", "请输入历史名称 (仅限字母/数字/_/-):", parent=self.root)
        if not name:
            return
        try:
            self.agent.create_history(name)
        except ValueError as exc:
            messagebox.showerror("创建历史失败", str(exc))
            return
        self.history_var.set(self.agent.history_name)
        self._refresh_history_selector()
        self.restore_conversation_history()
        self.clear_attachments(silent=True)
        self._set_text(self.request_preview, "(输入内容以生成预览)")
        self._set_text(self.response_preview, "(暂无响应记录)")
        self.schedule_request_preview()

    def _toggle_web_search(self) -> None:
        self.agent.set_web_search_enabled(self.web_search_var.get())
        self.schedule_request_preview()

    def _toggle_streaming(self) -> None:
        self.agent.set_streaming_enabled(self.streaming_var.get())
        self._reset_stream_display()
        self.schedule_request_preview()

    def _reset_stream_display(self) -> None:
        self._stream_thinking_buffer.clear()
        self._stream_output_buffer.clear()
        self._stream_rendered_text = ""
        self._stream_force_refresh = False
        placeholder = "(启用流式响应以查看实时内容)"
        if not hasattr(self, "stream_live_box"):
            return
        if not self.streaming_var.get():
            self.streaming_status_var.set("状态: 流式未启用")
            self._set_text(self.stream_live_box, placeholder)
            self._stream_rendered_text = placeholder
            return
        self.streaming_status_var.set("状态: 等待流式响应...")
        waiting = "(等待流式内容...)"
        self._set_text(self.stream_live_box, waiting)
        self._stream_rendered_text = waiting

    def _handle_stream_event(self, event: Dict[str, Any]) -> None:
        if not event or not self.streaming_var.get():
            return
        self.root.after(0, lambda e=event: self._apply_stream_event(e))

    def _apply_stream_event(self, event: Dict[str, Any]) -> None:
        if not self.streaming_var.get():
            return
        etype = event.get("type")
        if etype == "thinking":
            text = event.get("text")
            if text:
                self._stream_thinking_buffer.append(text)
        elif etype == "text":
            text = event.get("text")
            if text:
                self._stream_output_buffer.append(text)
        elif etype == "status":
            status = event.get("status") or "(未知状态)"
            self.streaming_status_var.set(f"状态: {status}")
        elif etype == "complete":
            self.streaming_status_var.set("状态: 完成")
        elif etype == "final_text":
            text = event.get("text") or ""
            self._stream_thinking_buffer.clear()
            self._stream_output_buffer = [text] if text else []
            self._stream_rendered_text = ""
            self._stream_force_refresh = True
        self._update_stream_box()

    def _update_stream_box(self) -> None:
        if not hasattr(self, "stream_live_box"):
            return
        if not self.streaming_var.get():
            placeholder = "(启用流式响应以查看实时内容)"
            self._set_text(self.stream_live_box, placeholder)
            self._stream_rendered_text = placeholder
            return
        if not (self._stream_thinking_buffer or self._stream_output_buffer):
            waiting = "(等待流式内容...)"
            self._set_text(self.stream_live_box, waiting)
            self._stream_rendered_text = waiting
            return
        sections: List[str] = []
        if self._stream_thinking_buffer:
            sections.append("【思考】\n" + "".join(self._stream_thinking_buffer).strip())
        if self._stream_output_buffer:
            sections.append("【输出】\n" + "".join(self._stream_output_buffer))
        text = "\n\n".join(sections)
        at_bottom = self._is_widget_near_bottom(self.stream_live_box)
        previous = self._stream_rendered_text
        if getattr(self, "_stream_force_refresh", False):
            self._set_text(
                self.stream_live_box,
                text,
                keep_view=not at_bottom,
                follow_tail=at_bottom,
            )
            self._stream_force_refresh = False
        elif text.startswith(previous):
            delta = text[len(previous):]
            if delta:
                self._insert_text(self.stream_live_box, delta, follow_tail=at_bottom)
        else:
            self._set_text(
                self.stream_live_box,
                text,
                keep_view=not at_bottom,
                follow_tail=at_bottom,
            )
        self._stream_rendered_text = text

    def _update_operations_preview(self, operations: List[Dict[str, Any]]) -> None:
        if not hasattr(self, "operations_box"):
            return
        if not operations:
            self._set_text(self.operations_box, "(暂无文件操作)")
            if hasattr(self, "apply_operations_button"):
                self.apply_operations_button.config(state=tk.DISABLED)
            return
        if hasattr(self, "apply_operations_button"):
            self.apply_operations_button.config(state=tk.NORMAL)
        lines: List[str] = []
        for idx, op in enumerate(operations, start=1):
            action = op.get("action") or "(未知操作)"
            rel_path = op.get("path") or "(缺少路径)"
            lines.append(f"{idx}. {action} -> {rel_path}")
            if action == "writeFile":
                content = op.get("content")
                if isinstance(content, str):
                    diff = self._build_operation_diff(rel_path, content)
                    lines.append(diff if diff else "   (无法生成 diff)")
                else:
                    lines.append("   (无内容，无法显示 diff)")
            else:
                lines.append("   (暂不支持预览该操作)")
        self._set_text(self.operations_box, "\n".join(lines))

    def _build_operation_diff(self, rel_path: str, new_content: str) -> str:
        target = (self.workspace_root / rel_path).resolve()
        try:
            original = read_text_for_diff(target)
        except Exception as exc:  # noqa: BLE001 - 仅用于预览
            return f"   (读取原文件失败: {exc})"
        diff = build_unified_diff(rel_path, original, new_content)
        if not diff:
            return "   (内容未变化)"
        return "   Diff:\n" + "\n".join(f"   {line}" for line in diff.splitlines())

    def _handle_read_operations(self, operations: List[Dict[str, Any]]) -> None:
        if self._auto_read_active:
            return
        if self._auto_read_counter >= self._auto_read_limit:
            messagebox.showwarning("自动读取", "已达到自动读取上限，请手动提供文件内容。")
            return
        lines: List[str] = ["[自动读取] 以下是请求的文件内容:"]
        attachments: List[str] = []
        has_result = False
        for op in operations:
            rel_path = (op.get("path") or "").strip()
            if not rel_path:
                lines.append("- (缺少路径)")
                continue
            try:
                target = self._resolve_workspace_path(rel_path)
            except ValueError as exc:
                lines.append(f"- {rel_path}: {exc}")
                continue
            if not target.exists():
                lines.append(f"- {rel_path}: 文件不存在")
                continue
            size = target.stat().st_size
            lines.append(f"- {rel_path}: 已附加 ({size} bytes)")
            attachments.append(target.as_posix())
            has_result = True
        summary_text = "\n".join(lines)
        if not has_result and len(lines) == 1:
            return
        self.append_history("user", summary_text)
        self._start_auto_read_request(summary_text, attachments)

    def _start_auto_read_request(self, text: str, attachments: List[str]) -> None:
        if self._auto_read_active:
            return
        self._auto_read_active = True
        self._auto_read_counter += 1
        stream_callback = None
        if self.streaming_var.get():
            stream_callback = self._handle_stream_event
            self._reset_stream_display()
            self.streaming_status_var.set("状态: 自动读取中...")
        threading.Thread(
            target=self._worker_auto_read,
            args=(text, attachments, stream_callback),
            daemon=True,
        ).start()

    def _worker_auto_read(
        self,
        text: str,
        attachments: List[str],
        stream_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        try:
            reply, ops = self.agent.send_message(
                text,
                attachments=attachments,
                stream_callback=stream_callback,
            )
            error = None
        except Exception as exc:  # noqa: BLE001
            reply = f"(自动读取失败: {exc})"
            error = exc
            ops = []
        self.root.after(0, self._on_auto_read_response, reply, error, ops)

    def _on_auto_read_response(self, reply: str, error: Exception | None, operations: List[Dict[str, Any]]) -> None:
        self._auto_read_active = False
        self._process_agent_response(
            reply,
            error,
            operations,
            suppress_spinner=True,
            update_stream=self.streaming_var.get(),
        )

    def browse_read_file(self) -> None:
        initial = str(self._workspace_root_resolved)
        path = filedialog.askopenfilename(title="选择要读取的文件", initialdir=initial)
        if not path:
            return
        try:
            rel = os.path.relpath(path, start=initial)
        except ValueError:
            messagebox.showerror("读取文件", "所选文件不在工作目录内。")
            return
        self.read_path_var.set(rel.replace("\\", "/"))
        self.read_file_content()

    def read_file_content(self) -> None:
        rel = (self.read_path_var.get() or "").strip()
        if not rel:
            messagebox.showinfo("读取文件", "请输入相对路径或使用浏览按钮选择文件。")
            return
        try:
            target = self._resolve_workspace_path(rel)
        except ValueError as exc:
            messagebox.showerror("读取文件", str(exc))
            return
        if not target.exists():
            self._set_text(self.read_output_box, f"(文件不存在: {rel})")
            return
        try:
            content = read_text_for_diff(target)
        except Exception as exc:  # noqa: BLE001 - 反馈给用户
            self._set_text(self.read_output_box, f"(读取失败: {exc})")
            return
        if content is None:
            self._set_text(self.read_output_box, "(该文件无法作为文本读取)")
            return
        limit = 20000
        display = content if len(content) <= limit else content[:limit] + "\n...(已截断)"
        header = f"路径: {rel}\n大小: {target.stat().st_size} bytes\n---\n"
        self._set_text(self.read_output_box, header + display, follow_tail=False)

    def apply_pending_operations(self) -> None:
        if not self._pending_operations:
            messagebox.showinfo("应用操作", "当前没有可应用的文件操作。")
            return
        applied: List[str] = []
        failed: List[str] = []
        remaining: List[Dict[str, Any]] = []
        for op in self._pending_operations:
            action = op.get("action")
            rel_path = op.get("path")
            if action != "writeFile" or not rel_path:
                failed.append(f"不支持的操作: {action or '未知'}")
                remaining.append(op)
                continue
            try:
                target = self._resolve_workspace_path(rel_path)
            except ValueError as exc:
                failed.append(f"{rel_path}: {exc}")
                remaining.append(op)
                continue
            content = op.get("content")
            if not isinstance(content, str):
                failed.append(f"{rel_path}: 缺少文本内容")
                remaining.append(op)
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                applied.append(rel_path)
            except Exception as exc:  # noqa: BLE001 - 展示给用户
                failed.append(f"{rel_path}: {exc}")
                remaining.append(op)
        self._pending_operations = remaining
        self._update_operations_preview(self._pending_operations)
        if applied:
            message = "\n".join(applied)
            messagebox.showinfo("应用操作成功", f"已写入 {len(applied)} 个文件:\n{message}")
        if failed:
            detail = "\n".join(failed)
            messagebox.showerror("部分操作失败", detail)

    def _resolve_workspace_path(self, rel_path: str) -> Path:
        candidate = (self.workspace_root / rel_path).resolve()
        if not str(candidate).startswith(str(self._workspace_root_resolved)):
            raise ValueError("路径超出工作目录范围")
        return candidate

    def _set_reasoning(self, effort: str) -> None:
        try:
            self.agent.set_reasoning_effort(effort)
        except ValueError as exc:
            messagebox.showerror("无效推理配置", str(exc))
            return
        self.reasoning_var.set(effort)
        self.reasoning_label.config(text=f"推理: {effort}")
        self.schedule_request_preview()

    # ------------------------------------------------------------------
    def add_attachments(self) -> None:
        paths = filedialog.askopenfilenames(title="选择附件")
        if not paths:
            return
        added = False
        for path_str in paths:
            path = Path(path_str)
            if not path.exists():
                messagebox.showwarning("文件不存在", f"{path}")
                continue
            entry = self._build_attachment_entry(path)
            if entry is None:
                continue
            self.attachments_data.append(entry)
            self.attachments_list.insert(tk.END, entry["summary"])
            added = True
        if added:
            if self.attachments_list.size() > 0 and not self.attachments_list.curselection():
                self.attachments_list.selection_set(0)
                self._show_attachment_preview(self.attachments_data[0])
            self.schedule_request_preview()

    def clear_attachments(self, silent: bool = False) -> None:
        if not silent and not self.attachments_data:
            return
        self.attachments_data.clear()
        self.attachments_list.delete(0, tk.END)
        self._show_attachment_preview(None)
        if not silent:
            self.schedule_request_preview()

    def _build_attachment_entry(self, path: Path) -> Optional[Dict[str, Any]]:
        if is_image_file(path):
            try:
                meta = preprocess_image(path)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("图片预处理失败", f"{path.name}: {exc}")
                return None
            payload = {
                "kind": "image",
                "path": path.as_posix(),
                "mime": meta.get("mime"),
                "base64": meta.get("base64"),
                "width": meta.get("width"),
                "height": meta.get("height"),
                "source_size": meta.get("source_size"),
                "output_size": meta.get("output_size"),
                "ratio": meta.get("ratio"),
                "processed_path": meta.get("output"),
                "source_path": meta.get("source"),
            }
            preview_text = (
                f"图片: {path.name}\n"
                f"尺寸: {meta.get('width')} x {meta.get('height')}\n"
                f"大小: {_human_size(meta.get('output_size'))} (原 {_human_size(meta.get('source_size'))})"
            )
            photo = self._create_photo_from_base64(meta.get("base64"))
            summary = f"{path.name} (图片)"
            return {
                "payload": payload,
                "summary": summary,
                "preview_text": preview_text,
                "preview_photo": photo,
            }

        size = path.stat().st_size
        payload = path.as_posix()
        preview_text = f"文件: {path.name}\n大小: {_human_size(size)}\n路径: {path.as_posix()}"
        summary = f"{path.name} ({_human_size(size)})"
        return {
            "payload": payload,
            "summary": summary,
            "preview_text": preview_text,
            "preview_photo": None,
        }

    def _create_photo_from_base64(self, payload: Optional[str]) -> Optional[ImageTk.PhotoImage]:
        if not payload:
            return None
        try:
            data = base64.b64decode(payload)
            with Image.open(BytesIO(data)) as img:
                img.thumbnail((240, 180))
                return ImageTk.PhotoImage(img)
        except Exception:
            return None

    def _on_attachment_select(self, _event=None) -> None:  # noqa: ANN001 - Tkinter callback
        selection = self.attachments_list.curselection()
        if not selection:
            self._show_attachment_preview(None)
            return
        index = selection[0]
        if 0 <= index < len(self.attachments_data):
            self._show_attachment_preview(self.attachments_data[index])

    def _show_attachment_preview(self, entry: Optional[Dict[str, Any]]) -> None:
        if entry and entry.get("preview_photo"):
            photo = entry["preview_photo"]
            self.attachment_image_label.config(image=photo, text="")
            self.attachment_image_label.image = photo
        else:
            self.attachment_image_label.config(image="", text="(无图片预览)")
            self.attachment_image_label.image = None
        text = entry["preview_text"] if entry else "(未选择附件)"
        self.attachment_info_label.config(text=text)

    def _attachment_payloads(self) -> List[Any]:
        return [entry["payload"] for entry in self.attachments_data]

    def _compose_history_text(self, text: str, entries: List[Dict[str, Any]]) -> str:
        text = (text or "").strip()
        tokens = self._attachment_tokens(entries)
        if tokens:
            attachments_line = "附件: " + " ".join(tokens)
            if text:
                return f"{text}\n{attachments_line}"
            return attachments_line
        return text or "(无文本内容)"

    def _attachment_tokens(self, entries: List[Dict[str, Any]]) -> List[str]:
        tokens: List[str] = []
        for entry in entries:
            payload = entry.get("payload")
            name: Optional[str] = None
            if isinstance(payload, dict):
                path_value = (
                    payload.get("path")
                    or payload.get("processed_path")
                    or payload.get("source_path")
                )
                if path_value:
                    name = Path(path_value).name
                else:
                    name = payload.get("mime") or payload.get("kind")
            elif isinstance(payload, str):
                name = Path(payload).name
            if not name:
                summary = entry.get("summary")
                if summary:
                    name = summary.split(" ")[0]
            if not name:
                name = "附件"
            tokens.append(f"[{name}]")
        return tokens

    def _set_model(self, model_name: str) -> None:
        model_name = (model_name or "").strip()
        if not model_name:
            return
        self.agent.model = model_name
        self.model_var.set(model_name)
        self.model_label.config(text=f"模型: {model_name}")
        self.schedule_request_preview()

    # ------------------------------------------------------------------
    def start_spinner(self) -> None:
        self._spinner_running = True
        self.spinner_label.config(fg="#f6d365")
        self._animate_spinner()

    def stop_spinner(self) -> None:
        self._spinner_running = False
        if self._spinner_job is not None:
            self.root.after_cancel(self._spinner_job)
            self._spinner_job = None
        self.spinner_label.config(text="就绪", fg="#888")

    def _animate_spinner(self) -> None:
        if not self._spinner_running:
            return
        frame = self._spinner_frames[self._spinner_index % len(self._spinner_frames)]
        self.spinner_label.config(text=f"等待回复 {frame}")
        self._spinner_index += 1
        self._spinner_job = self.root.after(120, self._animate_spinner)


if __name__ == "__main__":
    ChatWindow().run()