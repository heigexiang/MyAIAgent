"""Simple desktop chat window for the NetworkAgent.

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
            reply, _ops = self.agent.send_message(
                text,
                attachments=attachments,
                stream_callback=stream_callback,
            )
            error = None
        except Exception as exc:  # noqa: BLE001 - 展示真实错误
            reply = "(请求失败: {} )".format(exc)
            error = exc
        self.root.after(0, self._on_response, reply, error)

    def _on_response(self, reply: str, error: Exception | None) -> None:
        self.stop_spinner()
        self.send_button.config(state=tk.NORMAL)
        if error:
            messagebox.showerror("调用失败", str(error))
            if self.streaming_var.get():
                self.streaming_status_var.set("状态: 调用失败")
                if not (self._stream_output_buffer or self._stream_thinking_buffer):
                    self._set_text(self.stream_live_box, f"(错误详情) {error}")
        elif self.streaming_var.get():
            self.streaming_status_var.set("状态: 完成")
            self._update_stream_box()
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
        if text.startswith(previous):
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