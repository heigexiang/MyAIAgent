import json, sys, os, traceback

ROOT = os.getcwd()

HELP = "示例命令:\n  创建示例 -> 生成 examples/hello.txt\n  列目录 -> 列出根目录文件\n  读取 <相对路径> -> 返回文件内容\n  写入 <相对路径> <内容> -> 写入文件\n"

def op_write_file(rel_path: str, content: str):
    return {"action": "writeFile", "path": rel_path.replace('\\', '/'), "content": content}

def handle_chat(text: str):
    text = text.strip()
    operations = []
    if text.startswith("创建示例"):
        sample_path = "examples/hello.txt"
        operations.append(op_write_file(sample_path, "Hello from agent\n"))
        response = f"已生成写文件操作: {sample_path}"
    elif text.startswith("列目录"):
        try:
            entries = os.listdir(ROOT)
            response = "当前目录包含 (最多50): " + ", ".join(entries[:50])
        except Exception as e:
            response = f"列目录失败: {e}"
    elif text.startswith("读取 "):
        rel = text[3:].strip()
        target = os.path.join(ROOT, rel)
        if os.path.isfile(target):
            try:
                with open(target, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(5000)
                response = f"文件 {rel} 内容前5000字符:\n" + content
            except Exception as e:
                response = f"读取失败: {e}"
        else:
            response = f"文件不存在: {rel}"
    elif text.startswith("写入 "):
        parts = text.split(" ", 2)
        if len(parts) < 3:
            response = "用法: 写入 <路径> <内容>"
        else:
            rel_path, content = parts[1], parts[2]
            operations.append(op_write_file(rel_path, content))
            response = f"准备写入文件: {rel_path}"
    else:
        response = "未识别命令\n" + HELP
    return {"type": "response", "text": response, "operations": operations}


def main():
    # 发送 ready 信号，供扩展识别 Agent 已启动
    print(json.dumps({"type": "ready", "text": "agent started"}))
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception as e:
            print(json.dumps({"type": "response", "text": f"无法解析输入: {e}"}))
            sys.stdout.flush()
            continue
        try:
            if data.get("type") == "chat":
                msg = handle_chat(data.get("text", ""))
                print(json.dumps(msg, ensure_ascii=False))
                sys.stdout.flush()
        except Exception as e:
            err = traceback.format_exc()
            print(json.dumps({"type": "response", "text": f"处理异常: {e}\n{err}"}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
