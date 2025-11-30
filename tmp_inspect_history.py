import json
import sys
from textwrap import shorten
from pathlib import Path

path = Path(r"e:/桌面251125/MyAIAgent/agent_data/histories/SFedAvg_GoLore_01.json")
entries = json.loads(path.read_text(encoding="utf-8"))

if len(sys.argv) > 1 and sys.argv[1] == "--raw":
    idx = int(sys.argv[2]) - 1
    print(json.dumps(entries[idx], ensure_ascii=False, indent=2))
    sys.exit(0)

print(f"entries {len(entries)}")
for i, entry in enumerate(entries, 1):
    role = entry.get("role")
    content = entry.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
    else:
        text = str(content)
    print(f"--- entry {i} role={role} chars={len(text)}")
    print(shorten(text, width=200, placeholder="..."))
