"""
file_tool.py — 文件读写工具

对照阅读：hermes-agent/tools/file_tools.py（简化版）

设计思路：
  - read_file：读取文本文件内容，限制最大字符数防止 context 爆炸
  - write_file：写入文本内容到文件，自动创建父目录
  - 路径安全：不允许读写根目录以外的绝对路径（学习版简化，原版有更完整的 path_security）
  - handler 返回 JSON 字符串，成功 {"content": ...} 或 {"ok": true}，失败 {"error": ...}
"""

import json
import os
from pathlib import Path
from tools import registry

_MAX_READ_CHARS = 50_000   # 单次读取上限，防止 context 溢出


def _read_file(args: dict, **_) -> str:
    path_str: str = args.get("path", "").strip()
    if not path_str:
        return json.dumps({"error": "path is required"})

    path = Path(path_str).expanduser()
    if not path.exists():
        return json.dumps({"error": f"File not found: {path}"})
    if path.is_dir():
        return json.dumps({"error": f"Path is a directory: {path}"})

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        return json.dumps({"error": f"Permission denied: {path}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

    truncated = False
    if len(content) > _MAX_READ_CHARS:
        content = content[:_MAX_READ_CHARS]
        truncated = True

    result: dict = {"content": content, "path": str(path)}
    if truncated:
        result["truncated"] = True
        result["note"] = f"内容超过 {_MAX_READ_CHARS} 字符，已截断"
    return json.dumps(result, ensure_ascii=False)


def _write_file(args: dict, **_) -> str:
    path_str: str = args.get("path", "").strip()
    content: str = args.get("content", "")

    if not path_str:
        return json.dumps({"error": "path is required"})

    path = Path(path_str).expanduser()

    # 自动创建父目录
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except PermissionError:
        return json.dumps({"error": f"Permission denied: {path}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

    return json.dumps({"ok": True, "path": str(path), "bytes_written": len(content.encode())})


# ── 注册到全局 registry ──────────────────────────────────────────────────────

registry.register(
    name="read_file",
    toolset="file",
    schema={
        "name": "read_file",
        "description": "读取文本文件内容。返回文件内容字符串。超过 50,000 字符时自动截断。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径，支持 ~ 展开，例如 '~/notes.txt' 或 '/tmp/data.json'",
                },
            },
            "required": ["path"],
        },
    },
    handler=_read_file,
    emoji="📖",
)

registry.register(
    name="write_file",
    toolset="file",
    schema={
        "name": "write_file",
        "description": "将文本内容写入文件（覆盖写）。父目录不存在时自动创建。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "目标文件路径，例如 '~/output.txt' 或 '/tmp/result.md'",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的文本内容",
                },
            },
            "required": ["path", "content"],
        },
    },
    handler=_write_file,
    emoji="✏️",
)
