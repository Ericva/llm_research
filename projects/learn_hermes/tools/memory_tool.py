"""
tools/memory_tool.py — memory 工具注册

对照阅读：hermes-agent/tools/memory_tool.py（registry.register 部分）

设计思路：
  - handler 通过 **kwargs 接收 store（MemoryStore 实例）
  - store 由 AIAgent 在 dispatch 时注入（见 agent/core.py 的 handle_function_call 调用）
  - 若 store=None（memory 工具集未启用），返回错误而非崩溃

  对照原版的关键细节：
    handler=lambda args, **kw: memory_tool(..., store=kw.get("store"))
    这个 **kw 传递机制让工具不依赖全局 store，每个 AIAgent 实例有独立的 MemoryStore，
    多 agent 并发时不会互相污染。
"""

import json
from typing import Optional
from tools import registry


def _memory_handler(args: dict, **kwargs) -> str:
    """
    memory 工具的统一入口，分发到 MemoryStore 的对应方法。

    store 通过 kwargs 注入，而非全局变量。
    这与原版 memory_tool() 函数的 store=kw.get("store") 一致。
    """
    store = kwargs.get("store")
    if store is None:
        return json.dumps({
            "success": False,
            "error": "Memory store not available. Pass enabled_toolsets=['memory'] and ensure MemoryStore is loaded."
        })

    action = args.get("action", "")
    target = args.get("target", "memory")
    content = args.get("content")
    old_text = args.get("old_text")

    if target not in ("memory", "user"):
        return json.dumps({"success": False, "error": f"Invalid target '{target}'. Use 'memory' or 'user'."})

    if action == "add":
        if not content:
            return json.dumps({"success": False, "error": "content is required for 'add'."})
        result = store.add(target, content)

    elif action == "replace":
        if not old_text:
            return json.dumps({"success": False, "error": "old_text is required for 'replace'."})
        if not content:
            return json.dumps({"success": False, "error": "content is required for 'replace'."})
        result = store.replace(target, old_text, content)

    elif action == "remove":
        if not old_text:
            return json.dumps({"success": False, "error": "old_text is required for 'remove'."})
        result = store.remove(target, old_text)

    elif action == "read":
        result = store.read(target)

    else:
        return json.dumps({"success": False, "error": f"Unknown action '{action}'. Use: add, replace, remove, read"})

    return json.dumps(result, ensure_ascii=False)


registry.register(
    name="memory",
    toolset="memory",
    schema={
        "name": "memory",
        "description": (
            "将信息持久化保存到记忆文件，跨会话生效。记忆会注入到下次会话的 system prompt。\n\n"
            "何时主动保存（不要等用户要求）：\n"
            "- 用户纠正你，或说「记住这个」/「下次别这样」\n"
            "- 用户透露了偏好、习惯或个人信息（姓名、职业、时区、编码风格）\n"
            "- 你发现了环境信息（操作系统、已安装工具、项目结构）\n"
            "- 你学到了用户特定的惯例、API 特性或工作流\n\n"
            "两个存储目标：\n"
            "- 'memory'：你的个人笔记（环境、项目惯例、工具特性、教训）\n"
            "- 'user'：关于用户的信息（姓名、偏好、沟通风格、讨厌的事）\n\n"
            "操作：add（新增）、replace（用 old_text 定位后替换）、remove（用 old_text 定位后删除）、read（查看当前内容）"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "replace", "remove", "read"],
                    "description": "要执行的操作",
                },
                "target": {
                    "type": "string",
                    "enum": ["memory", "user"],
                    "description": "存储目标：'memory' 是 Agent 的笔记，'user' 是用户画像",
                },
                "content": {
                    "type": "string",
                    "description": "要保存的内容，add 和 replace 操作必填",
                },
                "old_text": {
                    "type": "string",
                    "description": "用于定位条目的唯一子串，replace 和 remove 操作必填",
                },
            },
            "required": ["action", "target"],
        },
    },
    handler=_memory_handler,
    emoji="🧠",
)
