"""
ToolRegistry — 工具注册与分发的核心数据结构

对照阅读：hermes-agent/tools/registry.py

设计思路：
  - ToolEntry 存储工具元数据（schema + handler + 检查函数）
  - ToolRegistry 是一个字典包装器，提供 register / dispatch / get_schema_list 三个核心操作
  - schema 使用 OpenAI function calling 格式（type="function"）
  - dispatch 将 name + args 路由到对应 handler，返回 JSON 字符串供 LLM 消费

里程碑 2 新增：
  - get_definitions(tool_names) — 按名称集合过滤，对应原版 registry.get_definitions()
  - 供 model_tools.py 的工具集系统使用
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set
import json


@dataclass
class ToolEntry:
    name: str
    toolset: str
    schema: dict          # OpenAI function calling 格式的 JSON Schema
    handler: Callable     # (args: dict, **kwargs) -> str (JSON 字符串)
    check_fn: Optional[Callable] = None   # 运行前置检查，返回 None 表示通过，否则返回错误信息
    emoji: str = ""


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolEntry] = {}

    def register(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Optional[Callable] = None,
        emoji: str = "",
    ) -> None:
        """注册一个工具到 registry。"""
        self._tools[name] = ToolEntry(
            name=name,
            toolset=toolset,
            schema=schema,
            handler=handler,
            check_fn=check_fn,
            emoji=emoji,
        )

    def get_schema_list(self, toolset: Optional[str] = None) -> List[dict]:
        """
        返回 OpenAI tools 参数格式的 schema 列表。
        若指定 toolset，只返回该工具集的工具。
        """
        tools = []
        for entry in self._tools.values():
            if toolset is not None and entry.toolset != toolset:
                continue
            tools.append({
                "type": "function",
                "function": entry.schema,
            })
        return tools

    def dispatch(self, name: str, args: dict, **kwargs) -> str:
        """
        按名称分发工具调用。
        返回 JSON 字符串（工具执行结果），供 LLM 的 tool message 使用。
        """
        if name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {name}"})

        entry = self._tools[name]

        # 运行前置检查
        if entry.check_fn is not None:
            error = entry.check_fn(args, **kwargs)
            if error:
                return json.dumps({"error": error})

        try:
            result = entry.handler(args, **kwargs)
            # handler 可以返回 str（已是 JSON）或 dict
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_definitions(self, tool_names: Set[str]) -> List[dict]:
        """
        按名称集合过滤，返回 OpenAI tools 格式的 schema 列表。
        对应原版的 registry.get_definitions(tool_names)。

        与 get_schema_list() 的区别：
          - get_schema_list(toolset=x)：按 toolset 字段过滤（用于单工具集）
          - get_definitions(tool_names)：按具体名称集合过滤（用于 resolve_toolset 之后）
        """
        result = []
        for name in sorted(tool_names):
            entry = self._tools.get(name)
            if not entry:
                continue
            # check_fn：若存在则运行，False 则跳过该工具（工具不满足运行条件）
            if entry.check_fn is not None and not entry.check_fn():
                continue
            result.append({
                "type": "function",
                "function": {**entry.schema, "name": entry.name},
            })
        return result

    def list_tools(self) -> List[str]:
        """返回所有已注册工具的名称。"""
        return list(self._tools.keys())
