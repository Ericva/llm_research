"""
model_tools.py — 工具编排层

对照阅读：hermes-agent/model_tools.py

设计思路：
  - 这一层是 AIAgent 和 ToolRegistry 之间的桥梁
  - discover_tools() 自动扫描 tools/ 目录，导入有 registry.register() 调用的模块
    （对应原版的 discover_builtin_tools()，使用 ast 静态分析判断）
  - get_tool_definitions(enabled_toolsets) 通过 toolsets.resolve_multiple_toolsets()
    将工具集名称展开为工具名集合，再调用 registry.get_definitions() 过滤 schema
  - handle_function_call() 接收 LLM 返回的工具调用，分发到 registry.dispatch()

里程碑 2 新增：
  - enabled_toolsets 参数：控制哪些工具集对 LLM 可见
  - discover_tools()：自动发现 tools/ 目录中的工具模块（不再手动 import）
  - 默认工具集从 toolsets.TOOLSETS["default"] 读取
"""

import ast
import importlib
from pathlib import Path
from typing import List, Optional

from tools import registry
from toolsets import resolve_multiple_toolsets, TOOLSETS


def discover_tools() -> None:
    """
    自动扫描 tools/ 目录，导入含有 registry.register() 调用的模块。

    对照原版：hermes-agent/tools/registry.py 的 discover_builtin_tools()

    工作原理：
      1. glob tools/*.py，排除 __init__.py 和 registry.py
      2. 用 ast.parse() 静态分析，检查模块顶层是否有 registry.register(...) 调用
      3. 只 import 有注册调用的模块（避免 import 无关辅助模块的副作用）
    """
    tools_path = Path(__file__).resolve().parent / "tools"

    for module_path in sorted(tools_path.glob("*.py")):
        if module_path.name in {"__init__.py", "registry.py"}:
            continue

        # 静态分析：判断模块顶层是否有 registry.register(...) 调用
        try:
            source = module_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(module_path))
        except (OSError, SyntaxError):
            continue

        has_register = any(
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and stmt.value.func.attr == "register"
            and isinstance(stmt.value.func.value, ast.Name)
            and stmt.value.func.value.id == "registry"
            for stmt in tree.body
        )
        if not has_register:
            continue

        mod_name = f"tools.{module_path.stem}"
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            print(f"[警告] 无法导入工具模块 {mod_name}: {e}")


# 在模块加载时自动发现并注册所有工具
discover_tools()


def get_tool_definitions(enabled_toolsets: Optional[List[str]] = None) -> List[dict]:
    """
    返回供 LLM 使用的工具 schema 列表（OpenAI tools 格式）。

    对照原版：hermes-agent/model_tools.py 的 get_tool_definitions()

    enabled_toolsets 为 None 时使用 "default" 工具集。
    通过 resolve_multiple_toolsets() 将工具集名展开为工具名集合，
    再调用 registry.get_definitions() 按名称集合过滤。
    """
    if enabled_toolsets is None:
        enabled_toolsets = ["default"]

    tool_names = set(resolve_multiple_toolsets(enabled_toolsets))
    return registry.get_definitions(tool_names)


def handle_function_call(name: str, args: dict, **kwargs) -> str:
    """
    将 LLM 的工具调用分发到对应的 handler，返回结果字符串。
    对应原版的 handle_function_call()。
    """
    return registry.dispatch(name, args, **kwargs)
