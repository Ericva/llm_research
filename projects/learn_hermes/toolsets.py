"""
toolsets.py — 工具集定义与解析

对照阅读：hermes-agent/toolsets.py

设计思路：
  TOOLSETS 是一个静态字典，每个条目描述一组工具。
  工具集支持两种组合方式：
    - "tools"：直接列出工具名（叶子节点）
    - "includes"：引用其他工具集（组合/菱形继承，用 visited 集合防循环）

  resolve_toolset() 是核心函数：递归展开 includes，去重后返回工具名列表。
  这让「debugging 包含 web + file」只需在 TOOLSETS 声明，不需要手动复制工具名。

  与原版的简化：
    - 去掉了 MCP / 插件动态扩展
    - 去掉了 hermes-* 平台工具集（里程碑 6 引入 gateway 时再加）
    - 保留了核心的 TOOLSETS / resolve_toolset / resolve_multiple_toolsets
"""

from typing import Dict, List, Set, Optional, Any


TOOLSETS: Dict[str, Dict[str, Any]] = {
    # ── 叶子工具集（单一职责） ──────────────────────────────────────────────
    "math": {
        "description": "数学计算工具",
        "tools": ["calculate"],
        "includes": [],
    },
    "file": {
        "description": "文件读写工具（read_file / write_file）",
        "tools": ["read_file", "write_file"],
        "includes": [],
    },
    "memory": {
        "description": "持久化记忆工具（跨会话，注入 system prompt）",
        "tools": ["memory"],
        "includes": [],
    },

    # ── 组合工具集（includes 引用上面的叶子） ──────────────────────────────
    "default": {
        "description": "默认工具集：数学 + 文件操作 + 记忆",
        "tools": [],
        "includes": ["math", "file", "memory"],
    },
    "math_only": {
        "description": "仅数学计算，不含文件操作",
        "tools": [],
        "includes": ["math"],
    },
}


def resolve_toolset(name: str, visited: Optional[Set[str]] = None) -> List[str]:
    """
    递归解析工具集，返回去重后的工具名列表（已排序）。

    visited 用于检测循环引用（菱形依赖不是 bug，正常处理）：
      default → math, file
      如果将来 file 也 includes math，math 只会被收集一次。

    对照原版：hermes-agent/toolsets.py resolve_toolset()
    """
    if visited is None:
        visited = set()

    if name in visited:
        return []
    visited.add(name)

    toolset = TOOLSETS.get(name)
    if not toolset:
        return []

    tools: Set[str] = set(toolset.get("tools", []))

    for included_name in toolset.get("includes", []):
        tools.update(resolve_toolset(included_name, visited))

    return sorted(tools)


def resolve_multiple_toolsets(names: List[str]) -> List[str]:
    """合并多个工具集，去重后返回工具名列表。"""
    all_tools: Set[str] = set()
    for name in names:
        all_tools.update(resolve_toolset(name))
    return sorted(all_tools)


def get_toolset_names() -> List[str]:
    """返回所有已定义的工具集名称。"""
    return sorted(TOOLSETS.keys())
