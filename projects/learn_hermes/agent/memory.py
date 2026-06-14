"""
agent/memory.py — 持久化记忆存储

对照阅读：hermes-agent/tools/memory_tool.py（MemoryStore 类）

设计思路：
  记忆系统的核心是"冻结快照"模式（frozen snapshot pattern）：
  1. 会话启动时从磁盘加载记忆文件，捕获 system prompt 快照（_snapshot）
  2. 整个会话期间 system prompt 不变（保持 LLM prefix cache 稳定）
  3. 工具调用写入磁盘立即生效，但不修改本轮 system prompt
  4. 下次会话启动时，快照刷新，包含上次写入的内容

两个记忆文件（对照原版）：
  - MEMORY.md：Agent 的个人笔记（环境事实、项目惯例、工具特性、学到的教训）
  - USER.md：关于用户的信息（偏好、沟通风格、期望、工作习惯）

条目分隔符：§（section sign），与原版保持一致。
每个条目是一段纯文本，可多行。

字符限制（非 token）：
  - MEMORY.md：2200 字符
  - USER.md：1375 字符
  字符数比 token 数保守，与模型无关。

操作（对照原版 action 参数）：
  - add：追加新条目
  - replace：用 old_text 子串定位条目，替换为 new_content
  - remove：用 old_text 子串定位条目，删除

存储路径：~/.learn-hermes/memories/MEMORY.md 和 USER.md
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# 默认记忆目录
DEFAULT_MEMORY_DIR = Path.home() / ".learn-hermes" / "memories"

# 条目分隔符（与原版一致）
ENTRY_DELIMITER = "\n§\n"

# 字符上限（与原版一致）
MEMORY_CHAR_LIMIT = 2200
USER_CHAR_LIMIT = 1375


class MemoryStore:
    """
    持久化记忆管理，对应原版 MemoryStore。

    两个并行状态：
      - _snapshot：会话启动时冻结的快照，用于 system prompt 注入，整个会话不变
      - memory_entries / user_entries：实时状态，工具写入后立即更新并持久化到磁盘

    这样设计的原因（对照原版注释）：
      "keeps prefix cache stable" —— 如果 system prompt 在会话中改变，
      LLM 无法复用 KV cache，每轮都要重算，增加延迟和费用。
    """

    def __init__(self, memory_dir: Optional[Path] = None):
        self.memory_dir = memory_dir or DEFAULT_MEMORY_DIR
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        # 冻结快照：会话启动时从磁盘读取后固定，不随 mid-session 写入改变
        self._snapshot: Dict[str, str] = {"memory": "", "user": ""}

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        从磁盘加载记忆文件，捕获 system prompt 快照。
        在 AIAgent.__init__ 中调用一次。
        """
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_entries = self._read_file(self.memory_dir / "MEMORY.md")
        self.user_entries = self._read_file(self.memory_dir / "USER.md")

        # 去重（保持顺序，保留第一次出现）
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))

        # 冻结快照
        self._snapshot = {
            "memory": self._render_block("memory", self.memory_entries),
            "user": self._render_block("user", self.user_entries),
        }

    # ── System prompt 注入 ────────────────────────────────────────────────────

    def build_system_prompt_suffix(self) -> str:
        """
        返回注入 system prompt 的记忆块（冻结快照）。
        会话期间调用多次也安全，始终返回同一快照。
        若两个文件均为空，返回空字符串。
        """
        parts = []
        memory_block = self._snapshot.get("memory", "")
        user_block = self._snapshot.get("user", "")
        if memory_block:
            parts.append(memory_block)
        if user_block:
            parts.append(user_block)
        return "\n\n" + "\n\n".join(parts) if parts else ""

    # ── 工具操作（由 memory_tool.py 调用） ────────────────────────────────────

    def add(self, target: str, content: str) -> Dict[str, Any]:
        """追加新条目。若超出字符上限则拒绝。"""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        entries = self._entries_for(target)
        limit = self._char_limit(target)

        # 拒绝完全重复的条目
        if content in entries:
            return self._success_resp(target, "Entry already exists (no duplicate added).")

        new_entries = entries + [content]
        new_total = len(ENTRY_DELIMITER.join(new_entries))
        if new_total > limit:
            current = self._char_count(target)
            return {
                "success": False,
                "error": (
                    f"Memory at {current:,}/{limit:,} chars. "
                    f"Adding this entry ({len(content)} chars) would exceed the limit. "
                    f"Replace or remove existing entries first."
                ),
                "current_entries": entries,
            }

        entries.append(content)
        self._set_entries(target, entries)
        self._write_to_disk(target)
        return self._success_resp(target, "Entry added.")

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """用 old_text 子串定位条目并替换。"""
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete."}

        entries = self._entries_for(target)
        matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

        if not matches:
            return {"success": False, "error": f"No entry matched '{old_text}'."}
        if len(matches) > 1:
            unique = {e for _, e in matches}
            if len(unique) > 1:
                previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                return {"success": False, "error": f"Multiple entries matched '{old_text}'. Be more specific.", "matches": previews}

        idx = matches[0][0]
        test = entries.copy()
        test[idx] = new_content
        if len(ENTRY_DELIMITER.join(test)) > self._char_limit(target):
            return {"success": False, "error": "Replacement would exceed character limit."}

        entries[idx] = new_content
        self._set_entries(target, entries)
        self._write_to_disk(target)
        return self._success_resp(target, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """用 old_text 子串定位条目并删除。"""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        entries = self._entries_for(target)
        matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

        if not matches:
            return {"success": False, "error": f"No entry matched '{old_text}'."}
        if len(matches) > 1:
            unique = {e for _, e in matches}
            if len(unique) > 1:
                previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                return {"success": False, "error": f"Multiple entries matched '{old_text}'. Be more specific.", "matches": previews}

        entries.pop(matches[0][0])
        self._set_entries(target, entries)
        self._write_to_disk(target)
        return self._success_resp(target, "Entry removed.")

    def read(self, target: str) -> Dict[str, Any]:
        """返回当前实时记忆内容（非快照，含 mid-session 写入）。"""
        return self._success_resp(target)

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _entries_for(self, target: str) -> List[str]:
        return self.user_entries if target == "user" else self.memory_entries

    def _set_entries(self, target: str, entries: List[str]) -> None:
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_limit(self, target: str) -> int:
        return USER_CHAR_LIMIT if target == "user" else MEMORY_CHAR_LIMIT

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        return len(ENTRY_DELIMITER.join(entries)) if entries else 0

    def _path_for(self, target: str) -> Path:
        return self.memory_dir / ("USER.md" if target == "user" else "MEMORY.md")

    def _write_to_disk(self, target: str) -> None:
        """将条目写回磁盘（覆盖写）。"""
        path = self._path_for(target)
        entries = self._entries_for(target)
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        path.write_text(content, encoding="utf-8")

    def _render_block(self, target: str, entries: List[str]) -> str:
        """生成注入 system prompt 的记忆块（带标题和用量指示）。"""
        if not entries:
            return ""
        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int(current / limit * 100)) if limit > 0 else 0

        if target == "user":
            header = f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"

        sep = "═" * 46
        return f"{sep}\n{header}\n{sep}\n{content}"

    def _success_resp(self, target: str, message: str = "") -> Dict[str, Any]:
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int(current / limit * 100)) if limit > 0 else 0
        resp: Dict[str, Any] = {
            "success": True,
            "target": target,
            "entries": entries,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        return resp

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """读取记忆文件，按 ENTRY_DELIMITER 分割。"""
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return []
        if not raw.strip():
            return []
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]
