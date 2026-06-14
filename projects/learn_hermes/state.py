"""
state.py — Session 持久化层（SQLite）

对照阅读：hermes-agent/hermes_state.py

设计思路：
  原版 SessionDB 非常完整（WAL、FTS5、schema migration、WAL checkpoint、
  jitter retry 等），适合生产多进程并发写。

  学习版保留核心结构和关键设计决策，去掉生产级复杂度：
  ✅ 保留：WAL mode（理解为何需要 journal_mode=WAL）
  ✅ 保留：schema 表 + messages 表（对照原版字段学习数据建模）
  ✅ 保留：session 生命周期（create → append_messages → end）
  ✅ 保留：load_messages（跨对话恢复历史）
  ✅ 保留：list_sessions（会话列表 / 切换）
  ✅ 保留：自动生成 title（取第一条 user 消息前 40 字）

  简化：
  ❌ 去掉 FTS5（学习版不做 session_search）
  ❌ 去掉 jitter retry（单进程不需要）
  ❌ 去掉 schema migration 链（从头全新 schema）
  ❌ 去掉 token 统计、cost 字段

SQLite 表结构（精简版）：
  sessions(id, title, source, model, created_at, ended_at)
  messages(id, session_id, role, content, tool_calls, tool_call_id, tool_name, ts)

关键学习点（对照原版）：
  1. WAL 模式：journal_mode=WAL 允许并发读，单写；适合多平台 gateway
  2. row_factory=sqlite3.Row：查询结果可按列名访问，代替下标
  3. isolation_level=None：自己管理事务（BEGIN/COMMIT），避免隐式提交混乱
  4. messages 的 tool_calls 列：存 JSON 字符串，还原时 json.loads()
  5. title 自动生成：第一条 user 消息截断，帮助用户识别历史会话
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any


# 默认存储目录：~/.learn-hermes/state.db
DEFAULT_DB_PATH = Path.home() / ".learn-hermes" / "state.db"

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    source      TEXT NOT NULL DEFAULT 'cli',
    model       TEXT,
    created_at  REAL NOT NULL,
    ended_at    REAL
);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id),
    role         TEXT NOT NULL,
    content      TEXT,
    tool_calls   TEXT,
    tool_call_id TEXT,
    tool_name    TEXT,
    ts           REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, ts);
"""


class SessionDB:
    """
    SQLite 会话持久化。

    使用方式：
        db = SessionDB()
        sid = db.create_session(model="gpt-4o-mini")
        db.append_messages(sid, new_messages)
        db.end_session(sid)

        # 下次启动：
        sessions = db.list_sessions(limit=10)
        history = db.load_messages(session_id)
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # isolation_level=None → 手动管理事务（BEGIN/COMMIT）
        # 对照原版：避免 Python sqlite3 隐式事务与 BEGIN IMMEDIATE 冲突
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row

        # WAL 模式：允许读写并发，减少锁争用
        # 对照原版注释："WAL mode for concurrent readers + one writer"
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._init_schema()

    def _init_schema(self) -> None:
        """建表（幂等），写入 schema_version。"""
        self._conn.executescript(_SCHEMA_SQL)

        row = self._conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            self._conn.commit()

    def close(self) -> None:
        """关闭连接前做一次 PASSIVE WAL checkpoint，对照原版的 close()。"""
        try:
            self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except Exception:
            pass
        self._conn.close()

    # ── Session 生命周期 ──────────────────────────────────────────────────────

    def create_session(
        self,
        model: Optional[str] = None,
        source: str = "cli",
        session_id: Optional[str] = None,
    ) -> str:
        """
        创建新会话，返回 session_id（UUID4 字符串）。

        对照原版：HermesState.create_session()
        session_id 可以外部传入（测试用），默认自动生成 uuid4。
        """
        sid = session_id or str(uuid.uuid4())
        now = time.time()
        self._conn.execute(
            "INSERT INTO sessions (id, source, model, created_at) VALUES (?, ?, ?, ?)",
            (sid, source, model, now),
        )
        self._conn.commit()
        return sid

    def end_session(self, session_id: str) -> None:
        """标记会话结束时间。"""
        self._conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (time.time(), session_id),
        )
        self._conn.commit()

    def set_title(self, session_id: str, title: str) -> None:
        """设置会话标题（通常取第一条 user 消息前 40 字）。"""
        self._conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?",
            (title[:80], session_id),
        )
        self._conn.commit()

    # ── 消息存取 ──────────────────────────────────────────────────────────────

    def append_messages(self, session_id: str, messages: List[dict]) -> None:
        """
        将一批消息追加到会话。
        messages 是 OpenAI Chat 格式的 dict 列表（role/content/tool_calls/…）。

        对照原版：HermesState.append_messages()

        序列化规则：
          - content：字符串或 None
          - tool_calls：list → json.dumps，存为 TEXT
          - tool_call_id：透传字符串
          - tool_name：从 tool_calls[0].function.name 提取（方便搜索）
        """
        now = time.time()
        rows = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content")
            tool_calls_raw = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

            # 序列化 tool_calls list → JSON 字符串
            tool_calls_str = None
            tool_name = None
            if tool_calls_raw:
                tool_calls_str = json.dumps(tool_calls_raw, ensure_ascii=False)
                # 提取第一个工具名（方便日后搜索/展示）
                try:
                    tool_name = tool_calls_raw[0]["function"]["name"]
                except (KeyError, IndexError, TypeError):
                    pass

            rows.append((
                session_id, role, content,
                tool_calls_str, tool_call_id, tool_name, now,
            ))

        self._conn.executemany(
            """INSERT INTO messages
               (session_id, role, content, tool_calls, tool_call_id, tool_name, ts)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

        # 自动设置 title：取本批消息中第一条 user 消息（若 session 还没 title）
        self._maybe_set_title(session_id, messages)

    def load_messages(self, session_id: str) -> List[dict]:
        """
        加载会话的完整消息历史，返回 OpenAI Chat 格式的 list。

        对照原版：HermesState.get_messages()

        反序列化规则：
          - tool_calls：JSON 字符串 → list
          - content 为 None 时保留 None（assistant tool_calls 消息 content 通常是 None）
        """
        rows = self._conn.execute(
            """SELECT role, content, tool_calls, tool_call_id
               FROM messages
               WHERE session_id = ?
               ORDER BY ts, id""",
            (session_id,),
        ).fetchall()

        messages = []
        for row in rows:
            msg: Dict[str, Any] = {"role": row["role"]}

            # content：可能是 None（assistant 发起工具调用时）
            if row["content"] is not None:
                msg["content"] = row["content"]
            else:
                msg["content"] = None

            # tool_calls：还原为 list
            if row["tool_calls"]:
                try:
                    msg["tool_calls"] = json.loads(row["tool_calls"])
                except json.JSONDecodeError:
                    pass

            # tool_call_id：仅 role=tool 时存在
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]

            messages.append(msg)

        return messages

    # ── 会话列表 ──────────────────────────────────────────────────────────────

    def list_sessions(self, limit: int = 20, source: Optional[str] = None) -> List[dict]:
        """
        返回最近的会话列表（按创建时间倒序）。
        每条包含 id, title, model, created_at, ended_at。
        """
        if source:
            rows = self._conn.execute(
                """SELECT id, title, model, source, created_at, ended_at
                   FROM sessions WHERE source = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (source, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT id, title, model, source, created_at, ended_at
                   FROM sessions
                   ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> Optional[dict]:
        """按 ID 获取单个会话元数据。"""
        row = self._conn.execute(
            "SELECT id, title, model, source, created_at, ended_at FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    def _maybe_set_title(self, session_id: str, messages: List[dict]) -> None:
        """
        若会话尚无标题，从本批消息中取第一条 user 消息自动生成标题。
        对照原版：title 由第一条 user 消息截断生成（40字）。
        """
        row = self._conn.execute(
            "SELECT title FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row and row["title"]:
            return  # 已有 title，不覆盖

        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content") or ""
                if content:
                    title = content.strip().replace("\n", " ")[:40]
                    self.set_title(session_id, title)
                    return
