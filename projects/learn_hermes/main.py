"""
main.py — 入口：带 Session 持久化 + 记忆系统的交互式终端对话循环

使用方式：
  OPENAI_API_KEY=sk-xxx python main.py
  OPENAI_API_KEY=sk-xxx OPENAI_BASE_URL=https://... python main.py
  OPENAI_API_KEY=sk-xxx TOOLSETS=math_only python main.py

内置命令（以 / 开头）：
  /new          开启新会话（丢弃当前历史，在 db 中创建新 session）
  /sessions     列出最近 10 个会话（id 前8位 + title）
  /resume <id>  恢复指定会话（输入 id 前缀即可）
  /memory       查看当前记忆内容（MEMORY.md + USER.md）
  /exit /quit   退出

会话历史存储在 ~/.learn-hermes/state.db（SQLite）
记忆文件存储在 ~/.learn-hermes/memories/MEMORY.md 和 USER.md
"""

import os
import sys
import time
from agent import AIAgent
from agent.memory import MemoryStore
from state import SessionDB
from toolsets import get_toolset_names


def _fmt_time(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _print_sessions(sessions: list) -> None:
    if not sessions:
        print("  （无历史会话）")
        return
    for s in sessions:
        title = s["title"] or "（无标题）"
        sid_short = s["id"][:8]
        created = _fmt_time(s["created_at"])
        model = s["model"] or "?"
        print(f"  [{sid_short}]  {created}  {model:20}  {title}")


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("错误：请设置 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("OPENAI_BASE_URL")
    toolsets_env = os.environ.get("TOOLSETS", "").strip()
    enabled_toolsets = [t.strip() for t in toolsets_env.split(",") if t.strip()] or None

    db = SessionDB()
    # MemoryStore 全局共享：所有会话共享同一份记忆文件
    # 对照原版：MemoryStore 绑定到 AIAgent，每次 new_session 重新 load() 刷新快照
    memory_store = MemoryStore()

    def new_session() -> tuple:
        """创建新会话。每次都新建 AIAgent（触发 memory_store.load() 刷新快照）。"""
        sid = db.create_session(model=model)
        # 每次新会话重新构建 AIAgent，使 system prompt 包含最新记忆快照
        agent = AIAgent(
            api_key=api_key,
            model=model,
            base_url=base_url or None,
            enabled_toolsets=enabled_toolsets,
            session_id=sid,
            db=db,
            memory_store=memory_store,
        )
        return sid, [], agent

    def resume_session(prefix: str) -> tuple:
        sessions = db.list_sessions(limit=50)
        matches = [s for s in sessions if s["id"].startswith(prefix)]
        if not matches:
            print(f"  找不到以 '{prefix}' 开头的会话")
            return None, None, None
        if len(matches) > 1:
            print(f"  前缀 '{prefix}' 匹配多个会话：")
            _print_sessions(matches)
            return None, None, None
        s = matches[0]
        sid = s["id"]
        history = db.load_messages(sid)
        # 恢复会话同样需要新的 AIAgent（刷新记忆快照）
        agent = AIAgent(
            api_key=api_key,
            model=model,
            base_url=base_url or None,
            enabled_toolsets=enabled_toolsets,
            session_id=sid,
            db=db,
            memory_store=memory_store,
        )
        print(f"  已恢复会话 [{sid[:8]}]：{s['title'] or '（无标题）'}（共 {len(history)} 条消息）")
        return sid, history, agent

    active = enabled_toolsets or ["default"]
    print(f"Hermes Agent (里程碑 4) — 模型：{model}  工具集：{', '.join(active)}")
    print(f"可用工具集：{', '.join(get_toolset_names())}")
    print("内置命令：/new  /sessions  /resume <id前缀>  /memory  /exit")
    print()

    session_id, history, agent = new_session()
    print(f"新会话已创建 [{session_id[:8]}]\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            db.end_session(session_id)
            print("再见！")
            break

        if user_input.lower() == "/new":
            db.end_session(session_id)
            session_id, history, agent = new_session()
            print(f"新会话已创建 [{session_id[:8]}]\n")
            continue

        if user_input.lower() == "/sessions":
            sessions = db.list_sessions(limit=10)
            print(f"最近 {len(sessions)} 个会话：")
            _print_sessions(sessions)
            print()
            continue

        if user_input.lower().startswith("/resume "):
            prefix = user_input[8:].strip()
            new_sid, new_history, new_agent = resume_session(prefix)
            if new_sid:
                db.end_session(session_id)
                session_id, history, agent = new_sid, new_history, new_agent
            print()
            continue

        if user_input.lower() == "/memory":
            mem_result = memory_store.read("memory")
            user_result = memory_store.read("user")
            print(f"MEMORY ({mem_result['usage']}):")
            for i, e in enumerate(mem_result["entries"], 1):
                print(f"  {i}. {e[:100]}")
            print(f"USER ({user_result['usage']}):")
            for i, e in enumerate(user_result["entries"], 1):
                print(f"  {i}. {e[:100]}")
            print()
            continue

        result = agent.run_conversation(user_input, history)
        print(f"Agent: {result['response']}\n")
        history.extend(result["new_messages"])


if __name__ == "__main__":
    main()
