"""
test_gateway.py — Gateway 层本地测试（不需要真实 Telegram Token）

使用 MockAdapter 替代 TelegramAdapter，验证 GatewayRunner 的核心逻辑：
  - per-chat 会话隔离（不同 chat_id 互不干扰）
  - /new 命令重置会话
  - 正常消息 → Agent 处理 → 回复

运行方式：
  OPENAI_API_KEY=sk-xxx python test_gateway.py
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Awaitable

# 把 learn_hermes 目录加到 path
sys.path.insert(0, str(Path(__file__).parent))

from gateway.platforms.base import BasePlatformAdapter, MessageEvent
from gateway.run import GatewayRunner


# ── Mock Adapter ────────────────────────────────────────────────────────────

class MockAdapter(BasePlatformAdapter):
    """
    模拟平台适配器：不连接真实平台，由测试代码直接注入 MessageEvent。
    """

    def __init__(self):
        super().__init__()
        self.sent: List[dict] = []          # 记录所有发出的消息
        self._stop = asyncio.Event()
        self._message_queue: asyncio.Queue = asyncio.Queue()

    async def connect(self) -> None:
        """阻塞直到 disconnect() 被调用，期间从 queue 处理消息。"""
        while not self._stop.is_set():
            try:
                event = await asyncio.wait_for(self._message_queue.get(), timeout=0.1)
                await self._dispatch(event)
            except asyncio.TimeoutError:
                continue

    async def disconnect(self) -> None:
        self._stop.set()

    async def send(self, chat_id: str, text: str) -> None:
        self.sent.append({"chat_id": chat_id, "text": text})
        print(f"  [Bot → {chat_id}] {text[:80]}")

    async def inject(self, chat_id: str, text: str, username: str = "test_user") -> None:
        """向 queue 注入一条消息（模拟用户发消息）。"""
        event = MessageEvent(
            chat_id=chat_id,
            user_id="u_" + chat_id,
            username=username,
            text=text,
            message_id="msg_1",
        )
        await self._message_queue.put(event)


# ── 测试 ────────────────────────────────────────────────────────────────────

async def run_tests():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("错误：请设置 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    # 用临时目录，不污染真实 db / 记忆
    tmp = Path(tempfile.mkdtemp())
    print(f"临时目录：{tmp}\n")

    adapter = MockAdapter()

    # 通过 monkey-patch 让 GatewayRunner 用临时目录
    from state import SessionDB
    from agent.memory import MemoryStore

    runner = GatewayRunner(
        api_key=api_key,
        model="gpt-4o-mini",
        adapter=adapter,
        enabled_toolsets=["math_only"],  # 只用 math，减少噪音
    )
    # 替换为临时路径的 db 和 memory
    runner.db = SessionDB(db_path=tmp / "state.db")
    runner.memory_store = MemoryStore(memory_dir=tmp / "mem")

    # 启动 runner（后台运行）
    runner_task = asyncio.create_task(runner.run())
    await asyncio.sleep(0.2)  # 等 runner 启动

    # ── 测试 1：基础对话 ──────────────────────────────────────────────────
    print("=" * 50)
    print("测试 1：chat_id=chat_A 发送普通消息")
    adapter.sent.clear()
    await adapter.inject("chat_A", "你好，请用一句话介绍自己")
    await asyncio.sleep(8)  # 等 LLM 响应
    assert any(m["chat_id"] == "chat_A" for m in adapter.sent), "chat_A 应该收到回复"
    print(f"  PASS - 收到 {len(adapter.sent)} 条回复")

    # ── 测试 2：工具调用 ──────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("测试 2：chat_id=chat_A 请求计算")
    adapter.sent.clear()
    await adapter.inject("chat_A", "请计算 sqrt(256) + 2**8")
    await asyncio.sleep(10)
    assert any("chat_A" == m["chat_id"] for m in adapter.sent), "应收到计算结果"
    print(f"  PASS - 收到回复: {adapter.sent[-1]['text'][:60]}")

    # ── 测试 3：per-chat 隔离（不同 chat_id 互不干扰）──────────────────
    print("\n" + "=" * 50)
    print("测试 3：chat_B 发消息，不影响 chat_A 的历史")
    adapter.sent.clear()
    await adapter.inject("chat_B", "你好")
    await asyncio.sleep(8)
    assert any(m["chat_id"] == "chat_B" for m in adapter.sent), "chat_B 应该收到回复"
    assert "chat_A" not in runner._agents or runner._agents.get("chat_B") is not runner._agents.get("chat_A"), \
        "chat_A 和 chat_B 应该是不同的 agent"
    print(f"  PASS - chat_A 和 chat_B 各自独立")

    # ── 测试 4：/new 重置会话 ─────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("测试 4：/new 命令重置 chat_A 的会话")
    old_sid = runner._session_ids.get("chat_A")
    adapter.sent.clear()
    await adapter.inject("chat_A", "/new")
    await asyncio.sleep(2)
    assert any("新会话" in m["text"] for m in adapter.sent), "/new 应回复'已开启新会话'"
    # chat_A 的 agent 应该被清除（等下次消息时重建）
    assert "chat_A" not in runner._agents, "chat_A agent 应已被清除"
    print(f"  PASS - /new 成功，旧会话 [{old_sid[:8] if old_sid else '?'}] 已关闭")

    # ── 清理 ──────────────────────────────────────────────────────────────
    await adapter.disconnect()
    await asyncio.wait_for(runner_task, timeout=3)

    print("\n" + "=" * 50)
    print("所有测试通过！")


if __name__ == "__main__":
    asyncio.run(run_tests())
