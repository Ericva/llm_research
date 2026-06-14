"""
gateway/run.py — GatewayRunner：异步消息网关核心

对照阅读：hermes-agent/gateway/run.py（GatewayRunner 类）

设计思路：
  GatewayRunner 是 Gateway 层的核心调度器：
  1. 持有一个（或多个）平台适配器（Telegram / Discord / …）
  2. 从适配器接收 MessageEvent
  3. 查找或创建对应的 AIAgent（per-chat 会话隔离）
  4. 调用 agent.run_conversation()，得到回复
  5. 通过适配器将回复发回给用户

  Per-session Agent 缓存（对照原版 LRU cache + TTL）：
    - 每个 chat_id 对应一个 AIAgent 实例（保持会话历史）
    - 简化版：用 dict 存储，不做 LRU 淘汰
    - 对照原版：max_agents=128，idle_ttl=3600s

  并发处理（对照原版 per-session 任务隔离）：
    - 每个 chat_id 同时只能有一个消息在处理（asyncio.Lock per chat）
    - 如果正在处理，新消息等待（不丢弃）
    - 对照原版：_session_tasks dict + asyncio.Event interrupt

  会话持久化：
    - GatewayRunner 持有 SessionDB + MemoryStore
    - 每个 chat_id 对应一个 DB session_id
    - chat 历史从 db.load_messages() 恢复

对照原版的简化：
  - 去掉了多平台同时运行（这里只支持一个 adapter）
  - 去掉了 skill/command 路由
  - 去掉了 streaming 流式回复（等 agent 完成后一次性发送）
  - 保留了核心的 per-session agent cache + async 消息处理
"""

import asyncio
import logging
import os
import signal
from typing import Dict, Optional

from agent import AIAgent
from agent.memory import MemoryStore
from state import SessionDB
from gateway.platforms.base import BasePlatformAdapter, MessageEvent

logger = logging.getLogger(__name__)


class GatewayRunner:
    """
    异步消息网关核心调度器。

    对照原版：hermes-agent/gateway/run.py GatewayRunner

    职责：
      - 注册平台适配器，监听入站消息
      - per-chat 会话隔离（每个 chat_id 独立 agent + history）
      - 调用 AIAgent 处理消息，将结果发回平台
      - 优雅关闭（SIGINT/SIGTERM）

    使用方式：
      runner = GatewayRunner(api_key=..., model=..., adapter=TelegramAdapter(token))
      await runner.run()  # 阻塞直到 SIGINT
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        adapter: BasePlatformAdapter,
        base_url: Optional[str] = None,
        enabled_toolsets: Optional[list] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.adapter = adapter
        self.base_url = base_url
        self.enabled_toolsets = enabled_toolsets

        # 持久化层（所有 chat 共享同一 DB + MemoryStore）
        self.db = SessionDB()
        self.memory_store = MemoryStore()

        # per-chat 状态
        # _agents[chat_id] = AIAgent（持有 session_id + history 引用）
        self._agents: Dict[str, AIAgent] = {}
        # _histories[chat_id] = List[dict]（本地 history 缓存）
        self._histories: Dict[str, list] = {}
        # _session_ids[chat_id] = str（DB session_id）
        self._session_ids: Dict[str, str] = {}
        # _locks[chat_id] = asyncio.Lock（per-chat 串行化）
        self._locks: Dict[str, asyncio.Lock] = {}

        # 运行状态
        self._running = False
        self._stop_event = asyncio.Event()

    # ── 会话管理 ─────────────────────────────────────────────────────────────

    def _get_or_create_agent(self, chat_id: str) -> AIAgent:
        """
        获取或创建 chat_id 对应的 AIAgent。

        对照原版：GatewayRunner._get_or_create_agent()
        原版维护 LRU cache + idle TTL 淘汰。
        这里简化为普通 dict，不做淘汰。

        关键细节：
          每次创建新 Agent 都调用 memory_store.load()，刷新记忆快照。
          这与 main.py 的 new_session() 函数逻辑一致。
        """
        if chat_id not in self._agents:
            sid = self.db.create_session(model=self.model, source="telegram")
            agent = AIAgent(
                api_key=self.api_key,
                model=self.model,
                base_url=self.base_url or None,
                enabled_toolsets=self.enabled_toolsets,
                session_id=sid,
                db=self.db,
                memory_store=self.memory_store,
            )
            self._agents[chat_id] = agent
            self._session_ids[chat_id] = sid
            self._histories[chat_id] = []
            logger.info("chat_id=%s 创建新会话 [%s]", chat_id, sid[:8])
        return self._agents[chat_id]

    def _reset_session(self, chat_id: str) -> None:
        """
        为 chat_id 开启新会话（/new 命令触发）。

        对照原版：/reset 命令触发 session 重置 + agent 替换。
        """
        old_sid = self._session_ids.get(chat_id)
        if old_sid:
            self.db.end_session(old_sid)
            logger.info("chat_id=%s 关闭会话 [%s]", chat_id, old_sid[:8])

        # 清除缓存，下次 _get_or_create_agent 会重建
        self._agents.pop(chat_id, None)
        self._histories.pop(chat_id, None)
        self._session_ids.pop(chat_id, None)

    def _get_lock(self, chat_id: str) -> asyncio.Lock:
        """
        获取 chat_id 对应的 asyncio.Lock（per-chat 串行化）。

        对照原版：_session_tasks dict 实现同等效果（同一 session 同时只处理一条消息）。
        asyncio.Lock 更简洁：同一 chat_id 的消息顺序处理，不并发。
        """
        if chat_id not in self._locks:
            self._locks[chat_id] = asyncio.Lock()
        return self._locks[chat_id]

    # ── 消息处理 ─────────────────────────────────────────────────────────────

    async def _handle_message(self, event: MessageEvent) -> None:
        """
        处理一条入站消息（由 adapter 通过 on_message 回调触发）。

        对照原版：GatewayRunner._handle_message() —— 核心分发逻辑。

        流程：
          1. 识别特殊命令（/new 重置会话）
          2. 获取 per-chat 锁（串行化同一 chat 的消息）
          3. 查找/创建 AIAgent
          4. 调用 agent.run_conversation()
          5. 将回复发回平台
          6. 更新本地 history

        注意：async with lock 保证同一 chat_id 的消息不并发处理，
        但不同 chat_id 的消息完全并发（asyncio 并发，非多线程）。
        """
        chat_id = event.chat_id

        # ── /new 命令：重置当前会话 ───────────────────────────────────────────
        if event.text.strip() == "/new":
            self._reset_session(chat_id)
            await self.adapter.send(chat_id, "已开启新会话。")
            return

        # ── 获取 per-chat 锁（串行化） ────────────────────────────────────────
        lock = self._get_lock(chat_id)

        async with lock:
            agent = self._get_or_create_agent(chat_id)
            history = self._histories[chat_id]

            logger.info(
                "chat_id=%s user=%s 处理消息: %s",
                chat_id, event.username, event.text[:50]
            )

            try:
                # 调用 Agent（同步阻塞 → 在 asyncio 中需要 run_in_executor 或接受阻塞）
                # 对照原版：run_agent 是 async，这里 AIAgent.run_conversation 是同步的。
                # 解决方案：run_in_executor 将同步 IO-bound 调用放到线程池。
                # 对照原版设计：原版 AIAgent 是全异步的，我们的是同步的，
                # 所以需要 loop.run_in_executor 避免阻塞事件循环。
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,  # 使用默认线程池
                    agent.run_conversation,
                    event.text,
                    list(history),  # 传入 history 快照，避免并发修改
                )

                response = result["response"]
                new_messages = result["new_messages"]

                # 更新本地 history
                history.extend(new_messages)

                logger.info(
                    "chat_id=%s 回复（%d 字）: %s",
                    chat_id, len(response), response[:80]
                )

                # 发送回复
                await self.adapter.send(chat_id, response or "（无回复）")

            except Exception as e:
                logger.error("chat_id=%s 处理消息出错: %s", chat_id, e, exc_info=True)
                try:
                    await self.adapter.send(chat_id, f"处理消息时出错：{e}")
                except Exception:
                    pass

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        启动 Gateway：注册消息处理器，运行平台适配器，直到收到停止信号。

        对照原版：GatewayRunner.start() 启动多个平台 + SIGINT handler。
        这里只运行一个 adapter。

        两个并发任务：
          1. adapter.connect()  — 阻塞，处理平台消息（长轮询）
          2. _wait_for_stop()   — 监听 SIGINT/SIGTERM，触发 disconnect
        """
        # 注册消息处理回调
        self.adapter.on_message(self._handle_message)

        # 注册信号处理（优雅退出）
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._on_stop_signal)

        logger.info("GatewayRunner 启动中...")
        self._running = True

        try:
            # 并发：adapter.connect()（阻塞）+ 等待停止信号
            connect_task = asyncio.create_task(self.adapter.connect())
            stop_task = asyncio.create_task(self._stop_event.wait())

            # 等待任一完成
            done, pending = await asyncio.wait(
                [connect_task, stop_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # 如果是停止信号触发，主动断开 adapter
            if stop_task in done:
                logger.info("收到停止信号，正在关闭 adapter...")
                await self.adapter.disconnect()
                # 等待 connect_task 完成清理
                try:
                    await asyncio.wait_for(connect_task, timeout=5.0)
                except asyncio.TimeoutError:
                    connect_task.cancel()

        finally:
            self._running = False
            # 关闭所有活跃的 DB session
            for chat_id, sid in list(self._session_ids.items()):
                try:
                    self.db.end_session(sid)
                except Exception:
                    pass
            logger.info("GatewayRunner 已停止")

    def _on_stop_signal(self) -> None:
        """收到 SIGINT/SIGTERM 时触发停止。"""
        logger.info("收到停止信号")
        self._stop_event.set()
