"""
gateway/platforms/telegram.py — Telegram 平台适配器

对照阅读：hermes-agent/gateway/platforms/telegram.py（TelegramAdapter 类）

设计思路：
  使用 python-telegram-bot v20+（原生 asyncio）实现：
  - 长轮询（polling）模式：Application.run_polling()
  - 收到消息 → 封装为 MessageEvent → 调用 _dispatch()
  - 发送回复 → bot.send_message()

  python-telegram-bot v20 的核心 asyncio 模式：
    Application = 整个 Bot 生命周期（handlers、updater、job_queue）
    Updater      = 长轮询线程（内部管理 asyncio loop）
    Application.run_polling() = 阻塞，直到收到 stop_signal

  对照原版简化：
    - 去掉了 webhook 模式（只保留 polling）
    - 去掉了媒体消息处理（/start 命令 + 纯文本）
    - 去掉了消息批处理/聚合（原版对快速连续消息做合并）
    - 去掉了流式"打字机"编辑效果
    - 保留了 connect/disconnect/send 核心结构

  关于 Application 生命周期（v20 关键）：
    v20 将 Application 设计为异步上下文管理器：
      async with Application.builder().token(token).build() as app:
          app.run_polling()  # 但这里不能直接 await

    在外部 asyncio.run() 驱动时，需要手动管理：
      await app.initialize()
      await app.start()
      await app.updater.start_polling()
      ...（等待 stop signal）
      await app.updater.stop()
      await app.stop()
      await app.shutdown()
"""

import asyncio
import logging
from typing import Optional

from telegram import Update, Bot
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler as TGMessageHandler,
    ContextTypes,
    filters,
)

from gateway.platforms.base import BasePlatformAdapter, MessageEvent

logger = logging.getLogger(__name__)


class TelegramAdapter(BasePlatformAdapter):
    """
    Telegram 平台适配器（polling 模式）。

    对照原版：hermes-agent/gateway/platforms/telegram.py TelegramAdapter

    生命周期：
      __init__   → 保存 token，不建立连接
      connect()  → 构建 Application，注册 handler，启动长轮询（阻塞）
      disconnect()→ 停止轮询，关闭 Application
      send()     → bot.send_message()（异步）
    """

    def __init__(self, token: str):
        """
        Args:
            token: Telegram Bot Token（从 @BotFather 获取）
        """
        super().__init__()
        self.token = token
        self._app: Optional[Application] = None
        self._stop_event: asyncio.Event = asyncio.Event()

    # ── 核心接口实现 ──────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        启动 Telegram 长轮询。

        对照原版 connect() 流程：
          1. ApplicationBuilder().token().build() 构建 Application
          2. 注册 handlers（text + command）
          3. await app.initialize()
          4. await app.updater.start_polling()
          5. await app.start()
          6. 等待 _stop_event（直到 disconnect() 触发）
          7. 清理

        注意：不使用 app.run_polling()，因为它内部会调用 asyncio.run()，
        而我们已经在一个 asyncio 事件循环中。
        需要手动管理 initialize/start/stop/shutdown。
        """
        logger.info("TelegramAdapter: 正在连接...")

        self._app = (
            ApplicationBuilder()
            .token(self.token)
            .build()
        )

        # 注册消息处理器
        # /start 命令：欢迎消息
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        # /new 命令：开启新会话（对照原版 /reset）
        self._app.add_handler(CommandHandler("new", self._cmd_new))
        # 普通文本消息（非命令）→ 发给 Agent 处理
        self._app.add_handler(
            TGMessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text)
        )

        self._stop_event.clear()

        # 手动管理生命周期（对照原版，避免 run_polling() 内部 asyncio.run 冲突）
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,  # 忽略 Bot 离线期间积压的消息
        )

        logger.info("TelegramAdapter: 已连接，开始接收消息")

        # 阻塞直到 disconnect() 被调用
        await self._stop_event.wait()

        # 优雅停止
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        logger.info("TelegramAdapter: 已断开连接")

    async def disconnect(self) -> None:
        """触发停止信号，让 connect() 中的等待退出。"""
        logger.info("TelegramAdapter: 正在断开...")
        self._stop_event.set()

    async def send(self, chat_id: str, text: str) -> None:
        """
        发送文本消息到指定 chat。

        对照原版：send() 含重试、分段（>4096 字符自动切割）。
        这里只做基础实现：超长消息截断。

        Telegram 单条消息上限：4096 字符
        """
        if not self._app:
            logger.error("send() 调用但 App 未初始化")
            return

        MAX_LEN = 4096
        # 超长消息分段发送
        chunks = [text[i:i + MAX_LEN] for i in range(0, len(text), MAX_LEN)] if text else ["（无回复）"]

        for chunk in chunks:
            try:
                await self._app.bot.send_message(
                    chat_id=int(chat_id),
                    text=chunk,
                )
            except Exception as e:
                logger.error("send() 失败 chat_id=%s: %s", chat_id, e)

    # ── Telegram Handler 回调 ─────────────────────────────────────────────────

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        处理普通文本消息。

        对照原版：_handle_text_message() 含消息批处理（聚合快速连续消息）。
        这里简化：直接封装并分发。
        """
        if not update.message or not update.message.text:
            return

        msg = update.message
        event = MessageEvent(
            chat_id=str(msg.chat_id),
            user_id=str(msg.from_user.id) if msg.from_user else "unknown",
            username=(
                msg.from_user.username
                or msg.from_user.first_name
                or "用户"
            ) if msg.from_user else "用户",
            text=msg.text,
            message_id=str(msg.message_id),
        )

        logger.debug(
            "收到消息 chat_id=%s user=%s: %s",
            event.chat_id, event.username, event.text[:50]
        )

        # 发送"正在处理…"提示（对照原版 send_typing）
        try:
            await context.bot.send_chat_action(
                chat_id=msg.chat_id,
                action="typing",
            )
        except Exception:
            pass  # typing 提示失败不影响主流程

        # 分发给 GatewayRunner 处理
        await self._dispatch(event)

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/start 命令处理：发送欢迎消息。"""
        if not update.message:
            return
        username = (
            update.message.from_user.first_name
            if update.message.from_user
            else "用户"
        )
        await update.message.reply_text(
            f"你好，{username}！我是 Hermes Agent（学习版）。\n"
            "直接发送消息即可对话，输入 /new 开启新会话。"
        )

    async def _cmd_new(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /new 命令：通知 GatewayRunner 为该 chat 开启新会话。

        对照原版：/reset 命令触发 session 重置。
        实现方式：发一个特殊标记消息给 GatewayRunner。
        """
        if not update.message:
            return
        msg = update.message
        event = MessageEvent(
            chat_id=str(msg.chat_id),
            user_id=str(msg.from_user.id) if msg.from_user else "unknown",
            username=(msg.from_user.username or msg.from_user.first_name or "用户")
            if msg.from_user
            else "用户",
            text="/new",  # GatewayRunner 识别此标记
            message_id=str(msg.message_id),
        )
        await self._dispatch(event)
