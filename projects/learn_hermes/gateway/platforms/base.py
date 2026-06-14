"""
gateway/platforms/base.py — 平台适配器抽象基类

对照阅读：hermes-agent/gateway/platforms/base.py（BasePlatformAdapter）

设计思路：
  每种消息平台（Telegram、Discord 等）都实现这个接口。
  GatewayRunner 通过这个接口与平台通信，不关心底层协议细节。

  核心抽象（3 个方法）：
    connect()    — 启动平台连接（轮询 / webhook）
    disconnect() — 关闭连接
    send()       — 向指定 chat 发送文本回复

  对照原版的简化：
    - 去掉了 media send（图片/语音/视频）
    - 去掉了 send_typing（正在输入…）
    - 去掉了 streaming message edit（流式打字机效果）
    - 保留了核心的 connect/disconnect/send + MessageEvent 数据类

MessageEvent 数据类：
  封装一条入站消息，平台适配器填充后交给 GatewayRunner 处理。
  对照原版：gateway/platforms/base.py MessageEvent dataclass
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable


# ── 数据类 ────────────────────────────────────────────────────────────────────

@dataclass
class MessageEvent:
    """
    一条入站消息的统一封装。

    对照原版：MessageEvent 含 text, source, message_id, media_urls 等字段。
    这里只保留里程碑 6 需要的字段。
    """
    chat_id: str           # 消息来源：群组/私聊的唯一 ID（Telegram chat.id）
    user_id: str           # 发送者 ID（用于区分群内不同用户）
    username: str          # 发送者用户名（展示用）
    text: str              # 消息正文
    message_id: Optional[str] = None  # 平台消息 ID（用于回复/编辑）


# ── 回调类型 ──────────────────────────────────────────────────────────────────

# GatewayRunner 传给适配器的消息处理器：接收 MessageEvent，返回 None
MessageHandler = Callable[[MessageEvent], Awaitable[None]]


# ── 抽象基类 ──────────────────────────────────────────────────────────────────

class BasePlatformAdapter(ABC):
    """
    所有平台适配器的抽象基类。

    对照原版：hermes-agent/gateway/platforms/base.py BasePlatformAdapter
    原版有 3000+ 行（含流式发送、媒体处理、per-session 任务跟踪等）。
    这里保留最核心的 3 个抽象方法 + on_message 注册机制。

    使用方式：
      adapter = TelegramAdapter(token=...)
      adapter.on_message(handler)   # 注册消息处理回调
      await adapter.connect()       # 开始接收消息
      ...
      await adapter.disconnect()    # 优雅关闭
    """

    def __init__(self):
        # GatewayRunner 注册的消息处理回调
        self._message_handler: Optional[MessageHandler] = None

    def on_message(self, handler: MessageHandler) -> None:
        """注册入站消息处理器（GatewayRunner 调用）。"""
        self._message_handler = handler

    async def _dispatch(self, event: MessageEvent) -> None:
        """
        内部方法：将 MessageEvent 分发给已注册的处理器。
        平台适配器在收到消息后调用此方法。
        """
        if self._message_handler:
            await self._message_handler(event)

    @abstractmethod
    async def connect(self) -> None:
        """
        启动平台连接（阻塞直到 stop() 或出错）。
        对照原版：connect() 启动长轮询或 webhook 服务器。
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """
        优雅关闭连接（等待进行中的消息处理完成）。
        对照原版：disconnect() 停止 Application + updater。
        """
        ...

    @abstractmethod
    async def send(self, chat_id: str, text: str) -> None:
        """
        向指定 chat 发送文本消息。
        对照原版：send() 调用 bot.send_message()，含重试逻辑。
        """
        ...
