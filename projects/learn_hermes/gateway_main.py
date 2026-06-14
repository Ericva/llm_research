"""
gateway_main.py — 里程碑 6 入口：Telegram Gateway

使用方式：
  OPENAI_API_KEY=sk-xxx TELEGRAM_BOT_TOKEN=xxx python gateway_main.py
  OPENAI_API_KEY=sk-xxx TELEGRAM_BOT_TOKEN=xxx OPENAI_BASE_URL=https://... python gateway_main.py
  OPENAI_API_KEY=sk-xxx TELEGRAM_BOT_TOKEN=xxx TOOLSETS=math_only python gateway_main.py

环境变量：
  OPENAI_API_KEY       必填：OpenAI API 密钥
  TELEGRAM_BOT_TOKEN   必填：从 @BotFather 获取的 Bot Token
  OPENAI_MODEL         可选：模型名称（默认 gpt-4o-mini）
  OPENAI_BASE_URL      可选：API Base URL（用于 OpenRouter 等）
  TOOLSETS             可选：逗号分隔的工具集名称

对照原版：hermes-agent/gateway/run.py main()
"""

import asyncio
import logging
import os
import sys

from gateway.run import GatewayRunner
from gateway.platforms.telegram import TelegramAdapter


def setup_logging() -> None:
    """配置日志：INFO 级别，简洁格式。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # 降低 httpx / telegram 库的日志噪音
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


async def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("错误：请设置 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        print("错误：请设置 TELEGRAM_BOT_TOKEN 环境变量")
        sys.exit(1)

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("OPENAI_BASE_URL")
    toolsets_env = os.environ.get("TOOLSETS", "").strip()
    enabled_toolsets = [t.strip() for t in toolsets_env.split(",") if t.strip()] or None

    logger.info("启动 Hermes Gateway (里程碑 6)")
    logger.info("模型：%s  工具集：%s", model, enabled_toolsets or ["default"])

    adapter = TelegramAdapter(token=bot_token)
    runner = GatewayRunner(
        api_key=api_key,
        model=model,
        adapter=adapter,
        base_url=base_url or None,
        enabled_toolsets=enabled_toolsets,
    )

    print(f"Hermes Gateway 已启动 — 模型：{model}")
    print("发消息给你的 Telegram Bot 开始对话，Ctrl+C 退出")

    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
