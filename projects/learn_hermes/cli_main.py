"""
cli_main.py — 里程碑 5 入口：带 prompt_toolkit + rich + 流式输出的 CLI

使用方式：
  OPENAI_API_KEY=sk-xxx python cli_main.py
  OPENAI_API_KEY=sk-xxx OPENAI_BASE_URL=https://... python cli_main.py
  OPENAI_API_KEY=sk-xxx TOOLSETS=math_only python cli_main.py

对照 main.py（里程碑 1-4 入口）：
  - main.py：简单 input() 循环，无依赖
  - cli_main.py：prompt_toolkit + rich + 流式输出（需要额外依赖）
"""

import os
import sys
from cli import HermesCLI


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("错误：请设置 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("OPENAI_BASE_URL")
    toolsets_env = os.environ.get("TOOLSETS", "").strip()
    enabled_toolsets = [t.strip() for t in toolsets_env.split(",") if t.strip()] or None

    cli = HermesCLI(
        api_key=api_key,
        model=model,
        base_url=base_url or None,
        enabled_toolsets=enabled_toolsets,
    )
    cli.run()


if __name__ == "__main__":
    main()
