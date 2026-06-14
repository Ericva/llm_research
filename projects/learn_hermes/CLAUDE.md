# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

learn_hermes 是仿照 [hermes-agent](https://github.com/coleam00/hermes-agent) 源码构建的学习版 Mini Agent。目标是"对照阅读 + 自己实现"来理解 Agent 系统的设计。

## Quick Commands

```bash
# 所有命令需要在 learn_hermes/ 目录下运行，并先加载环境变量
export $(grep -v '^#' .env.local | xargs)

# 简单 CLI 入口（M1-M4，input() 循环）
python main.py

# Rich CLI 入口（M5，流式输出 + 彩色终端，需要真实 TTY）
python cli_main.py

# Telegram Gateway（M6，需要 TELEGRAM_BOT_TOKEN）
python gateway_main.py

# Gateway Mock 测试（不需要 Telegram Token）
python test_gateway.py

# 快速验证各层
python -c "from model_tools import handle_function_call; print(handle_function_call('calculate', {'expression': '2**10'}))"
python -c "from agent.memory import MemoryStore; s = MemoryStore(); s.load(); print(s.add('memory', 'test'))"
python -c "from state import SessionDB; db = SessionDB(); sid = db.create_session(); print(sid[:8])"
```

## Architecture

```
learn_hermes/
├── agent/
│   ├── core.py          # AIAgent 核心类（LLM 主循环 _loop）
│   ├── memory.py        # MemoryStore（冻结快照记忆）
│   └── __init__.py      # 只导出 AIAgent
├── tools/
│   ├── registry.py      # ToolRegistry 全局单例（register + dispatch）
│   ├── calculator.py    # 数学计算工具
│   ├── file_tool.py     # 文件读写工具
│   ├── memory_tool.py   # 记忆工具（通过 kwargs 注入 store）
│   └── __init__.py
├── gateway/
│   ├── run.py           # GatewayRunner（异步消息调度，per-chat 会话隔离）
│   └── platforms/
│       ├── base.py      # BasePlatformAdapter（抽象基类 + MessageEvent）
│       └── telegram.py  # TelegramAdapter（python-telegram-bot v20+ 原生 asyncio）
├── main.py              # M1-M4 入口（简单 CLI，input() 循环）
├── cli.py               # HermesCLI（prompt_toolkit + rich + 流式输出）
├── cli_main.py          # M5 入口
├── gateway_main.py      # M6 入口
├── state.py             # SessionDB（SQLite WAL mode）
├── toolsets.py          # TOOLSETS 定义 + resolve_toolset() 递归展开
├── model_tools.py       # 工具编排层：discover_tools() + get_tool_definitions() + handle_function_call()
├── test_gateway.py      # Gateway Mock 测试
└── .env.local           # 本地环境变量（OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, etc.）
```

## Key Design Decisions

### 1. 工具调用流程（agent/core.py `_loop`）
- LLM 返回 `finish_reason == "tool_calls"` → 先 append assistant 消息（含 tool_calls），再 append tool 结果消息
- `tool_call_id` 由 LLM 生成，OpenAI API 要求 tool 消息必须关联到对应的 assistant tool_calls
- 支持一次多个 tool_calls，按序执行

### 2. 冻结快照记忆（agent/memory.py）
- 会话启动时 `load()` 捕获快照 → 注入 system prompt
- 整个会话期间 system prompt 不变（保持 LLM prefix cache 稳定）
- 工具写入立即持久化到磁盘，但不修改本轮快照
- 下次会话启动时快照刷新

### 3. 循环导入问题（已修复）
- `memory_tool.py` 不能顶层 `from agent.memory import MemoryStore`，会触发循环导入
- 解决方案：通过 `kwargs.get("store")` duck-typing 获取 MemoryStore 实例

### 4. SQLite WAL mode（state.py）
- `journal_mode=WAL`：允许并发读 + 单写
- `isolation_level=None`：手动管理事务
- `row_factory=sqlite3.Row`：查询结果按列名访问

### 5. Telegram Adapter 生命周期（gateway/platforms/telegram.py）
- 不能用 `run_polling()`（内部调用 asyncio.run()，与外层事件循环冲突）
- 手动管理：`initialize() → start() → updater.start_polling() → wait stop_event → stop() → shutdown()`

## 数据存储路径

- Session DB：`~/.learn-hermes/state.db`（SQLite）
- 记忆文件：`~/.learn-hermes/memories/MEMORY.md` 和 `USER.md`

## Milestones Status

| M1 | 基础 Agent（AIAgent + LLM 主循环） | ✅ |
| M2 | 工具集系统（ToolRegistry + toolsets） | ✅ |
| M3 | Session 持久化（SessionDB） | ✅ |
| M4 | 记忆系统（MemoryStore + memory_tool） | ✅ |
| M5 | Rich CLI（流式输出 + 彩色终端） | ✅ |
| M6 | Gateway（Telegram Bot + per-chat 隔离） | ✅ |

全部 6 个里程碑代码已完成，并通过分层测试验证。