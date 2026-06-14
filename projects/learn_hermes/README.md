# learn_hermes

**通过仿写学习 Agent 系统设计** —— 对照 [hermes-agent](https://github.com/coleam00/hermes-agent) 源码，从零搭建一个精简版 Mini Agent。

---

## 项目目标

- **学习导向**：每个文件头部都有详细注释，说明设计思路、对照原版的关键决策、简化的地方
- **里程碑制**：按功能模块递进（M1 基础 Agent → M6 Gateway），每个里程碑可独立运行
- **生产级设计**：保留原版核心架构（WAL mode SQLite、冻结快照记忆、per-chat 会话隔离），去掉生产复杂度（FTS5、retry、streaming）

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env.local`（不要提交到 git）：

```bash
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o-mini
TOOLSETS=default
```

### 3. 运行

**CLI 模式（里程碑 1-4，简单 input 循环）**：
```bash
export $(grep -v '^#' .env.local | xargs)
python main.py
```

**Rich CLI 模式（里程碑 5，流式输出 + 彩色终端）**：
```bash
export $(grep -v '^#' .env.local | xargs)
python cli_main.py
```

**Telegram Gateway 模式（里程碑 6）**：
```bash
# 先在 .env.local 添加 TELEGRAM_BOT_TOKEN=xxx
export $(grep -v '^#' .env.local | xargs)
python gateway_main.py
```

---

## 里程碑架构

| 里程碑 | 内容 | 核心文件 | 对照原版 |
|--------|------|----------|----------|
| **M1** | 基础 Agent：LLM + 工具调用主循环 | `agent/core.py` | `run_agent.py` (AIAgent) |
| **M2** | 工具集系统：注册表 + 动态发现 | `tools/registry.py`<br>`toolsets.py`<br>`model_tools.py` | `tools/registry.py`<br>`toolsets.py` |
| **M3** | Session 持久化：SQLite WAL mode | `state.py` | `hermes_state.py` |
| **M4** | 记忆系统：冻结快照 + 跨会话 | `agent/memory.py`<br>`tools/memory_tool.py` | `tools/memory_tool.py` (MemoryStore) |
| **M5** | Rich CLI：流式输出 + 彩色终端 | `cli.py`<br>`cli_main.py` | `cli.py` (HermesCLI) |
| **M6** | Gateway：Telegram Bot + per-chat 隔离 | `gateway/run.py`<br>`gateway/platforms/telegram.py`<br>`gateway_main.py` | `gateway/run.py`<br>`gateway/platforms/telegram.py` |

---

## 目录结构

```
learn_hermes/
├── agent/
│   ├── core.py          # AIAgent 核心类（LLM 主循环）
│   ├── memory.py        # MemoryStore（冻结快照记忆）
│   └── __init__.py
├── tools/
│   ├── registry.py      # ToolRegistry（工具注册表）
│   ├── calculator.py    # 数学计算工具
│   ├── file_tool.py     # 文件读写工具
│   ├── memory_tool.py   # 记忆工具（add/replace/remove）
│   └── __init__.py
├── gateway/
│   ├── run.py           # GatewayRunner（异步消息调度）
│   └── platforms/
│       ├── base.py      # BasePlatformAdapter（抽象基类）
│       └── telegram.py  # TelegramAdapter（Telegram 平台）
├── main.py              # 里程碑 1-4 入口（简单 CLI）
├── cli.py               # HermesCLI（流式输出 + rich）
├── cli_main.py          # 里程碑 5 入口（Rich CLI）
├── gateway_main.py      # 里程碑 6 入口（Telegram Gateway）
├── state.py             # SessionDB（SQLite 会话持久化）
├── toolsets.py          # 工具集定义与解析
├── model_tools.py       # 工具编排层（discover + get_definitions + dispatch）
├── test_gateway.py      # Gateway 层 Mock 测试
└── requirements.txt
```

---

## 核心设计决策（对照原版）

### 1. **工具注册表（ToolRegistry）**
- **原版**：全局 `registry` 单例，工具模块导入时自动注册
- **学习版**：保留相同机制，用 `ast.parse()` 静态分析自动发现工具模块
- **关键学习点**：`discover_tools()` 避免 import 无关模块的副作用

### 2. **工具集系统（Toolsets）**
- **原版**：支持 `includes` 菱形继承（如 `debugging` 包含 `web + file`）
- **学习版**：保留 `resolve_toolset()` 递归展开逻辑，用 `visited` 集合防循环
- **简化**：去掉 MCP 插件动态扩展

### 3. **Session 持久化（SessionDB）**
- **原版**：WAL mode + FTS5 全文搜索 + jitter retry + schema migration
- **学习版**：保留 WAL mode + schema 表 + messages 表，去掉 FTS5 和 retry
- **关键学习点**：
  - `journal_mode=WAL`：允许并发读 + 单写，适合多平台 gateway
  - `isolation_level=None`：手动管理事务，避免隐式提交混乱
  - `row_factory=sqlite3.Row`：查询结果按列名访问

### 4. **记忆系统（MemoryStore）**
- **原版**：冻结快照模式（frozen snapshot pattern）
- **学习版**：完整保留此设计
- **关键学习点**：
  - 会话启动时 `load()` 捕获快照 → 注入 system prompt
  - 整个会话期间 system prompt 不变（保持 LLM prefix cache 稳定）
  - 工具写入立即持久化到磁盘，但不修改本轮快照
  - 下次会话启动时快照刷新，包含上次写入的内容

### 5. **Gateway 层（GatewayRunner）**
- **原版**：多平台并发 + LRU agent cache + streaming 流式回复
- **学习版**：单平台 + 简单 dict cache + 非流式回复
- **关键学习点**：
  - per-chat 会话隔离：每个 `chat_id` 独立 `AIAgent` + `history`
  - `asyncio.Lock` per chat：同一 chat 的消息串行处理，不同 chat 并发
  - `run_in_executor`：同步 `agent.run_conversation()` 在 asyncio 中需要线程池

### 6. **Telegram Adapter**
- **原版**：polling + webhook 双模式 + 消息批处理 + 流式编辑
- **学习版**：只保留 polling + 基础文本消息
- **关键学习点**：
  - `python-telegram-bot` v20+ 是原生 asyncio
  - 手动管理生命周期：`initialize() → start() → updater.start_polling() → wait → stop() → shutdown()`
  - 不能用 `run_polling()`（内部会调用 `asyncio.run()`，与外层事件循环冲突）

---

## 测试

### 单元测试（分层验证）

```bash
# 工具层
python -c "from model_tools import handle_function_call; print(handle_function_call('calculate', {'expression': '2**10'}))"

# MemoryStore
python -c "from agent.memory import MemoryStore; s = MemoryStore(); s.load(); print(s.add('memory', 'test'))"

# SessionDB
python -c "from state import SessionDB; db = SessionDB(); sid = db.create_session(); print(sid[:8])"

# AIAgent + 真实 LLM
OPENAI_API_KEY=sk-xxx python -c "from agent import AIAgent; a = AIAgent(); print(a.chat('你好'))"
```

### Gateway Mock 测试（不需要 Telegram Token）

```bash
OPENAI_API_KEY=sk-xxx python test_gateway.py
```

### Telegram 真实测试

1. 从 [@BotFather](https://t.me/BotFather) 获取 Bot Token
2. 添加到 `.env.local`：`TELEGRAM_BOT_TOKEN=xxx`
3. 运行 `python gateway_main.py`
4. 在 Telegram 里找到你的 Bot，发 `/start` 开始对话

---

## 学习路径

### 阶段 1：理解工具调用机制（M1-M2）

**目标**：掌握 LLM 工具调用的完整流程

**推荐阅读顺序**：
1. `tools/calculator.py` — 最简单的工具，理解 schema + handler 结构
2. `tools/registry.py` — 工具注册表，理解 `register()` 和 `dispatch()` 机制
3. `model_tools.py` — 工具编排层，理解 `discover_tools()` 自动发现
4. `toolsets.py` — 工具集系统，理解 `resolve_toolset()` 递归展开
5. `agent/core.py` — **核心**，理解 `_loop()` 主循环：
   - LLM 返回 `finish_reason == "tool_calls"` 时如何处理
   - `tool_calls` 消息格式（必须先 append assistant，再 append tool results）
   - `tool_call_id` 的作用（关联 tool 消息和 assistant 消息）

**动手实验**：
```bash
# 1. 直接调用工具（绕过 LLM）
python -c "from model_tools import handle_function_call; print(handle_function_call('calculate', {'expression': 'sqrt(144)'}))"

# 2. 让 LLM 调用工具
OPENAI_API_KEY=sk-xxx python -c "
from agent import AIAgent
agent = AIAgent(enabled_toolsets=['math_only'])
print(agent.chat('请计算 2的10次方'))
"

# 3. 添加自己的工具
# 在 tools/ 下新建 my_tool.py，参考 calculator.py 的结构
```

**关键问题**（阅读代码时思考）：
- Q1：为什么 `tool_calls` 消息必须先 append assistant，再 append tool results？
- Q2：`tool_call_id` 是谁生成的？有什么作用？
- Q3：如果 LLM 一次返回多个 `tool_calls`，执行顺序是怎样的？
- Q4：`discover_tools()` 为什么用 `ast.parse()` 而不是直接 `import *`？

---

### 阶段 2：理解持久化机制（M3-M4）

**目标**：掌握会话历史和记忆的存储与恢复

**推荐阅读顺序**：
1. `state.py` — SessionDB，理解 SQLite WAL mode + messages 表结构
2. `agent/memory.py` — MemoryStore，理解冻结快照模式
3. `tools/memory_tool.py` — memory 工具，理解如何通过 `kwargs` 注入 `store`
4. `main.py` — 完整 CLI 入口，理解 `/new` `/resume` 命令如何操作 DB

**动手实验**：
```bash
# 1. 观察 SQLite 数据库
OPENAI_API_KEY=sk-xxx python main.py
# 对话几轮后退出，查看 ~/.learn-hermes/state.db
sqlite3 ~/.learn-hermes/state.db "SELECT * FROM sessions;"
sqlite3 ~/.learn-hermes/state.db "SELECT role, content FROM messages LIMIT 10;"

# 2. 测试记忆持久化
OPENAI_API_KEY=sk-xxx python main.py
# 输入：我叫小明，请记住
# 退出，重新运行 main.py（新会话）
# 输入：我叫什么名字？
# 观察 ~/.learn-hermes/memories/USER.md 文件内容
```

**关键问题**：
- Q1：为什么用 `journal_mode=WAL` 而不是默认的 DELETE mode？
- Q2：`isolation_level=None` 是什么意思？为什么要手动管理事务？
- Q3：MemoryStore 的"冻结快照"是什么意思？为什么不在会话中实时更新 system prompt？
- Q4：如果在会话中写入记忆，当前会话能看到吗？下次会话呢？

---

### 阶段 3：理解 Gateway 架构（M6）

**目标**：掌握多平台消息网关的设计

**推荐阅读顺序**：
1. `gateway/platforms/base.py` — 抽象基类，理解 `connect()` / `disconnect()` / `send()` 三个核心接口
2. `gateway/platforms/telegram.py` — Telegram 适配器，理解 `python-telegram-bot` v20 的 asyncio 生命周期
3. `gateway/run.py` — GatewayRunner，理解 per-chat 会话隔离 + `asyncio.Lock`
4. `gateway_main.py` — 入口，理解如何组装 adapter + runner

**动手实验**：
```bash
# 1. Mock 测试（不需要 Telegram Token）
OPENAI_API_KEY=sk-xxx python test_gateway.py

# 2. 真实 Telegram 测试
# 从 @BotFather 获取 Token，添加到 .env.local
export $(grep -v '^#' .env.local | xargs)
python gateway_main.py
# 在 Telegram 里找到你的 Bot，发消息测试
```

**关键问题**：
- Q1：为什么每个 `chat_id` 需要独立的 `AIAgent` 实例？
- Q2：`asyncio.Lock` per chat 的作用是什么？为什么不用全局锁？
- Q3：`run_in_executor` 的作用是什么？为什么需要它？
- Q4：如果要支持 Discord，需要改哪些文件？

---

### 阶段 4：深入某个模块（可选）

选择一个你感兴趣的模块深入研究：

**选项 A：流式输出（M5）**
- 阅读 `cli.py` 的 `_stream_loop()` 函数
- 对比 `agent/core.py` 的 `_loop()`，理解流式和非流式的差异
- 关键问题：为什么 `tool_calls` 需要从增量字符串拼接？

**选项 B：工具集系统**
- 阅读 `toolsets.py` 的 `resolve_toolset()` 递归逻辑
- 尝试定义一个新的组合工具集（如 `debugging = web + file + memory`）
- 关键问题：如何防止循环引用？菱形依赖如何处理？

**选项 C：记忆系统**
- 阅读 `agent/memory.py` 的 `add()` / `replace()` / `remove()` 实现
- 理解字符限制的设计（为什么是字符数而不是 token 数？）
- 关键问题：如果记忆满了，如何优雅地处理？

---

## 常见问题

### Q: 为什么有 `main.py` 和 `cli_main.py` 两个入口？

- `main.py`（M1-M4）：简单 `input()` 循环，无额外依赖，适合学习基础架构
- `cli_main.py`（M5）：`prompt_toolkit` + `rich` + 流式输出，体验更好但依赖更多

### Q: 为什么 `memory_tool.py` 不直接 `from agent.memory import MemoryStore`？

会导致循环导入：
```
model_tools → discover_tools() → import tools.memory_tool
tools.memory_tool → from agent.memory import MemoryStore
agent/__init__.py → from .core import AIAgent
agent/core.py → from model_tools import ...  ← 循环！
```

解决方案：`memory_tool.py` 通过 `kwargs.get("store")` duck-typing 获取 `MemoryStore`，不在模块顶层导入。

### Q: 为什么 Telegram Adapter 不能用 `run_polling()`？

`run_polling()` 内部会调用 `asyncio.run()`，而我们已经在 `gateway_main.py` 的 `asyncio.run(main())` 中了，嵌套会报错。需要手动管理 `initialize()` → `start()` → `updater.start_polling()` → `stop()` → `shutdown()`。

### Q: 如何添加新工具？

1. 在 `tools/` 下新建 `my_tool.py`
2. 参考 `calculator.py` 的结构：定义 handler + schema + `registry.register()`
3. 在 `toolsets.py` 中添加到某个工具集的 `tools` 列表
4. 重启 agent，工具会自动被 `discover_tools()` 发现

---

## 对照原版学习建议

每个文件头部都有 `对照阅读：hermes-agent/xxx.py` 注释，建议：

1. **先读学习版**：理解核心流程（去掉了生产复杂度）
2. **再读原版对应文件**：看生产级实现加了哪些细节（retry、FTS5、streaming 等）
3. **对比差异**：理解为什么原版需要这些复杂度，什么场景下必须加

**推荐对比阅读清单**：

| 学习版 | 原版 | 关键差异 |
|--------|------|----------|
| `agent/core.py` | `run_agent.py` (AIAgent) | 原版是全异步，学习版是同步 |
| `state.py` | `hermes_state.py` | 原版有 FTS5 全文搜索 + jitter retry |
| `agent/memory.py` | `tools/memory_tool.py` (MemoryStore) | 基本一致，学习版去掉了 token 统计 |
| `gateway/run.py` | `gateway/run.py` (GatewayRunner) | 原版有 LRU cache + streaming + 多平台 |
| `toolsets.py` | `toolsets.py` | 原版有 MCP 插件动态扩展 |

---

## 下一步

学完 M1-M6 后，可以尝试：

- **添加新工具**：web search、shell exec、image generation
- **实现流式回复**：让 Telegram Bot 逐字显示（编辑消息）
- **多平台支持**：添加 Discord adapter
- **MCP 集成**：对照原版实现 Model Context Protocol 工具插件
- **优化记忆系统**：自动摘要、向量检索、LRU 淘汰

---

## 致谢

本项目是 [hermes-agent](https://github.com/coleam00/hermes-agent) 的学习版精简实现，感谢原作者的开源贡献。

---

## License

MIT
