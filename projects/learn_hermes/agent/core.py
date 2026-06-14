"""
AIAgent — 核心 Agent 类，实现 LLM + 工具调用主循环

对照阅读：hermes-agent/run_agent.py（AIAgent 类）

设计思路：
  主循环（_loop）是整个 Agent 系统的心脏：
  1. 调用 LLM，传入当前消息历史 + 工具 schema
  2. 若 finish_reason == "stop"：LLM 直接给出回答，循环结束
  3. 若 finish_reason == "tool_calls"：
     a. 将 assistant 消息（含 tool_calls）追加到历史
     b. 逐一执行每个工具调用
     c. 将每个工具结果以 role="tool" 追加到历史
     d. 继续下一轮循环
  4. 超出 max_iterations 时强制退出，防止无限循环

消息格式遵循 OpenAI Chat Completions API：
  - system: 系统提示
  - user: 用户消息
  - assistant: LLM 回复（可含 tool_calls）
  - tool: 工具执行结果（需要对应的 tool_call_id）

里程碑 2 新增：
  - enabled_toolsets 参数：传给 get_tool_definitions()，控制 LLM 可用的工具集

里程碑 3 新增：
  - session_id 参数：关联 SessionDB 会话
  - db 参数：SessionDB 实例（可选，None 表示不持久化）
  - run_conversation() 在调用前后持久化消息到 SQLite
  - 对话历史从 db.load_messages() 加载，实现跨进程记忆

里程碑 4 新增：
  - memory_store 参数：MemoryStore 实例（可选）
  - __init__ 中调用 memory_store.load()，构建 system prompt 快照
  - system prompt 末尾追加记忆块（冻结快照，整个会话不变）
  - _loop 中调用工具时通过 **kwargs 将 store 传给 memory handler
    （对照原版：handle_function_call 的 kw 参数机制）
"""

import json
import os
from typing import List, Optional
from openai import OpenAI
from model_tools import get_tool_definitions, handle_function_call


class AIAgent:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        enabled_toolsets: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        db=None,              # Optional[SessionDB]
        memory_store=None,    # Optional[MemoryStore]
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.enabled_toolsets = enabled_toolsets
        self.session_id = session_id
        self.db = db
        self.memory_store = memory_store

        # ── 构建 system prompt ──────────────────────────────────────────────
        # 对照原版：run_agent.py 在 __init__ 中将记忆快照追加到 system prompt
        # 顺序：基础提示 + 记忆块（冻结快照）
        base_prompt = system_prompt or (
            "你是一个有帮助的 AI 助手。当需要计算、操作文件或保存记忆时，请使用提供的工具。"
        )

        if memory_store is not None:
            # 先加载磁盘记忆，捕获快照
            memory_store.load()
            # 追加记忆块到 system prompt
            memory_suffix = memory_store.build_system_prompt_suffix()
            self.system_prompt = base_prompt + memory_suffix
        else:
            self.system_prompt = base_prompt

        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
        )

    def chat(self, message: str, history: Optional[List[dict]] = None) -> str:
        """
        简单的单轮对话接口。
        history 为 None 时若有 db 则从 db 加载历史。
        """
        if history is None and self.db and self.session_id:
            history = self.db.load_messages(self.session_id)
        result = self.run_conversation(message, history or [])
        return result["response"]

    def run_conversation(self, user_message: str, history: List[dict]) -> dict:
        """
        完整的一次对话：构建消息列表，运行主循环，返回结果。
        返回 dict 包含：
          - response: 最终文本回复
          - messages: 完整消息历史（含工具调用记录）
          - new_messages: 本轮新增的消息（含 user）
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)

        user_msg = {"role": "user", "content": user_message}
        messages.append(user_msg)

        if self.db and self.session_id:
            self.db.append_messages(self.session_id, [user_msg])

        prev_len = len(messages)
        response_text = self._loop(messages)

        new_messages = messages[prev_len:]
        if self.db and self.session_id and new_messages:
            self.db.append_messages(self.session_id, new_messages)

        return {
            "response": response_text,
            "messages": messages,
            "new_messages": [user_msg] + new_messages,
        }

    def _loop(self, messages: List[dict]) -> str:
        """
        Agent 主循环：不断调用 LLM 直到得到最终回答。

        关键细节：
        - tool_calls 消息必须先 append assistant 消息（含 tool_calls 字段），
          再 append tool 结果，顺序不能乱，否则 API 报错
        - 每个 tool_call 有独立的 id，tool 消息需要用 tool_call_id 对应

        里程碑 4 新增：
        - handle_function_call 传入 store=self.memory_store
          对照原版：handle_function_call(name, args, **ctx) 中 ctx 包含 store
          memory_tool 的 handler 通过 kwargs.get("store") 取到 MemoryStore 实例
          这样 memory tool 就能读写记忆，而无需访问全局变量
        """
        tool_schemas = get_tool_definitions(self.enabled_toolsets)

        for iteration in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                tool_choice="auto" if tool_schemas else None,
            )

            choice = response.choices[0]

            if choice.finish_reason == "stop":
                return choice.message.content or ""

            if choice.finish_reason == "tool_calls":
                assistant_msg = choice.message
                messages.append({
                    "role": "assistant",
                    "content": assistant_msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_msg.tool_calls
                    ],
                })

                for tool_call in assistant_msg.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    print(f"  [工具调用] {tool_name}({tool_args})")

                    # 将 memory_store 注入 kwargs，memory_tool handler 通过 kw.get("store") 取用
                    result = handle_function_call(
                        tool_name, tool_args, store=self.memory_store
                    )
                    print(f"  [工具结果] {result}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

                continue

            break

        return "[达到最大迭代次数，对话终止]"
