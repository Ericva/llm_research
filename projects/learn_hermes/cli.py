"""
cli.py — 带 prompt_toolkit 输入 + rich 格式化输出的交互式终端界面

对照阅读：hermes-agent/cli.py（HermesCLI 类）

设计思路：
  里程碑 5 在里程碑 4 的基础上替换 input() 为 prompt_toolkit.PromptSession，
  将 print() 替换为 rich.Console，并加入流式输出（streaming=True）。

  核心设计决策：
  1. PromptSession + WordCompleter：命令行补全（/new、/sessions 等）
  2. FileHistory：输入历史持久化到 ~/.learn-hermes/history
  3. rich.Console：彩色输出、Markdown 渲染（agent 回复）、Panel/Rule 分隔
  4. 流式输出：client.chat.completions.create(stream=True)，逐 delta 打印
     对照原版：HermesCLI.streaming_enabled + _consume_stream()
  5. HermesCLI 不继承 AIAgent，而是持有 agent 实例（组合优于继承）

里程碑 5 新增文件：
  - cli.py（本文件）— HermesCLI 类 + 流式 Agent 循环
  对照原版新增：
  - main.py 保持不变（非流式终端入口）
  - 新增 cli_main.py 作为带 rich/prompt_toolkit 的入口

流式循环与非流式循环的关键差异：
  非流式：client.chat.completions.create()
    → response.choices[0].finish_reason == "stop" → 直接返回 .content
  流式：client.chat.completions.create(stream=True)
    → 逐个 chunk 处理 → 积累 delta.content → finish_reason == "stop" 结束
    → 工具调用需要积累 tool_calls（delta 中是增量字符串，需拼接）
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich import print as rprint

from agent import AIAgent
from agent.memory import MemoryStore
from state import SessionDB
from model_tools import get_tool_definitions, handle_function_call
from toolsets import get_toolset_names


# ── 常量 ────────────────────────────────────────────────────────────────────

HISTORY_PATH = Path.home() / ".learn-hermes" / "history"
COMMANDS = ["/new", "/sessions", "/resume", "/memory", "/history", "/help", "/exit", "/quit"]

# prompt_toolkit 输入框样式
PT_STYLE = Style.from_dict({
    "prompt": "ansigreen bold",
    "": "ansiwhite",
})


# ── 流式主循环 ──────────────────────────────────────────────────────────────

def _stream_loop(
    client: OpenAI,
    model: str,
    messages: List[dict],
    enabled_toolsets: Optional[List[str]],
    memory_store: Optional[MemoryStore],
    console: Console,
    max_iterations: int = 10,
) -> str:
    """
    流式 Agent 主循环，对照 agent/core.py 的 _loop()，区别在于：
    - 使用 stream=True 获取 SSE chunk 流
    - 逐 delta 打印到终端（实时显示）
    - tool_calls 需要从增量字符串拼接还原

    对照原版：HermesCLI._consume_stream() + run_agent.py _loop()

    返回最终回复文本（用于存入历史）。
    """
    tool_schemas = get_tool_definitions(enabled_toolsets)

    for iteration in range(max_iterations):
        # 发起流式请求
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas if tool_schemas else None,
            tool_choice="auto" if tool_schemas else None,
            stream=True,
        )

        # ── 积累流式响应 ────────────────────────────────────────────────────
        # 与非流式不同，这里需要手动拼接 delta
        finish_reason = None
        full_content = ""            # 普通文本回复
        tool_calls_acc: dict = {}    # {index: {id, name, arguments}}

        # 流式打印时显示 Agent 前缀
        response_started = False

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            finish_reason = chunk.choices[0].finish_reason if chunk.choices else finish_reason

            if delta is None:
                continue

            # ── 文本内容 delta ──────────────────────────────────────────────
            if delta.content:
                if not response_started:
                    # 第一个 token 到来时，打印 Agent 前缀
                    console.print()
                    console.print("[bold cyan]Agent:[/]", end=" ")
                    response_started = True
                # 逐字符打印（不换行，flush=True）
                print(delta.content, end="", flush=True)
                full_content += delta.content

            # ── 工具调用 delta ─────────────────────────────────────────────
            # tool_calls 是增量的：每个 chunk 只含部分 arguments 字符串
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_calls_acc[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_acc[idx]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc_delta.function.arguments

        # 流结束，换行（如果有文本内容）
        if response_started:
            print()  # 换行，结束流式文本

        # ── 根据 finish_reason 处理结果 ─────────────────────────────────────

        if finish_reason == "stop":
            # LLM 直接给出最终回答，不需要工具
            if full_content and not response_started:
                # 极端情况：content 全在一个非流式块里（理论上不会，but 防御性处理）
                console.print(f"[bold cyan]Agent:[/] {full_content}")
            return full_content

        if finish_reason == "tool_calls":
            # ── 将 assistant 消息（含 tool_calls）追加到历史 ──────────────
            # 注意：顺序必须先 assistant，再 tool results
            tc_list = sorted(tool_calls_acc.items())
            messages.append({
                "role": "assistant",
                "content": full_content or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for _, tc in tc_list
                ],
            })

            # ── 逐一执行工具调用 ──────────────────────────────────────────
            for _, tc in tc_list:
                tool_name = tc["name"]
                try:
                    tool_args = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    tool_args = {}

                # 打印工具调用（带 emoji 颜色）
                console.print(f"  [dim yellow]⚙ 工具调用[/] [yellow]{tool_name}[/]([dim]{tool_args}[/])")

                result = handle_function_call(tool_name, tool_args, store=memory_store)

                console.print(f"  [dim green]✓ 工具结果[/] [dim]{result[:200]}{'...' if len(result) > 200 else ''}[/]")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            continue  # 继续下一轮 LLM 调用

        # 其他 finish_reason（content_filter 等），中断
        break

    return full_content or "[达到最大迭代次数，对话终止]"


# ── HermesCLI ────────────────────────────────────────────────────────────────

class HermesCLI:
    """
    带 prompt_toolkit + rich + 流式输出的交互式终端界面。

    对照原版：hermes-agent/cli.py HermesCLI 类

    主要职责：
    - 维护会话状态（session_id, history, agent）
    - 用 PromptSession 读取输入（带历史、自动补全）
    - 用 rich.Console 渲染输出（彩色、Markdown、Panel）
    - 处理 /xxx 内置命令
    - 调用 _stream_loop() 进行流式对话
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        enabled_toolsets: Optional[List[str]] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self.enabled_toolsets = enabled_toolsets

        # rich console（输出端）
        self.console = Console()

        # OpenAI client（流式循环直接使用）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url or os.environ.get("OPENAI_BASE_URL") or None,
        )

        # 持久化层
        self.db = SessionDB()
        self.memory_store = MemoryStore()

        # 会话状态（由 _new_session / _resume_session 初始化）
        self.session_id: Optional[str] = None
        self.history: List[dict] = []
        self.agent: Optional[AIAgent] = None  # 仅用于持久化，不用于 _loop

        # prompt_toolkit session（输入端）
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        completer = WordCompleter(COMMANDS, ignore_case=True, sentence=True)
        self.prompt_session = PromptSession(
            history=FileHistory(str(HISTORY_PATH)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
            style=PT_STYLE,
            complete_while_typing=False,
        )

    # ── 会话管理 ─────────────────────────────────────────────────────────────

    def _new_session(self) -> None:
        """创建新会话，刷新 memory_store 快照。"""
        sid = self.db.create_session(model=self.model)
        self.agent = AIAgent(
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url or None,
            enabled_toolsets=self.enabled_toolsets,
            session_id=sid,
            db=self.db,
            memory_store=self.memory_store,
        )
        self.session_id = sid
        self.history = []

    def _resume_session(self, prefix: str) -> bool:
        """恢复会话，返回是否成功。"""
        sessions = self.db.list_sessions(limit=50)
        matches = [s for s in sessions if s["id"].startswith(prefix)]
        if not matches:
            self.console.print(f"  [red]找不到以 '{prefix}' 开头的会话[/]")
            return False
        if len(matches) > 1:
            self.console.print(f"  [red]前缀 '{prefix}' 匹配多个会话[/]")
            self._print_sessions(matches)
            return False

        s = matches[0]
        sid = s["id"]
        history = self.db.load_messages(sid)
        self.agent = AIAgent(
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url or None,
            enabled_toolsets=self.enabled_toolsets,
            session_id=sid,
            db=self.db,
            memory_store=self.memory_store,
        )
        self.session_id = sid
        self.history = history
        title = s["title"] or "（无标题）"
        self.console.print(
            f"  [green]已恢复会话 [{sid[:8]}]：{title}（共 {len(history)} 条消息）[/]"
        )
        return True

    # ── 显示辅助 ─────────────────────────────────────────────────────────────

    def _print_sessions(self, sessions: list) -> None:
        if not sessions:
            self.console.print("  [dim]（无历史会话）[/]")
            return
        for s in sessions:
            title = s["title"] or "（无标题）"
            sid_short = s["id"][:8]
            created = time.strftime("%Y-%m-%d %H:%M", time.localtime(s["created_at"]))
            model = s["model"] or "?"
            self.console.print(
                f"  [cyan]{sid_short}[/]  [dim]{created}[/]  [yellow]{model:20}[/]  {title}"
            )

    def _print_banner(self) -> None:
        active = self.enabled_toolsets or ["default"]
        toolsets_str = ", ".join(active)
        available_str = ", ".join(get_toolset_names())

        banner_text = (
            f"[bold cyan]Hermes Agent[/] [dim](里程碑 5)[/]\n"
            f"模型：[yellow]{self.model}[/]  "
            f"工具集：[green]{toolsets_str}[/]\n"
            f"可用工具集：[dim]{available_str}[/]\n"
            f"命令：[dim]/new  /sessions  /resume <id>  /memory  /history  /help  /exit[/]"
        )
        self.console.print(Panel(banner_text, border_style="cyan", padding=(0, 1)))

    def _print_help(self) -> None:
        help_text = (
            "[bold]/new[/]             开启新会话\n"
            "[bold]/sessions[/]        列出最近 10 个会话\n"
            "[bold]/resume <id>[/]     恢复指定会话（输入 id 前缀即可）\n"
            "[bold]/memory[/]          查看当前记忆内容\n"
            "[bold]/history[/]         查看当前会话消息数\n"
            "[bold]/help[/]            显示此帮助\n"
            "[bold]/exit, /quit[/]     退出"
        )
        self.console.print(Panel(help_text, title="内置命令", border_style="blue"))

    # ── 命令处理 ─────────────────────────────────────────────────────────────

    def _handle_command(self, cmd: str) -> bool:
        """
        处理 / 开头的内置命令。
        返回 True 表示命令已处理（跳过 LLM 调用），False 表示退出。
        """
        cmd_lower = cmd.strip().lower()

        if cmd_lower in ("/exit", "/quit", "exit", "quit"):
            self.db.end_session(self.session_id)
            self.console.print("[dim]再见！[/]")
            return False  # 信号：退出主循环

        if cmd_lower == "/new":
            self.db.end_session(self.session_id)
            self._new_session()
            self.console.print(
                Rule(f"[cyan]新会话 [{self.session_id[:8]}][/]", style="cyan")
            )
            return True

        if cmd_lower == "/sessions":
            sessions = self.db.list_sessions(limit=10)
            self.console.print(f"[bold]最近 {len(sessions)} 个会话：[/]")
            self._print_sessions(sessions)
            self.console.print()
            return True

        if cmd_lower.startswith("/resume "):
            prefix = cmd[8:].strip()
            old_sid = self.session_id
            if self._resume_session(prefix):
                self.db.end_session(old_sid)
            self.console.print()
            return True

        if cmd_lower == "/memory":
            mem_result = self.memory_store.read("memory")
            user_result = self.memory_store.read("user")
            self.console.print(f"[bold]MEMORY[/] ({mem_result['usage']}):")
            for i, e in enumerate(mem_result["entries"], 1):
                self.console.print(f"  {i}. [dim]{e[:120]}[/]")
            self.console.print(f"[bold]USER[/] ({user_result['usage']}):")
            for i, e in enumerate(user_result["entries"], 1):
                self.console.print(f"  {i}. [dim]{e[:120]}[/]")
            self.console.print()
            return True

        if cmd_lower == "/history":
            msg_count = len(self.history)
            self.console.print(
                f"[bold]当前会话[/] [{self.session_id[:8]}]：[cyan]{msg_count}[/] 条消息"
            )
            self.console.print()
            return True

        if cmd_lower == "/help":
            self._print_help()
            return True

        # 未知命令
        self.console.print(f"[red]未知命令：{cmd}[/]  输入 /help 查看帮助")
        return True

    # ── 对话主流程 ───────────────────────────────────────────────────────────

    def _run_chat(self, user_input: str) -> None:
        """
        执行一轮对话（流式）。

        与非流式版本（main.py）的关键差异：
        - 不调用 agent.run_conversation()
        - 直接调用 _stream_loop()（本文件中定义）
        - 持久化由本函数手动处理（和 run_conversation 等价）

        为什么不复用 agent.run_conversation()？
        因为流式输出需要在 _loop 内部逐 delta 打印，
        而 run_conversation() 是黑盒接口，返回完整文本后才能处理。
        对照原版：HermesCLI._run_streaming() 也绕过 AIAgent 直接调用 client。
        """
        # 构建当前轮次的完整消息列表
        # system prompt 由 agent 持有（包含记忆快照）
        system_prompt = self.agent.system_prompt
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history)

        user_msg = {"role": "user", "content": user_input}
        messages.append(user_msg)

        # 持久化用户消息
        if self.db and self.session_id:
            self.db.append_messages(self.session_id, [user_msg])

        prev_len = len(messages)

        # 流式主循环
        self.console.print(Rule(style="dim"))
        response_text = _stream_loop(
            client=self.client,
            model=self.model,
            messages=messages,
            enabled_toolsets=self.enabled_toolsets,
            memory_store=self.memory_store,
            console=self.console,
            max_iterations=self.agent.max_iterations,
        )
        self.console.print()

        # 持久化 assistant + tool 消息（_stream_loop 已将它们追加到 messages）
        new_messages = messages[prev_len:]
        if self.db and self.session_id and new_messages:
            self.db.append_messages(self.session_id, new_messages)

        # 更新本地 history（含 user_msg + 新增消息）
        self.history.append(user_msg)
        self.history.extend(new_messages)

    # ── 主入口 ────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """启动交互式 REPL 主循环。"""
        self._new_session()
        self._print_banner()
        self.console.print(f"[dim]新会话已创建 [{self.session_id[:8]}][/]\n")

        while True:
            try:
                user_input = self.prompt_session.prompt(
                    [("class:prompt", "You")],
                    rprompt=f"[{self.session_id[:8]}]",
                )
            except KeyboardInterrupt:
                # Ctrl+C 清空当前行，不退出
                self.console.print()
                continue
            except EOFError:
                # Ctrl+D 退出
                self.db.end_session(self.session_id)
                self.console.print("\n[dim]再见！[/]")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # 命令路由
            if user_input.startswith("/") or user_input.lower() in ("exit", "quit"):
                should_continue = self._handle_command(user_input)
                if not should_continue:
                    break
                continue

            # 正常对话
            try:
                self._run_chat(user_input)
            except KeyboardInterrupt:
                self.console.print("\n[yellow]（已中断）[/]")
            except Exception as e:
                self.console.print(f"\n[red]错误：{e}[/]")
