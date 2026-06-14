"""
Calculator 工具 — 第一个工具，用于验证 ToolRegistry 工作流

对照阅读：hermes-agent/tools/ 目录下各工具文件

设计思路：
  - 工具文件只做两件事：定义计算逻辑 + 向 registry 注册
  - schema 遵循 OpenAI function calling 格式
  - handler 是一个普通函数，接收 args dict，返回 JSON 字符串
  - 危险操作（eval）在学习阶段使用，生产中应换用 ast.literal_eval 或 numexpr
"""

import json
import math
from tools import registry


def _calculate(expression: str) -> str:
    """安全地计算数学表达式，返回 JSON 字符串结果。"""
    # 只允许数字、运算符、空格、括号、小数点，以及 math 函数
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed_names["abs"] = abs
    allowed_names["round"] = round

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return json.dumps({"result": result, "expression": expression})
    except ZeroDivisionError:
        return json.dumps({"error": "Division by zero", "expression": expression})
    except Exception as e:
        return json.dumps({"error": f"Invalid expression: {e}", "expression": expression})


# 注册到全局 registry
registry.register(
    name="calculate",
    toolset="math",
    schema={
        "name": "calculate",
        "description": "计算数学表达式。支持加减乘除、幂运算、括号，以及 math 模块中的函数（sin、cos、sqrt、log 等）。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，例如 '(123 * 456) + 789' 或 'sqrt(144)'",
                },
            },
            "required": ["expression"],
        },
    },
    handler=lambda args, **kw: _calculate(args["expression"]),
    emoji="🧮",
)
