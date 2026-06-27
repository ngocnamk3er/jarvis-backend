from langchain_core.tools import tool

from app.agents.messages import CalculatorMsg


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression (e.g. '2 + 2', '10 * 5 / 2').
    Only supports +, -, *, / and parentheses.
    """
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return CalculatorMsg.INVALID_CHARS
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return CalculatorMsg.EVAL_ERROR.format(error=e)
