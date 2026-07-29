from langchain_core.tools import tool
from langgraph.types import interrupt


@tool
async def ask_user(question: str, options: list[str] | None = None) -> str:
    """Ask the user a clarifying question and wait for their answer before continuing.

    Use this only when the request is genuinely ambiguous or underspecified in a
    way that matters for getting the task right — not for trivial preferences you
    could reasonably assume yourself. Overusing this interrupts the user's flow.

    Args:
        question: The question to ask the user.
        options: Optional list of suggested answers, if this is naturally a
            multiple-choice question (e.g. ["Bar chart", "Pie chart", "Line chart"]).
            Omit for an open-ended, free-text question.
    """
    return interrupt({"question": question, "options": options})
