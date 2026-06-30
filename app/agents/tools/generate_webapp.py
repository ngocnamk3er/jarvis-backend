import json
from langchain_core.tools import tool


@tool
def generate_webapp(html: str, title: str = "") -> str:
    """Render an interactive web app, game, or HTML/JS/CSS demo directly in the chat.

    Use this when the user asks for something interactive: games, simulations,
    calculators, visualizations with controls, animations with user input, etc.

    Write a complete self-contained HTML document. All CSS and JS must be inline
    (no external URLs). The page runs in a sandboxed iframe.

    Tips for games/interactive apps:
    - Use requestAnimationFrame for game loops
    - Handle keyboard events with addEventListener
    - Use canvas for graphics-heavy content
    - Keep the page background dark or neutral for readability

    Args:
        html: Complete HTML document (<!DOCTYPE html> ... </html>)
        title: Short title shown above the app (optional)
    """
    return json.dumps({"__viz__": "webapp", "code": html, "title": title})
