import json
from langchain_core.tools import tool


@tool
def generate_animation(html: str, label: str, title: str = "") -> str:
    """Create an interactive animation rendered directly in the chat window.

    Write a complete self-contained HTML page with embedded JS/CSS that produces
    the animation. Use the Canvas API, CSS keyframe animations, SVG animations,
    or any vanilla JS technique. Do NOT reference external URLs — all scripts,
    styles, and assets must be inline.

    Tips:
    - Animations MUST loop indefinitely — use requestAnimationFrame loop for canvas,
      animation-iteration-count: infinite for CSS, or repeatCount="indefinite" for SVG
    - Never let the animation stop on its own
    - Set body margin:0; overflow:hidden for a clean frame
    - The frame is 680×400px — design your animation to fit

    Args:
        html: Full HTML document (<html>…</html>) with all JS/CSS inline
        title: Short title shown above the animation (optional)
    """
    return json.dumps({"__viz__": "html", "code": html, "title": title})
