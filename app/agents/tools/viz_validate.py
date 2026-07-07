"""Render-and-check validation for the generate_* viz tools.

Mirrors what the frontend actually does with each artifact (same Mermaid
build, same sanitization, same iframe sandbox flags) inside a headless
Chromium page, so the agent can catch a broken diagram/app/animation and
retry before it ever reaches the user.

Uses Playwright's *sync* API inside a worker thread (via asyncio.to_thread).
On Windows, uvicorn's --reload forces a SelectorEventLoop for psycopg
compatibility, but Playwright's async API needs ProactorEventLoop for its
subprocess — the sync API sidesteps that conflict by not touching the
caller's event loop at all.
"""

import asyncio
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

_MERMAID_JS = (Path(__file__).parent.parent.parent / "static" / "vendor" / "mermaid.min.js").read_text(
    encoding="utf-8"
)

_RENDER_TIMEOUT_MS = 8000

# Mirrors svg-diagram.tsx's sanitizeSvg — HTML entities are invalid in SVG/XML.
_HTML_ENTITIES = {
    "&nbsp;": " ", "&mdash;": "—", "&ndash;": "–",
    "&hellip;": "…", "&laquo;": "«", "&raquo;": "»",
    "&ldquo;": "“", "&rdquo;": "”", "&lsquo;": "‘", "&rsquo;": "’",
    "&copy;": "©", "&reg;": "®", "&trade;": "™",
    "&times;": "×", "&divide;": "÷", "&plusmn;": "±",
    "&deg;": "°", "&micro;": "µ", "&middot;": "·",
}


def _sanitize_svg_entities(svg: str) -> str:
    return re.sub(r"&[a-zA-Z]+;", lambda m: _HTML_ENTITIES.get(m.group(0), m.group(0)), svg)


def _validate_mermaid_sync(code: str) -> str | None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.set_default_timeout(_RENDER_TIMEOUT_MS)
            page.set_content(
                f"<!DOCTYPE html><html><head><script>{_MERMAID_JS}</script></head><body></body></html>"
            )
            return page.evaluate(
                """async (code) => {
                    try {
                        await window.mermaid.parse(code);
                        return null;
                    } catch (e) {
                        return String((e && e.message) || e);
                    }
                }""",
                code,
            )
        finally:
            browser.close()


def _validate_svg_sync(code: str) -> str | None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.set_default_timeout(_RENDER_TIMEOUT_MS)
            page.set_content("<!DOCTYPE html><html><body></body></html>")
            return page.evaluate(
                """({svg}) => new Promise((resolve) => {
                    const img = new Image();
                    const dataUri = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
                    const timer = setTimeout(() => resolve('Timed out loading SVG image.'), 5000);
                    img.onload = () => {
                        clearTimeout(timer);
                        if (img.naturalWidth === 0) resolve('SVG rendered with zero width — likely malformed.');
                        else resolve(null);
                    };
                    img.onerror = () => {
                        clearTimeout(timer);
                        resolve('Browser failed to decode SVG (malformed XML/markup).');
                    };
                    img.src = dataUri;
                })""",
                {"svg": code},
            )
        finally:
            browser.close()


def _validate_html_sync(html: str, sandbox: str, wait_ms: int) -> str | None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.set_default_timeout(_RENDER_TIMEOUT_MS)
            errors: list[str] = []
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
            page.goto("about:blank")
            page.evaluate(
                """({html, sandbox}) => {
                    const iframe = document.createElement('iframe');
                    iframe.sandbox = sandbox;
                    iframe.srcdoc = html;
                    document.body.appendChild(iframe);
                }""",
                {"html": html, "sandbox": sandbox},
            )
            page.wait_for_timeout(wait_ms)
            if errors:
                return "\n".join(dict.fromkeys(errors))[:2000]
            return None
        finally:
            browser.close()


async def validate_mermaid(code: str) -> str | None:
    """Return an error message if the Mermaid source fails to parse, else None."""
    try:
        return await asyncio.to_thread(_validate_mermaid_sync, code)
    except Exception as e:
        return f"Could not validate Mermaid diagram: {e}"


async def validate_svg(code: str) -> str | None:
    """Return an error message if the SVG fails to render as an image, else None."""
    trimmed = code.strip()
    if not trimmed.startswith("<svg"):
        return "Invalid SVG: content must start with '<svg'."
    try:
        return await asyncio.to_thread(_validate_svg_sync, _sanitize_svg_entities(trimmed))
    except Exception as e:
        return f"Could not validate SVG: {e}"


async def validate_html(html: str, sandbox: str, wait_ms: int = 1200) -> str | None:
    """Load `html` inside a sandboxed iframe like the frontend does, and report
    any uncaught JS exception or console error seen within `wait_ms`."""
    try:
        return await asyncio.to_thread(_validate_html_sync, html, sandbox, wait_ms)
    except Exception as e:
        return f"Could not validate rendering: {e}"
