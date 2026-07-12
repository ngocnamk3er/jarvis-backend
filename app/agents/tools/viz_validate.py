"""Render-and-check validation for generate_visualization_svg.

Mirrors what the frontend actually does with the SVG (same sanitization,
same iframe sandbox flags) inside a headless Chromium page, so the agent
can catch a broken SVG and retry before it ever reaches the user.

Uses Playwright's *sync* API inside a worker thread (via asyncio.to_thread).
On Windows, uvicorn's --reload forces a SelectorEventLoop for psycopg
compatibility, but Playwright's async API needs ProactorEventLoop for its
subprocess — the sync API sidesteps that conflict by not touching the
caller's event loop at all.
"""

import asyncio
import re

from playwright.sync_api import sync_playwright

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


def _validate_svg_script_sync(svg: str, sandbox: str, wait_ms: int) -> str | None:
    # Must navigate the iframe's `src` to the SVG as its own document (not
    # `srcdoc` with an HTML wrapper) — a sandboxed HTML-wrapped SVG runs its
    # script and builds the DOM correctly but silently fails to paint. This
    # mirrors the fix in svg-diagram.tsx.
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
                """({svg, sandbox}) => {
                    const iframe = document.createElement('iframe');
                    iframe.sandbox = sandbox;
                    iframe.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
                    document.body.appendChild(iframe);
                }""",
                {"svg": svg, "sandbox": sandbox},
            )
            page.wait_for_timeout(wait_ms)
            if errors:
                return "\n".join(dict.fromkeys(errors))[:2000]
            return None
        finally:
            browser.close()


async def validate_svg(code: str, sandbox: str) -> str | None:
    """Return an error message if the SVG fails to render as an image, or if it
    throws a runtime JS error inside the sandboxed iframe the frontend uses to
    render SVGs (which allows CSS :hover, <a>, SMIL <animate>, and inline
    <script> event handlers), else None."""
    trimmed = code.strip()
    if not trimmed.startswith("<svg"):
        return "Invalid SVG: content must start with '<svg'."
    sanitized = _sanitize_svg_entities(trimmed)
    try:
        structural_error = await asyncio.to_thread(_validate_svg_sync, sanitized)
        if structural_error:
            return structural_error
        return await asyncio.to_thread(_validate_svg_script_sync, sanitized, sandbox, 1200)
    except Exception as e:
        return f"Could not validate SVG: {e}"
