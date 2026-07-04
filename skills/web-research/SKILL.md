---
name: web-research
description: Web research workflow combining web_search and web_fetch. Use when the user asks to research a topic, find current information, read an article, or gather data from the internet.
---

# web-research

## Tool selection

- **`web_search`** — find URLs and a quick answer for a topic; call at most twice per task
- **`web_fetch`** — read the full content of a specific URL as markdown

## Typical workflow

1. Use `web_search` to find relevant URLs
2. Use `web_fetch` on the most relevant URL(s) to read full content
3. Synthesize and answer — do not keep searching after you have enough

```
web_search("Python asyncio best practices 2024")
→ get URLs + direct answer

web_fetch("https://docs.python.org/3/library/asyncio.html")
→ get full page content as markdown
```

## When to use only web_search

For factual/current questions where the direct answer is enough:
- "What is the latest version of React?"
- "Who won the 2024 World Cup?"
- "Current USD to EUR rate"

## When to also use web_fetch

When you need full article/documentation content:
- Reading a specific documentation page
- Extracting details from a blog post or article
- Getting full technical specs from a page

## Combining with bash for data extraction

Fetch a page and process it with Python:
```bash
python - << 'EOF'
# Process fetched content that was saved to a variable
import re
# ... extraction logic
EOF
```

## Discipline

- One `web_search` call per topic — use a specific query, not multiple vague ones
- `web_fetch` truncates at 20,000 characters — if content is cut off, the most important info is usually in the first part
- Do not call both tools for the same URL — `web_fetch` already returns the full page
- Stop researching once you have enough to answer; do not over-research
