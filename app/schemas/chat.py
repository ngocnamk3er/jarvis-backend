from pydantic import BaseModel
from typing import Literal

# Pricing/context sourced from openrouter.ai model pages.
# Sorted by input price ascending — cheapest first.
#
# Removed after testing against the same multi-step research prompt:
# - tencent/hy3-preview: repeatedly announced an action ("Let me fetch...")
#   without ever issuing the tool call, leaving the turn silently incomplete.
# - openai/gpt-5.4-nano: ignored the "don't call web_search more than twice
#   on the same topic" prompt guidance entirely — delegated to a research
#   subagent AND kept searching itself, 46 tool calls for one question,
#   exhausting the shared Tavily search quota for everyone.
# "effortOptions" lists which thinking_effort levels are actually meaningful
# (or safe) for each model — verified live this session, not assumed:
# - deepseek-v4-flash/pro: DeepSeek's own API docs (api-docs.deepseek.com/
#   guides/thinking_mode) confirm "low"/"medium" are silently remapped to
#   "high" server-side — real test showed no measurable difference (617-661
#   reasoning chars regardless). Only "high" vs "xhigh" (which DeepSeek maps
#   to its own "max") showed a real, if modest, difference (556 -> 691).
#   So "low"/"medium" are dropped here — selecting them would be a no-op.
# - qwen3.5-9b/qwen3.7-plus: "xhigh" (OpenRouter's docs: allocates ~95% of
#   max_tokens to reasoning) is officially documented as supported only for
#   OpenAI o-series/GPT-5/Grok — Qwen models declare no `supported_efforts`
#   at all in OpenRouter's own model listing. Live-tested anyway: produced
#   84,652+ chars of reasoning and still hadn't finished after 150s (vs.
#   ~6-7k chars for low/medium/high) — a real, reproducible runaway, not
#   noise. Dropped here to prevent users from triggering it by accident.
#   low/medium/high themselves collapsed to a single "high" option: Qwen's
#   own chat_template.jinja only has a binary enable_thinking true/false
#   toggle (verified — no per-level branch exists in the template at all),
#   and repeated live tests showed no reliable scaling between low/medium/
#   high (7271/6079/7308 reasoning chars — noise, not a trend). Presenting
#   3 separate options would imply a graduated control that doesn't exist
#   at the model level.
# - claude-opus-4.8: kept at all 4 graduated levels — Anthropic is one of
#   only two providers (with Gemini 3) OpenRouter documents an explicit
#   effort -> native-param mapping table for, unlike Qwen. Our own test was
#   a single small sample (133 vs 107 reasoning chars) — not enough to
#   justify removing options from a model OpenRouter's docs treat as
#   properly supported.
# - "none" (thinking off entirely) added for every model below after live
#   testing it directly: reasoning_chars dropped to exactly 0 (a real answer
#   still came back, not an error) for deepseek-v4-flash, deepseek-v4-pro,
#   and qwen3.5-9b. Not individually tested for qwen3.7-plus (closed
#   weights, no template to inspect either) or claude-opus-4.8 (both
#   extended by reasonable inference from the same family/officially
#   documented support, not verified the same way).
# "huggingFaceId" is null for closed/proprietary models, or the real HF repo
# path for open-weight ones — sourced from OpenRouter's own /api/v1/models
# `hugging_face_id` field (verified live this session), which itself is what
# we used to confirm each model's chat_template/encoding-script mechanism
# above (deepseek-ai/DeepSeek-V4-Flash, deepseek-ai/DeepSeek-V4-Pro,
# Qwen/Qwen3.5-9B all resolved to real, fetchable repos; qwen3.7-plus and
# claude-opus-4.8 both returned null there).
AVAILABLE_MODELS = [
    {"id": "deepseek/deepseek-v4-flash",  "name": "DeepSeek Flash",     "desc": "Fast, cost-effective MoE",        "inputPrice": "$0.09",  "outputPrice": "$0.18",  "context": "1M",  "contextWindow": 1_000_000, "size": "13B active / 284B total", "default": True, "effortOptions": ["none", "high", "xhigh"], "huggingFaceId": "deepseek-ai/DeepSeek-V4-Flash"},
    {"id": "qwen/qwen3.5-9b",             "name": "Qwen 3.5 9B",        "desc": "Compact, efficient multimodal",   "inputPrice": "$0.10",  "outputPrice": "$0.15",  "context": "262K", "contextWindow": 262_000, "size": "9B", "effortOptions": ["none", "high"], "huggingFaceId": "Qwen/Qwen3.5-9B"},
    {"id": "qwen/qwen3.7-plus",           "name": "Qwen 3.7+",          "desc": "High quality",                     "inputPrice": "$0.32",  "outputPrice": "$1.28",  "context": "1M",  "contextWindow": 1_000_000, "size": "Undisclosed", "effortOptions": ["none", "high"], "huggingFaceId": None},
    {"id": "deepseek/deepseek-v4-pro",    "name": "DeepSeek Pro",       "desc": "Advanced reasoning, coding, agents", "inputPrice": "$0.435", "outputPrice": "$0.87",  "context": "1M", "contextWindow": 1_000_000, "size": "49B active / 1.6T total", "effortOptions": ["none", "high", "xhigh"], "huggingFaceId": "deepseek-ai/DeepSeek-V4-Pro"},
    {"id": "anthropic/claude-opus-4.8",   "name": "Claude Opus 4.8",    "desc": "Frontier reasoning",               "inputPrice": "$1.70",  "outputPrice": "$25.00", "context": "1M", "contextWindow": 1_000_000, "size": "Undisclosed", "effortOptions": ["none", "low", "medium", "high", "xhigh"], "huggingFaceId": None},
]

DEFAULT_MODEL = next(m["id"] for m in AVAILABLE_MODELS if m.get("default"))


class ChatRequest(BaseModel):
    thread_id: str
    content: str
    thinking_effort: Literal["none", "low", "medium", "high", "xhigh"] = "high"
    model: str = DEFAULT_MODEL
    # None means "inherit the main model" — LangGraph merges the root run's
    # configurable into the research subagent's own config automatically, so
    # the subagent already follows `model` above unless this is set to
    # something different.
    subagent_model: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    decision: Literal["approve", "reject"]
    # Without this, resume used to silently fall back to DEFAULT_MODEL for
    # the rest of the turn — confirmed live: a conversation selected as
    # qwen3.5-9b, once a HITL approval triggered /chat/resume, kept running
    # (and wrote the final synthesized answer) on deepseek-v4-flash instead.
    model: str = DEFAULT_MODEL
    subagent_model: str | None = None


class ClarifyResumeRequest(BaseModel):
    thread_id: str
    answer: str
    model: str = DEFAULT_MODEL
    subagent_model: str | None = None


class StopRequest(BaseModel):
    thread_id: str
