"""Agentic retrieval over the user's file workspace.

Plain vector search is one shot: embed the query, take the nearest chunks,
hope they were the right ones. That breaks in two common cases — a question
that needs facts from several documents at once, and a question whose wording
shares no vocabulary with the text that answers it.

This wraps the same vector search in a loop that can notice both:

    decompose -> retrieve (per sub-query) -> rerank -> reflect -> refine?
                      ^                                              |
                      +----------------------------------------------+

`decompose` splits a compound question so each part gets its own embedding
rather than one averaged vector that matches nothing well. `rerank` sends the
merged pool through a cross-encoder, which reads query and passage together
instead of comparing two independent embeddings. `reflect` then asks the one
thing a reranker structurally cannot: not "which of these is best" but "does
any of this answer the question at all" — and if not, how the query should
have been worded.

Cost: two LLM calls (decompose, reflect) plus one reranker call, and one more
of each if a refine round fires. `_MAX_ROUNDS` bounds the loop. Those calls go
to a deliberately fast model — see `_llm()`, where using the agent's own
reasoning model turned a 3s search into a 47s one.

Every LLM step degrades to plain vector search on failure. This path must
never return worse results than the one-shot it replaces — a reranker that
errors should cost latency, not recall.
"""

import asyncio
import json
import logging

from langchain_core.messages import HumanMessage

from app.agents.llm import build_llm
from app.clients import file_client
from app.core.config import settings

logger = logging.getLogger(__name__)

# One extra retrieve+rerank round at most. A second refine has never been worth
# the latency in practice: if a reworded query still misses, the content
# usually isn't there.
_MAX_ROUNDS = 2
# Guards against a decomposition that shatters one question into a dozen
# searches, each costing an embedding call.
_MAX_SUBQUERIES = 3
# How many candidates go to the reranker. Larger means better recall for it
# to work with, at the cost of a bigger rerank request.
_CANDIDATE_POOL = 12
# Only the best few go to the reflection step. If the top passages miss the
# question the tail will too, and a short prompt keeps that call cheap.
_REFLECT_SAMPLE = 4
_SNIPPET_CHARS = 400


def _llm():
    """A fast, non-reasoning model for the two helper steps.

    Deliberately not the agent's own model. That one defaults to a reasoning
    model with `reasoning.effort` at high, which measured ~20s per call here —
    47s for a search that plain vector answers in 3. Splitting a query and
    judging whether passages answer it are short classification tasks; chain
    of thought buys nothing and costs everything.

    `honor_model_override=False` matters: without it a user picking a heavy
    model for the conversation would silently drag these calls along with it.
    """
    return build_llm(
        settings.RETRIEVAL_MODEL, honor_model_override=False
    ).bind(extra_body={"reasoning": {"effort": "low", "exclude": True}})


def _parse_json(text: str) -> dict | list | None:
    """Pull JSON out of a model reply, fenced or not."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    if start == -1:
        return None
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        return None


async def _decompose(query: str) -> list[str]:
    """Split a compound question into independently-searchable parts.

    Returns `[query]` unchanged for anything simple — which is most queries.
    Over-splitting is worse than not splitting: each part costs an embedding
    call and dilutes the candidate pool with near-duplicates.
    """
    prompt = (
        "Split this search query into independent sub-queries, but only if it "
        "genuinely asks about separate things that would live in different "
        "documents.\n\n"
        f"Query: {query}\n\n"
        f"Reply with a JSON array of strings, at most {_MAX_SUBQUERIES}. If the "
        'query asks about one thing, reply with just ["<the original query>"]. '
        "No explanation."
    )
    try:
        reply = await _llm().ainvoke([HumanMessage(content=prompt)])
        parsed = _parse_json(reply.content)
        if isinstance(parsed, list) and parsed:
            subs = [str(q).strip() for q in parsed[:_MAX_SUBQUERIES] if str(q).strip()]
            if subs:
                return subs
    except Exception:
        logger.warning("query decomposition failed, using the query as-is", exc_info=True)
    return [query]


async def _retrieve(user_id: str, queries: list[str], per_query: int) -> list[dict]:
    """Vector search for each query, merged and deduplicated.

    Keeps the best score when the same chunk surfaces for several sub-queries —
    that overlap is a signal the chunk is central, not a reason to drop it.
    """
    results = await asyncio.gather(
        *(file_client.search_vector(user_id, q, per_query) for q in queries),
        return_exceptions=True,
    )
    merged: dict[tuple, dict] = {}
    for res in results:
        if isinstance(res, BaseException):
            logger.warning("a sub-query failed during retrieval: %s", res)
            continue
        for chunk in res:
            key = (chunk["file_id"], chunk["chunk_text"][:120])
            if key not in merged or chunk["score"] > merged[key]["score"]:
                merged[key] = chunk
    return sorted(merged.values(), key=lambda c: c["score"], reverse=True)


async def _rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Reorder candidates with a cross-encoder, best first.

    Embedding search encodes query and passage separately, so it compares two
    summaries of meaning — good at topic, weak at "does this actually answer
    the question". A cross-encoder reads both together and answers that.

    An unconfigured or failing reranker hands back zero scores, which leaves
    the embedding order intact.
    """
    try:
        ranked = await file_client.rerank(
            query, [c["chunk_text"] for c in candidates], top_n=len(candidates)
        )
    except Exception:
        logger.warning("rerank call failed, keeping vector order", exc_info=True)
        return candidates

    out = []
    for item in ranked:
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(candidates):
            chunk = candidates[idx]
            chunk["rerank_score"] = float(item.get("score", 0.0))
            out.append(chunk)
    return out or candidates


async def _reflect(query: str, candidates: list[dict]) -> tuple[bool, str | None]:
    """Decide whether the top candidates actually answer the question.

    This is the half a reranker cannot do: it ranks what it was given but has
    no notion of "none of this is the answer, go look differently". Only the
    top few are shown — if the best passages miss, the rest will too, and a
    shorter prompt keeps this cheap.

    Returns (sufficient, follow-up query).
    """
    listing = "\n\n".join(
        f"[{i}] {c['path']}\n{c['chunk_text'][:_SNIPPET_CHARS]}"
        for i, c in enumerate(candidates)
    )
    prompt = (
        f"Question: {query}\n\n"
        f"Best passages found:\n\n{listing}\n\n"
        "Do these contain what is needed to answer the question?\n\n"
        'Reply with JSON only: {"sufficient": true|false, "followup": '
        '"a differently-worded search query, or null"}\n\n'
        "Answer false only when none of them answers it. Then suggest a "
        "follow-up phrased the way the answer would be written, not the way "
        "the question was asked — that difference in wording is usually why "
        "the first search missed."
    )
    try:
        reply = await _llm().ainvoke([HumanMessage(content=prompt)])
        parsed = _parse_json(reply.content)
        if not isinstance(parsed, dict):
            raise ValueError("reflection did not return an object")
        return bool(parsed.get("sufficient", True)), parsed.get("followup") or None
    except Exception:
        # Treat a failed reflection as "good enough" — retrying on a signal we
        # never got would burn latency for nothing.
        logger.warning("reflection failed, accepting current results", exc_info=True)
        return True, None


async def agentic_search(user_id: str, query: str, top_k: int) -> tuple[list[dict], list[str]]:
    """Decompose, retrieve, rerank, and refine once if the first pass missed.

    Returns the chosen chunks plus a short trace of what the pipeline did, so
    the caller can show its work instead of silently returning different
    results than a plain search would have.
    """
    trace: list[str] = []

    subqueries = await _decompose(query)
    if len(subqueries) > 1:
        trace.append(f"split into {len(subqueries)}: {'; '.join(subqueries)}")

    candidates = await _retrieve(user_id, subqueries, _CANDIDATE_POOL)
    if not candidates:
        return [], trace

    for round_no in range(_MAX_ROUNDS):
        ranked = await _rerank(query, candidates[:_CANDIDATE_POOL])
        top = ranked[:top_k]

        if round_no == _MAX_ROUNDS - 1:
            return top, trace

        sufficient, followup = await _reflect(query, ranked[:_REFLECT_SAMPLE])
        if sufficient or not followup:
            return top, trace

        trace.append(f"top hits missed the question, retrying with: {followup}")
        extra = await _retrieve(user_id, [followup], _CANDIDATE_POOL)
        seen = {(c["file_id"], c["chunk_text"][:120]) for c in candidates}
        candidates += [
            c for c in extra if (c["file_id"], c["chunk_text"][:120]) not in seen
        ]

    return candidates[:top_k], trace


def enabled() -> bool:
    return settings.AGENTIC_RAG_ENABLED
