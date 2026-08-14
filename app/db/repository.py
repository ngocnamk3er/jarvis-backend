from uuid import uuid4

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.connection import get_sessionmaker
from app.db.models import Conversation, SubagentTrace


def _conversation_dict(c: Conversation) -> dict:
    return {
        "id": c.id,
        "title": c.title,
        "last_model": c.last_model,
        "last_subagent_model": c.last_subagent_model,
        "context_tokens": c.context_tokens,
        "created_at": c.created_at,
        "updated_at": c.updated_at,
    }


async def list_conversations(user_id: str) -> list[dict]:
    async with get_sessionmaker()() as session:
        result = await session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
        )
        return [_conversation_dict(c) for c in result.scalars()]


async def create_conversation(title: str, user_id: str) -> dict:
    conversation = Conversation(id=str(uuid4()), title=title, user_id=user_id)
    async with get_sessionmaker()() as session:
        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)
        return _conversation_dict(conversation)


async def get_conversation(thread_id: str) -> Conversation | None:
    async with get_sessionmaker()() as session:
        result = await session.execute(select(Conversation).where(Conversation.id == thread_id))
        return result.scalar_one_or_none()


async def update_conversation_title(thread_id: str, title: str) -> None:
    async with get_sessionmaker()() as session:
        await session.execute(
            update(Conversation).where(Conversation.id == thread_id).values(title=title, updated_at=func.now())
        )
        await session.commit()


async def touch_conversation(
    thread_id: str,
    model: str | None = None,
    subagent_model: str | None = None,
) -> None:
    values: dict = {"updated_at": func.now()}
    if model is not None:
        values["last_model"] = model
    if subagent_model is not None:
        values["last_subagent_model"] = subagent_model
    async with get_sessionmaker()() as session:
        await session.execute(update(Conversation).where(Conversation.id == thread_id).values(**values))
        await session.commit()


async def set_context_tokens(thread_id: str, tokens: int) -> None:
    """Overwrite (not accumulate) the conversation's current-context-size
    gauge — see Conversation.context_tokens for why this is last-call-wins
    rather than a running total."""
    async with get_sessionmaker()() as session:
        await session.execute(
            update(Conversation).where(Conversation.id == thread_id).values(context_tokens=tokens, updated_at=func.now())
        )
        await session.commit()


async def delete_conversation(thread_id: str) -> None:
    async with get_sessionmaker()() as session:
        await session.execute(delete(Conversation).where(Conversation.id == thread_id))
        await session.commit()


async def save_subagent_trace(conversation_id: str, tool_call_id: str, events: list[dict]) -> None:
    """Save a subagent's own tool_start/tool_end events, purely for display —
    never read back into the model's context on later turns."""
    # Upsert: a graph replay/resume can re-run the same `task` tool call, which
    # re-triggers this save with the same tool_call_id — overwrite rather than
    # error, since the replayed trace supersedes the old one.
    stmt = (
        pg_insert(SubagentTrace)
        .values(tool_call_id=tool_call_id, conversation_id=conversation_id, events=events)
        .on_conflict_do_update(index_elements=[SubagentTrace.tool_call_id], set_={"events": events})
    )
    async with get_sessionmaker()() as session:
        await session.execute(stmt)
        await session.commit()


async def get_subagent_traces(conversation_id: str) -> dict[str, list[dict]]:
    async with get_sessionmaker()() as session:
        result = await session.execute(
            select(SubagentTrace.tool_call_id, SubagentTrace.events).where(
                SubagentTrace.conversation_id == conversation_id
            )
        )
        return {tool_call_id: events for tool_call_id, events in result}
