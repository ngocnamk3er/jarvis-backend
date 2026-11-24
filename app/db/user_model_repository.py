"""The models users have added, and the result of the last connection check on each."""

import json

from app.db import connection

_FIELDS = ("id", "user_id", "name", "kind", "model", "base_url", "api_key_enc", "vision", "context_window",
           "send_reasoning", "last_status", "last_checked_at", "last_result", "created_at")
_COLS = "id::text, user_id, name, kind, model, base_url, api_key_enc, vision, context_window, send_reasoning, " \
        "last_status, last_checked_at, last_result, created_at"


def _row(r) -> dict:
    return dict(zip(_FIELDS, r))


def _pool():
    if connection.pool is None:
        raise RuntimeError("Database not initialised")
    return connection.pool


async def list_models(user_id: str) -> list[dict]:
    async with _pool().connection() as conn:
        cur = await conn.execute(
            f"SELECT {_COLS} FROM {connection.SCHEMA}.user_models WHERE user_id = %s ORDER BY created_at",
            (user_id,),
        )
        return [_row(r) for r in await cur.fetchall()]


async def get_model(user_id: str, model_id: str) -> dict | None:
    async with _pool().connection() as conn:
        cur = await conn.execute(
            f"SELECT {_COLS} FROM {connection.SCHEMA}.user_models WHERE user_id = %s AND id = %s::uuid",
            (user_id, model_id),
        )
        r = await cur.fetchone()
    return _row(r) if r else None


async def create_model(user_id: str, name: str, kind: str, model: str, base_url: str | None,
                       api_key_enc: str | None, context_window: int, send_reasoning: bool) -> dict | None:
    """None if the user already has a model by that name."""
    async with _pool().connection() as conn:
        cur = await conn.execute(
            f"""
            INSERT INTO {connection.SCHEMA}.user_models
                (user_id, name, kind, model, base_url, api_key_enc, context_window, send_reasoning)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, name) DO NOTHING
            RETURNING {_COLS}
            """,
            (user_id, name, kind, model, base_url, api_key_enc, context_window, send_reasoning),
        )
        r = await cur.fetchone()
    return _row(r) if r else None


async def delete_model(user_id: str, model_id: str) -> bool:
    async with _pool().connection() as conn:
        cur = await conn.execute(
            f"DELETE FROM {connection.SCHEMA}.user_models WHERE user_id = %s AND id = %s::uuid",
            (user_id, model_id),
        )
        return cur.rowcount > 0


async def save_check(user_id: str, model_id: str, status: str, result: dict, vision: bool | None) -> None:
    """Keep the outcome of a check; `vision` (when the check could tell) is what the agent will believe."""
    async with _pool().connection() as conn:
        await conn.execute(
            f"""
            UPDATE {connection.SCHEMA}.user_models
            SET last_status = %s, last_checked_at = now(), last_result = %s::jsonb,
                vision = COALESCE(%s, vision), updated_at = now()
            WHERE user_id = %s AND id = %s::uuid
            """,
            (status, json.dumps(result), vision, user_id, model_id),
        )
