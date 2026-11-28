"""The flows users have built (app/flows/). Every query is scoped to the user: a flow id alone reaches nothing."""

import json

from psycopg.errors import UniqueViolation

from app.db import connection

_SUMMARY = ("id", "name", "slug", "description", "version", "enabled", "created_at", "updated_at")
_SUMMARY_COLS = "id::text, name, slug, description, version, enabled, created_at, updated_at"
_FULL = _SUMMARY + ("definition",)
_FULL_COLS = _SUMMARY_COLS + ", definition"


class SlugTaken(Exception):
    """The user already has a flow with that slug."""


def _pool():
    if connection.pool is None:
        raise RuntimeError("Database not initialised")
    return connection.pool


async def list_flows(user_id: str) -> list[dict]:
    async with _pool().connection() as conn:
        cur = await conn.execute(
            f"SELECT {_SUMMARY_COLS} FROM {connection.SCHEMA}.user_flows WHERE user_id = %s ORDER BY updated_at DESC",
            (user_id,),
        )
        return [dict(zip(_SUMMARY, r, strict=True)) for r in await cur.fetchall()]


async def count_flows(user_id: str) -> int:
    async with _pool().connection() as conn:
        cur = await conn.execute(f"SELECT count(*) FROM {connection.SCHEMA}.user_flows WHERE user_id = %s", (user_id,))
        return (await cur.fetchone())[0]


async def get_flow(user_id: str, flow_id: str) -> dict | None:
    async with _pool().connection() as conn:
        cur = await conn.execute(
            f"SELECT {_FULL_COLS} FROM {connection.SCHEMA}.user_flows WHERE user_id = %s AND id = %s::uuid",
            (user_id, flow_id),
        )
        r = await cur.fetchone()
    return dict(zip(_FULL, r, strict=True)) if r else None


async def create_flow(user_id: str, name: str, slug: str, description: str, definition: dict) -> dict:
    """Raises SlugTaken."""
    try:
        async with _pool().connection() as conn:
            cur = await conn.execute(
                f"""
                INSERT INTO {connection.SCHEMA}.user_flows (user_id, name, slug, description, definition)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                RETURNING {_FULL_COLS}
                """,
                (user_id, name, slug, description, json.dumps(definition)),
            )
            return dict(zip(_FULL, await cur.fetchone(), strict=True))
    except UniqueViolation:
        raise SlugTaken(slug) from None


async def update_flow(user_id: str, flow_id: str, name: str, slug: str, description: str, definition: dict,
                      enabled: bool) -> dict | None:
    """The saved flow, its version one up; None if the user has no such flow. Raises SlugTaken."""
    try:
        async with _pool().connection() as conn:
            cur = await conn.execute(
                f"""
                UPDATE {connection.SCHEMA}.user_flows
                SET name = %s, slug = %s, description = %s, definition = %s::jsonb, enabled = %s,
                    version = version + 1, updated_at = now()
                WHERE user_id = %s AND id = %s::uuid
                RETURNING {_FULL_COLS}
                """,
                (name, slug, description, json.dumps(definition), enabled, user_id, flow_id),
            )
            r = await cur.fetchone()
    except UniqueViolation:
        raise SlugTaken(slug) from None
    return dict(zip(_FULL, r, strict=True)) if r else None


async def delete_flow(user_id: str, flow_id: str) -> bool:
    async with _pool().connection() as conn:
        cur = await conn.execute(
            f"DELETE FROM {connection.SCHEMA}.user_flows WHERE user_id = %s AND id = %s::uuid",
            (user_id, flow_id),
        )
        return cur.rowcount > 0


async def list_enabled(user_id: str) -> list[dict]:
    """The user's switched-on flows with their definitions: what their conversations are offered as tools."""
    async with _pool().connection() as conn:
        cur = await conn.execute(
            f"SELECT {_FULL_COLS} FROM {connection.SCHEMA}.user_flows WHERE user_id = %s AND enabled ORDER BY name",
            (user_id,),
        )
        return [dict(zip(_FULL, r, strict=True)) for r in await cur.fetchall()]

