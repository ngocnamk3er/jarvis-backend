from typing import Literal

from app.db import connection


async def save_feedback(trace_id: str, rating: Literal["up", "down"], comment: str | None) -> None:
    if connection.pool is None:
        raise RuntimeError("Database not initialised")
    async with connection.pool.connection() as conn:
        await conn.execute(
            f"""
            INSERT INTO {connection.SCHEMA}.message_feedback (trace_id, rating, comment, updated_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (trace_id) DO UPDATE
            SET rating = EXCLUDED.rating, comment = EXCLUDED.comment, updated_at = EXCLUDED.updated_at
            """,
            (trace_id, rating, comment),
        )


async def get_feedback(trace_id: str) -> dict | None:
    if connection.pool is None:
        raise RuntimeError("Database not initialised")
    async with connection.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT rating, comment FROM {connection.SCHEMA}.message_feedback WHERE trace_id = %s",
                (trace_id,),
            )
            row = await cur.fetchone()
    if row is None:
        return None
    return {"rating": row[0], "comment": row[1]}


async def get_feedback_bulk(trace_ids: list[str]) -> dict[str, dict]:
    """Keyed by trace_id — lets callers attach feedback to many messages in
    one round trip instead of one get_feedback() call per message."""
    if not trace_ids:
        return {}
    if connection.pool is None:
        raise RuntimeError("Database not initialised")
    async with connection.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT trace_id, rating, comment FROM {connection.SCHEMA}.message_feedback "
                "WHERE trace_id = ANY(%s)",
                (trace_ids,),
            )
            rows = await cur.fetchall()
    return {row[0]: {"rating": row[1], "comment": row[2]} for row in rows}
