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
