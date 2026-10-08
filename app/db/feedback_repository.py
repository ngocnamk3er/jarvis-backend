from typing import Literal

from app.db.connection import SCHEMA, pool


async def save_feedback(trace_id: str, rating: Literal["up", "down"], comment: str | None) -> None:
    if pool is None:
        raise RuntimeError("Database not initialised")
    async with pool.connection() as conn:
        await conn.execute(
            f"""
            INSERT INTO {SCHEMA}.message_feedback (trace_id, rating, comment, updated_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (trace_id) DO UPDATE
            SET rating = EXCLUDED.rating, comment = EXCLUDED.comment, updated_at = EXCLUDED.updated_at
            """,
            (trace_id, rating, comment),
        )


async def get_feedback(trace_id: str) -> dict | None:
    if pool is None:
        raise RuntimeError("Database not initialised")
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT rating, comment FROM {SCHEMA}.message_feedback WHERE trace_id = %s",
                (trace_id,),
            )
            row = await cur.fetchone()
    if row is None:
        return None
    return {"rating": row[0], "comment": row[1]}
