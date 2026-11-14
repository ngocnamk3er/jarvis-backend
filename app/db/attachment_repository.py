"""Images the agent has viewed, kept so they outlive the sandbox file and the process."""

import hashlib

from app.db import connection


def _pool():
    if connection.pool is None:
        raise RuntimeError("Database not initialised")
    return connection.pool


async def save(user_id: str, thread_id: str, data: bytes, mime: str) -> str:
    """The id of these bytes in this conversation; saving the same bytes again returns the same id."""
    sha = hashlib.sha256(data).hexdigest()
    async with _pool().connection() as conn:
        cur = await conn.execute(
            f"""
            INSERT INTO {connection.SCHEMA}.attachments (user_id, thread_id, sha256, mime, size_bytes, data)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, thread_id, sha256) DO UPDATE SET mime = EXCLUDED.mime
            RETURNING id::text
            """,
            (user_id, thread_id, sha, mime, len(data), data),
        )
        return (await cur.fetchone())[0]


async def get(attachment_id: str, user_id: str) -> tuple[bytes, str] | None:
    """(bytes, mime) if the attachment exists and belongs to `user_id`."""
    try:
        async with _pool().connection() as conn:
            cur = await conn.execute(
                f"SELECT data, mime FROM {connection.SCHEMA}.attachments WHERE id = %s::uuid AND user_id = %s",
                (attachment_id, user_id),
            )
            row = await cur.fetchone()
    except Exception as e:  # noqa: BLE001 — a malformed id is "not found", not a server error
        if "invalid input syntax for type uuid" in str(e):
            return None
        raise
    return (bytes(row[0]), row[1]) if row else None


async def delete_by_thread(thread_id: str) -> None:
    async with _pool().connection() as conn:
        await conn.execute(f"DELETE FROM {connection.SCHEMA}.attachments WHERE thread_id = %s", (thread_id,))
