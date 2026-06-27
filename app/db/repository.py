from uuid import uuid4
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


async def list_conversations(pool: AsyncConnectionPool) -> list[dict]:
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT id, title, created_at, updated_at "
                "FROM conversations ORDER BY updated_at DESC"
            )
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def create_conversation(pool: AsyncConnectionPool, title: str = "New conversation") -> dict:
    thread_id = str(uuid4())
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "INSERT INTO conversations (id, title) VALUES (%s, %s) "
                "RETURNING id, title, created_at, updated_at",
                (thread_id, title),
            )
            row = await cur.fetchone()
            return dict(row)


async def update_conversation_title(pool: AsyncConnectionPool, thread_id: str, title: str) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE conversations SET title = %s, updated_at = NOW() WHERE id = %s",
            (title, thread_id),
        )


async def touch_conversation(pool: AsyncConnectionPool, thread_id: str) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE conversations SET updated_at = NOW() WHERE id = %s",
            (thread_id,),
        )


async def delete_conversation(pool: AsyncConnectionPool, thread_id: str) -> None:
    async with pool.connection() as conn:
        await conn.execute("DELETE FROM conversations WHERE id = %s", (thread_id,))
