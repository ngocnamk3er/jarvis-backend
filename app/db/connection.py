from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.core.config import settings

pool: AsyncConnectionPool | None = None
checkpointer: AsyncPostgresSaver | None = None


async def init_db() -> AsyncPostgresSaver:
    global pool, checkpointer
    pool = AsyncConnectionPool(
        conninfo=settings.DATABASE_URL,
        open=False,
        kwargs={"autocommit": True},
    )
    await pool.open()

    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()

    async with pool.connection() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT        PRIMARY KEY,
                title       TEXT        NOT NULL DEFAULT 'New conversation',
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

    return checkpointer


async def close_db() -> None:
    if pool:
        await pool.close()


def get_pool() -> AsyncConnectionPool:
    if pool is None:
        raise RuntimeError("Database not initialised")
    return pool
