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

    return checkpointer


async def close_db() -> None:
    if pool:
        await pool.close()


def get_pool() -> AsyncConnectionPool:
    if pool is None:
        raise RuntimeError("Database not initialised")
    return pool
