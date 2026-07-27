from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.core.config import settings
from app.db.models import SCHEMA

pool: AsyncConnectionPool | None = None
checkpointer: AsyncPostgresSaver | None = None


async def init_db() -> AsyncPostgresSaver:
    global pool, checkpointer
    pool = AsyncConnectionPool(
        conninfo=settings.DATABASE_URL,
        open=False,
        kwargs={
            "autocommit": True,
            # LangGraph's checkpoint tables (checkpoints, checkpoint_blobs,
            # checkpoint_writes, checkpoint_migrations) are created with bare,
            # unqualified names — this is the only way to steer them into our
            # own schema instead of the connection's default (public).
            "options": f"-c search_path={SCHEMA},public",
        },
    )
    await pool.open()

    # checkpointer.setup() only does CREATE TABLE IF NOT EXISTS, never CREATE
    # SCHEMA — so the schema must already exist before search_path can resolve
    # into it. Alembic's migration env.py bootstraps this same schema too;
    # this repeats it defensively in case the app starts before `make migrate`
    # has ever run.
    async with pool.connection() as conn:
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

        # One-time move for databases that ran with these tables in public
        # before SCHEMA existed — real checkpoint history lives here, so it
        # must follow the tables rather than get silently orphaned once
        # search_path resolves unqualified names into SCHEMA instead. A fresh
        # database has none of these yet: checkpointer.setup() below creates
        # them directly in SCHEMA via the search_path set above.
        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"):
            await conn.execute(
                f"""
                DO $$
                BEGIN
                    IF to_regclass('public.{table}') IS NOT NULL
                       AND to_regclass('{SCHEMA}.{table}') IS NULL THEN
                        ALTER TABLE public.{table} SET SCHEMA {SCHEMA};
                    END IF;
                END $$;
                """
            )

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
