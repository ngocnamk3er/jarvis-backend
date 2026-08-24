from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from app.core.config import settings

# LangGraph's own checkpoint/store tables still live under this schema name —
# the conversations/subagent_traces tables that used to share it moved out to
# jarvis-conversation-service's own database (see Chapter 2 decomposition).
SCHEMA = "jarvis"

pool: AsyncConnectionPool | None = None
checkpointer: AsyncPostgresSaver | None = None
store: AsyncPostgresStore | None = None


async def init_db() -> AsyncPostgresSaver:
    global pool, checkpointer, store
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
    # into it.
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

    # LangGraph's Store (cross-thread, persistent key-value storage — used for
    # per-user agent memory, see app/agents/memory.py) is independent of the
    # checkpointer above but shares the same pool/schema/search_path setup.
    store = AsyncPostgresStore(pool)
    await store.setup()

    return checkpointer


async def close_db() -> None:
    if pool:
        await pool.close()


def get_store() -> AsyncPostgresStore:
    if store is None:
        raise RuntimeError("Database not initialised")
    return store
