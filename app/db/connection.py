from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from app.core.config import settings
from app.db.models import SCHEMA

pool: AsyncConnectionPool | None = None
checkpointer: AsyncPostgresSaver | None = None
store: AsyncPostgresStore | None = None
engine: AsyncEngine | None = None
async_session: async_sessionmaker[AsyncSession] | None = None


async def init_db() -> AsyncPostgresSaver:
    global pool, checkpointer, store, engine, async_session
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

    # LangGraph's Store (cross-thread, persistent key-value storage — used for
    # per-user agent memory, see app/agents/memory.py) is independent of the
    # checkpointer above but shares the same pool/schema/search_path setup.
    store = AsyncPostgresStore(pool)
    await store.setup()

    # Separate SQLAlchemy engine for ORM queries (repository.py) — LangGraph's
    # checkpointer needs its own raw psycopg AsyncConnectionPool above, so this
    # runs as a second, independent pool to the same database rather than
    # trying to share one pool between the two libraries.
    db_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = create_async_engine(db_url)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    return checkpointer


async def close_db() -> None:
    if pool:
        await pool.close()
    if engine:
        await engine.dispose()


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if async_session is None:
        raise RuntimeError("Database not initialised")
    return async_session


def get_store() -> AsyncPostgresStore:
    if store is None:
        raise RuntimeError("Database not initialised")
    return store
