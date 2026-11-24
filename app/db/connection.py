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

    # Message feedback (thumbs up/down + comment) — the read side used to be
    # Langfuse's own Score API, but self-hosted Langfuse's ingestion queue
    # delays every write by a hardcoded 5s before it's queryable (confirmed
    # live, not configurable), which made a vote vanish on an immediate page
    # reload. This table is the fast, read-your-own-write source of truth;
    # app/core/observability.create_score() still mirrors writes to Langfuse
    # best-effort, purely so feedback is still visible there for analysis.
    async with pool.connection() as conn:
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.message_feedback (
                trace_id TEXT PRIMARY KEY,
                rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
                comment TEXT,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

    # MCP servers users add from the chat UI (see app/agents/user_mcp.py):
    # per user, with the headers (tokens) encrypted; plus the conversations a
    # server is switched off for — on by default everywhere.
    async with pool.connection() as conn:
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.user_mcp_servers (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                transport TEXT NOT NULL,
                headers_enc TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (user_id, name)
            )
            """
        )
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.conversation_mcp_disabled (
                conversation_id TEXT NOT NULL,
                server_id UUID NOT NULL REFERENCES {SCHEMA}.user_mcp_servers(id) ON DELETE CASCADE,
                PRIMARY KEY (conversation_id, server_id)
            )
            """
        )

    # Models users add from the chat UI (see app/agents/user_models.py): per user, the
    # API key of an own endpoint encrypted; the last connection check is kept so the list
    # can show a status without calling every model on every page load.
    async with pool.connection() as conn:
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.user_models (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL CHECK (kind IN ('openrouter', 'endpoint')),
                model TEXT NOT NULL,
                base_url TEXT,
                api_key_enc TEXT,
                vision BOOLEAN NOT NULL DEFAULT FALSE,
                context_window INTEGER NOT NULL DEFAULT 128000,
                send_reasoning BOOLEAN NOT NULL DEFAULT FALSE,
                last_status TEXT,
                last_checked_at TIMESTAMPTZ,
                last_result JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (user_id, name)
            )
            """
        )

    # Images the agent has looked at (view_image) — DeepSeek Harness's "attachment
    # store". Saved once, content-addressed per conversation (the same bytes are
    # one row), and referenced by id from the tool result, so a later turn, a
    # restart, a deleted sandbox file or a page reload all still find the picture.
    # Removed with the conversation.
    async with pool.connection() as conn:
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.attachments (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                mime TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                data BYTEA NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (user_id, thread_id, sha256)
            )
            """
        )
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS attachments_thread_idx ON {SCHEMA}.attachments (thread_id)"
        )

    return checkpointer


async def close_db() -> None:
    if pool:
        await pool.close()


def get_store() -> AsyncPostgresStore:
    if store is None:
        raise RuntimeError("Database not initialised")
    return store
