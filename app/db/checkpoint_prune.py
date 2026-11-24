"""One-off cleanup of checkpoint history that built up before chat turns pruned after themselves.

    python -m app.db.checkpoint_prune            # dry run: what would go, nothing is deleted
    python -m app.db.checkpoint_prune --apply    # keep only each thread's newest checkpoint

Run it in the backend pod. Only checkpoints other than each thread's newest are removed (with the writes and
state blobs that only they used); the messages of every conversation stay, since the newest checkpoint holds them.
"""

import asyncio
import sys

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings
from app.db.connection import SCHEMA

_SUMMARY_SQL = """
SELECT count(DISTINCT thread_id) AS threads, count(*) AS checkpoints,
       count(*) - count(DISTINCT (thread_id, checkpoint_ns)) AS old_checkpoints
FROM checkpoints
"""
_SIZE_SQL = """
SELECT pg_size_pretty(sum(pg_total_relation_size(c.oid))::bigint) AS size
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = %s AND c.relname IN ('checkpoints', 'checkpoint_blobs', 'checkpoint_writes')
"""


async def _report(pool: AsyncConnectionPool, label: str) -> dict:
    async with pool.connection() as conn:
        conn.row_factory = dict_row
        summary = await (await conn.execute(_SUMMARY_SQL)).fetchone()
        size = await (await conn.execute(_SIZE_SQL, (SCHEMA,))).fetchone()
    print(f"{label}: {summary['threads']} threads, {summary['checkpoints']} checkpoints "
          f"({summary['old_checkpoints']} not the newest of their thread), tables {size['size']}")
    return summary


async def main(apply: bool) -> None:
    pool = AsyncConnectionPool(
        conninfo=settings.DATABASE_URL, open=False,
        kwargs={"autocommit": True, "options": f"-c search_path={SCHEMA},public"},
    )
    await pool.open()
    try:
        before = await _report(pool, "before")
        if not apply:
            print("dry run — nothing deleted. Re-run with --apply to prune.")
            return
        async with pool.connection() as conn:
            rows = await (await conn.execute("SELECT DISTINCT thread_id FROM checkpoints")).fetchall()
        thread_ids = [r[0] for r in rows]
        saver = AsyncPostgresSaver(pool)
        for i in range(0, len(thread_ids), 20):
            await saver.aprune(thread_ids[i : i + 20], strategy="keep_latest")
        after = await _report(pool, "after ")
        assert after["threads"] == before["threads"], "a thread lost all its checkpoints"
        # DELETE only frees space inside the tables for reuse; this gives it back to the disk.
        async with pool.connection() as conn:
            for table in ("checkpoint_blobs", "checkpoints", "checkpoint_writes"):
                await conn.execute(f"VACUUM (FULL) {table}")
        await _report(pool, "vacuumed")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main("--apply" in sys.argv))
