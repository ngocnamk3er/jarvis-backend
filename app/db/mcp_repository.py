"""The MCP servers users have added, and where they are switched off."""

from app.db import connection

_COLS = "id::text, user_id, name, url, transport, headers_enc, created_at, updated_at"


def _row(r) -> dict:
    return dict(zip(("id", "user_id", "name", "url", "transport", "headers_enc", "created_at", "updated_at"), r))


def _pool():
    if connection.pool is None:
        raise RuntimeError("Database not initialised")
    return connection.pool


async def list_servers(user_id: str) -> list[dict]:
    async with _pool().connection() as conn:
        cur = await conn.execute(
            f"SELECT {_COLS} FROM {connection.SCHEMA}.user_mcp_servers WHERE user_id = %s ORDER BY created_at",
            (user_id,),
        )
        return [_row(r) for r in await cur.fetchall()]


async def get_server(user_id: str, server_id: str) -> dict | None:
    async with _pool().connection() as conn:
        cur = await conn.execute(
            f"SELECT {_COLS} FROM {connection.SCHEMA}.user_mcp_servers WHERE user_id = %s AND id = %s::uuid",
            (user_id, server_id),
        )
        r = await cur.fetchone()
    return _row(r) if r else None


async def create_server(user_id: str, name: str, url: str, transport: str, headers_enc: str | None) -> dict | None:
    """None if the user already has a server by that name."""
    async with _pool().connection() as conn:
        cur = await conn.execute(
            f"""
            INSERT INTO {connection.SCHEMA}.user_mcp_servers (user_id, name, url, transport, headers_enc)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id, name) DO NOTHING
            RETURNING {_COLS}
            """,
            (user_id, name, url, transport, headers_enc),
        )
        r = await cur.fetchone()
    return _row(r) if r else None


async def delete_server(user_id: str, server_id: str) -> bool:
    async with _pool().connection() as conn:
        cur = await conn.execute(
            f"DELETE FROM {connection.SCHEMA}.user_mcp_servers WHERE user_id = %s AND id = %s::uuid",
            (user_id, server_id),
        )
        return cur.rowcount > 0


async def disabled_ids(conversation_id: str) -> set[str]:
    async with _pool().connection() as conn:
        cur = await conn.execute(
            f"SELECT server_id::text FROM {connection.SCHEMA}.conversation_mcp_disabled WHERE conversation_id = %s",
            (conversation_id,),
        )
        return {r[0] for r in await cur.fetchall()}


async def set_enabled(conversation_id: str, server_id: str, enabled: bool) -> None:
    async with _pool().connection() as conn:
        if enabled:
            await conn.execute(
                f"DELETE FROM {connection.SCHEMA}.conversation_mcp_disabled "
                "WHERE conversation_id = %s AND server_id = %s::uuid",
                (conversation_id, server_id),
            )
        else:
            await conn.execute(
                f"INSERT INTO {connection.SCHEMA}.conversation_mcp_disabled (conversation_id, server_id) "
                "VALUES (%s, %s::uuid) ON CONFLICT DO NOTHING",
                (conversation_id, server_id),
            )


async def list_enabled(user_id: str, conversation_id: str) -> list[dict]:
    """The user's servers that are on for this conversation."""
    off = await disabled_ids(conversation_id)
    return [s for s in await list_servers(user_id) if s["id"] not in off]
