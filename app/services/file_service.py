"""Thin pass-through to file_client, kept as its own layer mainly for
consistency with conversation_service.py's shape. Unlike conversations
(looked up by a bare thread_id, so this backend has to check
conv.user_id itself — see conversation_service.py's _get_owned_conversation),
every file_client call already carries user_id and jarvis-file-service scopes
every query by it at the DB layer, so there's no separate ownership check to
add here — a caller can never reach another user's node even by guessing an id.
"""

from app.clients import file_client


async def list_tree(user_id: str, path: str) -> list[dict]:
    return await file_client.list_tree(user_id, path)


async def get_node(user_id: str, node_id: str) -> dict | None:
    return await file_client.get_node(user_id, node_id)


async def get_content(user_id: str, node_id: str) -> tuple[bytes, str, str]:
    return await file_client.get_content(user_id, node_id)


async def create_folder(user_id: str, parent_path: str, name: str) -> dict:
    return await file_client.create_folder(user_id, parent_path, name)


async def upload_file(
    user_id: str, parent_path: str, filename: str, content: bytes, content_type: str | None
) -> dict:
    return await file_client.upload_file(user_id, parent_path, filename, content, content_type)


async def rename_or_move(user_id: str, node_id: str, name: str | None, parent_path: str | None) -> dict:
    return await file_client.rename_or_move(user_id, node_id, name, parent_path)


async def delete_node(user_id: str, node_id: str) -> None:
    await file_client.delete_node(user_id, node_id)
