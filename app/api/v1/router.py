from fastapi import APIRouter
from app.api.v1.endpoints import health, chat, conversations, debug, files, mcp, user_models, flows, sandbox_files

router = APIRouter()

router.include_router(health.router, prefix="/health", tags=["health"])
router.include_router(chat.router, prefix="/chat", tags=["chat"])
router.include_router(conversations.router, prefix="/conversations", tags=["conversations"])
router.include_router(files.router, prefix="/files", tags=["files"])
router.include_router(debug.router, prefix="/debug", tags=["debug"])
router.include_router(mcp.router, prefix="/mcp", tags=["mcp"])
router.include_router(user_models.router, prefix="/user-models", tags=["user-models"])
router.include_router(flows.router, prefix="/flows", tags=["flows"])
router.include_router(sandbox_files.router, prefix="/chat/sandbox-files", tags=["sandbox"])
