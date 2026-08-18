from fastapi import APIRouter
from app.api.v1.endpoints import health, chat, conversations, debug

router = APIRouter()

router.include_router(health.router, prefix="/health", tags=["health"])
router.include_router(chat.router, prefix="/chat", tags=["chat"])
router.include_router(conversations.router, prefix="/conversations", tags=["conversations"])
router.include_router(debug.router, prefix="/debug", tags=["debug"])
