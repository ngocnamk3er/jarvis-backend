import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1.router import router as api_v1_router
from app.db.connection import init_db, close_db
from app.agents.graph import build_graph
from app.agents.tools.sandbox_manager import cleanup_expired_sandboxes
from app.agents.llm import enable_llm_cache

_SANDBOX_TTL_MINUTES = 30
_CLEANUP_INTERVAL_SECONDS = 5 * 60  # check every 5 minutes


async def _sandbox_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
        try:
            await asyncio.to_thread(cleanup_expired_sandboxes, _SANDBOX_TTL_MINUTES)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.LLM_CACHE:
        enable_llm_cache()
    checkpointer = await init_db()
    app.state.graph = build_graph(checkpointer=checkpointer)
    cleanup_task = asyncio.create_task(_sandbox_cleanup_loop())
    yield
    cleanup_task.cancel()
    await close_db()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1_router, prefix=settings.API_PREFIX)


@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.APP_NAME}"}
