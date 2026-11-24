import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core import observability
from app.core.load_shed import LoadShedMiddleware, backend_classifier, install_downstream_passthrough
from app.api.v1.router import router as api_v1_router
from app.core.conversation_lock import conversation_locks
from app.core.sandbox_coordination import coordination
from app.core.run_control import run_control
from app.services.chat_service import chat_service
from app.db.connection import init_db, close_db, get_store
from app.clients import conversation_client, file_client
from app.agents.async_research import research_runner
from app.agents.tools import sandbox_manager
from app.agents.graph import build_graph
from app.agents.llm import enable_llm_cache
from app.agents.mcp_tools import load_mcp_tools, parse_mcp_servers


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.LLM_CACHE:
        enable_llm_cache()
    observability.init_tracing()
    conversation_client.init_client()
    file_client.init_client()
    sandbox_manager.init_client()
    checkpointer = await init_db()
    store = get_store()
    mcp_tools = await load_mcp_tools(parse_mcp_servers(settings.MCP_SERVERS))
    app.state.graph = build_graph(checkpointer=checkpointer, store=store, extra_tools=mcp_tools)
    # No tools bound — used only for conversation_service.get_messages(),
    # which just calls graph.aget_state() to read a checkpoint. That never
    # invokes the model/tools, so this is functionally identical to
    # app.state.graph for that read-only path, without paying to bind the
    # full tool list. See build_graph()'s docstring for why this is safe to
    # share the same checkpointer as app.state.graph above.
    app.state.history_graph = build_graph(
        checkpointer=checkpointer, store=store, include_tools=False, extra_tools=mcp_tools
    )
    research_runner.init(checkpointer)
    # So that a Stop, or a cancel of a research task, that lands on another replica reaches this one.
    await run_control.start(
        on_cancel_run=chat_service.stop_local,
        on_cancel_research=research_runner.cancel,
        active_runs=chat_service.active_runs,
    )
    yield
    # A replica being replaced has stopped taking requests by now (uvicorn waited for the open streams first). Let what
    # still runs, detached from any browser, finish before the process goes; the Stop path keeps working meanwhile.
    await asyncio.gather(
        chat_service.drain(settings.SHUTDOWN_DRAIN_SECONDS), research_runner.drain(settings.SHUTDOWN_DRAIN_SECONDS)
    )
    await run_control.close()
    await research_runner.shutdown()
    observability.shutdown_tracing()
    await close_db()
    await conversation_locks.close()
    await coordination.close()
    await conversation_client.close_client()
    await file_client.close_client()
    await sandbox_manager.close_client()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Added first = innermost. CORS is added after it so that it wraps it: a refusal (503) then still carries the CORS
# headers, and Retry-After is exposed, which is what lets the browser hand the frontend the status and the header
# instead of an opaque network error.
if settings.LOAD_SHED_ENABLED:
    app.add_middleware(
        LoadShedMiddleware,
        classify=backend_classifier(settings.API_PREFIX),
        limits=lambda: {"default": settings.LOAD_SHED_MAX_INFLIGHT, "chat": settings.LOAD_SHED_MAX_INFLIGHT_CHAT},
        retry_after=lambda: settings.LOAD_SHED_RETRY_AFTER_SECONDS,
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)

install_downstream_passthrough(app)

app.include_router(api_v1_router, prefix=settings.API_PREFIX)


@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.APP_NAME}"}
