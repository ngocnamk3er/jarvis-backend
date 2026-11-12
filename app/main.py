from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core import observability
from app.api.v1.router import router as api_v1_router
from app.db.connection import init_db, close_db, get_store
from app.clients import conversation_client, file_client
from app.agents.async_research import research_runner
from app.agents.tools import sandbox_manager
from app.agents.graph import build_graph
from app.agents.llm import enable_llm_cache
from app.agents.mcp_tools import load_mcp_tools, parse_mcp_servers


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
    yield
    await research_runner.shutdown()
    observability.shutdown_tracing()
    await close_db()
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1_router, prefix=settings.API_PREFIX)


@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.APP_NAME}"}
