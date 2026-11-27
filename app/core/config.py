from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Jarvis"
    APP_VERSION: str = "0.1.4"
    API_PREFIX: str = "/api/v1"

    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "deepseek/deepseek-r1-0528-qwen3-8b:free"
    # Comma-separated model IDs tried, in order, if the requested model errors
    # (rate-limited, down, etc.) — see build_llm_with_fallback() in llm.py,
    # which chains these via LangChain's Runnable.with_fallbacks().
    OPENROUTER_FALLBACK_MODELS: str = "deepseek/deepseek-v4-flash,deepseek/deepseek-v4-pro"

    TAVILY_API_KEY: str = ""

    DATABASE_URL: str = "postgresql://jarvis:jarvis@localhost:5433/jarvis"

    # Redis for the per-conversation run lock (app/core/conversation_lock.py). Empty = an in-process lock,
    # which is all one backend replica needs; it also is the fallback if this Redis cannot be reached.
    REDIS_URL: str = ""

    # Shutdown (a rolling update, a scale-down): how long a replica that is being replaced waits for the chat runs and
    # research tasks it has to finish. The pod's terminationGracePeriodSeconds must be longer, since uvicorn first waits
    # for open connections (the streams of runs the browser is still attached to) and only then drains the rest.
    SHUTDOWN_DRAIN_SECONDS: int = 600
    # A chat stream sends an SSE comment after this many silent seconds, so that no proxy on the way (ingress nginx 300 s,
    # Kong 60 s by default) closes it mid-turn. 0 turns it off. See app/core/sse.py.
    SSE_HEARTBEAT_SECONDS: float = 15.0

    # Flows users build from nodes and run (app/flows/). A run is stopped after FLOW_RUN_TIMEOUT_SECONDS.
    FLOWS_ENABLED: bool = True
    FLOWS_MAX_PER_USER: int = 50
    FLOW_RUN_TIMEOUT_SECONDS: int = 600

    # jarvis-conversation-service — owns the conversations/subagent_traces
    # tables since the Chapter 2 decomposition; see app/clients/conversation_client.py.
    CONVERSATION_SERVICE_URL: str = "http://localhost:8001"
    # jarvis-file-service — owns the per-user folder/file workspace (tree
    # metadata + MinIO blobs + Qdrant embeddings); see app/clients/file_client.py.
    # Shares INTERNAL_API_KEY below with conversation-service.
    FILE_SERVICE_URL: str = "http://localhost:8002"
    INTERNAL_API_KEY: str = ""

    FRONTEND_URL: str = "http://localhost:3000"

    OIDC_ISSUER: str = "http://localhost:8180/realms/jarvis"
    OIDC_AUDIENCE: str = "jarvis-frontend"
    OIDC_JWKS_URL: str = "http://localhost:8180/realms/jarvis/protocol/openid-connect/certs"
    OIDC_JWKS_CACHE_TTL_SECONDS: int = 3600

    LLM_CACHE: bool = False

    # Langfuse — the self-hosted trace store in jarvis-deploy/langfuse/.
    # Empty host disables tracing entirely (see app/core/observability.py),
    # which is what local runs and CI get by default.
    LANGFUSE_HOST: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    # Stamped on every trace so one Langfuse project can hold more than one
    # cluster without the dashboards averaging them together. Langfuse
    # restricts this to lowercase letters, digits, "-" and "_".
    LANGFUSE_ENVIRONMENT: str = "development"

    # kubernetes-sigs/agent-sandbox — backs the bash tool + present_file, one
    # dedicated sandbox pod per conversation. See sandbox_manager.py and
    # SANDBOX-SETUP.md. The warm pool name must match a SandboxWarmPool that
    # exists in AGENTSANDBOX_NAMESPACE; `python-sandbox-pool` is the one
    # upstream's own quickstart YAML creates.
    AGENTSANDBOX_NAMESPACE: str = "default"
    AGENTSANDBOX_WARMPOOL: str = "python-sandbox-pool"
    # How long a conversation's sandbox may sit idle before the cluster stops
    # it. Behaves as an idle timeout rather than a fixed deadline because
    # sandbox_manager pushes the expiry out on every use. Expiry deletes the
    # Pod but keeps the volume, so the next message revives the same
    # filesystem in a few seconds — this is a pause, not a teardown, and the
    # value is a CPU/memory-reclaim knob, not a data-retention one.
    AGENTSANDBOX_TTL_SECONDS: int = 1_800
    # The second tier, and the only thing that ever reclaims disk. Expiry
    # above frees the pod but keeps the volume, so a conversation nobody ever
    # deletes would hold its 1Gi forever. This deadline is set on the *claim*,
    # where expiry cascades down and removes the volume too. Also pushed
    # forward on every use, so it measures abandonment rather than age.
    AGENTSANDBOX_MAX_IDLE_SECONDS: int = 604_800  # 7 days
    # A ceiling on how many conversations may have a sandbox *pod* at the same time (the one spare pod the warm pool
    # keeps ready is not counted; paused sandboxes, whose pod is gone and only the volume is kept, are not either).
    # Past it a conversation that needs a pod first makes room by pausing the sandbox that has been idle the longest
    # (never one that is running something, nor one idle for less than EVICT_MIN_IDLE), and if nothing can be paused
    # it waits up to ADMIT_WAIT for a slot before the tool call fails with a "sandbox capacity" error. 0 = no ceiling.
    # See sandbox_manager._admission.
    AGENTSANDBOX_MAX_PODS: int = 10
    # How many sandbox pods one user may have at once, across their conversations. Past it a new one is made room for
    # by pausing that user's own idlest sandbox, so nobody can take the shared ceiling for themselves by opening many
    # conversations. 0 = no per-user limit. See sandbox_manager._admission.
    AGENTSANDBOX_MAX_PODS_PER_USER: int = 3
    AGENTSANDBOX_ADMIT_WAIT_SECONDS: int = 45
    AGENTSANDBOX_EVICT_MIN_IDLE_SECONDS: int = 120

    # Background research: start_research/check_research/cancel_research/
    # list_research, which run the research subagent as a task that outlives the
    # chat turn (app/agents/async_research.py). False leaves the tools out.
    # Admission control (app/core/load_shed.py): past this many requests in flight in this process, the next one is
    # refused at once with 503 + Retry-After instead of queueing behind the rest. DEFAULT covers everything quick; 30
    # comes from the throughput curve on the test cluster (it stops rising at ~10 concurrent, so ~30 keeps the median
    # near 250 ms). CHAT is one long-lived stream per run and is counted apart so it cannot crowd the quick requests
    # out; 20 is a guess, nothing measured the cost of a run. Both are per process: with more workers or replicas the
    # total is that many times as much. 0 = no limit for that class.
    LOAD_SHED_ENABLED: bool = True
    LOAD_SHED_MAX_INFLIGHT: int = 30
    LOAD_SHED_MAX_INFLIGHT_CHAT: int = 20
    LOAD_SHED_RETRY_AFTER_SECONDS: int = 1

    BACKGROUND_RESEARCH: bool = True
    # How many background research tasks may run at once in this process. Each
    # is a full search/fetch loop of LLM calls, so this bounds what one user
    # asking for ten can cost everyone else.
    BACKGROUND_RESEARCH_MAX_RUNNING: int = 4

    # Agentic retrieval for search_files — decompose, rerank, and retry once
    # when the first pass misses. See app/agents/retrieval.py. Costs two LLM
    # calls on the subagent model per search, three when a retry fires, so
    # it's a quality-for-latency trade; turn it off to fall back to one-shot
    # vector search.
    AGENTIC_RAG_ENABLED: bool = True
    # The helper model for decomposition and reflection. Separate from the
    # agent's own model on purpose: that one defaults to a reasoning model at
    # high effort, which made each of these short calls take ~20s. Pick
    # something fast and non-reasoning — these are classification tasks.
    RETRIEVAL_MODEL: str = "deepseek/deepseek-v4-flash"

    # Skills (app/agents/skills.py): SKILL.md instructions loaded on demand.
    # Empty SKILLS_DIR means the app/skills directory shipped with the image.
    SKILLS_ENABLED: bool = True
    SKILLS_DIR: str = ""
    # Deferred tools (app/agents/tool_search.py): comma-separated tool names
    # whose schemas are only sent once the model asks for them with
    # tool_search. Chosen for being large and rarely needed.
    # ocr_image, draw_boxes and view_image are deliberately NOT deferred: a model asked to "draw boxes" on an
    # image did not think of loading them, and drew from coordinates it guessed or skipped the
    # boxes altogether. Offered in every request, their descriptions are in front of it. view_image
    # joined them after a test where, asked what an attached photo showed, the model answered from
    # ocr_image's text alone (it drops diacritics and digits) instead of loading view_image first.
    DEFERRED_TOOLS: str = "generate_visualization_svg,cancel_research,list_research,check_pdf,read_pdf,office_convert,office_recalculate"
    # MCP (app/agents/mcp_tools.py): a JSON object, or a path to a file
    # holding one, naming the servers to load tools from. Empty = none.
    MCP_SERVERS: str = ""
    # Defer every MCP tool, so a server with many tools costs nothing until
    # one is wanted.
    MCP_DEFER: bool = True

    # MCP servers users add themselves from the chat UI (app/agents/user_mcp.py).
    MCP_USER_ENABLED: bool = True
    MCP_USER_MAX_SERVERS: int = 10
    # Fernet key that encrypts the headers (tokens) users give their servers.
    # Unset: servers can still be added, but not with headers.
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    MCP_SECRET_KEY: str = ""
    # Comma-separated hosts exempt from the public-address rule in
    # app/agents/mcp_guard.py — e.g. an MCP server run inside the cluster.
    MCP_USER_ALLOWED_HOSTS: str = ""

    # Models users add themselves (app/agents/user_models.py). Their API keys are encrypted with
    # MCP_SECRET_KEY (the same Fernet key the MCP headers use); without it only OpenRouter models
    # (which use the server's key) can be added. Hosts to allow past the public-address check
    # reuse MCP_USER_ALLOWED_HOSTS.
    USER_MODELS_ENABLED: bool = True
    USER_MODELS_MAX: int = 20

    # LangGraph checkpoints. "exit" writes one checkpoint when a chat turn ends instead of one per step;
    # KEEP_LATEST then deletes the thread's older ones once the turn is over (nothing here reads history).
    # Not applied to the background research tasks, whose progress is read while they run.
    CHECKPOINT_DURABILITY: str = "exit"
    CHECKPOINT_KEEP_LATEST: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
