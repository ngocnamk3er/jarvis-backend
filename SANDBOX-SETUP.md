# Sandbox setup: kubernetes-sigs/agent-sandbox ↔ jarvis-backend

How jarvis-backend gives each conversation its own sandboxed shell +
filesystem, end to end: cluster-side CRDs/controller/router, then the
jarvis-backend RBAC/code/config that talks to them.

Uses agent-sandbox's own **stock `python-runtime-sandbox` image**
throughout (`AGENTSANDBOX_WARMPOOL=python-sandbox-pool`) — nothing custom
to build for the sandbox side itself. Every command below is exactly what's
live on this cluster right now (checked against the running objects, not
copied from memory).

## 0. Prerequisites

- A Kubernetes cluster with `kubectl` pointed at it.
- `docker`, and a way to get built images into the cluster. This doc uses
  `minikube image load` (this cluster is minikube with a local GitLab
  registry that has an unreliable `access forbidden` pull bug — pushing to
  a registry and pulling normally is simpler if you have one that works).
- jarvis-backend already deployed to this cluster, namespace `jarvis`.

## 1. Install the CRDs + controller

```bash
kubectl config current-context   # sanity check before applying anything

VERSION=v1.0.2

kubectl apply -f https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${VERSION}/sandbox.yaml
kubectl apply -f https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${VERSION}/extensions.yaml

kubectl -n agent-sandbox-system rollout status deployment/agent-sandbox-controller --timeout=90s
kubectl get crd | grep x-k8s.io
```

Expect four CRDs: `sandboxes.agents.x-k8s.io` (core), and
`sandboxclaims`/`sandboxtemplates`/`sandboxwarmpools.extensions.agents.x-k8s.io`.
Note the split — `sandboxes` is **not** under `extensions.*`; easy to get
wrong when writing RBAC later (step 4 got this wrong once).

## 2. Deploy the router

The router proxies each exec/file call to the right Sandbox pod. Use
agent-sandbox's own published image and their quickstart config — there is
nothing to build:

```bash
VERSION=v1.0.2

curl -sSL https://raw.githubusercontent.com/kubernetes-sigs/agent-sandbox/refs/tags/${VERSION}/clients/python/agentic-sandbox-client/sandbox-router/sandbox_router.yaml \
  | sed 's|${ROUTER_IMAGE}|us-central1-docker.pkg.dev/k8s-staging-images/agent-sandbox/sandbox-router:latest-main|g' \
  | sed '/ALLOW_UNAUTHENTICATED_ROUTER/{n;s/value: "false"/value: "true"/}' \
  | kubectl -n agent-sandbox-system apply -f -

kubectl -n agent-sandbox-system rollout status deployment/sandbox-router-deployment --timeout=90s
```

That creates both the `sandbox-router-deployment` and the
`sandbox-router-svc` Service the client talks to. **No ServiceAccount, no
RBAC** — the router never calls the Kubernetes API. It doesn't need to:
the SDK resolves the pod IP itself from the Sandbox CR's
`.status.podIPs` and passes it along as `X-Sandbox-Pod-IP`, so the router
only has to forward. (That is also why step 4 grants jarvis-backend
`sandboxes: get` — it is what makes routing work, not just a status check.)

### The auth flag

The second `sed` is the one worth understanding. Upstream's file ships
`ALLOW_UNAUTHENTICATED_ROUTER: "false"`; the quickstart flips it to
`"true"` so you don't have to provision a Secret. Both work — tested:

| Setting | Behavior |
|---|---|
| `"true"` | No credential required. Anything that can reach the Service can address any sandbox. |
| `"false"` | Requires `ROUTER_AUTH_TOKEN` (a single shared token, wired from a Secret — the env block is commented out in upstream's YAML). Callers without it get `401`. Clients send it as `Authorization: Bearer <token>`. |

**Per-conversation isolation does not depend on this flag.** Each
conversation gets its own pod either way — verified with two conversations
where B could neither list nor read A's files. What `"false"` buys is
rejecting callers that present no token at all. This setup runs `"true"`,
since only jarvis-backend calls the router; if you want `"false"`, create
the Secret, uncomment the `ROUTER_AUTH_TOKEN` env block, and have
`sandbox_manager.py` send that token (see step 5 — the header-injection
seam is already there in `_get_or_create_sandbox()`).

> If the router is managed by GitOps (ArgoCD with `selfHeal`), don't run
> the `kubectl apply` above — it will be reverted on the next sync. Put the
> YAML's *content* into the tracked manifest instead.

## 3. Create the SandboxTemplate + SandboxWarmPool (default image)

Stock `python-runtime-sandbox` image, no jarvis-specific build. A warm
pool keeps `replicas` pods pre-started so a conversation's first sandbox
call doesn't pay full pod-startup latency.

```bash
kubectl apply -f - <<'EOF'
apiVersion: extensions.agents.x-k8s.io/v1beta1
kind: SandboxTemplate
metadata:
  name: python-sandbox-template
  namespace: default
spec:
  podTemplate:
    spec:
      containers:
      - name: python-runtime
        image: us-central1-docker.pkg.dev/k8s-staging-images/agent-sandbox/python-runtime-sandbox:latest-main
        ports:
        - containerPort: 8888
        readinessProbe:
          httpGet: {path: "/", port: 8888}
          initialDelaySeconds: 0
          periodSeconds: 1
        livenessProbe:
          httpGet: {path: "/", port: 8888}
          initialDelaySeconds: 2
          periodSeconds: 10
        resources:
          requests: {cpu: "250m", memory: "512Mi", ephemeral-storage: "512Mi"}
      restartPolicy: OnFailure
  volumeClaimTemplates:
  - metadata: {name: workspace}
    spec:
      accessModes: ["ReadWriteOnce"]
      resources: {requests: {storage: "1Gi"}}
  volumeClaimTemplatesPolicy: Overrides
---
apiVersion: extensions.agents.x-k8s.io/v1beta1
kind: SandboxWarmPool
metadata:
  name: python-sandbox-pool
  namespace: default
spec:
  replicas: 1
  sandboxTemplateRef:
    name: python-sandbox-template
EOF

kubectl get pods -n default -w   # wait for python-sandbox-pool-xxx to hit 1/1 Running
```

## 4. jarvis-backend side — RBAC + ServiceAccount

jarvis-backend needs permission to create/list/delete `SandboxClaim`s and
to read `Sandbox`es. The `sandboxes` grant does double duty and is easy to
mistake for dead weight: besides readiness checks, it is where the SDK
reads `.status.podIPs` to tell the router which pod to forward to. Drop it
and routing breaks, not just status reporting.

```bash
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: ServiceAccount
metadata:
  name: jarvis-backend
  namespace: jarvis
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: jarvis-backend-agentsandbox
  namespace: default
rules:
  - apiGroups: ["extensions.agents.x-k8s.io"]
    resources: ["sandboxclaims"]
    verbs: ["get", "list", "watch", "create", "delete"]
  - apiGroups: ["agents.x-k8s.io"]   # core CRD, not extensions.* — see step 1
    resources: ["sandboxes"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: jarvis-backend-agentsandbox
  namespace: default
roleRef: {apiGroup: rbac.authorization.k8s.io, kind: Role, name: jarvis-backend-agentsandbox}
subjects:
  - {kind: ServiceAccount, name: jarvis-backend, namespace: jarvis}
EOF
```

Then point the `backend` Deployment at it —
`jarvis-deploy/backend/base/backend.yaml`:

```yaml
spec:
  template:
    spec:
      serviceAccountName: jarvis-backend
```

## 5. jarvis-backend side — code

`requirements.txt`:

```
k8s-agent-sandbox[async]==1.0.2
```

`app/core/config.py`:

```python
AGENTSANDBOX_NAMESPACE: str = "default"
AGENTSANDBOX_WARMPOOL: str = "python-sandbox-pool"
```

`app/agents/tools/sandbox_manager.py` — the whole client. Connects
**through the router**, not straight to a pod IP: the pod itself checks
nothing, so pod-direct traffic is unauthenticated by anyone — confirmed
live that it lets one sandbox fully read and exec another's.

```python
import re
import shlex

from k8s_agent_sandbox import AsyncSandboxClient
from k8s_agent_sandbox.exceptions import SandboxRequestError
from k8s_agent_sandbox.models import SandboxDirectConnectionConfig

from app.core.config import settings

_client: AsyncSandboxClient | None = None
_THREAD_LABEL = "jarvis-thread"
_CLAIM_READY_TIMEOUT = 180
_ROUTER_URL = "http://sandbox-router-svc.agent-sandbox-system.svc.cluster.local:8080"


def init_client() -> None:
    global _client
    _client = AsyncSandboxClient(
        connection_config=SandboxDirectConnectionConfig(api_url=_ROUTER_URL, server_port=8888),
    )


def _label_value(thread_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", thread_id)[:63]
    return safe.strip("-_.") or "unknown"


async def _get_or_create_sandbox(thread_id: str):
    client = _client
    namespace = settings.AGENTSANDBOX_NAMESPACE
    label = _label_value(thread_id)
    # thread_id -> claim reattachment goes through a k8s label, not a
    # database: create_sandbox() always picks its own random claim name,
    # so a later call in the same conversation finds it back by label.
    existing = await client.list_all_sandboxes(namespace, label_selector=f"{_THREAD_LABEL}={label}")
    if existing:
        sandbox = await client.get_sandbox(existing[0], namespace)
    else:
        sandbox = await client.create_sandbox(
            warmpool=settings.AGENTSANDBOX_WARMPOOL,
            namespace=namespace,
            sandbox_ready_timeout=_CLAIM_READY_TIMEOUT,
            labels={_THREAD_LABEL: label},
        )
    # If the router runs with ALLOW_UNAUTHENTICATED_ROUTER="false", inject the
    # shared token here — SandboxDirectConnectionConfig has no headers field,
    # so reach into the httpx.AsyncClient the connector actually uses:
    #   sandbox.connector.client.headers.update({"Authorization": f"Bearer {token}"})
    return sandbox


async def exec_bash(thread_id: str, command: str) -> dict:
    """Returns {stdout, stderr, exit_code, timed_out}."""
    sandbox = await _get_or_create_sandbox(thread_id)
    # python-runtime-sandbox runs commands via shlex.split()+subprocess,
    # not a real shell — &&/|/>/heredocs need bash -c to work at all.
    wrapped = "bash -c " + shlex.quote(command)
    try:
        result = await sandbox.commands.run(wrapped, timeout=300)
    except SandboxRequestError as e:
        return {"stdout": "", "stderr": str(e), "exit_code": None, "timed_out": True}
    return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.exit_code, "timed_out": False}


async def read_file(thread_id: str, name: str) -> bytes:
    sandbox = await _get_or_create_sandbox(thread_id)
    return await sandbox.files.read(name)


async def reset(thread_id: str) -> None:
    """Tear the conversation's sandbox down (deletes its claim + pod)."""
    namespace = settings.AGENTSANDBOX_NAMESPACE
    label = _label_value(thread_id)
    existing = await _client.list_all_sandboxes(namespace, label_selector=f"{_THREAD_LABEL}={label}")
    for claim_name in existing:
        await _client.delete_sandbox(claim_name, namespace)
```

(Trimmed for this doc — the real file also handles the `k8s_agent_sandbox`
client lifecycle, mime-type guessing on `read_file`, and a couple of edge
cases; see `app/agents/tools/sandbox_manager.py` itself for the exact
current version.)

`bash.py` / `present_file.py` / `chat.py`'s `/sandbox-file` endpoint call
`exec_bash`/`read_file`/`reset` and catch `k8s_agent_sandbox.exceptions.SandboxRequestError`
around file reads — **not** `httpx.HTTPError`, even though the SDK uses
`httpx` internally; it wraps every non-2xx response into its own exception
type.

## 6. Verify end to end

From inside a real `backend` pod, through the actual `bash` tool:

```bash
kubectl exec -n jarvis deploy/backend -- python3 -c "
import asyncio, sys
sys.path.insert(0, '/app')

async def main():
    from app.agents.tools.bash import bash
    from app.agents.tools import sandbox_manager
    sandbox_manager.init_client()
    thread_id = 'verify-test'
    config = {'configurable': {'thread_id': thread_id}}
    try:
        result = await bash.ainvoke(
            {'command': 'echo ok && python3 -c \"print(6*7)\"', 'label': 'verify'},
            config=config,
        )
        print(result)
    finally:
        await sandbox_manager.reset(thread_id)

asyncio.run(main())
"
```

Expect `ok` then `42`, no traceback.

To confirm the isolation itself rather than just that a command runs, use
two different `thread_id`s: write a file from one, then `ls` from the
other. The second must not see it, and the two must report different
`hostname`s.

## Known limitation, accepted as-is

Nothing stops one sandbox pod from reaching another sandbox pod's IP
directly — that traffic never touches the router, so no router setting
affects it. NetworkPolicy would close it but isn't enforced on this
cluster's CNI.

This is confirmed exploitable and knowingly left open. Two things keep it
narrow: it requires code *inside* a sandbox deliberately calling out to
another pod's IP (an injected payload, say — not anything ordinary tool
use does), and the attacker still has to find the target pod's IP.
Per-conversation isolation on the normal FE → BE → sandbox path is intact
and verified.

Note that running the router with `ALLOW_UNAUTHENTICATED_ROUTER="false"`
would *not* close this either, nor would the Go router's
`--authz-mode=tokenreview` — the latter only authenticates that a caller
is some valid cluster principal, and every conversation here shares one
ServiceAccount, so it never distinguished between them. The mode that
actually binds a credential to a single sandbox is `scoped-token`, which
needs something to mint per-sandbox tokens at creation time. See
`jarvis-sandbox/AGENTSANDBOX-MIGRATION.md` step G.
