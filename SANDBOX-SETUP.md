# Sandbox setup: kubernetes-sigs/agent-sandbox ↔ jarvis-backend

How jarvis-backend gives each conversation its own sandboxed shell and
filesystem, end to end — the cluster side (CRDs, controller, router,
template) and the jarvis-backend side (RBAC, code, config) that drives it.

Everything runs on **published images**: agent-sandbox's controller, their
`sandbox-router`, and their stock `python-runtime-sandbox` as the sandbox
itself. Nothing here needs building. Commands and manifests below were
checked against the running cluster rather than written from memory.

## How it works

A `bash` tool call turns into two separate kinds of traffic, and keeping
them apart explains nearly every design decision further down:

```
bash tool (thread_id = the conversation)
  │
  ├─ [k8s API]  find or create this conversation's sandbox
  │               SandboxClaim labelled jarvis-thread=<thread_id>
  │               → controller hands out a pod from the warm pool
  │               → read the Sandbox CR's .status.podIPs
  │
  └─ [HTTP]     run the command
                  POST sandbox-router-svc:8080/execute
                    headers: X-Sandbox-ID, X-Sandbox-Namespace,
                             X-Sandbox-Port, X-Sandbox-Pod-IP
                  → router forwards to <pod-ip>:8888
                  → sandbox pod runs it and replies
```

Three consequences worth internalising:

**Per-conversation isolation comes from the label, nothing else.** Each
`thread_id` gets its own `SandboxClaim`, so its own pod, so its own
filesystem. Coming back to the same conversation finds the same pod by
label selector, with its files still there. No router setting, no
credential, and no network policy participates in this — which is why it
behaves identically no matter how the router is configured.

**The router needs no Kubernetes access.** The *client* resolves the pod IP
(from the Sandbox CR it already has permission to read) and passes it as
`X-Sandbox-Pod-IP`; the router only forwards. That is why upstream's own
router YAML ships no ServiceAccount and no RBAC, and why the
`sandboxes: get` grant in step 4 is load-bearing for routing rather than
just for status checks.

**The sandbox pod authenticates nothing.** `/execute`, `/download/<path>`
and the rest are open to anything that can reach the pod's IP. Upstream's
model puts the boundary at the router plus NetworkPolicy, not at the pod.
Hence: always go through the router, never pod-direct. See
[Known limitation](#known-limitation-accepted-as-is) for what this still
leaves open.

## 0. Prerequisites

- A Kubernetes cluster with `kubectl` pointed at it.
- jarvis-backend already deployed to it, in namespace `jarvis`.
- Outbound access to `registry.k8s.io` and
  `us-central1-docker.pkg.dev` for the published images.

## 1. Install the CRDs + controller

```bash
kubectl config current-context   # sanity check before applying anything

VERSION=v1.0.2

kubectl apply -f https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${VERSION}/sandbox.yaml
kubectl apply -f https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${VERSION}/extensions.yaml

kubectl -n agent-sandbox-system rollout status deployment/agent-sandbox-controller --timeout=90s
kubectl get crd | grep x-k8s.io
```

Expect four CRDs. Note the split in API groups, because it is easy to get
wrong in RBAC later and fails silently when you do:

| CRD | API group |
|---|---|
| `sandboxes` | `agents.x-k8s.io` (core) |
| `sandboxclaims`, `sandboxtemplates`, `sandboxwarmpools` | `extensions.agents.x-k8s.io` |

## 2. Deploy the router

Upstream's published image and their quickstart manifest:

```bash
VERSION=v1.0.2

curl -sSL https://raw.githubusercontent.com/kubernetes-sigs/agent-sandbox/refs/tags/${VERSION}/clients/python/agentic-sandbox-client/sandbox-router/sandbox_router.yaml \
  | sed 's|${ROUTER_IMAGE}|us-central1-docker.pkg.dev/k8s-staging-images/agent-sandbox/sandbox-router:latest-main|g' \
  | sed '/ALLOW_UNAUTHENTICATED_ROUTER/{n;s/value: "false"/value: "true"/}' \
  | kubectl -n agent-sandbox-system apply -f -

kubectl -n agent-sandbox-system rollout status deployment/sandbox-router-deployment --timeout=90s
```

That creates `sandbox-router-deployment` and the `sandbox-router-svc`
Service the client talks to. Configuration is by **env var** — the image
takes no CLI flags.

> **Under GitOps** (this deployment: ArgoCD with `selfHeal: true`), don't
> run that `kubectl apply` — it gets reverted on the next sync. Put the
> manifest's content into the tracked path instead; here that is
> `jarvis-deploy/agent-sandbox-router/base/deployment.yaml`.

### The auth flag

The second `sed` is the part to understand before copying it. Upstream's
file ships `ALLOW_UNAUTHENTICATED_ROUTER: "false"`; the quickstart flips it
to `"true"` so you needn't provision a Secret. Both were tested here:

| Setting | Behavior | Extra setup |
|---|---|---|
| `"true"` | No credential checked. Anything that can reach the Service can address any sandbox. | none |
| `"false"` | Callers without a credential get `401`. | Create a Secret, uncomment the `ROUTER_AUTH_TOKEN` env block, and have the client send `Authorization: Bearer <token>` |

**Per-conversation isolation does not depend on this flag** — see
[How it works](#how-it-works). Verified both ways with two conversations:
different pods, and B could neither list nor read a file A had written.
What `"false"` buys is narrower than it sounds: it rejects callers with no
token at all. It does not scope a caller to one sandbox.

This deployment runs `"true"`, scoped deliberately to the FE → BE →
sandbox path where jarvis-backend is the only caller. To switch, see the
injection seam noted in step 5.

## 3. Create the SandboxTemplate + SandboxWarmPool

Upstream's own template and warm pool, applied straight from the repo —
they only carry placeholders, so there is nothing to hand-maintain here.
The stock `python-runtime-sandbox` image needs no build; the warm pool
keeps pods pre-started so a conversation's first call doesn't pay full pod
startup.

```bash
VERSION=v1.0.2
BASE=https://raw.githubusercontent.com/kubernetes-sigs/agent-sandbox/refs/tags/${VERSION}/clients/python/agentic-sandbox-client

curl -sSL ${BASE}/python-sandbox-template.yaml \
  | sed -e 's|${SANDBOX_NAMESPACE}|default|g' \
        -e 's|${SANDBOX_TEMPLATE_NAME}|python-sandbox-template|g' \
  | kubectl apply -f -

# The warm pool ships replicas: 0, which pre-starts nothing.
curl -sSL ${BASE}/python-sandbox-warmpool.yaml \
  | sed -e 's|${SANDBOX_NAMESPACE}|default|g' \
        -e 's|${SANDBOX_TEMPLATE_NAME}|python-sandbox-template|g' \
        -e 's|${SANDBOX_WARMPOOL_NAME}|python-sandbox-pool|g' \
        -e 's|replicas: 0|replicas: 1|' \
  | kubectl apply -f -

kubectl get pods -n default -w   # wait for python-sandbox-pool-xxx to hit 1/1 Running
```

The two names must agree: the warm pool's `sandboxTemplateRef` has to match
the template's name, and `AGENTSANDBOX_WARMPOOL` in step 5 has to match the
pool's. Substituting a throwaway template name here is how you end up with
an orphan template that nothing references.

Upstream's template also carries commented-out `runtimeClassName` lines for
gVisor and Kata — uncomment one for kernel-level isolation between
sandboxes instead of namespace-level, after installing that RuntimeClass.

The controller also stamps `networkPolicyManagement: Managed` on the
Template and creates a NetworkPolicy per template. On a cluster whose CNI
doesn't enforce NetworkPolicy — this one — that object exists but does
nothing. Don't mistake its presence for isolation.

## 4. jarvis-backend side — RBAC + ServiceAccount

jarvis-backend needs to create/list/delete `SandboxClaim`s and read
`Sandbox`es. The `sandboxes` grant looks like a status-check nicety and
isn't: it is where the SDK reads `.status.podIPs` to tell the router which
pod to forward to. Remove it and routing breaks, not just reporting.

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

Verify it actually took — `kubectl apply` accepts a Role with the wrong
API group without complaint, and the gap only shows up as a failure at
runtime:

```bash
kubectl auth can-i get sandboxes    --as=system:serviceaccount:jarvis:jarvis-backend -n default
kubectl auth can-i create sandboxclaims --as=system:serviceaccount:jarvis:jarvis-backend -n default
```

Then point the `backend` Deployment at the ServiceAccount —
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

`app/agents/tools/sandbox_manager.py` is the entire client. Abridged —
the real file also handles client lifecycle, mime-type guessing, and
timeout translation:

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
    # Through the router, never pod-direct — see "How it works".
    _client = AsyncSandboxClient(
        connection_config=SandboxDirectConnectionConfig(api_url=_ROUTER_URL, server_port=8888),
    )


def _label_value(thread_id: str) -> str:
    """A label value must be <=63 chars and match a restricted charset."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", thread_id)[:63]
    return safe.strip("-_.") or "unknown"


async def _get_or_create_sandbox(thread_id: str):
    namespace = settings.AGENTSANDBOX_NAMESPACE
    label = _label_value(thread_id)
    # The thread_id -> claim mapping lives in a k8s label, not a database:
    # create_sandbox() always picks its own random claim name, so a later
    # call in the same conversation has to find it back by selector.
    existing = await _client.list_all_sandboxes(namespace, label_selector=f"{_THREAD_LABEL}={label}")
    if existing:
        sandbox = await _client.get_sandbox(existing[0], namespace)
    else:
        sandbox = await _client.create_sandbox(
            warmpool=settings.AGENTSANDBOX_WARMPOOL,
            namespace=namespace,
            sandbox_ready_timeout=_CLAIM_READY_TIMEOUT,
            labels={_THREAD_LABEL: label},
        )
    # This is the seam for router auth, if ALLOW_UNAUTHENTICATED_ROUTER="false".
    # SandboxDirectConnectionConfig has no headers field, but the connector's
    # httpx client is reachable and shared by .commands and .files:
    #   sandbox.connector.client.headers.update({"Authorization": f"Bearer {token}"})
    return sandbox


async def exec_bash(thread_id: str, command: str) -> dict:
    """Returns {stdout, stderr, exit_code, timed_out}."""
    sandbox = await _get_or_create_sandbox(thread_id)
    # python-runtime-sandbox runs commands through shlex.split() + subprocess,
    # with no shell in between, so &&, |, > and heredocs all misbehave
    # ("pwd && ls -la" hands pwd a bogus -la flag). Wrapping makes that same
    # shlex.split produce ['bash', '-c', '<command>'] so bash does the parsing.
    wrapped = "bash -c " + shlex.quote(command)
    try:
        result = await sandbox.commands.run(wrapped, timeout=300)
    except SandboxRequestError as e:
        # The SDK has no graceful timeout result — it only ever raises.
        return {"stdout": "", "stderr": str(e), "exit_code": None, "timed_out": True}
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
        "timed_out": False,
    }


async def reset(thread_id: str) -> None:
    """Tear this conversation's sandbox down (deletes its claim + pod)."""
    namespace = settings.AGENTSANDBOX_NAMESPACE
    label = _label_value(thread_id)
    for claim_name in await _client.list_all_sandboxes(
        namespace, label_selector=f"{_THREAD_LABEL}={label}"
    ):
        await _client.delete_sandbox(claim_name, namespace)
```

Two things that bite when adding a call site:

- **Catch `SandboxRequestError`, not `httpx.HTTPError`.** The SDK uses
  `httpx` internally but wraps every non-2xx into its own exception, with
  `.status_code` carried across. Code that caught the httpx types silently
  never matched.
- **Don't hardcode a working directory.** The stock image has no
  `/workspace`; commands land in the image's own working directory. Use
  relative paths.

## 6. Verify

Run a command end to end, from a real `backend` pod through the real tool:

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

Expect `ok` then `42`. If `&&` misbehaves, the `bash -c` wrapping is
missing. A `502` means the router couldn't reach the pod — check that the
Sandbox CR has `.status.podIPs` populated and that jarvis-backend can read
`sandboxes`.

That only proves a command runs. To check the property that matters, use
two different `thread_id`s: write a file from one, then `ls` and `cat` it
from the other. The second must see neither, and the two must report
different `hostname`s. Expect something like:

```
A: python-sandbox-pool-2wh22 | B: python-sandbox-pool-68l2r | different: True
A ls: __pycache__  main.py  requirements.txt  secret-of-a.txt
B ls: __pycache__  main.py  requirements.txt
B cat A's file: cat: secret-of-a.txt: No such file or directory
```

Then clean up — a failed test run leaves claims behind, and each one holds
a pod:

```bash
kubectl get sandboxclaim -n default
```

## Known limitation, accepted as-is

One sandbox pod can reach another sandbox pod's IP directly. That traffic
never touches the router, so **no router setting affects it** —
`ALLOW_UNAUTHENTICATED_ROUTER="false"` doesn't, and neither does the Go
router's `--authz-mode=tokenreview` (which only authenticates that a caller
is *some* valid cluster principal; every conversation here shares one
ServiceAccount, so it never told them apart). NetworkPolicy would close it,
but isn't enforced on this cluster's CNI.

This is confirmed exploitable and knowingly left open. What keeps it
narrow: it takes code running *inside* a sandbox that deliberately calls
another pod's IP — an injected payload, not anything ordinary tool use
does — and that code still has to find the target IP. Per-conversation
isolation on the normal FE → BE → sandbox path is intact and verified.

Closing it properly would mean one of: NetworkPolicy on a CNI that
enforces it; authentication on the sandbox pods' own HTTP server (which
means forking the stock image); or the router's `scoped-token` mode, the
only option that makes the router aware of *which* sandbox a caller may
touch — it needs a component to mint per-sandbox tokens at creation time,
and a self-built router image.

