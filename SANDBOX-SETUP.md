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

## 2. Build and deploy the router, with auth on

The router is what actually proxies client requests to the right Sandbox
pod, and it's also where **all** of agent-sandbox's authorization lives —
individual sandbox pods have no auth of their own, by design, so this step
is not optional for anything beyond a throwaway local test. Build it from
source instead of using `kubectl apply` on upstream's own
`sandbox_router.yaml` quickstart — that YAML defaults to
`--authz-mode=allow-all` (anyone can call any sandbox), which is fine for a
five-minute smoke test and not fine for anything real.

**Build** (context must be the repo root — the router's Dockerfile imports
shared Go packages from outside `sandbox-router/`):

```bash
git clone --branch v1.0.2 --depth 1 https://github.com/kubernetes-sigs/agent-sandbox.git
cd agent-sandbox
docker build -f sandbox-router/Dockerfile --build-arg GIT_VERSION=v1.0.2 \
  -t sandbox-router-go:local .
```

**Get the image into the cluster** (minikube path — swap for a real
`docker push` if your registry doesn't have the local `access forbidden`
bug this one does):

```bash
docker tag sandbox-router-go:local \
  host.minikube.internal:5050/root/jarvis-sandbox:sandbox-router-v1.0.2
minikube image load host.minikube.internal:5050/root/jarvis-sandbox:sandbox-router-v1.0.2
```

**RBAC** — the router needs to read Pods (to resolve a Sandbox to its pod
IP) and to call the TokenReview API (to validate the bearer tokens clients
send it):

```bash
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: ServiceAccount
metadata:
  name: sandbox-router
  namespace: agent-sandbox-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: sandbox-router
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: sandbox-router
roleRef: {apiGroup: rbac.authorization.k8s.io, kind: ClusterRole, name: sandbox-router}
subjects:
- {kind: ServiceAccount, name: sandbox-router, namespace: agent-sandbox-system}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: sandbox-router-auth-delegator
roleRef: {apiGroup: rbac.authorization.k8s.io, kind: ClusterRole, name: system:auth-delegator}
subjects:
- {kind: ServiceAccount, name: sandbox-router, namespace: agent-sandbox-system}
EOF
```

**Deployment + Service** — `--authz-mode=tokenreview` is the whole point;
`--cache-enabled=true` is a pod-IP cache that also fixes 502s on a
freshly-claimed sandbox:

```bash
kubectl apply -f - <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sandbox-router-deployment
  namespace: agent-sandbox-system
  labels: {app: sandbox-router}
spec:
  replicas: 2
  selector: {matchLabels: {app: sandbox-router}}
  template:
    metadata: {labels: {app: sandbox-router}}
    spec:
      serviceAccountName: sandbox-router
      securityContext: {runAsNonRoot: true, seccompProfile: {type: RuntimeDefault}}
      containers:
      - name: sandbox-router
        image: host.minikube.internal:5050/root/jarvis-sandbox:sandbox-router-v1.0.2
        imagePullPolicy: IfNotPresent
        args:
        - --http-bind-address=:8080
        - --metrics-bind-address=:9090
        - --health-probe-bind-address=:8081
        - --cluster-domain=cluster.local
        - --proxy-timeout=180s
        - --upstream-max-retries=3
        - --cache-enabled=true
        - --authz-mode=tokenreview
        - --authz-tokenreview-require-token=true
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          capabilities: {drop: ["ALL"]}
        ports:
        - {name: http, containerPort: 8080}
        - {name: metrics, containerPort: 9090}
        - {name: healthz, containerPort: 8081}
        livenessProbe: {httpGet: {path: /healthz, port: healthz}, initialDelaySeconds: 5, periodSeconds: 10}
        readinessProbe: {httpGet: {path: /readyz, port: healthz}, initialDelaySeconds: 1, periodSeconds: 5}
        resources:
          requests: {cpu: 100m, memory: 128Mi}
          limits: {cpu: "1", memory: 512Mi}
      terminationGracePeriodSeconds: 45
---
apiVersion: v1
kind: Service
metadata:
  name: sandbox-router-svc
  namespace: agent-sandbox-system
spec:
  type: ClusterIP
  selector: {app: sandbox-router}
  ports:
  - {name: http, port: 8080, targetPort: 8080}
EOF

kubectl -n agent-sandbox-system rollout status deployment/sandbox-router-deployment --timeout=90s
```

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

jarvis-backend needs two things from the k8s API: permission to
create/list/delete `SandboxClaim`s and read `Sandbox`es, and — separately —
a token the router will accept (step 2's `--authz-tokenreview-require-token`).
Both are covered by the same ServiceAccount; no extra secret needed, its
normal projected token doubles as the router's bearer credential.

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
**through the router**, not straight to a pod IP (the pod itself checks
nothing; going pod-direct skips step 2's auth entirely — confirmed live
that it lets one conversation's sandbox fully read/exec another's). Sends
the ServiceAccount's own projected token as the bearer credential on every
call, read fresh each time since kubelet rotates it (~1h default):

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
_SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"


def _auth_headers() -> dict[str, str]:
    try:
        with open(_SA_TOKEN_PATH) as f:
            token = f.read().strip()
    except OSError:
        return {}
    return {"Authorization": f"Bearer {token}"}


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
    # SandboxDirectConnectionConfig has no headers/auth field — reach into
    # the shared httpx.AsyncClient the connector actually uses instead.
    sandbox.connector.client.headers.update(_auth_headers())
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

Expect `ok` then `42`, no traceback. Also worth confirming the router's
auth is actually doing something — an unauthenticated call should be
rejected:

```bash
kubectl run curl-test --rm -it --image=curlimages/curl --restart=Never -- \
  curl -s -o /dev/null -w "%{http_code}\n" \
  http://sandbox-router-svc.agent-sandbox-system.svc.cluster.local:8080/
```

Expect `401`.

## Known limitation, accepted as-is

The router authenticates jarvis-backend's own calls, but it does nothing
to stop one sandbox pod from reaching another sandbox pod's IP directly —
that traffic never touches the router. NetworkPolicy would close this but
isn't enforced on this cluster's CNI. Confirmed exploitable, confirmed that
ordinary per-conversation usage never triggers it (it requires code
*inside* a sandbox deliberately calling out to another pod's IP). Left
open for now — see `jarvis-sandbox/AGENTSANDBOX-MIGRATION.md` step G for
the full writeup if this needs revisiting.
