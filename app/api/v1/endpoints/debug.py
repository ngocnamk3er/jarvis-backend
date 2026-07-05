import asyncio
import sys
import threading
import time
import traceback

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter()

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FastAPI Thread Monitor</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0f172a; color: #e2e8f0; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 13px; }

header { display: flex; align-items: center; gap: 16px; padding: 12px 20px; background: #1e293b; border-bottom: 1px solid #334155; }
header h1 { font-size: 16px; font-weight: 600; color: #38bdf8; }
#ts { color: #64748b; font-size: 11px; margin-left: auto; }
#rps-badge { background: #0f4c81; color: #7dd3fc; padding: 2px 8px; border-radius: 4px; font-size: 11px; }

.grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; padding: 14px; }

.panel { background: #1e293b; border: 1px solid #334155; border-radius: 8px; overflow: hidden; }
.panel-title { padding: 8px 12px; font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; border-bottom: 1px solid #334155; display: flex; align-items: center; gap: 6px; }
.panel-title .badge { padding: 1px 6px; border-radius: 10px; font-size: 10px; }
.panel-main   .panel-title { color: #a78bfa; }
.panel-asyncio .panel-title { color: #34d399; }
.panel-anyio  .panel-title { color: #fb923c; }
.badge-main   { background: #3b1f6e; color: #c4b5fd; }
.badge-asyncio { background: #064e3b; color: #6ee7b7; }
.badge-anyio  { background: #431407; color: #fdba74; }

.slot-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px; padding: 10px; }
.slot { height: 36px; border-radius: 4px; background: #0f172a; border: 1px solid #1e293b; display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 2px; overflow: hidden; cursor: default; transition: background .15s; position: relative; }
.slot:hover { border-color: #475569; z-index: 1; }
.slot .slot-id { font-size: 9px; color: #475569; }
.slot .slot-fn { font-size: 8px; color: #94a3b8; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 100%; text-align: center; }
.slot.busy { background: #14532d; border-color: #16a34a; animation: pulse 1.2s infinite; }
.slot.busy .slot-id { color: #86efac; }
.slot.busy .slot-fn { color: #bbf7d0; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.7} }

.task-list { padding: 8px; display: flex; flex-direction: column; gap: 4px; max-height: 420px; overflow-y: auto; }
.task { padding: 6px 8px; border-radius: 4px; background: #0f172a; border-left: 3px solid #6d28d9; }
.task.running { border-left-color: #a78bfa; }
.task-name { font-size: 10px; color: #7c3aed; margin-bottom: 2px; }
.task.running .task-name { color: #c4b5fd; }
.task-coro { font-size: 11px; color: #94a3b8; word-break: break-all; }
.task.running .task-coro { color: #e2e8f0; }

.tooltip { display: none; position: fixed; background: #1e293b; border: 1px solid #475569; border-radius: 6px; padding: 8px; z-index: 100; max-width: 320px; font-size: 11px; pointer-events: none; }
.tooltip .tf { color: #94a3b8; padding: 1px 0; }
.tooltip .tf span { color: #e2e8f0; }
.slot:hover .tooltip { display: block; }

.other-list { padding: 8px; }
.other-row { padding: 4px 8px; border-radius: 4px; background: #0f172a; margin-bottom: 3px; font-size: 11px; color: #64748b; }
</style>
</head>
<body>
<header>
  <h1>FastAPI Thread Monitor</h1>
  <span id="rps-badge">tasks: —</span>
  <span id="ts">—</span>
</header>
<div class="grid">
  <div class="panel panel-main">
    <div class="panel-title">
      <span>Event Loop</span>
      <span class="badge badge-main" id="task-count">0 tasks</span>
    </div>
    <div class="task-list" id="task-list"></div>
  </div>
  <div class="panel panel-asyncio">
    <div class="panel-title">
      <span>asyncio Thread Pool</span>
      <span class="badge badge-asyncio" id="asyncio-count">0/20</span>
    </div>
    <div class="slot-grid" id="asyncio-grid"></div>
  </div>
  <div class="panel panel-anyio">
    <div class="panel-title">
      <span>anyio Thread Pool</span>
      <span class="badge badge-anyio" id="anyio-count">0/40</span>
    </div>
    <div class="slot-grid" id="anyio-grid"></div>
  </div>
</div>
<script>
const MAX_ASYNCIO = 20, MAX_ANYIO = 40;

function buildGrid(containerId, maxSlots) {
  const el = document.getElementById(containerId);
  el.innerHTML = '';
  for (let i = 0; i < maxSlots; i++) {
    const s = document.createElement('div');
    s.className = 'slot';
    s.id = `${containerId}-${i}`;
    s.innerHTML = `<span class="slot-id">#${i}</span><span class="slot-fn"></span>`;
    el.appendChild(s);
  }
}
buildGrid('asyncio-grid', MAX_ASYNCIO);
buildGrid('anyio-grid', MAX_ANYIO);

function shortFn(stack) {
  for (const f of stack) {
    if (!['_bootstrap','threading','anyio','asyncio','uvicorn','starlette'].some(s => f.file.includes(s) || f.func.includes(s)))
      return f.func;
  }
  return stack[0] ? stack[0].func : '';
}

function stackHtml(stack) {
  return stack.map(f => `<div class="tf">${f.file}:${f.line} <span>${f.func}</span></div>`).join('');
}

function updateGrid(gridId, threads, maxSlots, countId) {
  let busy = 0;
  // reset all slots
  for (let i = 0; i < maxSlots; i++) {
    const s = document.getElementById(`${gridId}-${i}`);
    if (!s) continue;
    s.className = 'slot';
    s.querySelector('.slot-fn').textContent = '';
  }
  threads.forEach((t, i) => {
    if (i >= maxSlots) return;
    const s = document.getElementById(`${gridId}-${i}`);
    if (!s) return;
    const fn = shortFn(t.stack);
    const isIdle = !fn || ['_worker','wait','get','run'].includes(fn);
    if (!isIdle) {
      s.className = 'slot busy';
      busy++;
    } else {
      s.className = 'slot';
    }
    s.querySelector('.slot-fn').textContent = fn;
    s.title = t.stack.map(f => `${f.file}:${f.line} ${f.func}`).join('\\n');
  });
  document.getElementById(countId).textContent = `${busy}/${threads.length}`;
}

async function poll() {
  try {
    const res = await fetch('/api/v1/debug/threads');
    const data = await res.json();

    document.getElementById('ts').textContent = new Date(data.timestamp * 1000).toLocaleTimeString();

    // Tasks
    const tasks = data.tasks.filter(t => !t.done);
    document.getElementById('task-count').textContent = `${tasks.length} tasks`;
    document.getElementById('task-list').innerHTML = tasks.map(t => `
      <div class="task ${t.state}">
        <div class="task-name">${t.name}</div>
        <div class="task-coro">${t.coro}</div>
      </div>`).join('');

    // Thread pools
    const asyncio = data.threads.filter(t => t.category === 'asyncio_pool');
    const anyio   = data.threads.filter(t => t.category === 'anyio_pool');

    updateGrid('asyncio-grid', asyncio, MAX_ASYNCIO, 'asyncio-count');
    updateGrid('anyio-grid',   anyio,   MAX_ANYIO,   'anyio-count');

    document.getElementById('rps-badge').textContent = `threads: ${data.threads.length}`;
  } catch(e) { console.error(e); }
  setTimeout(poll, 600);
}
poll();
</script>
</body>
</html>"""


def _get_stack(frame, limit: int = 6) -> list[dict]:
    if frame is None:
        return []
    out = []
    for fs in reversed(traceback.extract_stack(frame)[-limit:]):
        out.append({"file": fs.filename.split("/")[-1], "line": fs.lineno, "func": fs.name})
    return out


@router.get("/threads", response_class=JSONResponse)
async def threads_api():
    frames = sys._current_frames()

    # Identify asyncio default executor threads
    asyncio_thread_names: set[str] = set()
    try:
        loop = asyncio.get_event_loop()
        ex = getattr(loop, "_default_executor", None)
        if ex:
            asyncio_thread_names = {t.name for t in getattr(ex, "_threads", set())}
    except Exception:
        pass

    threads_out = []
    for thread in threading.enumerate():
        name = thread.name
        if name == "MainThread":
            cat = "main"
        elif name in asyncio_thread_names:
            cat = "asyncio_pool"
        elif "ThreadPoolExecutor" in name or "AnyIO" in name or "anyio" in name.lower():
            cat = "anyio_pool"
        else:
            cat = "other"

        threads_out.append({
            "id": thread.ident,
            "name": name,
            "daemon": thread.daemon,
            "alive": thread.is_alive(),
            "category": cat,
            "stack": _get_stack(frames.get(thread.ident)),
        })

    tasks_out = []
    for task in asyncio.all_tasks():
        coro = task.get_coro()
        tasks_out.append({
            "name": task.get_name(),
            "done": task.done(),
            "coro": getattr(coro, "__qualname__", str(coro)),
            "state": "done" if task.done() else "running",
        })

    return {
        "threads": threads_out,
        "tasks": tasks_out,
        "timestamp": time.time(),
    }


@router.get("/threads/ui", response_class=HTMLResponse)
async def threads_ui():
    return HTMLResponse(content=_HTML)
