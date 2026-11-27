"""Heartbeats for Server-Sent Events.

Every proxy between the backend and the browser closes a response that has been silent for too long: the ingress's
nginx after `proxy-read-timeout` (300 s here, jarvis-deploy shared/base/ingress.yaml), Kong after a service's
`read_timeout` (60 s by default), a cloud load balancer after its idle timeout. These count the time between two reads,
not the length of the response. A chat turn can be silent for that long: a bash command that runs for minutes, a `task`
subagent (its own tokens are not streamed), the compaction before a turn. The proxy then closes the stream mid-turn, and
the browser cannot tell: the status was 200, the stream just ends. The run itself goes on and is saved, so a reload
shows the full answer.

An SSE comment line (one that starts with ':') is data to every proxy and nothing to an SSE reader: the frontend and
jarvis-eval only read `data: ` lines. Sending one whenever the stream has been quiet for a while keeps it open.
"""

import asyncio
from collections.abc import AsyncIterator

HEARTBEAT = ": ping\n\n"


async def with_heartbeat(lines: AsyncIterator[str], interval: float) -> AsyncIterator[str]:
    """Yield `lines`, and `HEARTBEAT` whenever `interval` seconds pass without one. `interval` <= 0 turns it off.

    `lines` is read by a task of its own, so that waiting for the next line can time out without cancelling it. That
    task is cancelled when this generator is (the browser went away), so `lines` ends the way it did when it was the
    response body itself; an exception it raises is raised here.
    """
    if interval <= 0:
        async for line in lines:
            yield line
        return

    queue: asyncio.Queue = asyncio.Queue()
    end = object()

    async def read() -> None:
        try:
            async for line in lines:
                queue.put_nowait(line)
        finally:
            queue.put_nowait(end)

    reader = asyncio.ensure_future(read())
    try:
        while True:
            try:
                line = await asyncio.wait_for(queue.get(), interval)
            except TimeoutError:
                yield HEARTBEAT
                continue
            if line is end:
                break
            yield line
        await reader  # raises what `lines` raised
    finally:
        if not reader.done():
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
