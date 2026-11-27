"""with_heartbeat: a silent chat stream sends SSE comments, so a proxy's idle timeout does not close it mid-turn."""

import asyncio

import pytest
from app.core.sse import HEARTBEAT, with_heartbeat


async def collect(lines, interval):
    return [line async for line in with_heartbeat(lines, interval)]


async def quiet_between(*gaps_and_lines):
    """Yield the strings, sleeping for each number."""
    for item in gaps_and_lines:
        if isinstance(item, str):
            yield item
        else:
            await asyncio.sleep(item)


async def test_a_silence_is_filled_with_heartbeats_and_every_line_comes_through_in_order():
    out = await collect(quiet_between("data: 1\n\n", 0.35, "data: 2\n\n", "data: 3\n\n"), 0.1)
    data = [line for line in out if line != HEARTBEAT]
    assert data == ["data: 1\n\n", "data: 2\n\n", "data: 3\n\n"]
    assert out[0] == "data: 1\n\n" and out[-1] == "data: 3\n\n"
    assert 2 <= out.count(HEARTBEAT) <= 4                   # ~one per 0.1 s of the 0.35 s silence


async def test_a_stream_that_keeps_talking_gets_no_heartbeat():
    out = await collect(quiet_between(*[x for i in range(10) for x in (0.01, f"data: {i}\n\n")]), 0.2)
    assert HEARTBEAT not in out and len(out) == 10


async def test_a_stream_silent_before_its_first_line_gets_heartbeats_too():
    """The compaction before a turn can take a while before anything is sent."""
    out = await collect(quiet_between(0.25, "data: 1\n\n"), 0.1)
    assert out[0] == HEARTBEAT and out[-1] == "data: 1\n\n"


async def test_an_error_in_the_stream_is_raised_after_the_lines_before_it():
    async def failing():
        yield "data: 1\n\n"
        raise ValueError("db down")

    got = []
    with pytest.raises(ValueError, match="db down"):
        async for line in with_heartbeat(failing(), 0.1):
            got.append(line)
    assert got == ["data: 1\n\n"]


async def test_when_the_browser_goes_away_the_stream_behind_it_ends_too():
    """Starlette cancels the response body when the client disconnects. The stream behind the heartbeat must see that
    too: its `finally` is where chat_service.guarded gives the conversation lock back."""
    ended = asyncio.Event()

    async def endless():
        try:
            yield "data: 1\n\n"
            await asyncio.sleep(60)
            yield "data: never\n\n"
        finally:
            ended.set()

    browser = asyncio.ensure_future(collect(endless(), 0.05))
    await asyncio.sleep(0.2)
    browser.cancel()
    await asyncio.gather(browser, return_exceptions=True)
    await asyncio.wait_for(ended.wait(), 1)


async def test_zero_turns_it_off():
    out = await collect(quiet_between("data: 1\n\n", 0.2, "data: 2\n\n"), 0)
    assert out == ["data: 1\n\n", "data: 2\n\n"]
