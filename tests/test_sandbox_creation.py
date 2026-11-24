"""A conversation gets one sandbox, even when the model asks for several bash calls in the same step.

Seen on the test cluster: two bash calls started at once on a conversation with no sandbox each found none and each
created one, so the conversation had two sandboxes that do not share files."""

import asyncio
import contextlib

import pytest

from app.agents.tools import sandbox_manager as sm


class FakeClient:
    """The slice of AsyncSandboxClient that _get_or_create_sandbox uses, with a slow create like the real one."""

    def __init__(self, create_seconds: float = 0.05, fail_first: bool = False):
        self.sandboxes: dict[str, str] = {}              # label -> sandbox id
        self.created = 0
        self.create_seconds = create_seconds
        self.fail_first = fail_first

    async def list_all_sandboxes(self, namespace, label_selector):
        label = label_selector.split("=", 1)[1]
        return [self.sandboxes[label]] if label in self.sandboxes else []

    async def get_sandbox(self, sandbox_id, namespace):
        return sandbox_id

    async def create_sandbox(self, warmpool, namespace, sandbox_ready_timeout, labels):
        self.created += 1
        await asyncio.sleep(self.create_seconds)
        if self.fail_first and self.created == 1:
            raise RuntimeError("the pod did not become ready")
        sid = f"sb-{labels[sm._THREAD_LABEL]}-{self.created}"
        self.sandboxes[labels[sm._THREAD_LABEL]] = sid
        return type("Sandbox", (), {"claim_name": "claim-" + sid, "sandbox_id": sid})()


@pytest.fixture
def fake(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(sm, "_client", client)

    async def no_owner(label):
        return None

    async def touch(*a, **kw):
        return None

    @contextlib.asynccontextmanager
    async def free_slot(label, coord=None):
        yield

    monkeypatch.setattr(sm, "_owner_of", no_owner)
    monkeypatch.setattr(sm, "_touch_expiry", touch)
    monkeypatch.setattr(sm, "_admission", free_slot)
    return client


async def test_parallel_first_calls_make_one_sandbox(fake):
    results = await asyncio.gather(*(sm._get_or_create_sandbox("conv-1") for _ in range(5)))
    assert fake.created == 1
    assert len({getattr(r, "sandbox_id", r) for r in results}) == 1   # all five got the same sandbox


async def test_a_later_call_finds_it_without_creating(fake):
    await sm._get_or_create_sandbox("conv-2")
    await sm._get_or_create_sandbox("conv-2")
    assert fake.created == 1


async def test_different_conversations_do_not_wait_for_each_other(fake):
    fake.create_seconds = 0.3
    start = asyncio.get_running_loop().time()
    await asyncio.gather(sm._get_or_create_sandbox("conv-a"), sm._get_or_create_sandbox("conv-b"))
    assert fake.created == 2
    assert asyncio.get_running_loop().time() - start < 0.5        # in parallel, not one after the other


async def test_when_the_first_creation_fails_the_next_call_tries_again(fake):
    fake.fail_first = True
    first, second = await asyncio.gather(sm._get_or_create_sandbox("conv-3"), sm._get_or_create_sandbox("conv-3"), return_exceptions=True)
    assert isinstance(first, RuntimeError)
    assert not isinstance(second, Exception)                      # it was not left stuck behind the failure
    assert fake.created == 2


async def test_the_per_conversation_lock_does_not_pile_up(fake):
    await asyncio.gather(*(sm._get_or_create_sandbox(f"conv-{i}") for i in range(20)))
    await sm._get_or_create_sandbox("conv-0")
    assert sm._creating == {}


async def test_a_call_cancelled_while_waiting_leaves_nothing_behind(fake):
    fake.create_seconds = 0.2
    first = asyncio.ensure_future(sm._get_or_create_sandbox("conv-x"))
    await asyncio.sleep(0.02)
    second = asyncio.ensure_future(sm._get_or_create_sandbox("conv-x"))
    await asyncio.sleep(0.02)
    second.cancel()
    await first
    with contextlib.suppress(asyncio.CancelledError):
        await second
    assert fake.created == 1
    assert sm._creating == {}
