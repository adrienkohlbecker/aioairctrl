"""Client behaviour against a fake device: lost replies, ignored commands, stale notifications."""

import asyncio
import json
import time

import pytest
from aiocoap.error import NetworkError

from aioairctrl.coap.client import Client
from aioairctrl.coap.encryption import EncryptionContext


class FakeDevice:
    """Answer sync and control requests and push status notifications.

    `replies` scripts each control reply ("success", "failed", None for a
    lost reply, "applied_lost" for a lost reply to an accepted command, or
    "unreachable" for a network error); `applies` scripts whether each
    accepted command changes the device; `lost_syncs` and `lost_statuses`
    count sync and status requests to drop; `reply_delay` holds control replies back, so the
    notification arrives first; `silent` drops status requests.
    """

    def __init__(
        self,
        status,
        replies=(),
        applies=(),
        lost_syncs=0,
        lost_statuses=0,
        reply_delay=0.0,
        silent=False,
    ):
        self.status = dict(status)
        self.replies = list(replies)
        self.applies = list(applies)
        self.lost_syncs = lost_syncs
        self.lost_statuses = lost_statuses
        self.reply_delay = reply_delay
        self.silent = silent
        self.controls = []
        self.syncs = 0
        self.notifications: asyncio.Queue | None = None
        self._enc = EncryptionContext()
        self._enc.set_client_key("00000001")

    def notify(self):
        payload = json.dumps({"state": {"reported": self.status}})
        assert self.notifications is not None
        self.notifications.put_nowait(FakeMessage(self._enc.encrypt(payload).encode()))

    def request(self, message):
        loop = asyncio.get_running_loop()
        requester = FakeRequester(loop.create_future())
        path = message.opt.uri_path
        if path == ("sys", "dev", "sync"):
            self.syncs += 1
            if self.lost_syncs:
                self.lost_syncs -= 1
            else:
                requester.response.set_result(FakeMessage(b"0000ABCD"))
        elif path == ("sys", "dev", "status") and self.silent:
            pass
        elif path == ("sys", "dev", "status") and self.lost_statuses:
            self.lost_statuses -= 1
        elif path == ("sys", "dev", "status"):
            self.notifications = asyncio.Queue()
            requester.observation = FakeObservation(self.notifications)
            self.notify()
            requester.response.set_result(self.notifications.get_nowait())
        elif path == ("sys", "dev", "control"):
            desired = json.loads(EncryptionContext().decrypt(message.payload.decode()))
            data = {
                k: v
                for k, v in desired["state"]["desired"].items()
                if k not in ("CommandType", "DeviceId", "EnduserId")
            }
            self.controls.append(data)
            reply = self.replies.pop(0) if self.replies else "success"
            if reply == "unreachable":
                requester.response.set_exception(NetworkError("host unreachable"))
            elif reply not in (None, "applied_lost"):
                message = FakeMessage(json.dumps({"status": reply}).encode())
                loop.call_later(self.reply_delay, requester.response.set_result, message)
            if reply in ("success", "applied_lost") and (
                self.applies.pop(0) if self.applies else True
            ):
                changed = any(self.status.get(k) != v for k, v in data.items())
                self.status.update(data)
                if changed and self.notifications is not None:
                    self.notify()
        return requester


class FakeMessage:
    def __init__(self, payload):
        self.payload = payload
        self.opt = type("Options", (), {"max_age": None})()


class FakeRequester:
    def __init__(self, response):
        self.response = response
        self.observation: FakeObservation | None = None


class FakeObservation:
    def __init__(self, queue):
        self._queue = queue

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._queue.get()

    def cancel(self):
        pass


def make_client(device):
    client = Client("192.0.2.1")
    client.REPLY_TIMEOUT = 0.01
    client.CONFIRM_TIMEOUT = 0.05
    client._client_context = device  # type: ignore[assignment]
    client._encryption_context = EncryptionContext()
    client._enc.set_client_key("00000001")
    return client


async def observed(client, device):
    """Start observing and return the task once the first status is in."""
    seen = asyncio.Event()

    async def observe():
        async for _status in client.observe_status():
            seen.set()

    task = asyncio.create_task(observe())
    await asyncio.wait_for(seen.wait(), 1)
    return task


def run(coroutine):
    return asyncio.run(asyncio.wait_for(coroutine, 5))


def test_sync_retries_a_lost_reply():
    device = FakeDevice({}, lost_syncs=2)
    client = make_client(device)
    run(client._sync())
    assert device.syncs == 3


def test_sync_raises_after_every_attempt_is_lost():
    device = FakeDevice({}, lost_syncs=3)
    client = make_client(device)
    with pytest.raises(TimeoutError):
        run(client._sync())
    assert device.syncs == 3


def test_control_retries_a_lost_reply():
    device = FakeDevice({"pwr": 0}, replies=[None, "success"])
    client = make_client(device)
    assert run(client.set_control_values({"pwr": 1})) is True
    assert device.controls == [{"pwr": 1}, {"pwr": 1}]


def test_control_fails_when_every_reply_is_lost():
    device = FakeDevice({"pwr": 0}, replies=[None] * 3)
    client = make_client(device)
    assert run(client.set_control_values({"pwr": 1}, retry_count=2)) is False
    assert len(device.controls) == 3


def test_unconfirmed_control_trusts_the_reply():
    device = FakeDevice({"mode": 17}, applies=[False])
    client = make_client(device)
    assert run(client.set_control_values({"mode": 0})) is True
    assert device.status["mode"] == 17


async def _confirmed(device, data):
    client = make_client(device)
    task = await observed(client, device)
    try:
        return await client.set_control_values(data, confirm=True)
    finally:
        task.cancel()


def test_confirmed_control_returns_once_reported():
    device = FakeDevice({"pwr": 1, "mode": 17})
    assert run(_confirmed(device, {"mode": 0})) is True
    assert device.controls == [{"mode": 0}]


def test_confirmed_control_resends_an_ignored_command():
    device = FakeDevice({"pwr": 1, "mode": 17}, applies=[False, True])
    assert run(_confirmed(device, {"mode": 0})) is True
    assert device.controls == [{"mode": 0}, {"mode": 0}]
    assert device.status["mode"] == 0


def test_confirmed_control_fails_when_never_applied():
    device = FakeDevice({"pwr": 1, "mode": 17}, applies=[False, False])
    assert run(_confirmed(device, {"mode": 0})) is False
    assert device.controls == [{"mode": 0}, {"mode": 0}]


def test_confirmed_control_ignores_a_stale_notification():
    # The device pushes its old status first, then applies the command.
    async def scenario():
        device = FakeDevice({"pwr": 1, "mode": 17}, applies=[False])
        client = make_client(device)
        task = await observed(client, device)

        async def apply_late():
            await asyncio.sleep(0.01)
            device.notify()  # stale: still mode 17
            await asyncio.sleep(0.01)
            device.status["mode"] = 0
            device.notify()

        late = asyncio.create_task(apply_late())
        try:
            return await client.set_control_values({"mode": 0}, confirm=True), device
        finally:
            await late
            task.cancel()

    result, device = run(scenario())
    assert result is True
    assert device.controls == [{"mode": 0}]


def test_confirmed_control_does_not_wait_for_an_unchanged_value():
    # The device sends no notification when nothing changes.
    device = FakeDevice({"pwr": 1, "mode": 0})
    assert run(_confirmed(device, {"mode": 0})) is True
    assert device.controls == [{"mode": 0}]


def test_confirmed_control_does_not_wait_for_unreported_keys():
    device = FakeDevice({"pwr": 1}, applies=[False])
    assert run(_confirmed(device, {"unreported": 1})) is True
    assert device.controls == [{"unreported": 1}]


def test_confirm_without_observation_trusts_the_reply():
    device = FakeDevice({"pwr": 1, "mode": 17}, applies=[False])
    client = make_client(device)
    assert run(client.set_control_values({"mode": 0}, confirm=True)) is True
    assert device.controls == [{"mode": 0}]


def test_observation_end_stops_confirming():
    async def scenario():
        device = FakeDevice({"pwr": 1, "mode": 17}, applies=[False])
        client = make_client(device)
        task = await observed(client, device)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return await client.set_control_values({"mode": 0}, confirm=True)

    assert run(scenario()) is True


def test_confirmed_control_resends_after_only_a_stale_notification():
    async def scenario():
        device = FakeDevice({"pwr": 1, "mode": 17}, applies=[False, True])
        client = make_client(device)
        task = await observed(client, device)
        device.notify()  # the old mode again, not a confirmation
        try:
            return await client.set_control_values({"mode": 0}, confirm=True), device
        finally:
            task.cancel()

    result, device = run(scenario())
    assert result is True
    assert device.controls == [{"mode": 0}, {"mode": 0}]


def test_confirmed_control_catches_a_notification_before_the_reply():
    device = FakeDevice({"pwr": 1, "mode": 17}, reply_delay=0.005)
    assert run(_confirmed(device, {"mode": 0})) is True
    assert device.controls == [{"mode": 0}]


def test_resend_does_not_undo_a_newer_command():
    async def scenario():
        device = FakeDevice({"pwr": 1, "mode": 17}, applies=[False, True, True])
        client = make_client(device)
        task = await observed(client, device)
        try:
            first = asyncio.create_task(client.set_control_values({"mode": 0}, confirm=True))
            await asyncio.sleep(0)
            second = asyncio.create_task(client.set_control_values({"mode": 18}, confirm=True))
            return await asyncio.gather(first, second), device
        finally:
            task.cancel()

    results, device = run(scenario())
    assert results == [True, True]
    assert device.controls == [{"mode": 0}, {"mode": 0}, {"mode": 18}]
    assert device.status["mode"] == 18


def test_editing_the_yielded_status_does_not_fake_a_confirmation():
    async def scenario():
        device = FakeDevice({"pwr": 1, "mode": 17}, applies=[False, False])
        client = make_client(device)
        statuses = []
        seen = asyncio.Event()

        async def observe():
            async for status in client.observe_status():
                statuses.append(status)
                seen.set()

        task = asyncio.create_task(observe())
        await asyncio.wait_for(seen.wait(), 1)
        statuses[-1]["mode"] = 0  # an optimistic write by the caller
        try:
            return await client.set_control_values({"mode": 0}, confirm=True), device
        finally:
            task.cancel()

    result, device = run(scenario())
    assert result is False
    assert device.controls == [{"mode": 0}, {"mode": 0}]


def test_a_notification_without_the_keys_does_not_confirm():
    async def scenario():
        device = FakeDevice({"pwr": 1, "mode": 17}, applies=[False, False])
        client = make_client(device)
        task = await observed(client, device)

        async def unrelated():
            await asyncio.sleep(0.01)
            assert device.notifications is not None
            payload = json.dumps({"state": {"reported": {"pm25": 4}}})
            device.notifications.put_nowait(FakeMessage(device._enc.encrypt(payload).encode()))

        side = asyncio.create_task(unrelated())
        try:
            return await client.set_control_values({"mode": 0}, confirm=True)
        finally:
            await side
            task.cancel()

    assert run(scenario()) is False


def test_observation_ending_mid_confirmation_fails():
    async def scenario():
        device = FakeDevice({"pwr": 1, "mode": 17}, applies=[False])
        client = make_client(device)
        task = await observed(client, device)

        async def stop():
            await asyncio.sleep(0.01)
            task.cancel()

        client.CONFIRM_TIMEOUT = 2
        stopper = asyncio.create_task(stop())
        start = time.monotonic()
        result = await client.set_control_values({"mode": 0}, confirm=True)
        await stopper
        return result, device, time.monotonic() - start

    result, device, elapsed = run(scenario())
    assert result is False
    assert device.controls == [{"mode": 0}]
    assert elapsed < 1  # failed when the observation ended, not at CONFIRM_TIMEOUT


def test_control_retries_an_unreachable_network():
    device = FakeDevice({"pwr": 0}, replies=["unreachable", "success"])
    client = make_client(device)
    assert run(client.set_control_values({"pwr": 1})) is True
    assert len(device.controls) == 2


def test_lost_control_reply_resyncs_once():
    device = FakeDevice({"pwr": 0}, replies=[None, None, "success"])
    client = make_client(device)
    assert run(client.set_control_values({"pwr": 1})) is True
    assert device.syncs == 1


def raises_promptly(coroutine):
    """Assert coroutine raises the client's TimeoutError, well before run()'s own."""
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        run(coroutine)
    assert time.monotonic() - start < 1


def test_get_status_times_out():
    device = FakeDevice({"pwr": 0}, silent=True)
    client = make_client(device)
    raises_promptly(client.get_status())


def test_observation_times_out_without_a_first_reply():
    async def scenario():
        client = make_client(FakeDevice({"pwr": 0}, silent=True))
        async for _status in client.observe_status():
            pass

    raises_promptly(scenario())


def test_a_notification_missing_a_command_key_does_not_confirm():
    """A notification reporting the power but not the mode of a power-and-mode command."""

    async def scenario():
        device = FakeDevice({"pwr": 1, "mode": 17}, applies=[False, False])
        client = make_client(device)
        task = await observed(client, device)

        async def partial():
            await asyncio.sleep(0.01)
            assert device.notifications is not None
            payload = json.dumps({"state": {"reported": {"pwr": 1, "pm25": 4}}})
            device.notifications.put_nowait(FakeMessage(device._enc.encrypt(payload).encode()))

        side = asyncio.create_task(partial())
        try:
            return await client.set_control_values({"pwr": 1, "mode": 0}, confirm=True)
        finally:
            await side
            task.cancel()

    assert run(scenario()) is False


def test_a_reported_command_stops_resending_lost_replies():
    device = FakeDevice({"pwr": 1, "mode": 17}, replies=["applied_lost"] + [None] * 10)
    assert run(_confirmed(device, {"mode": 0})) is True
    assert device.controls == [{"mode": 0}]
    assert device.syncs == 0


def test_get_status_retries_a_lost_reply():
    device = FakeDevice({"pwr": 0}, lost_statuses=2)
    client = make_client(device)
    assert run(client.get_status()) == ({"pwr": 0}, 60)


def test_shutdown_twice_shuts_the_context_down_once():
    class Context:
        shutdowns = 0

        async def shutdown(self):
            self.shutdowns += 1

    context = Context()
    client = Client("192.0.2.1")
    client._client_context = context  # type: ignore[assignment]
    run(client.shutdown())
    run(client.shutdown())
    assert context.shutdowns == 1
