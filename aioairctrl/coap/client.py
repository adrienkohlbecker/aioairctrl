"""CoAP client for Philips air purifiers."""

import asyncio
import json
import logging
import os
import struct

from aiocoap import (
    NON,
    Context,
    Message,
)
from aiocoap.error import NetworkError
from aiocoap.numbers.codes import (
    GET,
    POST,
)

from aioairctrl.coap.encryption import DigestMismatchException, EncryptionContext

logger = logging.getLogger(__name__)


class Client:
    STATUS_PATH = "/sys/dev/status"
    CONTROL_PATH = "/sys/dev/control"
    SYNC_PATH = "/sys/dev/sync"

    # Every request is NON, so CoAP never retransmits a lost request or reply:
    # the client gives up after REPLY_TIMEOUT and retries itself. A reply that
    # arrives later is ignored, though the device may have applied the command.
    REPLY_TIMEOUT: float = 5
    SYNC_ATTEMPTS = 3
    # How long a confirmed command waits for the status notification that
    # reports it, and how often it is sent before giving up.
    CONFIRM_TIMEOUT: float = 5
    CONFIRM_ATTEMPTS = 2
    # How long ping() waits for the device's reset.
    PING_TIMEOUT: float = 2

    def __init__(self, host, port=5683):
        self.host = host
        self.port = port
        # Both are set together in _init; kept as None so shutdown() is safe to
        # call even if _init never completed.
        self._client_context: Context | None = None
        self._encryption_context: EncryptionContext | None = None
        # Every key notified to this client, with its latest value, kept apart
        # from the yielded statuses (the caller may edit those) and across
        # observations, as the device notifies only changes; and the confirmed
        # commands waiting for a notification that reports them.
        self._observed_status: dict | None = None
        self._confirmations: dict[asyncio.Future, dict] = {}
        # Set while an observe_status() registration stands, even before its
        # first notification. A replacing registration bumps the generation,
        # so the replaced one's cleanup leaves the new state alone.
        self._observing = False
        self._observation_generation = 0
        # One command at a time: a resent command must not undo a newer one.
        self._control_lock = asyncio.Lock()

    @property
    def _ctx(self) -> Context:
        if self._client_context is None:
            raise RuntimeError("Client not initialized; use Client.create()")
        return self._client_context

    @property
    def _enc(self) -> EncryptionContext:
        if self._encryption_context is None:
            raise RuntimeError("Client not initialized; use Client.create()")
        return self._encryption_context

    async def _init(self):
        self._client_context = await Context.create_client_context(transports=["simple6"])
        self._encryption_context = EncryptionContext()
        try:
            await self._sync()
        except BaseException:
            # Ensure the aiocoap context is always cleaned up, even on
            # cancellation (asyncio.CancelledError is a BaseException).
            await self._client_context.shutdown()
            raise

    @classmethod
    async def create(cls, *args, **kwargs):
        """Async factory — use instead of the constructor."""
        obj = cls(*args, **kwargs)
        await obj._init()
        return obj

    def remember_status(self, status: dict) -> None:
        """Seed the status confirmations compare against, for a new client.

        Pass the status a previous client was notified, so that a command for
        values the device already has is not waited on: the device notifies
        nothing when nothing changes.
        """
        if self._observed_status is None:
            self._observed_status = dict(status)

    @property
    def observing(self) -> bool:
        """Whether an observe_status() registration stands, notified or not."""
        return self._observing

    async def ping(self) -> bool:
        """Return whether the device answers an empty CoAP ping, within PING_TIMEOUT.

        The device answers a confirmable empty message with a reset at once,
        unlike a status request, which it answers only once the status
        changes, so this is the liveness check. It uses its own socket: a ping
        sent through aiocoap that goes unanswered would fail every pending
        request to the device, observations included.
        """
        loop = asyncio.get_running_loop()
        mid = int.from_bytes(os.urandom(2), "big")
        answered = loop.create_future()

        class Protocol(asyncio.DatagramProtocol):
            def datagram_received(self, data, addr):
                # RST (type 3) carrying our message ID.
                if (
                    len(data) >= 4
                    and (data[0] >> 4) & 3 == 3
                    and data[2:4] == mid.to_bytes(2, "big")
                ):
                    if not answered.done():
                        answered.set_result(True)

            def error_received(self, exc):
                # ICMP unreachable on the connected socket: the device is gone.
                if not answered.done():
                    answered.set_result(False)

        try:
            transport, _ = await loop.create_datagram_endpoint(
                Protocol, remote_addr=(self.host, self.port)
            )
        except OSError:
            # No route to the device, for one.
            return False
        try:
            transport.sendto(struct.pack("!BBH", 0x40, 0x00, mid))  # ver 1, CON, no token, empty
            return await asyncio.wait_for(answered, self.PING_TIMEOUT)
        except TimeoutError:
            return False
        finally:
            transport.close()

    async def shutdown(self) -> None:
        # aiocoap's Context.shutdown() raises when called twice, so forget the
        # context before shutting it down.
        context, self._client_context = self._client_context, None
        if context:
            await context.shutdown()

    async def _request(self, request: Message) -> Message:
        """Send request and return its reply.

        Raises TimeoutError when no reply arrives in time, or when the network
        reports the device unreachable (aiocoap's NetworkError), since both
        mean the request may be retried.
        """
        try:
            return await asyncio.wait_for(self._ctx.request(request).response, self.REPLY_TIMEOUT)
        except NetworkError as ex:
            raise TimeoutError(f"{self.host} unreachable: {ex}") from ex

    async def _sync(self):
        """Exchange a nonce with the device to obtain the initial client key.

        The client sends a random 4-byte hex string; the device responds with
        the client key that must be used for all subsequent encrypted messages.
        A device in Wi-Fi power save can miss the request, so it is retried
        before raising TimeoutError.
        """
        for attempt in range(1, self.SYNC_ATTEMPTS + 1):
            logger.debug("syncing (attempt %d/%d)", attempt, self.SYNC_ATTEMPTS)
            sync_request = os.urandom(4).hex().upper()
            request = Message(
                code=POST,
                mtype=NON,
                uri=f"coap://{self.host}:{self.port}{self.SYNC_PATH}",
                payload=sync_request.encode(),
            )
            try:
                response = await self._request(request)
            except TimeoutError:
                continue
            client_key = response.payload.decode()
            logger.debug("synced: %s", client_key)
            self._enc.set_client_key(client_key)
            return
        raise TimeoutError(f"{self.host} did not answer {self.SYNC_ATTEMPTS} sync requests")

    async def get_status(self):
        """Return (state_reported, max_age) for the current device status.

        A lost reply is retried like a sync, before raising TimeoutError.
        """
        for attempt in range(1, self.SYNC_ATTEMPTS + 1):
            logger.debug("retrieving status (attempt %d/%d)", attempt, self.SYNC_ATTEMPTS)
            request = Message(
                code=GET,
                mtype=NON,
                uri=f"coap://{self.host}:{self.port}{self.STATUS_PATH}",
            )
            # observe=0 registers a CoAP observation; the first response carries
            # the current state, which is all we consume here.
            request.opt.observe = 0
            try:
                response = await self._request(request)
            except TimeoutError:
                continue
            break
        else:
            raise TimeoutError(f"{self.host} did not answer {self.SYNC_ATTEMPTS} status requests")
        payload_encrypted = response.payload.decode()
        payload = self._enc.decrypt(payload_encrypted)
        logger.debug("status: %s", payload)
        state_reported = json.loads(payload)
        max_age = 60
        if response.opt.max_age is not None:
            max_age = response.opt.max_age
            logger.debug("max age = %s", max_age)
        return state_reported["state"]["reported"], max_age

    async def observe_status(self):
        """Async generator that yields state_reported dicts as the device pushes updates.

        The device answers the registration only with its next notification,
        sent when its status changes, which can take hours on a quiet device,
        so the first status is awaited without a timeout; use ping() to tell a
        quiet device from a gone one. A new registration replaces the previous
        one from the same address.
        """

        def decrypt_status(response):
            payload_encrypted = response.payload.decode()
            payload = self._enc.decrypt(payload_encrypted)
            logger.debug("observation status: %s", payload)
            status = json.loads(payload)
            return status["state"]["reported"]

        logger.debug("observing status")
        request = Message(
            code=GET,
            mtype=NON,
            uri=f"coap://{self.host}:{self.port}{self.STATUS_PATH}",
        )
        request.opt.observe = 0
        requester = self._ctx.request(request)
        observation = requester.observation
        self._observation_generation += 1
        generation = self._observation_generation
        self._observing = True
        try:
            responses = _chain(await requester.response, observation)
            async for response in responses:
                try:
                    status = decrypt_status(response)
                except (DigestMismatchException, ValueError) as err:
                    logger.warning("dropping unreadable status notification: %r", err)
                    continue
                yield self._observed(status)
        finally:
            if generation == self._observation_generation:
                self._observing = False
                for confirmed in self._confirmations:
                    if not confirmed.done():
                        confirmed.set_result(False)
            # Cancel the observation when the caller stops iterating, so the
            # device stops sending notifications and aiocoap frees its resources.
            if observation is not None:
                observation.cancel()

    def _observed(self, status: dict) -> dict:
        """Record a notified status and resolve the confirmations it reports.

        Statuses are merged, so a notification that omits a key the device
        reported before neither drops it from a confirmation nor confirms it.
        """
        self._observed_status = observed = {**(self._observed_status or {}), **status}
        for confirmed, expected in self._confirmations.items():
            if not confirmed.done() and _reports(observed, expected):
                confirmed.set_result(True)
        return status

    async def set_control_value(
        self, key, value, retry_count=5, resync=True, confirm=False
    ) -> bool:
        return await self.set_control_values(
            data={key: value}, retry_count=retry_count, resync=resync, confirm=confirm
        )

    async def set_control_values(
        self, data: dict, retry_count=5, resync=True, confirm=False
    ) -> bool:
        """Send a control command to the device, returning whether it took effect.

        Without confirm, success means the device accepted the command. Some
        devices accept a command and then ignore it, so with confirm, success
        also needs a status notification reporting the new values, which only
        an observe_status() running on this client receives; the command is
        resent once if none arrives, and fails if that observation ends first.
        A notification confirms the command only if it reports every key of
        it that the device reported before, or every key when nothing was
        notified yet. Confirmation is skipped, and acceptance alone counts,
        when no observation runs, when the device does not report the
        command's keys, or when it already reports the values: the device
        sends nothing when nothing changes. For that reason a command sent
        before the first notification, for values the device already has,
        fails. Values compare with ==, so they must be sent in the type the
        device reports. Commands run one at a time, so a resend never undoes
        a newer command.
        """
        async with self._control_lock:
            if not confirm or not self._observing:
                return await self._send_control(data, retry_count, resync)
            status = self._observed_status
            if status is None:
                return await self._send_confirmed(data, dict(data), retry_count, resync)
            expected = {k: data[k] for k in data if k in status}
            if not expected or _reports(status, expected):
                return await self._send_control(data, retry_count, resync)
            return await self._send_confirmed(data, expected, retry_count, resync)

    async def _send_confirmed(
        self, data: dict, expected: dict, retry_count: int, resync: bool
    ) -> bool:
        """Send data until a notification reports the expected values."""
        for attempt in range(1, self.CONFIRM_ATTEMPTS + 1):
            if not self._observing:
                logger.warning("%s stopped notifying before confirming %s", self.host, data)
                return False
            # Registered before sending, as the notification can beat the reply.
            # Resolves to True once reported, or False when the observation ends.
            confirmed: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            self._confirmations[confirmed] = expected
            try:
                # A device whose replies get lost may still apply the command
                # and report it, which settles the command.
                if (
                    not await self._send_control(data, retry_count, resync, confirmed)
                    and not confirmed.done()
                ):
                    return False
                if not await asyncio.wait_for(confirmed, self.CONFIRM_TIMEOUT):
                    logger.warning("%s stopped notifying before confirming %s", self.host, data)
                    return False
                return True
            except TimeoutError:
                logger.warning(
                    "%s accepted %s but reported no change within %ss (attempt %d/%d)",
                    self.host,
                    data,
                    self.CONFIRM_TIMEOUT,
                    attempt,
                    self.CONFIRM_ATTEMPTS,
                )
            finally:
                del self._confirmations[confirmed]
        return False

    async def _send_control(
        self, data: dict, retry_count: int, resync: bool, settled: asyncio.Future | None = None
    ) -> bool:
        """Send a control command until the device accepts it, retrying on failure.

        A failed or missing reply is retried. On the first failure, if
        resync=True, the client re-syncs the encryption key before retrying.
        Subsequent failures retry without re-syncing (stale key is unlikely to
        be the cause after one resync). Sending stops, returning False, once
        the settled future is done.
        """
        state_desired = {
            "state": {
                "desired": {
                    "CommandType": "app",
                    "DeviceId": "",
                    "EnduserId": "",
                    **data,
                }
            }
        }
        payload = json.dumps(state_desired)
        logger.debug("REQUEST: %s", payload)
        for attempt in range(retry_count + 1):
            if settled is not None and settled.done():
                return False
            payload_encrypted = self._enc.encrypt(payload)
            request = Message(
                code=POST,
                mtype=NON,
                uri=f"coap://{self.host}:{self.port}{self.CONTROL_PATH}",
                payload=payload_encrypted.encode(),
            )
            try:
                response = await self._request(request)
            except TimeoutError:
                logger.debug("set_control_value got no reply")
            else:
                logger.debug("RESPONSE: %s", response.payload)
                result = json.loads(response.payload)
                if result.get("status") == "success":
                    return True
            if settled is not None and settled.done():
                return False
            if attempt == 0 and resync:
                logger.debug("set_control_value failed, resyncing...")
                try:
                    await self._sync()
                except TimeoutError:
                    logger.debug("resync got no reply")
            else:
                logger.debug(
                    "set_control_value failed, retrying (attempt %d/%d)...",
                    attempt + 1,
                    retry_count,
                )
        logger.error("set_control_value failed: %s", data)
        return False


def _reports(status: dict, expected: dict) -> bool:
    """Whether status carries every key of expected, with its value."""
    return all(k in status and status[k] == v for k, v in expected.items())


async def _chain(first, rest):
    """Yield first, then everything rest yields, if there is a rest."""
    yield first
    if rest is not None:
        async for item in rest:
            yield item
