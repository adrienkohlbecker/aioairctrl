"""CoAP client for Philips air purifiers."""

import asyncio
import json
import logging
import os

from aiocoap import (
    NON,
    Context,
    Message,
)
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

    # Every request is NON (no CoAP retransmission), so a lost request or reply
    # never arrives late: waiting longer than a round trip only delays the retry.
    REPLY_TIMEOUT: float = 5
    SYNC_ATTEMPTS = 3
    # How long a confirmed command waits for the status notification that
    # reports it, and how often it is sent before giving up.
    CONFIRM_TIMEOUT: float = 5
    CONFIRM_ATTEMPTS = 2

    def __init__(self, host, port=5683):
        self.host = host
        self.port = port
        # Both are set together in _init; kept as None so shutdown() is safe to
        # call even if _init never completed.
        self._client_context: Context | None = None
        self._encryption_context: EncryptionContext | None = None
        # The latest status an active observe_status() received, and the
        # confirmed commands waiting for a notification that reports them.
        self._observed_status: dict | None = None
        self._confirmations: dict[asyncio.Future, dict] = {}

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

    async def shutdown(self) -> None:
        if self._client_context:
            await self._client_context.shutdown()

    async def _request(self, request: Message) -> Message:
        """Send request and return its reply, raising TimeoutError if none arrives."""
        return await asyncio.wait_for(self._ctx.request(request).response, self.REPLY_TIMEOUT)

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
        """Return (state_reported, max_age) for the current device status."""
        logger.debug("retrieving status")
        request = Message(
            code=GET,
            mtype=NON,
            uri=f"coap://{self.host}:{self.port}{self.STATUS_PATH}",
        )
        # observe=0 registers a CoAP observation; the first response carries
        # the current state, which is all we consume here.
        request.opt.observe = 0
        response = await self._request(request)
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
        """Async generator that yields state_reported dicts as the device pushes updates."""

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
        try:
            response = await requester.response
            yield self._observed(decrypt_status(response))
            if observation is not None:
                async for response in observation:
                    try:
                        status = decrypt_status(response)
                    except (DigestMismatchException, ValueError) as err:
                        logger.warning("dropping unreadable status notification: %r", err)
                        continue
                    yield self._observed(status)
        finally:
            self._observed_status = None
            # Cancel the observation when the caller stops iterating, so the
            # device stops sending notifications and aiocoap frees its resources.
            if observation is not None:
                observation.cancel()

    def _observed(self, status: dict) -> dict:
        """Record a notified status and resolve the confirmations it reports."""
        self._observed_status = status
        for confirmed, data in self._confirmations.items():
            if not confirmed.done() and _reports(status, data):
                confirmed.set_result(None)
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
        devices accept a command and then ignore it, so with confirm and an
        active observe_status() on this client, success also needs a status
        notification reporting the new values; the command is resent once if
        none arrives. A command the device already reports, or one whose keys
        the device never reports, is not waited on: the device sends no
        notification when nothing changes.
        """
        for attempt in range(1, self.CONFIRM_ATTEMPTS + 1):
            status = self._observed_status
            if not confirm or status is None or not any(k in status for k in data):
                return await self._send_control(data, retry_count, resync)
            if _reports(status, data):
                return await self._send_control(data, retry_count, resync)
            # Registered before sending, as the notification can beat the reply.
            confirmed = asyncio.get_running_loop().create_future()
            self._confirmations[confirmed] = data
            try:
                if not await self._send_control(data, retry_count, resync):
                    return False
                await asyncio.wait_for(confirmed, self.CONFIRM_TIMEOUT)
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

    async def _send_control(self, data: dict, retry_count: int, resync: bool) -> bool:
        """Send a control command until the device accepts it, retrying on failure.

        A failed or missing reply is retried. On the first failure, if
        resync=True, the client re-syncs the encryption key before retrying.
        Subsequent failures retry without re-syncing (stale key is unlikely to
        be the cause after one resync).
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


def _reports(status: dict, data: dict) -> bool:
    """Whether status reports every key of data that it carries, with data's value."""
    return all(status[k] == v for k, v in data.items() if k in status)
