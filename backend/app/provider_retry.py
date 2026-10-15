"""One retry rule for every model call."""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

Result = TypeVar("Result")

TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})
RETRY_DELAYS_SECONDS = (1.0, 3.0)


def is_transient(error: BaseException) -> bool:
    """True when the same request may well succeed if it is simply sent again."""
    if getattr(error, "retryable", False):
        return True
    code = getattr(error, "code", None)
    if isinstance(code, int) and code in TRANSIENT_STATUS:
        return True
    return isinstance(error, TimeoutError) or "timeout" in type(error).__name__.casefold()


def _slow(error: BaseException) -> bool:
    """The failure already cost the full wait: a client timeout, or the provider's
    own deadline (504)."""
    return (
        getattr(error, "code", None) == 504
        or isinstance(error, TimeoutError)
        or "timeout" in type(error).__name__.casefold()
    )


def _overloaded(error: BaseException) -> bool:
    return is_transient(error) and not _slow(error)


async def call_with_retry(
    call: Callable[[], Awaitable[Result]],
    *,
    delays: tuple[float, ...] = RETRY_DELAYS_SECONDS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Result:
    """Run `call`, retrying only a transient provider failure."""
    for delay in (*delays, None):
        try:
            return await call()
        except Exception as error:
            if delay is None or not _overloaded(error):
                raise
            await sleep(delay + random.uniform(0, delay / 2))
    raise AssertionError("unreachable retry loop")
