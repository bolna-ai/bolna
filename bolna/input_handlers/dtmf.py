"""Digit collection shared by handlers whose transport delivers one DTMF digit per event.

Digits are submitted as one string either on the terminator key (not appended, like Twilio's
finishOnKey) or once the caller pauses for the inter-digit timeout, so input without a '#'
still gets through.
"""

import asyncio
import os
from collections.abc import Callable

from bolna.helpers.logger_config import configure_logger

logger = configure_logger(__name__)

DTMF_TERMINATOR = "#"
# SIP_DTMF_INTERDIGIT_TIMEOUT_S is the name the sip-trunk handler shipped with; still honoured.
DTMF_INTERDIGIT_TIMEOUT_S = float(
    os.environ.get("DTMF_INTERDIGIT_TIMEOUT_S") or os.environ.get("SIP_DTMF_INTERDIGIT_TIMEOUT_S") or "3"
)


class DtmfAccumulator:
    """Collects digits and hands the finished string to `submit`."""

    def __init__(self, submit: Callable[[str], None], terminator: str = DTMF_TERMINATOR):
        self._submit = submit
        self._terminator = terminator
        self._digits = ""
        self._timer: asyncio.Task | None = None

    @property
    def digits(self) -> str:
        return self._digits

    def press(self, digit: str) -> None:
        if not digit:
            return
        if digit == self._terminator:
            logger.info("DTMF termination key pressed")
            self._cancel_timer()
            self.flush()
            return
        self._digits += digit
        self._restart_timer()

    def flush(self) -> None:
        if not self._digits:
            return
        digits, self._digits = self._digits, ""
        self._submit(digits)

    def close(self) -> None:
        """Stop the pending timer without submitting; used at handler teardown."""
        self._cancel_timer()

    def _cancel_timer(self) -> None:
        if self._timer and not self._timer.done():
            self._timer.cancel()
        self._timer = None

    def _restart_timer(self) -> None:
        self._cancel_timer()
        self._timer = asyncio.create_task(self._expire())

    async def _expire(self) -> None:
        try:
            await asyncio.sleep(DTMF_INTERDIGIT_TIMEOUT_S)
        except asyncio.CancelledError:
            return
        self._timer = None
        if self._digits:
            logger.info(f"DTMF inter-digit timeout, submitting '{self._digits}'")
            self.flush()
