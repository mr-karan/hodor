"""Terminal safety helpers for Hodor runtimes.

Key responsibilities:
- Disable prompt_toolkit CPR handshakes (which leave ``^[##;#R`` junk behind).
- Drain any stray control-sequence replies that might still be queued in stdin
  before control returns to the user's shell.
"""

from __future__ import annotations

import atexit
import fcntl
import os
import sys
from typing import Final

_PROMPT_TOOLKIT_CPR_FLAG: Final[str] = "1"
_PROMPT_TOOLKIT_ENV: Final[str] = "PROMPT_TOOLKIT_NO_CPR"


def disable_prompt_toolkit_cpr() -> None:
    """Disable prompt_toolkit cursor-position queries for the current process."""
    if os.environ.get(_PROMPT_TOOLKIT_ENV) != _PROMPT_TOOLKIT_CPR_FLAG:
        os.environ[_PROMPT_TOOLKIT_ENV] = _PROMPT_TOOLKIT_CPR_FLAG


def _drain_pending_terminal_input() -> None:
    """Drop unread TTY input (e.g., CPR replies) without blocking."""
    stdin = sys.stdin
    if not stdin.isatty():
        return

    try:
        fd = stdin.fileno()
    except (AttributeError, OSError):
        return

    try:
        original_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    except OSError:
        return

    try:
        fcntl.fcntl(fd, fcntl.F_SETFL, original_flags | os.O_NONBLOCK)
        while True:
            try:
                chunk = os.read(fd, 1024)
                if not chunk:
                    break
            except BlockingIOError:
                break
            except OSError:
                break
    finally:
        try:
            fcntl.fcntl(fd, fcntl.F_SETFL, original_flags)
        except OSError:
            pass


def restore_terminal_state() -> None:
    """Best-effort reset for the parent TTY."""
    _drain_pending_terminal_input()


# Apply safeguards immediately and again right before interpreter shutdown.
disable_prompt_toolkit_cpr()
atexit.register(restore_terminal_state)
