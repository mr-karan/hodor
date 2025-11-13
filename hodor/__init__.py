"""Hodor - AI-powered code review agent that finds bugs and security issues."""

from . import _tty as _terminal_safety  # noqa: F401
from .agent import review_pr
from .cli import main

__version__ = "0.1.0"
__all__ = ["review_pr", "main"]
