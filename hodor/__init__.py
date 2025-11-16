"""Hodor - AI-powered code review agent that finds bugs and security issues."""

import warnings

from . import _tty as _terminal_safety  # noqa: F401
from .agent import review_pr
from .cli import main

# Suppress litellm's asyncio deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="litellm")

__version__ = "0.1.0"
__all__ = ["review_pr", "main"]
