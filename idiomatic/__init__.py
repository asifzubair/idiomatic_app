# This file makes the 'idiomatic' directory a Python package.

# Expose the main configuration class and application runner function
from .config import IdiomaticConfig
from .app import run_app

__all__ = [
    "IdiomaticConfig",
    "run_app",
]
