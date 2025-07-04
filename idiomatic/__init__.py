# This file makes Python treat the 'idiomatic' directory as a package.

# This file makes Python treat the 'idiomatic' directory as a package.

# For easier imports from the package:
from .config import IdiomaticConfig
from .main import run_app

__all__ = ['IdiomaticConfig', 'run_app']

print("idiomatic package initialized") # For testing import
