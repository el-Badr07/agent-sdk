import uuid
from typing import Type, TypeVar

# Generic TypeVar
T = TypeVar("T")

def generate_uuid(prefix: str = "") -> str:
    """Generates a unique UUID string, optionally prefixed."""
    return f"{prefix}{uuid.uuid4()}"

# Add other general-purpose utility functions here as needed.
# For example, type checking helpers, environment variable readers, etc.
