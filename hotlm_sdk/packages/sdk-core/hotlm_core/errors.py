"""Custom exception classes for the HotLM framework."""

class HotLMError(Exception):
    """Base class for all HotLM specific errors."""
    pass

class HotLMConfigurationError(HotLMError):
    """Error related to configuration issues."""
    pass

class HotLMRuntimeError(HotLMError):
    """Error occurring during runtime execution of a component."""
    pass

class HotLMToolError(HotLMRuntimeError):
    """Error related to tool execution."""
    pass

class HotLMModelError(HotLMRuntimeError):
    """Error related to model interaction (API calls, parsing, etc.)."""
    pass

class HotLMMemoryError(HotLMRuntimeError):
    """Error related to memory operations."""
    pass

class HotLMValidationError(HotLMError, ValueError):
    """Error related to validation (e.g., input validation)."""
    pass
