"""Custom exceptions for the AI Tutor application."""

class ExecutorError(Exception):
    """Custom exception for errors during Executor execution."""
    pass

class ToolInputError(ValueError):
    """Custom exception for tool input validation errors."""
    pass 