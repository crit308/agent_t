from ai_tutor.telemetry import log_tool

raise ImportError(
    "function_tool_logged has moved to ai_tutor.skills; import from there"
)

def function_tool_logged(*ft_args, **ft_kwargs):
    """Compose log_tool → function_tool(strict_mode=True), but return the original function so it's callable."""
    def decorator(fn):
        from agents import function_tool  # Local import to avoid circular dependency
        # Apply telemetry logging to the function
        wrapped_fn = log_tool(fn)
        # Register the telemetry-wrapped function as a tool for agents
        function_tool(strict_mode=True, *ft_args, **ft_kwargs)(wrapped_fn)
        # Return the wrapped function for normal use
        return wrapped_fn
    return decorator 