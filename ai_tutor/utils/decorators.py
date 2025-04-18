from ai_tutor.telemetry import log_tool

def function_tool_logged(*ft_args, **ft_kwargs):
    """Compose log_tool → function_tool(strict_mode=True)."""
    def wrapper(fn):
        from agents import function_tool  # Local import to avoid circular dependency
        return function_tool(strict_mode=True, *ft_args, **ft_kwargs)(
            log_tool(fn)
        )
    return wrapper 