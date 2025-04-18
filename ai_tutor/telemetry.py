import time
from ai_tutor.dependencies import SUPABASE_CLIENT
from openai.types import CompletionUsage
from ai_tutor.context import TutorContext

def log_tool(tool_fn):
    async def wrapper(ctx, *a, **kw):
        start = time.perf_counter()
        res = await tool_fn(ctx, *a, **kw)
        elapsed = int((time.perf_counter() - start) * 1000)

        usage: CompletionUsage | None = getattr(res, "usage", None)
        if SUPABASE_CLIENT:
            SUPABASE_CLIENT.table("edge_logs").insert({
                "session_id": str(getattr(ctx, "session_id", None)),
                "user_id": str(getattr(ctx, "user_id", None)),
                "tool": tool_fn.__name__,
                "latency_ms": elapsed,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
            }).execute()
            # Persist context after every successful tool
            if hasattr(ctx, "context") and isinstance(ctx.context, TutorContext):
                SUPABASE_CLIENT.table("sessions").update({
                    "context_json": ctx.context.model_dump()
                }).eq("id", str(ctx.context.session_id)).eq("user_id", str(ctx.context.user_id)).execute()

        return res
    return wrapper 