from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# Core counters / histograms (feel free to extend)
TOKENS_TOTAL = Counter(
    "ai_tutor_tokens_total",
    "Count of prompt + completion tokens sent to the OpenAI API",
    ["model", "phase"],
)

TOOL_LATENCY = Histogram(
    "ai_tutor_tool_latency_seconds",
    "Duration of FunctionTool invocations",
    ["tool_name"],
)


def metrics_endpoint():
    """FastAPI route handler for /metrics (scraped by Prometheus)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# This module is intentionally import‑side‑effect free; counters are updated elsewhere. 