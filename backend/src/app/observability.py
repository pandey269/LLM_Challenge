"""Observability helpers for tracing, metrics, and logging."""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Histogram

from .config import settings

logger = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

if settings.observability.enable_tracing:
    resource = Resource.create({"service.name": "intelligent-doc-qa"})
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.observability.otlp_endpoint))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

REQUEST_LATENCY = Histogram("rag_request_latency_ms", "Latency of QA requests", buckets=(50, 100, 250, 500, 1000, 2000, 5000))
TOKEN_USAGE = Counter("rag_token_usage", "Token usage", labelnames=("type",))


@contextmanager
def traced_span(name: str) -> Iterator[None]:
    start = perf_counter()
    with tracer.start_as_current_span(name):
        yield
    duration_ms = (perf_counter() - start) * 1000
    REQUEST_LATENCY.observe(duration_ms)


def record_tokens(prompt: int, completion: int) -> None:
    TOKEN_USAGE.labels("prompt").inc(prompt)
    TOKEN_USAGE.labels("completion").inc(completion)
