"""Monotonic stage latency measurement."""
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

@contextmanager
def measure() -> Iterator[dict[str, float]]:
    start = perf_counter(); state = {}
    try: yield state
    finally: state["latency_ms"] = (perf_counter() - start) * 1000
