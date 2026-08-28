"""Structured runtime request traces."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from time import perf_counter
from contextlib import contextmanager
from typing import Iterator
from typing import Any
from uuid import uuid4

@dataclass(slots=True)
class RequestTrace:
    request_id: str; query: str; status: str = "running"; stage_timings_ms: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float | None = None; model_name: str | None = None; usage: Any = None; source_ids: tuple[str, ...] = (); error: dict[str, str] | None = None

class RuntimeTracer:
    def __init__(self, enabled: bool = True, persist_path: str | Path | None = None): self.enabled, self.persist_path, self._starts = enabled, persist_path, {}
    def start(self, query: str) -> RequestTrace:
        trace = RequestTrace(str(uuid4()), query) if self.enabled else RequestTrace("", query)
        self._starts[trace.request_id] = perf_counter(); return trace
    def finish(self, trace: RequestTrace, **fields: Any) -> RequestTrace:
        if not self.enabled: return trace
        trace.status = "success"; trace.total_latency_ms = (perf_counter() - self._starts.pop(trace.request_id, perf_counter())) * 1000
        for key, value in fields.items(): setattr(trace, key, value)
        self._persist(trace); return trace
    def fail(self, trace: RequestTrace, stage: str, exc: Exception) -> RequestTrace:
        if self.enabled:
            trace.status = "error"; trace.error = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}; trace.total_latency_ms = (perf_counter() - self._starts.pop(trace.request_id, perf_counter())) * 1000; self._persist(trace)
        return trace
    @contextmanager
    def stage(self, trace: RequestTrace, name: str) -> Iterator[None]:
        if not self.enabled:
            yield; return
        start = perf_counter()
        try: yield
        finally: trace.stage_timings_ms[name] = (perf_counter() - start) * 1000
    def _persist(self, trace: RequestTrace) -> None:
        if not self.persist_path: return
        path = Path(self.persist_path); path.parent.mkdir(parents=True, exist_ok=True)
        path.open("a", encoding="utf-8").write(json.dumps(asdict(trace), default=lambda x: asdict(x), sort_keys=True) + "\n")
