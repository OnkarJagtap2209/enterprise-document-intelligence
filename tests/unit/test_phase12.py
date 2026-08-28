import unittest
from enterprise_rag.observability import RuntimeTracer, UsageInfo

class Phase12Tests(unittest.TestCase):
    def test_trace_success_stage_and_usage(self):
        tracer = RuntimeTracer()
        trace = tracer.start("revenue")
        with tracer.stage(trace, "retrieval"): pass
        done = tracer.finish(trace, model_name="gemini-2.5-flash", source_ids=("c1",), usage=UsageInfo(1, 2, 3))
        self.assertEqual(done.status, "success"); self.assertIn("retrieval", done.stage_timings_ms); self.assertEqual(done.usage.total_tokens, 3)

    def test_error_trace_and_disabled(self):
        tracer = RuntimeTracer(); trace = tracer.start("q"); failed = tracer.fail(trace, "generation", RuntimeError("safe"))
        self.assertEqual(failed.status, "error"); self.assertEqual(failed.error["stage"], "generation")
        disabled = RuntimeTracer(enabled=False).start("q"); self.assertEqual(disabled.request_id, "")

    def test_usage_unavailable_is_none(self):
        self.assertEqual(UsageInfo.from_response(object()), UsageInfo())

if __name__ == "__main__": unittest.main()
