import unittest
from enterprise_rag.generation import ContextBuilder, GeminiGenerationError, GeminiService, GroundedGenerator, build_grounded_prompt, CitationValidationError, GenerationResult
from enterprise_rag.retrieval import HybridRetrievalResult

def result(cid, content): return HybridRetrievalResult(cid, "doc", content, {"source_filename": "x.pdf"}, ({"ref": cid},), .1, {}, 1)

class FakeResponse:
    text = '{"answer":"Grounded answer","source_ids":["a"]}'
class FakeModels:
    def generate_content(self, **kwargs): return FakeResponse()
class FakeClient:
    models = FakeModels()

class Phase11Tests(unittest.TestCase):
    def test_context_order_bounds_and_dedup(self):
        items = ContextBuilder(5).build((result("a", "abc"), result("a", "abc"), result("b", "xyz")))
        self.assertEqual([item.chunk_id for item in items], ["a"])
        self.assertEqual(items[0].provenance, ({"ref": "a"},))

    def test_prompt_and_fake_generation(self):
        context = ContextBuilder().build((result("a", "Revenue was 10"),))
        self.assertIn("[a]", build_grounded_prompt("What was revenue?", context))
        output, returned = GroundedGenerator(GeminiService(api_key="x", client=FakeClient())).generate("What was revenue?", (result("a", "Revenue was 10"),))
        self.assertEqual(output.answer, "Grounded answer")
        self.assertEqual(returned[0].chunk_id, "a")
        self.assertEqual(output.source_ids, ("a",))

    def test_missing_key_and_unknown_source(self):
        with self.assertRaises(GeminiGenerationError): GeminiService(client=FakeClient()).generate("x")
        class SourceGemini(GeminiService):
            def generate(self, prompt): return GenerationResult("x", ("unknown",))
        with self.assertRaises(CitationValidationError): GroundedGenerator(SourceGemini(api_key="x", client=FakeClient())).generate("q", (result("a", "A"),))

    def test_citation_alias_is_resolved_only_to_known_context_id(self):
        class AliasGemini(GeminiService):
            def generate(self, prompt): return GenerationResult("x", ("abc",))
        item = result("chunk_abc", "A")
        generated, _ = GroundedGenerator(AliasGemini(api_key="x", client=FakeClient())).generate("q", (item,))
        self.assertEqual(generated.source_ids, ("chunk_abc",))

if __name__ == "__main__": unittest.main()
