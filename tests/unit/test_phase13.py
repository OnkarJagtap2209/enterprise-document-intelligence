import unittest
from fastapi.testclient import TestClient
from enterprise_rag.api import create_app
from enterprise_rag.application import QueryOutcome

class Service:
    def __init__(self, outcome=None, error=None): self.outcome, self.error = outcome, error
    def query(self, query):
        if self.error: raise self.error
        return self.outcome

class Phase13Tests(unittest.TestCase):
    def test_health_and_query(self):
        outcome = QueryOutcome("answer", (type("S", (), {"chunk_id":"c", "document_id":"d", "metadata":{"source_filename":"x.pdf", "page_start":1, "page_end":1}})(),), "req")
        client = TestClient(create_app(Service(outcome)))
        self.assertEqual(client.get("/health").json(), {"status":"ok"})
        response = client.post("/query", json={"query":"revenue"})
        self.assertEqual(response.status_code, 200); self.assertEqual(response.json()["request_id"], "req")

    def test_validation_clarification_and_safe_error(self):
        client = TestClient(create_app(Service(QueryOutcome(None, (), "req", "Which year?"))))
        self.assertEqual(client.post("/query", json={"query":"x"}).status_code, 200)
        self.assertEqual(client.post("/query", json={"query":""}).status_code, 422)
        failing = TestClient(create_app(Service(error=RuntimeError("secret"))))
        response = failing.post("/query", json={"query":"x"})
        self.assertEqual(response.status_code, 500); self.assertNotIn("secret", response.text)

if __name__ == "__main__": unittest.main()
