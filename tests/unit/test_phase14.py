import unittest
from enterprise_rag.ui.api_client import APIClient, APIClientError

class Response:
    def __init__(self, code, payload): self.status_code, self.payload = code, payload
    def json(self): return self.payload
class Session:
    def __init__(self, response): self.response, self.calls = response, []
    def request(self, method, url, **kwargs): self.calls.append((method, url, kwargs)); return self.response

class Phase14Tests(unittest.TestCase):
    def test_client_health_query_and_payload(self):
        session = Session(Response(200, {"answer":"ok", "sources":[], "request_id":"r"})); client = APIClient("http://api", session=session)
        self.assertEqual(client.health()["answer"], "ok"); client.query(" revenue ")
        self.assertEqual(session.calls[-1][2]["json"], {"query":"revenue"})
    def test_client_errors_and_empty_question(self):
        with self.assertRaises(APIClientError): APIClient(session=Session(Response(500, {"detail":"bad"}))).health()
        with self.assertRaises(APIClientError): APIClient().query(" ")

if __name__ == "__main__": unittest.main()
