import os
import tempfile
import unittest
from unittest.mock import patch

from src.brain import (
    _extract_alpha_id,
    _recover_alpha_id_from_history,
    get_alpha_performance,
    read_simulations_csv,
    simulate,
)


class DummyResponse:
    def __init__(self, status_code, payload, *, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


class DummySession:
    def __init__(self, response=None, *, get_responses=None, post_responses=None):
        self.response = response
        self._get_responses = list(get_responses or [])
        self._post_responses = list(post_responses or [])
        self.get_calls = 0
        self.post_calls = 0

    def get(self, _url):
        self.get_calls += 1
        if self._get_responses:
            return self._get_responses.pop(0)
        return self.response

    def post(self, _url, json=None):
        self.post_calls += 1
        if self._post_responses:
            return self._post_responses.pop(0)
        return self.response


class TestBrainResilience(unittest.TestCase):
    def test_extract_alpha_id_handles_nested_payload(self):
        payload = {"result": {"alpha": {"id": "A123"}}}
        self.assertEqual(_extract_alpha_id(payload), "A123")

    def test_recover_alpha_id_from_history_matches_expression(self):
        session = DummySession(
            DummyResponse(
                200,
                {
                    "results": [
                        {
                            "id": "X1",
                            "regular": {"code": "rank(ts_zscore(abs(close-vwap),21))"},
                        },
                        {
                            "id": "X2",
                            "regular": {"code": "rank(close)"},
                        },
                    ]
                },
            )
        )

        alpha_id = _recover_alpha_id_from_history(
            session,
            "rank( ts_zscore( abs(close-vwap), 21 ) )",
        )
        self.assertEqual(alpha_id, "X1")

    def test_get_alpha_performance_returns_none_for_non_200(self):
        session = DummySession(DummyResponse(429, {}, text="rate limited"))
        self.assertIsNone(get_alpha_performance(session, "A123"))

    def test_simulate_writes_pending_row_when_timeout_would_be_exceeded(self):
        session = DummySession(
            post_responses=[DummyResponse(201, {}, headers={"Location": "/progress"})],
            get_responses=[
                DummyResponse(200, {"alpha": "A123"}, headers={"Retry-After": "120"}),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = simulate(session, "rank(close)", timeout=60)
                df = read_simulations_csv("simulation_results.csv")
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(result["LOW_SHARPE"], "PENDING")
        self.assertEqual(result["alpha_id"], "A123")
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "LOW_SHARPE"], "PENDING")

    @patch("src.brain.sleep", return_value=None)
    def test_simulate_treats_retry_after_string_zero_as_finished(self, _sleep):
        session = DummySession(
            post_responses=[DummyResponse(201, {}, headers={"Location": "/progress"})],
            get_responses=[
                DummyResponse(200, {"alpha": "A123"}, headers={"Retry-After": "0"}),
                DummyResponse(
                    200,
                    {
                        "regular": {"code": "rank(close)"},
                        "is": {
                            "turnover": 0.2,
                            "returns": 0.1,
                            "drawdown": 0.04,
                            "margin": 0.03,
                            "fitness": 1.5,
                            "sharpe": 1.9,
                            "checks": [
                                {"name": "LOW_SHARPE", "result": "PASS"},
                                {"name": "LOW_FITNESS", "result": "PASS"},
                                {"name": "LOW_TURNOVER", "result": "PASS"},
                                {"name": "HIGH_TURNOVER", "result": "PASS"},
                                {"name": "CONCENTRATED_WEIGHT", "result": "PASS"},
                                {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS"},
                                {"name": "SELF_CORRELATION", "result": "PASS"},
                                {"name": "MATCHES_COMPETITION", "result": "PASS"},
                            ],
                        },
                        "settings": {
                            "region": "USA",
                            "universe": "TOP3000",
                            "delay": 1,
                            "decay": 5,
                            "neutralization": "Market",
                            "truncation": 0.05,
                        },
                    },
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = simulate(session, "rank(close)", timeout=60)
                df = read_simulations_csv("simulation_results.csv")
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(result["alpha_id"], "A123")
        self.assertEqual(result["LOW_SHARPE"], "PASS")
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "alpha_id"], "A123")

    @patch("src.brain.sleep", return_value=None)
    def test_simulate_retries_submit_after_429_then_succeeds(self, _sleep):
        session = DummySession(
            post_responses=[
                DummyResponse(429, {}, headers={"Retry-After": "0"}, text="SIMULATION_LIMIT_EXCEEDED"),
                DummyResponse(201, {}, headers={"Location": "/progress"}),
            ],
            get_responses=[
                DummyResponse(200, {"alpha": "A123"}, headers={"Retry-After": "0"}),
                DummyResponse(
                    200,
                    {
                        "regular": {"code": "rank(close)"},
                        "is": {
                            "turnover": 0.2,
                            "returns": 0.1,
                            "drawdown": 0.04,
                            "margin": 0.03,
                            "fitness": 1.5,
                            "sharpe": 1.9,
                            "checks": [
                                {"name": "LOW_SHARPE", "result": "PASS"},
                                {"name": "LOW_FITNESS", "result": "PASS"},
                                {"name": "LOW_TURNOVER", "result": "PASS"},
                                {"name": "HIGH_TURNOVER", "result": "PASS"},
                                {"name": "CONCENTRATED_WEIGHT", "result": "PASS"},
                                {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS"},
                                {"name": "SELF_CORRELATION", "result": "PASS"},
                                {"name": "MATCHES_COMPETITION", "result": "PASS"},
                            ],
                        },
                        "settings": {
                            "region": "USA",
                            "universe": "TOP3000",
                            "delay": 1,
                            "decay": 5,
                            "neutralization": "Market",
                            "truncation": 0.05,
                        },
                    },
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = simulate(session, "rank(close)", timeout=60)
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(result["alpha_id"], "A123")
        self.assertEqual(result["LOW_SHARPE"], "PASS")
        self.assertEqual(session.post_calls, 2)


if __name__ == "__main__":
    unittest.main()
