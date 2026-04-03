import unittest
from unittest.mock import Mock, patch

import requests

from src.utils import authenticate_session


class TestAuthenticationRetries(unittest.TestCase):
    @patch("src.utils.time.sleep", return_value=None)
    def test_authenticate_session_retries_after_429(self, _sleep):
        session = Mock()
        first = Mock(status_code=429, headers={"Retry-After": "1"}, text='{"message":"API rate limit exceeded"}')
        second = Mock(status_code=201, headers={}, text="")
        session.post.side_effect = [first, second]

        response = authenticate_session(session, context="Authentication", max_retries=3)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(session.post.call_count, 2)

    @patch("src.utils.time.sleep", return_value=None)
    def test_authenticate_session_returns_last_429_after_exhausting_retries(self, _sleep):
        session = Mock()
        limited = Mock(status_code=429, headers={}, text='{"message":"API rate limit exceeded"}')
        session.post.side_effect = [limited, limited]

        response = authenticate_session(session, context="Authentication", max_retries=2)

        self.assertEqual(response.status_code, 429)
        self.assertEqual(session.post.call_count, 2)

    @patch("src.utils.time.sleep", return_value=None)
    def test_authenticate_session_retries_network_error(self, _sleep):
        session = Mock()
        session.post.side_effect = [
            requests.exceptions.RequestException("boom"),
            Mock(status_code=201, headers={}, text=""),
        ]

        response = authenticate_session(session, context="Authentication", max_retries=2)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(session.post.call_count, 2)

    @patch("src.utils.time.sleep", return_value=None)
    def test_authenticate_session_does_not_retry_non_retryable_401(self, _sleep):
        session = Mock()
        unauthorized = Mock(status_code=401, headers={}, text='{"message":"bad credentials"}')
        session.post.return_value = unauthorized

        response = authenticate_session(session, context="Authentication", max_retries=4)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(session.post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
