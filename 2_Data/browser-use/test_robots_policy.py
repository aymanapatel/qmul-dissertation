import asyncio
import unittest
from email.message import Message
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from robots_policy import RobotsPolicy


class _Response:
    def __init__(self, body: str):
        self.body = body.encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


class RobotsPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_enforces_disallow_and_allow_rules(self):
        policy = RobotsPolicy()
        response = _Response("User-agent: *\nDisallow: /private\nAllow: /private/public\n")

        with patch("robots_policy.urlopen", return_value=response) as request:
            self.assertFalse(await policy.can_fetch("https://example.test/private/data"))
            self.assertTrue(await policy.can_fetch("https://example.test/private/public"))

        request.assert_called_once()

    async def test_missing_robots_file_allows_fetching(self):
        policy = RobotsPolicy()
        error = HTTPError("https://example.test/robots.txt", 404, "Not Found", Message(), None)

        with patch("robots_policy.urlopen", side_effect=error):
            self.assertTrue(await policy.can_fetch("https://example.test/page"))

    async def test_retrieval_failure_denies_fetching(self):
        policy = RobotsPolicy()

        with patch("robots_policy.urlopen", side_effect=URLError("offline")):
            self.assertFalse(await policy.can_fetch("https://example.test/page"))

    async def test_concurrent_checks_share_one_robots_request(self):
        policy = RobotsPolicy()
        response = _Response("User-agent: *\nAllow: /\n")

        with patch("robots_policy.urlopen", return_value=response) as request:
            results = await asyncio.gather(
                policy.can_fetch("https://example.test/one"),
                policy.can_fetch("https://example.test/two"),
            )

        self.assertEqual(results, [True, True])
        request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
