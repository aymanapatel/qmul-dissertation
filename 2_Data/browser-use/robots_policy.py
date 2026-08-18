"""Fetch and enforce robots.txt rules before automated navigation."""

import asyncio
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser


ROBOTS_USER_AGENT = "QMULAccessibilityResearchBot"


class RobotsPolicy:
    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout_seconds = timeout_seconds
        self._parsers: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def can_fetch(self, url: str) -> bool:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            return False

        origin = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
        if origin not in self._parsers:
            lock = self._locks.setdefault(origin, asyncio.Lock())
            async with lock:
                if origin not in self._parsers:
                    self._parsers[origin] = await asyncio.to_thread(self._load, origin)
        return self._parsers[origin].can_fetch(ROBOTS_USER_AGENT, url)

    def _load(self, origin: str) -> RobotFileParser:
        robots_url = f"{origin}/robots.txt"
        parser = RobotFileParser(robots_url)
        request = Request(robots_url, headers={"User-Agent": ROBOTS_USER_AGENT})

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
        except HTTPError as error:
            if error.code in {401, 403}:
                parser.parse(["User-agent: *", "Disallow: /"])
            elif 400 <= error.code < 500:
                parser.parse(["User-agent: *", "Allow: /"])
            return parser
        except (URLError, TimeoutError, OSError):
            # A policy that cannot be retrieved is not safe to assume permissive.
            return parser

        parser.parse(body.splitlines())
        return parser
