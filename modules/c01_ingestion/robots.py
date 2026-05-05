"""
prahar/modules/c01_ingestion/robots.py
robots.txt compliance checker.
Blocks scraping any URL that robots.txt disallows for our agent.
Results cached in Redis for 24 hours to avoid re-fetching.
"""
import asyncio
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from loguru import logger
import aiohttp
import redis.asyncio as aioredis
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ROBOTS_TTL = 86400          # 24 hours
USER_AGENT = "PraharBot/2.0 (OSINT research; contact: admin@prahar.local)"

from typing import Optional
_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


async def fetch_robots_txt(base_url: str, session: aiohttp.ClientSession) -> str:
    """Download robots.txt for a domain, return raw text (empty string on failure)."""
    robots_url = f"{base_url}/robots.txt"
    try:
        async with session.get(robots_url, timeout=aiohttp.ClientTimeout(total=5)) as r:
            if r.status == 200:
                return await r.text()
    except Exception:
        pass
    return ""


async def is_allowed(url: str, session: aiohttp.ClientSession) -> bool:
    """
    Return True if USER_AGENT is allowed to fetch this URL per robots.txt.
    Caches per-domain in Redis. Defaults to True on any fetch error.
    """
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    cache_key = f"prahar:robots:{parsed.netloc}"

    r = await get_redis()
    cached = await r.get(cache_key)

    if cached is None:
        txt = await fetch_robots_txt(base, session)
        await r.setex(cache_key, ROBOTS_TTL, txt or "")
        cached = txt

    if not cached:
        return True     # no robots.txt = allowed

    parser = RobotFileParser()
    parser.parse(cached.splitlines())
    allowed = parser.can_fetch(USER_AGENT, url)

    if not allowed:
        logger.warning(f"[ROBOTS] Blocked by robots.txt: {url}")

    return allowed
