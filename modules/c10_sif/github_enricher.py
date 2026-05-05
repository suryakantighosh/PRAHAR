"""
prahar/modules/c10_sif/github_enricher.py
Pulls stylometry-rich text from GitHub for a username:
  - Profile bio + location + company + blog
  - README of each public repo (plain-text stripped)
  - Up to N recent commit messages per repo

Uses the GITHUB_TOKEN from .env (handles both authenticated and
unauthenticated requests — token doubles the rate limit and unlocks
private repos if scoped).
"""

import asyncio
import os
import re
from typing import Optional, Union

import aiohttp
from loguru import logger


GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")

# How many repos / commits to crawl per user (keep API budget sensible)
MAX_REPOS   = 20
MAX_COMMITS = 15    # per repo


def _headers() -> dict:
    h = {
        "Accept":     "application/vnd.github+json",
        "User-Agent": "PraharBot/2.0",
    }
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


async def _get_json(
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[dict] = None,
) -> Optional[Union[dict, list]]:
    try:
        async with session.get(
            url,
            headers=_headers(),
            params=params or {},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as r:
            if r.status == 200:
                return await r.json()
            if r.status == 404:
                return None
            if r.status == 403:
                logger.warning("[C-10/GitHub] 403 — rate-limited or missing scope")
                return None
            logger.warning(f"[C-10/GitHub] {url} → HTTP {r.status}")
            return None
    except Exception as e:
        logger.warning(f"[C-10/GitHub] {url} error: {e}")
        return None


async def _readme_text(
    session: aiohttp.ClientSession,
    owner: str,
    repo: str,
) -> str:
    """Fetch README and return plain text (strips Markdown syntax)."""
    data = await _get_json(
        session,
        f"https://api.github.com/repos/{owner}/{repo}/readme",
    )
    if not data or not isinstance(data, dict):
        return ""
    # README content is base64-encoded
    import base64
    try:
        raw = base64.b64decode(data.get("content", "")).decode("utf-8", errors="ignore")
    except Exception:
        return ""
    # Strip Markdown — headers, links, code blocks, images
    raw = re.sub(r"```[\s\S]*?```", " ", raw)    # code blocks
    raw = re.sub(r"`[^`]+`", " ", raw)           # inline code
    raw = re.sub(r"!\[.*?\]\(.*?\)", " ", raw)   # images
    raw = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", raw)  # links → text
    raw = re.sub(r"^#{1,6}\s+", "", raw, flags=re.MULTILINE)  # headers
    raw = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", raw)  # bold/italic
    raw = re.sub(r"^[-*+]\s+", "", raw, flags=re.MULTILINE)  # list markers
    return raw.strip()


async def _commit_messages(
    session: aiohttp.ClientSession,
    owner: str,
    repo: str,
) -> list[str]:
    data = await _get_json(
        session,
        f"https://api.github.com/repos/{owner}/{repo}/commits",
        params={"per_page": MAX_COMMITS},
    )
    if not data or not isinstance(data, list):
        return []
    messages = []
    for item in data:
        msg = item.get("commit", {}).get("message", "")
        if msg:
            messages.append(msg)
    return messages


async def fetch_github_writing_corpus(username: str) -> str:
    """
    Return a single concatenated text of all stylometry-useful writing
    by `username` on GitHub.  Returns empty string on total failure.
    """
    if not username:
        return ""

    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:

        # ── 1. Profile text ───────────────────────────────────────────
        profile = await _get_json(
            session, f"https://api.github.com/users/{username}"
        )
        corpus_parts: list[str] = []

        if profile and isinstance(profile, dict):
            for field in ("bio", "company", "blog", "location", "name"):
                val = profile.get(field) or ""
                if val:
                    corpus_parts.append(val)

        # ── 2. Repos ──────────────────────────────────────────────────
        repos_data = await _get_json(
            session,
            f"https://api.github.com/users/{username}/repos",
            params={"per_page": MAX_REPOS, "sort": "pushed", "type": "owner"},
        )
        repos: list[dict] = []
        if repos_data and isinstance(repos_data, list):
            # Include description text first (fast, no extra call)
            for repo in repos_data[:MAX_REPOS]:
                desc = repo.get("description") or ""
                if desc:
                    corpus_parts.append(desc)
                repos.append(repo)

        # ── 3. READMEs + commits (concurrent) ────────────────────────
        readme_tasks = [
            _readme_text(session, username, r["name"])
            for r in repos[:MAX_REPOS]
        ]
        commit_tasks = [
            _commit_messages(session, username, r["name"])
            for r in repos[:MAX_REPOS]
        ]

        readmes, commit_lists = await asyncio.gather(
            asyncio.gather(*readme_tasks),
            asyncio.gather(*commit_tasks),
        )

        for readme in readmes:
            if readme:
                corpus_parts.append(readme)

        for msgs in commit_lists:
            if msgs:
                corpus_parts.extend(msgs)

    corpus = "\n\n".join(corpus_parts)
    logger.info(
        f"[C-10/GitHub] {username}: {len(corpus):,} chars from "
        f"{len(repos)} repos"
    )
    return corpus
