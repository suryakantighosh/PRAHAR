"""
prahar/modules/c01_ingestion/engine.py
C-01 Ingestion Manager — main async orchestrator.
Updated to support person ingestion with full OSINT sweep.
"""
import asyncio
from typing import Optional
from uuid import UUID
from loguru import logger
import aiohttp

from prahar.modules.c01_ingestion.seed import make_sih, make_case_id
from prahar.modules.c01_ingestion.audit import store_record
from prahar.modules.c01_ingestion.robots import is_allowed
from prahar.modules.c01_ingestion.connectors import (
    fetch_crtsh, fetch_rdap, fetch_wayback,
    fetch_shodan, fetch_virustotal, fetch_abuseipdb,
    fetch_github_user, fetch_hackertarget_reverseip, resolve_dns,
)
from prahar.modules.c01_ingestion.osint_connectors import run_person_osint
from prahar.core.db import AsyncSessionLocal

CONNECTOR_LIMIT = 50
SESSION_TIMEOUT = aiohttp.ClientTimeout(total=120, connect=10)


async def _safe_run(coro, label: str) -> dict:
    try:
        result = await coro
        return result or {}
    except Exception as e:
        logger.error(f"[C-01] Connector '{label}' failed: {e}")
        return {}


async def ingest_domain(
    domain: str,
    case_id: Optional[UUID] = None,
) -> dict:
    if case_id is None:
        case_id = make_case_id()

    sih = make_sih("domain", domain)
    seed_hash = str(sih)

    connector = aiohttp.TCPConnector(limit=CONNECTOR_LIMIT, ssl=False)
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=SESSION_TIMEOUT,
        headers={"User-Agent": "PraharBot/2.0"},
    ) as session:

        tasks = {
            "crtsh":      fetch_crtsh(domain, session),
            "rdap":       fetch_rdap(domain, session),
            "wayback":    fetch_wayback(f"*.{domain}", session),
            "virustotal": fetch_virustotal(domain, session),
            "dns":        resolve_dns(domain),
        }

        gathered = await asyncio.gather(
            *[_safe_run(coro, name) for name, coro in tasks.items()]
        )
        results = dict(zip(tasks.keys(), gathered))

        ips = results.get("dns", {}).get("ips", [])
        ip_tasks = {}
        for ip in ips[:5]:
            ip_tasks[f"shodan_{ip}"]       = fetch_shodan(ip, session)
            ip_tasks[f"abuseipdb_{ip}"]    = fetch_abuseipdb(ip, session)
            ip_tasks[f"hackertarget_{ip}"] = fetch_hackertarget_reverseip(ip, session)

        if ip_tasks:
            ip_gathered = await asyncio.gather(
                *[_safe_run(coro, name) for name, coro in ip_tasks.items()]
            )
            results.update(dict(zip(ip_tasks.keys(), ip_gathered)))

        async with AsyncSessionLocal() as db:
            saved = 0
            for name, payload in results.items():
                if not payload:
                    continue
                source_url = payload.get("source_url", f"https://{domain}")
                robots_ok = True
                if source_url.startswith("http"):
                    robots_ok = await is_allowed(source_url, session)
                if not robots_ok:
                    continue
                await store_record(
                    db,
                    case_id=case_id,
                    seed_hash=seed_hash,
                    source_url=source_url,
                    source_name=payload.get("source", name),
                    content=payload,
                    robots_allowed=robots_ok,
                )
                saved += 1

    logger.success(
        f"[C-01] domain={domain} case={case_id} "
        f"records_saved={saved} ips_found={len(ips)}"
    )
    return {
        "case_id": str(case_id),
        "seed_hash": seed_hash,
        "domain": domain,
        "records_saved": saved,
        "ips": ips,
    }


async def ingest_username(
    username: str,
    case_id: Optional[UUID] = None,
) -> dict:
    if case_id is None:
        case_id = make_case_id()

    sih = make_sih("username", username)
    seed_hash = str(sih)

    connector = aiohttp.TCPConnector(limit=CONNECTOR_LIMIT)
    async with aiohttp.ClientSession(connector=connector,
                                     timeout=SESSION_TIMEOUT) as session:
        result = await _safe_run(fetch_github_user(username, session), "github")

        async with AsyncSessionLocal() as db:
            saved = 0
            if result:
                await store_record(
                    db,
                    case_id=case_id,
                    seed_hash=seed_hash,
                    source_url=f"https://api.github.com/users/{username}",
                    source_name="github",
                    content=result,
                )
                saved = 1

    logger.success(f"[C-01] username={username} case={case_id} records_saved={saved}")
    return {
        "case_id": str(case_id),
        "seed_hash": seed_hash,
        "username": username,
        "records_saved": saved,
    }


async def ingest_person(
    name: str,
    username: str = "",
    email: str = "",
    phone: str = "",
    image_b64: str = "",          # base64-encoded photo
    case_id: Optional[UUID] = None,
) -> dict:
    """
    Full OSINT sweep for a named person.
    Accepts: name (required), username, email, phone, photo (base64).
    Hits 25+ sources: Wikipedia, DuckDuckGo, GitHub, Reddit, HN,
    Keybase, GitLab, npm, HIBP, Gravatar, Google CSE (LinkedIn/Twitter/News),
    Google News RSS, Pastebin, DeepFace analysis, and more.
    """
    if case_id is None:
        case_id = make_case_id()

    sih = make_sih("name", name)
    seed_hash = str(sih)

    logger.info(f"[C-01] Starting person OSINT for: {name} (case={case_id})")

    connector = aiohttp.TCPConnector(limit=CONNECTOR_LIMIT, ssl=False)
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=SESSION_TIMEOUT,
    ) as session:
        # Ensure username is never the same as the full name
        clean_username = username.strip() if username and username.strip().lower() != name.strip().lower() else ""
        osint_results = await run_person_osint(
            name=name,
            username=clean_username,
            email=email,
            phone=phone,
            image_b64=image_b64,
            session=session,
        )

        async with AsyncSessionLocal() as db:
            saved = 0
            for label, payload in osint_results:
                source_url = payload.get("url") or payload.get("source_url") or \
                             f"https://osint/{label}/{name.replace(' ','_')}"
                await store_record(
                    db,
                    case_id=case_id,
                    seed_hash=seed_hash,
                    source_url=source_url,
                    source_name=payload.get("source", label),
                    content=payload,
                    robots_allowed=True,
                )
                saved += 1

    logger.success(
        f"[C-01] person={name} case={case_id} records_saved={saved}"
    )
    return {
        "case_id":       str(case_id),
        "seed_hash":     seed_hash,
        "name":          name,
        "username":      username,
        "email":         email,
        "phone":         phone,
        "has_photo":     bool(image_b64),
        "records_saved": saved,
        "sources_hit":   [label for label, _ in osint_results],
    }
