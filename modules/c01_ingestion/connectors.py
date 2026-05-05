"""
prahar/modules/c01_ingestion/connectors.py
All free-API connectors for C-01 Ingestion Manager.
Each connector is an async function returning a dict payload
ready to be passed to audit.store_record().
"""
import asyncio
import os
import socket
from datetime import datetime
from loguru import logger
import aiohttp


# ── Quota manager ────────────────────────────────────────────
# Tracks per-source monthly/daily call counts in Redis
import redis.asyncio as aioredis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

QUOTAS = {
    "shodan":      {"limit": 100,  "period": "monthly"},
    "virustotal":  {"limit": 500,  "period": "daily"},
    "abuseipdb":   {"limit": 1000, "period": "daily"},
    "google_cse":  {"limit": 100,  "period": "daily"},
    "hackertarget":{"limit": 100,  "period": "daily"},
}


async def quota_ok(source: str) -> bool:
    """Return True if quota for this source has not been exceeded."""
    if source not in QUOTAS:
        return True
    r = await aioredis.from_url(REDIS_URL, decode_responses=True)
    key = f"prahar:quota:{source}:{datetime.utcnow().strftime('%Y-%m')}"
    count = await r.get(key)
    limit = QUOTAS[source]["limit"]
    return int(count or 0) < limit


async def quota_increment(source: str):
    r = await aioredis.from_url(REDIS_URL, decode_responses=True)
    key = f"prahar:quota:{source}:{datetime.utcnow().strftime('%Y-%m')}"
    await r.incr(key)
    await r.expire(key, 33 * 86400)   # 33 days TTL


# ── crt.sh — SSL certificate transparency subdomains ─────────
async def fetch_crtsh(domain: str, session: aiohttp.ClientSession) -> dict:
    url = f"https://crt.sh/?q=%.{domain}&output=json"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status == 200:
                data = await r.json(content_type=None)
                subdomains = list({e["name_value"] for e in data})
                logger.info(f"[crt.sh] {domain} → {len(subdomains)} subdomains")
                return {"source": "crt.sh", "domain": domain, "subdomains": subdomains}
    except Exception as e:
        logger.warning(f"[crt.sh] {domain} error: {e}")
    return {}


# ── WHOIS / RDAP ──────────────────────────────────────────────
async def fetch_rdap(domain: str, session: aiohttp.ClientSession) -> dict:
    url = f"https://rdap.org/domain/{domain}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status == 200:
                data = await r.json()
                logger.info(f"[RDAP] {domain} → registrar fetched")
                return {"source": "rdap", "domain": domain, "rdap": data}
    except Exception as e:
        logger.warning(f"[RDAP] {domain} error: {e}")
    return {}


# ── Wayback Machine CDX API ───────────────────────────────────
async def fetch_wayback(url_target: str, session: aiohttp.ClientSession,
                        limit: int = 50) -> dict:
    cdx_url = (
        f"http://web.archive.org/cdx/search/cdx"
        f"?url={url_target}&output=json&limit={limit}&fl=timestamp,original,statuscode"
    )
    try:
        async with session.get(cdx_url, timeout=aiohttp.ClientTimeout(total=20)) as r:
            if r.status == 200:
                rows = await r.json()
                logger.info(f"[Wayback] {url_target} → {len(rows)} snapshots")
                return {"source": "wayback", "target": url_target, "snapshots": rows}
    except Exception as e:
        logger.warning(f"[Wayback] {url_target} error: {e}")
    return {}


# ── Shodan (free tier — 100 results/month) ───────────────────
async def fetch_shodan(ip_or_domain: str, session: aiohttp.ClientSession) -> dict:
    api_key = os.getenv("SHODAN_API_KEY", "")
    if not api_key:
        logger.debug("[Shodan] No API key — skipping")
        return {}
    if not await quota_ok("shodan"):
        logger.warning("[Shodan] Monthly quota reached — skipping")
        return {}
    url = f"https://api.shodan.io/shodan/host/{ip_or_domain}?key={api_key}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
            if r.status == 200:
                data = await r.json()
                await quota_increment("shodan")
                logger.info(f"[Shodan] {ip_or_domain} → {len(data.get('ports',[]))} ports")
                return {"source": "shodan", "target": ip_or_domain, "data": data}
    except Exception as e:
        logger.warning(f"[Shodan] {ip_or_domain} error: {e}")
    return {}


# ── VirusTotal (free — 500 req/day) ──────────────────────────
async def fetch_virustotal(domain: str, session: aiohttp.ClientSession) -> dict:
    api_key = os.getenv("VIRUSTOTAL_API_KEY", "")
    if not api_key:
        return {}
    if not await quota_ok("virustotal"):
        logger.warning("[VT] Daily quota reached")
        return {}
    url = f"https://www.virustotal.com/api/v3/domains/{domain}"
    headers = {"x-apikey": api_key}
    try:
        async with session.get(url, headers=headers,
                               timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status == 200:
                data = await r.json()
                await quota_increment("virustotal")
                return {"source": "virustotal", "domain": domain, "data": data}
    except Exception as e:
        logger.warning(f"[VT] {domain} error: {e}")
    return {}


# ── AbuseIPDB (free — 1000 req/day) ──────────────────────────
async def fetch_abuseipdb(ip: str, session: aiohttp.ClientSession) -> dict:
    api_key = os.getenv("ABUSEIPDB_API_KEY", "")
    if not api_key:
        return {}
    if not await quota_ok("abuseipdb"):
        logger.warning("[AbuseIPDB] Daily quota reached")
        return {}
    url = "https://api.abuseipdb.com/api/v2/check"
    headers = {"Key": api_key, "Accept": "application/json"}
    params = {"ipAddress": ip, "maxAgeInDays": 90}
    try:
        async with session.get(url, headers=headers, params=params,
                               timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status == 200:
                data = await r.json()
                await quota_increment("abuseipdb")
                return {"source": "abuseipdb", "ip": ip, "data": data}
    except Exception as e:
        logger.warning(f"[AbuseIPDB] {ip} error: {e}")
    return {}


# ── GitHub public API ─────────────────────────────────────────
async def fetch_github_user(username: str, session: aiohttp.ClientSession) -> dict:
    token = os.getenv("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://api.github.com/users/{username}"
    try:
        async with session.get(url, headers=headers,
                               timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status == 200:
                data = await r.json()
                repos_url = data.get("repos_url", "")
                repos = []
                if repos_url:
                    async with session.get(repos_url + "?per_page=30",
                                           headers=headers,
                                           timeout=aiohttp.ClientTimeout(total=10)) as r2:
                        if r2.status == 200:
                            repos = await r2.json()
                return {
                    "source": "github",
                    "username": username,
                    "profile": data,
                    "repos": repos,
                }
    except Exception as e:
        logger.warning(f"[GitHub] {username} error: {e}")
    return {}


# ── HackerTarget (free — 100 req/day) ────────────────────────
async def fetch_hackertarget_reverseip(ip: str,
                                       session: aiohttp.ClientSession) -> dict:
    if not await quota_ok("hackertarget"):
        return {}
    url = f"https://api.hackertarget.com/reverseiplookup/?q={ip}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            text = await r.text()
            await quota_increment("hackertarget")
            return {"source": "hackertarget_reverseip", "ip": ip,
                    "domains": text.strip().splitlines()}
    except Exception as e:
        logger.warning(f"[HackerTarget] {ip} error: {e}")
    return {}


# ── DNS resolver ──────────────────────────────────────────────
async def resolve_dns(domain: str) -> dict:
    loop = asyncio.get_event_loop()
    try:
        infos = await loop.getaddrinfo(domain, None)
        ips = list({i[4][0] for i in infos})
        return {"source": "dns_resolver", "domain": domain, "ips": ips}
    except Exception as e:
        logger.warning(f"[DNS] {domain} error: {e}")
    return {}
