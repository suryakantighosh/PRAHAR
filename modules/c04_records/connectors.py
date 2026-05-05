"""
prahar/modules/c04_records/connectors.py
Public records connectors:
  - MCA21 (company directors, CIN, filings)
  - eCourts (case filings, party names, judgments)
  - Gazette of India (notifications, appointments)
  - Google News scrape (name-based news archive)
All via Playwright headless browser + BeautifulSoup parsing.
"""
import asyncio
import re
from typing import Optional, List, Dict, Any
from loguru import logger
import aiohttp
from bs4 import BeautifulSoup


# ── MCA21 — Ministry of Corporate Affairs portal ─────────────
async def search_mca21(name: str) -> List[Dict[str, Any]]:
    """
    Search MCA21 for company directors and CIN numbers.
    Uses public search endpoint — no auth needed.
    """
    results = []
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()

            await page.goto(
                "https://www.mca.gov.in/mcafoportal/viewCompanyMasterData.do",
                timeout=30000
            )
            await page.wait_for_load_state("networkidle")

            # Fill search form
            try:
                await page.fill('input[name="companyName"]', name)
                await page.click('button[type="submit"], input[type="submit"]')
                await page.wait_for_load_state("networkidle", timeout=15000)

                html = await page.content()
                soup = BeautifulSoup(html, "lxml")
                rows = soup.select("table tr")
                for row in rows[1:]:   # skip header
                    cols = [td.get_text(strip=True) for td in row.select("td")]
                    if len(cols) >= 3:
                        results.append({
                            "cin":     cols[0] if len(cols) > 0 else "",
                            "company": cols[1] if len(cols) > 1 else "",
                            "status":  cols[2] if len(cols) > 2 else "",
                            "source":  "mca21",
                        })
            except Exception as inner_e:
                logger.debug(f"[MCA21] Form interaction failed: {inner_e}")

            await browser.close()
    except Exception as e:
        logger.warning(f"[MCA21] {name}: {e}")

    logger.info(f"[MCA21] {name} → {len(results)} records")
    return results


# ── eCourts public API ────────────────────────────────────────
async def search_ecourts(party_name: str) -> List[Dict[str, Any]]:
    """
    Search eCourts for case filings by party name.
    Uses public search — no authentication required.
    """
    results = []
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()

            await page.goto(
                "https://services.ecourts.gov.in/ecourtindiaHC/",
                timeout=30000
            )
            await page.wait_for_load_state("networkidle")

            try:
                # Navigate to party name search
                await page.click('text=Party Name', timeout=5000)
                await page.fill('input#petitioner_respondant_name', party_name)
                await page.click('button#searchbtn', timeout=5000)
                await page.wait_for_load_state("networkidle", timeout=15000)

                html = await page.content()
                soup = BeautifulSoup(html, "lxml")
                for row in soup.select("table.case_details tr"):
                    cols = [td.get_text(strip=True) for td in row.select("td")]
                    if len(cols) >= 2:
                        results.append({
                            "case_number": cols[0],
                            "parties":     cols[1] if len(cols) > 1 else "",
                            "court":       cols[2] if len(cols) > 2 else "",
                            "status":      cols[3] if len(cols) > 3 else "",
                            "source":      "ecourts",
                        })
            except Exception as inner_e:
                logger.debug(f"[eCourts] Interaction failed: {inner_e}")

            await browser.close()
    except Exception as e:
        logger.warning(f"[eCourts] {party_name}: {e}")

    logger.info(f"[eCourts] {party_name} → {len(results)} records")
    return results


# ── Gazette of India — PDF text extraction ────────────────────
async def search_gazette(
    name: str,
    session: aiohttp.ClientSession,
) -> List[Dict[str, Any]]:
    """
    Search Gazette of India for name mentions.
    Uses the egazette.nic.in search API.
    """
    results = []
    search_url = "https://egazette.gov.in/WriteReadData/SearchGazette.aspx"
    try:
        async with session.get(
            f"https://egazette.gov.in/SearchByKeyword.aspx?keyword={name}",
            timeout=aiohttp.ClientTimeout(total=20),
        ) as r:
            if r.status == 200:
                html = await r.text()
                soup = BeautifulSoup(html, "lxml")
                for link in soup.select("a[href*='.pdf']")[:10]:
                    results.append({
                        "title":  link.get_text(strip=True),
                        "url":    link.get("href", ""),
                        "source": "gazette",
                    })
    except Exception as e:
        logger.warning(f"[Gazette] {name}: {e}")

    logger.info(f"[Gazette] {name} → {len(results)} entries")
    return results


# ── Google News scrape ────────────────────────────────────────
async def search_google_news(
    name: str,
    session: aiohttp.ClientSession,
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    """
    Scrape Google News RSS for name-based news archive.
    No API key needed — uses RSS endpoint.
    """
    results = []
    rss_url = (
        f"https://news.google.com/rss/search"
        f"?q={name.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
    )
    try:
        async with session.get(
            rss_url,
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"User-Agent": "Mozilla/5.0"},
        ) as r:
            if r.status == 200:
                xml = await r.text()
                soup = BeautifulSoup(xml, "xml")
                for item in soup.select("item")[:max_results]:
                    results.append({
                        "title":     _text(item, "title"),
                        "snippet":   _text(item, "description"),
                        "url":       _text(item, "link"),
                        "publisher": _text(item, "source"),
                        "published": _text(item, "pubDate"),
                        "source":    "google_news",
                    })
    except Exception as e:
        logger.warning(f"[GNews] {name}: {e}")

    logger.info(f"[GNews] {name} → {len(results)} articles")
    return results


# ── DuckDuckGo dork search (fallback news source) ────────────
async def dork_search(
    query: str,
    session: aiohttp.ClientSession,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    DuckDuckGo HTML scrape — unlimited, no rate limits.
    Used for dork queries like 'site:mca.gov.in <name>'.
    """
    results = []
    url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"User-Agent": "Mozilla/5.0"},
        ) as r:
            if r.status == 200:
                soup = BeautifulSoup(await r.text(), "lxml")
                for result in soup.select(".result__body")[:max_results]:
                    title_el = result.select_one(".result__title")
                    snippet_el = result.select_one(".result__snippet")
                    url_el = result.select_one(".result__url")
                    results.append({
                        "title":   title_el.get_text(strip=True) if title_el else "",
                        "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                        "url":     url_el.get_text(strip=True) if url_el else "",
                        "source":  "duckduckgo",
                    })
    except Exception as e:
        logger.warning(f"[DDG] {query}: {e}")
    return results


def _text(tag: Any, selector: str) -> str:
    el = tag.find(selector)
    return el.get_text(strip=True) if el else ""
