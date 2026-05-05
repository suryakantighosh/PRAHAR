"""
prahar/modules/c02_identity/connectors.py
C-02 connectors: Sherlock, Maigret, HaveIBeenPwned, HackerTarget, phonenumbers.
"""
import asyncio
import subprocess
import json
import os
from typing import Optional, List, Dict
import aiohttp
import phonenumbers
from phonenumbers import geocoder, carrier, number_type, PhoneNumberType
from loguru import logger


# ── Sherlock (subprocess call) ───────────────────────────────
async def run_sherlock(username: str, timeout: int = 60) -> Dict:
    """
    Run Sherlock CLI to find username across 300+ platforms.
    Returns dict of {platform: url} for found profiles.
    """
    loop = asyncio.get_event_loop()

    def _sherlock_sync():
        try:
            result = subprocess.run(
                ["sherlock", username, "--print-found", "--no-color",
                 "--timeout", "10"],
                capture_output=True, text=True, timeout=timeout
            )
            found = {}
            for line in result.stdout.splitlines():
                # Sherlock output: "[+] Platform: https://..."
                if line.startswith("[+]"):
                    parts = line[4:].split(": ", 1)
                    if len(parts) == 2:
                        found[parts[0].strip()] = parts[1].strip()
            return found
        except subprocess.TimeoutExpired:
            logger.warning(f"[Sherlock] Timeout for {username}")
            return {}
        except FileNotFoundError:
            logger.warning("[Sherlock] Not installed — skipping")
            return {}
        except Exception as e:
            logger.error(f"[Sherlock] Error: {e}")
            return {}

    result = await loop.run_in_executor(None, _sherlock_sync)
    logger.info(f"[Sherlock] {username} → {len(result)} platforms found")
    return {"source": "sherlock", "username": username, "found": result}


# ── Maigret (library import) ─────────────────────────────────
async def run_maigret(username: str) -> Dict:
    """
    Run Maigret for deeper username + profile data from 2500+ sites.
    """
    loop = asyncio.get_event_loop()

    def _maigret_sync():
        try:
            import maigret
            # Maigret has a simple search API
            results = {}
            # Use subprocess for stability with async
            result = subprocess.run(
                ["maigret", username, "--no-color", "--timeout", "10",
                 "--format", "json"],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                # Parse JSON output if available
                try:
                    data = json.loads(result.stdout)
                    return {"source": "maigret", "username": username,
                            "data": data}
                except json.JSONDecodeError:
                    pass
            return {"source": "maigret", "username": username, "data": {}}
        except FileNotFoundError:
            logger.warning("[Maigret] Not installed — skipping")
            return {}
        except Exception as e:
            logger.error(f"[Maigret] Error: {e}")
            return {}

    result = await loop.run_in_executor(None, _maigret_sync)
    logger.info(f"[Maigret] {username} complete")
    return result


# ── HaveIBeenPwned (free, no key needed for basic check) ────
async def check_hibp(email: str,
                     session: aiohttp.ClientSession) -> Dict:
    """
    Check email against HaveIBeenPwned breach database.
    Free tier: 10 req/min, no API key needed for breach check.
    """
    url = f"https://haveibeenpwned.com/api/v3/breachedaccount/{email}"
    headers = {
        "User-Agent": "PraharOSINT/2.0",
        "hibp-api-key": os.getenv("HIBP_API_KEY", ""),
    }
    try:
        async with session.get(
            url, headers=headers,
            timeout=aiohttp.ClientTimeout(total=10)
        ) as r:
            if r.status == 200:
                breaches = await r.json()
                logger.info(f"[HIBP] {email} → {len(breaches)} breaches")
                return {
                    "source": "hibp",
                    "email": email,
                    "breaches": breaches,
                }
            elif r.status == 404:
                return {"source": "hibp", "email": email, "breaches": []}
            elif r.status == 429:
                logger.warning("[HIBP] Rate limited — wait 60s")
                await asyncio.sleep(60)
                return {}
    except Exception as e:
        logger.warning(f"[HIBP] {email} error: {e}")
    return {}


# ── Phone number enrichment ───────────────────────────────────
def enrich_phone(number_str: str, default_region: str = "IN") -> Dict:
    """
    Enrich a phone number with carrier, region, and type.
    Uses Google's libphonenumbers — fully local, no API needed.
    """
    try:
        parsed = phonenumbers.parse(number_str, default_region)
        if not phonenumbers.is_valid_number(parsed):
            return {"source": "phonenumbers", "number": number_str,
                    "valid": False}

        num_type_val = number_type(parsed)
        type_map = {
            PhoneNumberType.MOBILE:     "MOBILE",
            PhoneNumberType.FIXED_LINE: "FIXED_LINE",
            PhoneNumberType.VOIP:       "VOIP",
            PhoneNumberType.TOLL_FREE:  "TOLL_FREE",
        }

        return {
            "source":    "phonenumbers",
            "number":    phonenumbers.format_number(
                             parsed, phonenumbers.PhoneNumberFormat.E164),
            "carrier":   carrier.name_for_number(parsed, "en"),
            "region":    geocoder.description_for_number(parsed, "en"),
            "type":      type_map.get(num_type_val, "UNKNOWN"),
            "valid":     True,
        }
    except Exception as e:
        logger.warning(f"[Phone] {number_str} error: {e}")
        return {}
