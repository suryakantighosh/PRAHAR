"""
prahar/modules/c02_identity/engine.py
C-02 Identity Resolver + CPIF engine.
Resolves a seed into Identity Fragment Nodes, fuses into CINs.
"""
import asyncio
from typing import Optional, List
from uuid import UUID
from loguru import logger
import aiohttp

from prahar.modules.c02_identity.cpif import (
    IdentitySignal, cpif_score, fuse_fragments, DEFAULT_THETA
)
from prahar.modules.c02_identity.connectors import (
    run_sherlock, run_maigret, check_hibp, enrich_phone
)
from prahar.modules.c01_ingestion.seed import make_case_id
from prahar.modules.c01_ingestion.audit import store_record
from prahar.core.db import AsyncSessionLocal
from prahar.models.identity import (
    IdentityFragment, ConsolidatedIdentity, BreachRecord, PhoneRecord
)


async def resolve_username(
    username: str,
    case_id: Optional[UUID] = None,
    theta: float = DEFAULT_THETA,
) -> dict:
    """
    Full username resolution pipeline:
    1. Sherlock + Maigret → platform profiles
    2. Build IFN per platform
    3. CPIF fusion → CINs
    4. Persist all to DB
    """
    if case_id is None:
        case_id = make_case_id()

    logger.info(f"[C-02] Resolving username={username} case={case_id}")

    # Run Sherlock and Maigret concurrently
    sherlock_result, maigret_result = await asyncio.gather(
        run_sherlock(username),
        run_maigret(username),
    )

    # Build one IdentitySignal per platform found
    fragments: List[IdentitySignal] = []
    all_platforms = {}

    if sherlock_result.get("found"):
        all_platforms.update(sherlock_result["found"])
    if maigret_result.get("data"):
        for platform, data in maigret_result["data"].items():
            if isinstance(data, dict) and data.get("url_user"):
                all_platforms[platform] = data["url_user"]

    for platform, url in all_platforms.items():
        sig = IdentitySignal(
            platform=platform,
            username=username,
            uncertainty=0.8,   # platform-confirmed = lower uncertainty
        )
        fragments.append(sig)

    # CPIF fusion
    groups = fuse_fragments(fragments, theta=theta)
    logger.info(f"[C-02] {len(fragments)} fragments → {len(groups)} CINs")

    # Persist to DB
    async with AsyncSessionLocal() as db:
        # Save raw fragments
        for frag in fragments:
            ifn = IdentityFragment(
                case_id=case_id,
                platform=frag.platform,
                username=frag.username,
                uncertainty=frag.uncertainty,
                meta={"url": all_platforms.get(frag.platform)},
            )
            db.add(ifn)

        # Save consolidated identities
        cin_ids = []
        for group in groups:
            # Score within the group (average pairwise score)
            if len(group) == 1:
                group_score = 0.85   # solo fragment — confident single source
            else:
                scores = []
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        r = cpif_score(group[i], group[j])
                        scores.append(r["score"])
                group_score = sum(scores) / len(scores) if scores else 0.0

            cin = ConsolidatedIdentity(
                case_id=case_id,
                cpif_score=round(group_score, 4),
                signals_used=[f.platform for f in group],
            )
            db.add(cin)
            cin_ids.append(cin)

        await db.commit()

    return {
        "case_id": str(case_id),
        "username": username,
        "platforms_found": len(all_platforms),
        "fragments": len(fragments),
        "consolidated_identities": len(groups),
        "platforms": list(all_platforms.keys()),
    }


async def resolve_email(
    email: str,
    case_id: Optional[UUID] = None,
) -> dict:
    """Email seed: HIBP breach check + IFN creation."""
    if case_id is None:
        case_id = make_case_id()

    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        hibp_result = await check_hibp(email, session)

    breaches = hibp_result.get("breaches", [])

    async with AsyncSessionLocal() as db:
        # Save IFN for this email
        ifn = IdentityFragment(
            case_id=case_id,
            platform="email",
            email=email,
            uncertainty=0.1,   # email is high-confidence seed
        )
        db.add(ifn)

        # Save breach records
        for breach in breaches:
            br = BreachRecord(
                case_id=case_id,
                email=email,
                breach_name=breach.get("Name"),
                breach_date=breach.get("BreachDate"),
                data_classes=breach.get("DataClasses", []),
            )
            db.add(br)

        await db.commit()

    logger.success(
        f"[C-02] email={email} case={case_id} "
        f"breaches={len(breaches)}"
    )
    return {
        "case_id": str(case_id),
        "email": email,
        "breaches_found": len(breaches),
        "breach_names": [b.get("Name") for b in breaches],
    }


async def resolve_phone(
    phone: str,
    case_id: Optional[UUID] = None,
) -> dict:
    """Phone seed: carrier + region + type enrichment."""
    if case_id is None:
        case_id = make_case_id()

    enriched = enrich_phone(phone)

    async with AsyncSessionLocal() as db:
        ifn = IdentityFragment(
            case_id=case_id,
            platform="phone",
            phone=enriched.get("number", phone),
            uncertainty=0.1,
            meta=enriched,
        )
        db.add(ifn)

        pr = PhoneRecord(
            case_id=case_id,
            number=enriched.get("number", phone),
            carrier=enriched.get("carrier"),
            region=enriched.get("region"),
            num_type=enriched.get("type"),
        )
        db.add(pr)
        await db.commit()

    logger.success(f"[C-02] phone={phone} enriched={enriched.get('valid')}")
    return {
        "case_id": str(case_id),
        "phone": phone,
        "enriched": enriched,
    }
