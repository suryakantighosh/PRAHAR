"""
prahar/modules/c06_graph/writer.py
Graph edge writer — translates PRAHAR data into Neo4j nodes and edges.
All writes are idempotent (MERGE not CREATE).
"""
from datetime import datetime
from typing import List, Dict, Any
from uuid import UUID
from loguru import logger

from prahar.modules.c06_graph.driver import run_write
from prahar.modules.c06_graph.schema import (
    NODE_IDENTITY, NODE_FRAGMENT, NODE_ENTITY, NODE_EVIDENCE, NODE_CASE,
    REL_LINKED_TO, REL_MENTIONED_IN, REL_BELONGS_TO,
    REL_SHARES_PLATFORM, REL_CORROBORATES,
    PROP_CASE_ID, PROP_CPIF_SCORE, PROP_CONFIDENCE,
    PROP_LABEL, PROP_TEXT, PROP_PLATFORM, PROP_CREATED_AT,
)


async def write_case_node(case_id: str) -> None:
    await run_write(
        f"MERGE (c:{NODE_CASE} {{case_id: $case_id}}) "
        f"SET c.created_at = $ts",
        {"case_id": case_id, "ts": datetime.utcnow().isoformat()},
    )


async def write_identity_node(
    case_id: str,
    identity_id: str,
    cpif_score: float,
    platforms: List[str],
) -> None:
    await run_write(
        f"MERGE (i:{NODE_IDENTITY} {{identity_id: $iid}}) "
        f"SET i.case_id = $cid, i.cpif_score = $score, "
        f"    i.platforms = $platforms, i.updated_at = $ts",
        {
            "iid": identity_id, "cid": case_id,
            "score": cpif_score, "platforms": platforms,
            "ts": datetime.utcnow().isoformat(),
        },
    )
    # Link to case
    await run_write(
        f"MATCH (i:{NODE_IDENTITY} {{identity_id: $iid}}) "
        f"MATCH (c:{NODE_CASE} {{case_id: $cid}}) "
        f"MERGE (i)-[:{REL_BELONGS_TO}]->(c)",
        {"iid": identity_id, "cid": case_id},
    )


async def write_fragment_node(
    case_id: str,
    fragment_id: str,
    platform: str,
    username: str,
    identity_id: str,
    confidence: float,
) -> None:
    await run_write(
        f"MERGE (f:{NODE_FRAGMENT} {{fragment_id: $fid}}) "
        f"SET f.case_id = $cid, f.platform = $platform, "
        f"    f.username = $username, f.confidence = $conf",
        {
            "fid": fragment_id, "cid": case_id,
            "platform": platform, "username": username,
            "conf": confidence,
        },
    )
    await run_write(
        f"MATCH (f:{NODE_FRAGMENT} {{fragment_id: $fid}}) "
        f"MATCH (i:{NODE_IDENTITY} {{identity_id: $iid}}) "
        f"MERGE (f)-[:{REL_LINKED_TO}]->(i)",
        {"fid": fragment_id, "iid": identity_id},
    )


async def write_entity_node(
    case_id: str,
    entity_id: str,
    text: str,
    label: str,
    canonical_form: str,
    count: int,
) -> None:
    await run_write(
        f"MERGE (e:{NODE_ENTITY} {{entity_id: $eid}}) "
        f"SET e.case_id = $cid, e.text = $text, "
        f"    e.label = $label, e.canonical_form = $cf, e.count = $count",
        {
            "eid": entity_id, "cid": case_id, "text": text,
            "label": label, "cf": canonical_form, "count": count,
        },
    )


async def write_shares_platform_edge(
    identity_id_a: str,
    identity_id_b: str,
    platform: str,
    weight: float,
) -> None:
    """Two identities found on same platform — SHARES_PLATFORM edge."""
    await run_write(
        f"MATCH (a:{NODE_IDENTITY} {{identity_id: $a}}) "
        f"MATCH (b:{NODE_IDENTITY} {{identity_id: $b}}) "
        f"MERGE (a)-[r:{REL_SHARES_PLATFORM}]->(b) "
        f"SET r.platform = $platform, r.weight = $weight",
        {"a": identity_id_a, "b": identity_id_b,
         "platform": platform, "weight": weight},
    )


async def write_corroborates_edge(
    entity_id: str,
    evidence_id: str,
    confidence: float,
) -> None:
    await run_write(
        f"MATCH (e:{NODE_ENTITY} {{entity_id: $eid}}) "
        f"MATCH (v:{NODE_EVIDENCE} {{evidence_id: $vid}}) "
        f"MERGE (e)-[r:{REL_CORROBORATES}]->(v) "
        f"SET r.confidence = $conf",
        {"eid": entity_id, "vid": evidence_id, "conf": confidence},
    )
