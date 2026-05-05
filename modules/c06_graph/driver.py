"""
prahar/modules/c06_graph/driver.py
Neo4j async driver wrapper.
Handles connection, retries, and context management.
"""
import os
from typing import Optional, List, Dict, Any
from loguru import logger
from neo4j import AsyncGraphDatabase, AsyncDriver

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "prahar_neo4j")

_driver: Optional[AsyncDriver] = None


async def get_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_pool_size=20,
        )
        logger.info(f"[C-06] Neo4j driver connected to {NEO4J_URI}")
    return _driver


async def close_driver():
    global _driver
    if _driver:
        await _driver.close()
        _driver = None


async def run_query(
    cypher: str,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run a Cypher query, return list of record dicts."""
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(cypher, params or {})
        return [dict(record) async for record in result]


async def run_write(
    cypher: str,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """Run a write Cypher query (no return value needed)."""
    driver = await get_driver()
    async with driver.session() as session:
        await session.run(cypher, params or {})


async def ensure_indexes() -> None:
    """Create Neo4j indexes for fast lookups — idempotent."""
    indexes = [
        "CREATE INDEX identity_case IF NOT EXISTS FOR (n:Identity) ON (n.case_id)",
        "CREATE INDEX fragment_case IF NOT EXISTS FOR (n:Fragment) ON (n.case_id)",
        "CREATE INDEX entity_case   IF NOT EXISTS FOR (n:Entity)   ON (n.case_id)",
        "CREATE INDEX evidence_case IF NOT EXISTS FOR (n:Evidence) ON (n.case_id)",
        "CREATE CONSTRAINT case_id_unique IF NOT EXISTS FOR (n:Case) REQUIRE n.case_id IS UNIQUE",
    ]
    for cypher in indexes:
        try:
            await run_write(cypher)
        except Exception as e:
            logger.debug(f"[C-06] Index already exists or error: {e}")
    logger.info("[C-06] Neo4j indexes ensured")
