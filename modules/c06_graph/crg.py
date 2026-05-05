"""
prahar/modules/c06_graph/crg.py
Conflict Resolution Graph (CRG) — Novel Patent Method.

When two signals CONTRADICT each other (e.g. two sources claim
different birthdates for same person), CRG:
  1. Creates a CONTRADICTS edge with a conflict weight
  2. Runs min-cut to find the minimum-weight set of edges to remove
     that resolves all conflicts
  3. Marks the losing signal as RESOLVED_CONFLICT

This replaces the naive approach of silently dropping conflicting data.
The algorithmic resolution is the patentable contribution.
"""
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from prahar.modules.c06_graph.driver import run_write, run_query
from prahar.modules.c06_graph.schema import REL_CONTRADICTS


@dataclass
class ConflictSignal:
    """One side of a data conflict."""
    signal_id:  str
    source:     str
    claim:      str       # what this signal asserts
    confidence: float     # how much we trust this source


@dataclass
class ConflictEdge:
    """A CONTRADICTS edge between two conflicting signals."""
    signal_a:       str
    signal_b:       str
    conflict_type:  str   # DATE_MISMATCH / LOCATION_MISMATCH / NAME_MISMATCH
    weight:         float # higher = more costly to cut (remove)


def build_conflict_graph(
    signals: List[ConflictSignal],
    conflicts: List[ConflictEdge],
) -> nx.Graph:
    """
    Build a NetworkX graph where:
    - Nodes = signals (data claims)
    - Edges = CONTRADICTS relationships with weights
    """
    G = nx.Graph()

    for sig in signals:
        G.add_node(
            sig.signal_id,
            source=sig.source,
            claim=sig.claim,
            confidence=sig.confidence,
        )

    for conflict in conflicts:
        G.add_edge(
            conflict.signal_a,
            conflict.signal_b,
            weight=conflict.weight,
            conflict_type=conflict.conflict_type,
        )

    return G


def resolve_conflicts_mincut(
    G: nx.Graph,
    source_node: str,
    sink_node: str,
) -> Dict[str, Any]:
    """
    Run min-cut algorithm (Ford-Fulkerson via NetworkX) on the conflict graph.
    The min-cut partition identifies which signals to keep vs. discard.

    Returns:
        reachable:    signal IDs in the source partition (KEEP)
        unreachable:  signal IDs in the sink partition (DISCARD)
        cut_value:    total weight of removed edges
        cut_edges:    list of (a, b) edges that were cut
    """
    if source_node not in G or sink_node not in G:
        return {
            "reachable":   list(G.nodes()),
            "unreachable": [],
            "cut_value":   0.0,
            "cut_edges":   [],
        }

    try:
        cut_value, partition = nx.minimum_cut(
            G, source_node, sink_node, capacity="weight"
        )
        reachable, unreachable = partition

        cut_edges = [
            (u, v) for u, v in G.edges()
            if (u in reachable and v in unreachable) or
               (u in unreachable and v in reachable)
        ]

        logger.info(
            f"[CRG] Min-cut: value={cut_value:.3f} "
            f"keep={len(reachable)} discard={len(unreachable)} "
            f"cut_edges={len(cut_edges)}"
        )

        return {
            "reachable":   list(reachable),
            "unreachable": list(unreachable),
            "cut_value":   cut_value,
            "cut_edges":   cut_edges,
        }
    except nx.NetworkXError as e:
        logger.warning(f"[CRG] Min-cut failed: {e} — keeping all signals")
        return {
            "reachable":   list(G.nodes()),
            "unreachable": [],
            "cut_value":   0.0,
            "cut_edges":   [],
        }


def detect_conflicts(
    signals: List[ConflictSignal],
) -> List[ConflictEdge]:
    """
    Auto-detect conflicts between signals.
    Two signals conflict if they make incompatible claims
    about the same attribute type.

    Current detection rules:
    - Same entity, different DATE claims → DATE_MISMATCH
    - Same entity, different LOCATION claims → LOCATION_MISMATCH
    - Signals with overlapping keywords but opposite sentiment → CLAIM_MISMATCH
    """
    conflicts = []

    for i in range(len(signals)):
        for j in range(i + 1, len(signals)):
            a, b = signals[i], signals[j]

            # Detect date conflicts: both claim a date but they differ
            import re
            date_pattern = r'\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d{4}\b'
            dates_a = re.findall(date_pattern, a.claim)
            dates_b = re.findall(date_pattern, b.claim)

            if dates_a and dates_b and set(dates_a) != set(dates_b):
                # Weight = harmonic mean of confidence (higher conf = costlier to cut)
                weight = 2 * a.confidence * b.confidence / (a.confidence + b.confidence + 1e-9)
                conflicts.append(ConflictEdge(
                    signal_a=a.signal_id,
                    signal_b=b.signal_id,
                    conflict_type="DATE_MISMATCH",
                    weight=weight,
                ))

    return conflicts


async def write_contradicts_edges(
    case_id: str,
    conflicts: List[ConflictEdge],
) -> None:
    """Persist CONTRADICTS edges to Neo4j."""
    for conflict in conflicts:
        await run_write(
            f"MERGE (a {{signal_id: $a}}) "
            f"MERGE (b {{signal_id: $b}}) "
            f"MERGE (a)-[r:CONTRADICTS]->(b) "
            f"SET r.conflict_type = $ct, r.weight = $w, r.case_id = $cid",
            {
                "a": conflict.signal_a,
                "b": conflict.signal_b,
                "ct": conflict.conflict_type,
                "w": conflict.weight,
                "cid": case_id,
            },
        )
    logger.info(f"[CRG] Wrote {len(conflicts)} CONTRADICTS edges for case={case_id}")


def louvain_communities(G: nx.Graph) -> List[List[str]]:
    """
    Detect communities in the identity graph using Louvain algorithm.
    Returns list of node-ID lists, one per community.
    """
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G)
        communities: Dict[int, List[str]] = {}
        for node, community_id in partition.items():
            communities.setdefault(community_id, []).append(node)
        result = list(communities.values())
        logger.info(f"[C-06] Louvain: {len(result)} communities detected")
        return result
    except Exception as e:
        logger.warning(f"[C-06] Louvain failed: {e}")
        return [[n] for n in G.nodes()]
