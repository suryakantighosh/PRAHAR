"""
prahar/modules/c08_brief/phc.py
Provenance Hash Chain (PHC) — Novel Patent Method.

Links every claim in the intelligence brief back through the
processing chain to the original raw scraped record via SHA-256.
The brief is a cryptographically verifiable chain-of-custody artifact.
"""
import hashlib
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ProvenanceNode:
    """One link in the hash chain."""
    node_id:     str          # UUID of this record
    node_type:   str          # raw_data / entity / identity / score
    content_hash: str         # SHA-256 of this node's content
    parent_hash:  Optional[str] = None   # hash of parent node
    chain_hash:   Optional[str] = None   # SHA-256(parent_hash + content_hash)
    metadata:     Dict[str, Any] = field(default_factory=dict)


def sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def hash_content(content: Any) -> str:
    """Deterministic SHA-256 of any JSON-serialisable content."""
    serialised = json.dumps(content, sort_keys=True, ensure_ascii=False,
                            default=str)
    return sha256(serialised)


def build_chain_hash(parent_hash: Optional[str], content_hash: str) -> str:
    """
    Chain hash = SHA-256(parent_hash || content_hash).
    If no parent, chain_hash = content_hash.
    """
    if parent_hash is None:
        return content_hash
    return sha256(parent_hash + content_hash)


def build_provenance_chain(
    nodes: List[Dict[str, Any]],
) -> List[ProvenanceNode]:
    """
    Build a linked provenance chain from a list of processing nodes.
    Each node links to the previous via chain_hash.
    nodes: list of dicts with keys: node_id, node_type, content
    """
    chain: List[ProvenanceNode] = []
    prev_chain_hash: Optional[str] = None

    for node in nodes:
        content_hash = hash_content(node.get("content", {}))
        chain_hash   = build_chain_hash(prev_chain_hash, content_hash)

        pnode = ProvenanceNode(
            node_id=node["node_id"],
            node_type=node["node_type"],
            content_hash=content_hash,
            parent_hash=prev_chain_hash,
            chain_hash=chain_hash,
            metadata=node.get("metadata", {}),
        )
        chain.append(pnode)
        prev_chain_hash = chain_hash

    return chain


def verify_chain(chain: List[ProvenanceNode]) -> bool:
    """
    Verify integrity of a provenance chain.
    Returns True if every chain_hash is correctly derived.
    """
    prev_chain_hash: Optional[str] = None

    for node in chain:
        expected = build_chain_hash(prev_chain_hash, node.content_hash)
        if node.chain_hash != expected:
            return False
        prev_chain_hash = node.chain_hash

    return True


def chain_to_dict(chain: List[ProvenanceNode]) -> List[Dict[str, Any]]:
    """Serialise chain to JSON-ready list."""
    return [
        {
            "node_id":      n.node_id,
            "node_type":    n.node_type,
            "content_hash": n.content_hash,
            "parent_hash":  n.parent_hash,
            "chain_hash":   n.chain_hash,
            "metadata":     n.metadata,
        }
        for n in chain
    ]
