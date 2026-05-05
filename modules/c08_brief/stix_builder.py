"""
prahar/modules/c08_brief/stix_builder.py
STIX 2.1 JSON bundle builder.
Converts PRAHAR identity graph into standard threat-intel format.
"""
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _stix_id(obj_type: str) -> str:
    return f"{obj_type}--{uuid4()}"


def make_identity_object(
    name: str,
    identity_class: str = "individual",
    description: str = "",
    confidence: int = 50,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """STIX 2.1 identity SDO."""
    return {
        "type":             "identity",
        "spec_version":     "2.1",
        "id":               _stix_id("identity"),
        "created":          _now(),
        "modified":         _now(),
        "name":             name,
        "identity_class":   identity_class,
        "description":      description,
        "confidence":       confidence,
        "labels":           labels or [],
    }


def make_indicator_object(
    pattern: str,
    pattern_type: str = "stix",
    name: str = "",
    description: str = "",
    confidence: int = 50,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """STIX 2.1 indicator SDO."""
    return {
        "type":             "indicator",
        "spec_version":     "2.1",
        "id":               _stix_id("indicator"),
        "created":          _now(),
        "modified":         _now(),
        "name":             name,
        "description":      description,
        "pattern":          pattern,
        "pattern_type":     pattern_type,
        "valid_from":       _now(),
        "confidence":       confidence,
        "labels":           labels or ["osint"],
    }


def make_relationship(
    source_id: str,
    target_id: str,
    rel_type: str = "related-to",
    description: str = "",
    confidence: int = 50,
) -> Dict[str, Any]:
    """STIX 2.1 relationship SRO."""
    return {
        "type":              "relationship",
        "spec_version":      "2.1",
        "id":                _stix_id("relationship"),
        "created":           _now(),
        "modified":          _now(),
        "relationship_type": rel_type,
        "source_ref":        source_id,
        "target_ref":        target_id,
        "description":       description,
        "confidence":        confidence,
    }


def make_note(
    content: str,
    object_refs: List[str],
    authors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """STIX 2.1 note SDO — for analyst commentary."""
    return {
        "type":         "note",
        "spec_version": "2.1",
        "id":           _stix_id("note"),
        "created":      _now(),
        "modified":     _now(),
        "content":      content,
        "authors":      authors or ["PRAHAR/v2"],
        "object_refs":  object_refs,
    }


def build_stix_bundle(objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wrap STIX objects in a bundle."""
    return {
        "type":         "bundle",
        "id":           _stix_id("bundle"),
        "spec_version": "2.1",
        "objects":      objects,
    }


def brief_to_stix(
    case_id: str,
    subject_name: str,
    platforms: List[str],
    risk_level: str,
    confidence_score: float,
    risk_flags: List[str],
    breach_names: List[str],
    provenance_hash: str,
) -> str:
    """
    Convert a PRAHAR intelligence brief into a STIX 2.1 JSON bundle string.
    """
    confidence_int = int(confidence_score * 100)

    # Primary identity object
    subject_obj = make_identity_object(
        name=subject_name,
        identity_class="individual",
        description=f"PRAHAR OSINT subject. Risk: {risk_level}. "
                    f"Platforms: {', '.join(platforms[:5])}",
        confidence=confidence_int,
        labels=risk_flags[:5],
    )

    objects = [subject_obj]

    # Platform indicators
    for platform in platforms[:10]:
        indicator = make_indicator_object(
            pattern=f"[user-account:user_id = '{subject_name}' AND "
                    f"user-account:account_type = '{platform}']",
            name=f"{subject_name} on {platform}",
            confidence=confidence_int,
            labels=["osint", "social-media"],
        )
        objects.append(indicator)
        objects.append(make_relationship(
            subject_obj["id"], indicator["id"],
            rel_type="uses",
            description=f"Subject uses {platform} account",
            confidence=confidence_int,
        ))

    # Breach indicators
    for breach in breach_names[:5]:
        indicator = make_indicator_object(
            pattern=f"[email-message:subject = '{breach}']",
            name=f"Data breach: {breach}",
            description=f"Subject's data exposed in {breach} breach",
            confidence=min(95, confidence_int + 10),
            labels=["osint", "data-breach"],
        )
        objects.append(indicator)

    # Provenance note
    objects.append(make_note(
        content=f"PRAHAR v2 intelligence brief. "
                f"Case: {case_id}. "
                f"Provenance hash: {provenance_hash[:16]}...",
        object_refs=[subject_obj["id"]],
    ))

    bundle = build_stix_bundle(objects)
    return json.dumps(bundle, indent=2)
