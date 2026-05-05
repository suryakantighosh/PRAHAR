"""
prahar/modules/c01_ingestion/seed.py
Seed Identity Hash (SIH) — deterministic UUID from any seed input.
Same seed always produces the same case root UUID.
"""
import hashlib
import uuid
import re


SEED_TYPES = ("name", "email", "phone", "username", "domain", "face")


def normalise_seed(seed_type: str, value: str) -> str:
    """Canonicalise seed before hashing so typos don't split cases."""
    value = value.strip().lower()
    if seed_type == "email":
        # Strip plus-addressing: user+tag@example.com → user@example.com
        local, _, domain = value.partition("@")
        local = re.sub(r"\+.*$", "", local)
        value = f"{local}@{domain}"
    elif seed_type == "phone":
        value = re.sub(r"[^\d+]", "", value)    # digits and leading + only
    elif seed_type == "domain":
        value = re.sub(r"^www\.", "", value)    # strip www. prefix
    return value


def make_sih(seed_type: str, value: str) -> uuid.UUID:
    """
    Seed Identity Hash — deterministic UUID v5 from (seed_type, normalised_value).
    Anchors all downstream records to a stable root node.
    """
    if seed_type not in SEED_TYPES:
        raise ValueError(
            f"Unknown seed type '{seed_type}'. Choose from {SEED_TYPES}"
        )

    normalised = normalise_seed(seed_type, value)
    namespace = uuid.UUID("b1d2e3f4-a5b6-c7d8-e9f0-a1b2c3d4e5f6")  # PRAHAR namespace
    return uuid.uuid5(namespace, f"{seed_type}:{normalised}")


def make_case_id() -> uuid.UUID:
    """Random UUID for a new investigation case (not seed-derived)."""
    return uuid.uuid4()
