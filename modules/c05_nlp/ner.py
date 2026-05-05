"""
prahar/modules/c05_nlp/ner.py
Named Entity Recognition pipeline using spaCy en_core_web_sm.
Handles: extraction, deduplication, co-reference resolution (basic),
and canonical form normalisation.
"""
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger


# Lazy-loaded spaCy model — loads once per process
_nlp = None

# Labels we care about for OSINT
OSINT_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "NORP",
    "FAC", "PRODUCT", "EVENT", "LAW", "DATE",
    "MONEY", "CARDINAL",
}


def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        _nlp = spacy.load("en_core_web_sm")
        logger.info("[C-05] spaCy en_core_web_sm loaded")
    return _nlp


@dataclass
class ExtractedEntity:
    text:           str
    label:          str
    canonical_form: str
    count:          int = 1
    sources:        List[str] = field(default_factory=list)
    aliases:        List[str] = field(default_factory=list)


def normalise_entity(text: str, label: str) -> str:
    """
    Produce canonical form of an entity string.
    - Strip titles/honorifics from PERSON
    - Uppercase ORG acronyms
    - Normalise GPE to title case
    """
    text = text.strip()

    if label == "PERSON":
        # Strip common honorifics
        for title in ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Shri", "Smt.",
                      "Mr", "Mrs", "Ms", "Dr", "Prof"]:
            text = re.sub(rf"^{re.escape(title)}\s+", "", text, flags=re.IGNORECASE)
        return text.title()

    if label == "ORG":
        # Keep ALL-CAPS acronyms, title-case others
        if text.isupper() and len(text) <= 6:
            return text
        return text.title()

    if label in ("GPE", "LOC"):
        return text.title()

    return text


def extract_entities(text: str, source_id: str = "") -> List[ExtractedEntity]:
    """
    Run NER on text, return deduplicated entity list.
    Entities with identical canonical form are merged.
    """
    nlp = get_nlp()
    doc = nlp(text)

    raw: Dict[Tuple[str, str], ExtractedEntity] = {}

    for ent in doc.ents:
        if ent.label_ not in OSINT_LABELS:
            continue
        canonical = normalise_entity(ent.text, ent.label_)
        key = (canonical, ent.label_)

        if key in raw:
            raw[key].count += 1
            if ent.text not in raw[key].aliases:
                raw[key].aliases.append(ent.text)
        else:
            raw[key] = ExtractedEntity(
                text=ent.text,
                label=ent.label_,
                canonical_form=canonical,
                count=1,
                sources=[source_id] if source_id else [],
                aliases=[],
            )

    return list(raw.values())


def merge_entity_lists(
    lists: List[List[ExtractedEntity]],
) -> List[ExtractedEntity]:
    """
    Merge multiple entity lists (from different sources) into one master list.
    Entities with same canonical_form + label are merged, counts summed.
    """
    master: Dict[Tuple[str, str], ExtractedEntity] = {}

    for entity_list in lists:
        for ent in entity_list:
            key = (ent.canonical_form, ent.label)
            if key in master:
                master[key].count += ent.count
                master[key].sources.extend(ent.sources)
                for alias in ent.aliases:
                    if alias not in master[key].aliases:
                        master[key].aliases.append(alias)
            else:
                master[key] = ExtractedEntity(
                    text=ent.text,
                    label=ent.label,
                    canonical_form=ent.canonical_form,
                    count=ent.count,
                    sources=list(ent.sources),
                    aliases=list(ent.aliases),
                )

    # Sort by frequency descending
    return sorted(master.values(), key=lambda e: e.count, reverse=True)


def basic_coref_resolve(
    entities: List[ExtractedEntity],
) -> List[ExtractedEntity]:
    """
    Basic co-reference resolution:
    Link pronoun/partial mentions to most frequent PERSON entity.
    Example: 'He' → 'John Doe' if John Doe is the top PERSON.

    For full neural co-ref, use neuralcoref or spaCy experimental coref
    (not included — requires Python 3.10+).
    This basic version handles the most common OSINT case:
    short alias → full name resolution.
    """
    persons = [e for e in entities if e.label == "PERSON"]
    if not persons:
        return entities

    # Most-mentioned person is the primary subject
    primary = max(persons, key=lambda e: e.count)

    # Check for first-name-only or last-name-only aliases
    primary_parts = primary.canonical_form.split()
    if len(primary_parts) >= 2:
        first = primary_parts[0]
        last  = primary_parts[-1]
        for ent in entities:
            if ent.label == "PERSON" and ent is not primary:
                if ent.canonical_form in (first, last):
                    # Merge into primary
                    primary.count += ent.count
                    if ent.canonical_form not in primary.aliases:
                        primary.aliases.append(ent.canonical_form)
                    ent.count = 0   # mark for removal

    return [e for e in entities if e.count > 0]
