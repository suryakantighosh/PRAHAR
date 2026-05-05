"""
prahar/modules/c06_graph/schema.py
Neo4j node and relationship type constants.
Single source of truth for all Cypher queries.
"""

# ── Node labels ───────────────────────────────────────────────
NODE_IDENTITY  = "Identity"       # Consolidated Identity Node (CIN)
NODE_FRAGMENT  = "Fragment"       # Identity Fragment Node (IFN)
NODE_ENTITY    = "Entity"         # NLP entity (PERSON, ORG, GPE...)
NODE_EVIDENCE  = "Evidence"       # Raw data record
NODE_CASE      = "Case"           # Investigation case root

# ── Relationship types ────────────────────────────────────────
REL_SHARES_PLATFORM  = "SHARES_PLATFORM"   # two identities on same platform
REL_LINKED_TO        = "LINKED_TO"         # fragment → identity
REL_MENTIONED_IN     = "MENTIONED_IN"      # entity → evidence
REL_CONTRADICTS      = "CONTRADICTS"       # CRG conflict edge
REL_CORROBORATES     = "CORROBORATES"      # supporting evidence edge
REL_BELONGS_TO       = "BELONGS_TO"        # fragment/entity → case
REL_SHARES_ARF       = "SHARES_ARF"        # C-11 temporal rhythm match

# ── Property keys ─────────────────────────────────────────────
PROP_CASE_ID    = "case_id"
PROP_CPIF_SCORE = "cpif_score"
PROP_WEIGHT     = "weight"
PROP_CONFIDENCE = "confidence"
PROP_LABEL      = "label"
PROP_TEXT       = "text"
PROP_PLATFORM   = "platform"
PROP_SOURCE     = "source"
PROP_CREATED_AT = "created_at"
