"""
prahar/modules/c08_brief/pdf_builder.py
Intelligence brief PDF generator using ReportLab.
Produces a structured 6-section PDF with chain-of-custody header.
"""
import io
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# ── Colour palette ────────────────────────────────────────────
PRAHAR_DARK   = colors.HexColor("#1a1a2e")
PRAHAR_ACCENT = colors.HexColor("#e94560")
PRAHAR_BLUE   = colors.HexColor("#0f3460")
PRAHAR_LIGHT  = colors.HexColor("#f5f5f5")
RISK_COLORS = {
    "HIGH":     colors.HexColor("#c0392b"),
    "MEDIUM":   colors.HexColor("#e67e22"),
    "LOW":      colors.HexColor("#27ae60"),
    "VERY_LOW": colors.HexColor("#95a5a6"),
}


def _styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "BriefTitle", parent=base["Title"],
            fontSize=20, textColor=PRAHAR_DARK,
            spaceAfter=6, alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "BriefH1", parent=base["Heading1"],
            fontSize=13, textColor=PRAHAR_BLUE,
            spaceBefore=14, spaceAfter=4,
        ),
        "h2": ParagraphStyle(
            "BriefH2", parent=base["Heading2"],
            fontSize=11, textColor=PRAHAR_DARK,
            spaceBefore=8, spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "BriefBody", parent=base["Normal"],
            fontSize=9, leading=14,
        ),
        "mono": ParagraphStyle(
            "BriefMono", parent=base["Code"],
            fontSize=7, leading=10,
            textColor=colors.HexColor("#555555"),
        ),
        "caption": ParagraphStyle(
            "BriefCaption", parent=base["Normal"],
            fontSize=8, textColor=colors.grey,
            alignment=TA_CENTER,
        ),
    }
    return styles


def _kv_table(rows: List[tuple], col_widths=None) -> Table:
    """Simple two-column key-value table."""
    if col_widths is None:
        col_widths = [5*cm, 11*cm]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), PRAHAR_LIGHT),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",   (0, 0), (0, -1), PRAHAR_DARK),
        ("GRID",        (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def build_pdf(
    case_id:           str,
    subject_name:      str,
    generated_at:      str,
    risk_level:        str,
    final_score:       float,
    risk_flags:        List[str],
    platforms:         List[str],
    breach_names:      List[str],
    top_persons:       List[str],
    top_orgs:          List[str],
    amce_breakdown:    Dict[str, Any],
    provenance_chain:  List[Dict[str, Any]],
    provenance_hash:   str,
    analyst_notes:     str = "",
) -> bytes:
    """
    Generate a 6-section intelligence brief PDF.
    Returns PDF bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    S = _styles()
    story = []

    # ── Cover / Header ────────────────────────────────────────
    story.append(Paragraph("PRAHAR v2", S["title"]))
    story.append(Paragraph(
        "Intelligence Brief — OSINT Analysis Report", S["caption"]
    ))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=PRAHAR_ACCENT, spaceAfter=10))

    story.append(_kv_table([
        ("Case ID",       case_id),
        ("Subject",       subject_name),
        ("Generated",     generated_at),
        ("Risk Level",    risk_level),
        ("Confidence",    f"{final_score:.1%}"),
        ("Provenance",    f"{provenance_hash[:32]}..."),
    ]))
    story.append(Spacer(1, 0.4*cm))

    # ── Section 1: Executive Summary ─────────────────────────
    story.append(Paragraph("1. Executive Summary", S["h1"]))
    summary_text = (
        f"Subject <b>{subject_name}</b> was identified across "
        f"<b>{len(platforms)}</b> platforms with an overall confidence "
        f"score of <b>{final_score:.1%}</b> (Risk: {risk_level}). "
    )
    if breach_names:
        summary_text += (
            f"The subject's data was exposed in <b>{len(breach_names)}</b> "
            f"known data breach(es): {', '.join(breach_names[:3])}. "
        )
    if risk_flags:
        summary_text += f"Risk indicators: {', '.join(risk_flags[:4])}."
    story.append(Paragraph(summary_text, S["body"]))
    story.append(Spacer(1, 0.3*cm))

    # ── Section 2: Digital Footprint ─────────────────────────
    story.append(Paragraph("2. Digital Footprint", S["h1"]))
    if platforms:
        plat_rows = [("Platform", "Status")]
        for p in platforms[:20]:
            plat_rows.append((p, "Found"))
        t = Table(plat_rows, colWidths=[8*cm, 8*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), PRAHAR_BLUE),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 8),
            ("GRID",         (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1),
             [colors.white, PRAHAR_LIGHT]),
            ("TOPPADDING",   (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No platform profiles found.", S["body"]))
    story.append(Spacer(1, 0.3*cm))

    # ── Section 3: Entity Analysis ────────────────────────────
    story.append(Paragraph("3. Entity Analysis", S["h1"]))
    if top_persons:
        story.append(Paragraph("<b>Associated persons:</b> " +
                                ", ".join(top_persons), S["body"]))
    if top_orgs:
        story.append(Paragraph("<b>Associated organisations:</b> " +
                                ", ".join(top_orgs), S["body"]))
    story.append(Spacer(1, 0.3*cm))

    # ── Section 4: Confidence Score Breakdown ─────────────────
    story.append(Paragraph("4. Confidence Score Breakdown (AMCE)", S["h1"]))
    contrib = amce_breakdown.get("contributions", {})
    score_rows = [
        ("Layer", "Score", "Contribution"),
        ("L1 — Raw signals",         "", str(contrib.get("l1_weighted", ""))),
        ("L2 — Structural",          "", str(contrib.get("l2_weighted", ""))),
        ("L3 — Behavioural",         "", str(contrib.get("l3_weighted", ""))),
        ("L4 — Conflict penalty (−)", "", str(contrib.get("l4_penalty", ""))),
        ("FINAL", "", f"{final_score:.4f}"),
    ]
    t2 = Table(score_rows, colWidths=[7*cm, 3*cm, 6*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), PRAHAR_BLUE),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",    (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND",  (0, -1), (-1, -1), PRAHAR_LIGHT),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("GRID",        (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.3*cm))

    # ── Section 5: Analyst Notes ──────────────────────────────
    story.append(Paragraph("5. Analyst Notes", S["h1"]))
    story.append(Paragraph(
        analyst_notes if analyst_notes else "No analyst notes recorded.",
        S["body"]
    ))
    story.append(Spacer(1, 0.3*cm))

    # ── Section 6: Chain of Custody ───────────────────────────
    story.append(Paragraph("6. Chain of Custody", S["h1"]))
    story.append(Paragraph(
        "Every claim in this brief is cryptographically linked to its "
        "originating source record via a SHA-256 Provenance Hash Chain (PHC).",
        S["body"]
    ))
    story.append(Spacer(1, 0.2*cm))

    chain_rows = [("Node Type", "Content Hash", "Chain Hash")]
    for node in provenance_chain[:10]:
        chain_rows.append((
            node.get("node_type", ""),
            node.get("content_hash", "")[:16] + "...",
            node.get("chain_hash",  "")[:16] + "...",
        ))
    t3 = Table(chain_rows, colWidths=[4*cm, 7*cm, 5*cm])
    t3.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), PRAHAR_DARK),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 7),
        ("FONTNAME",    (0, 1), (-1, -1), "Courier"),
        ("GRID",        (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.white, PRAHAR_LIGHT]),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]))
    story.append(t3)

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.lightgrey))
    story.append(Paragraph(
        f"Generated by PRAHAR v2 | {generated_at} | "
        f"Provenance: {provenance_hash[:48]}",
        S["caption"]
    ))

    doc.build(story)
    return buffer.getvalue()
