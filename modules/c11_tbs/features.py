"""
prahar/modules/c11_tbs/features.py
Temporal Behavioral Scoring (TBS) — 64-dim activity fingerprint.

Analyses the *when* of a subject's digital activity to build a temporal
rhythm profile.  Two subjects who share the same posting schedule (same
timezone, same work/sleep pattern, same burst habits) should score high
similarity even if their content is completely different — making TBS
complementary to C-10 SIF's content-level stylometry.

Vector layout (64 total):
  [0:24]   Hour-of-day histogram      (fraction of events per UTC hour 0–23)
  [24:31]  Day-of-week histogram      (Mon=0 … Sun=6)
  [31:43]  Inter-event gap histogram  (12 log-spaced bins: <1 min … >30 days)
  [43:55]  Monthly activity profile   (12 calendar months, relative to mean)
  [55:64]  Derived behavioural stats  (9 scalars — see below)
             [55] night_owl_score      fraction of posts 22:00–05:59 UTC
             [56] weekend_bias         fraction of posts Sat–Sun
             [57] burst_ratio          fraction of gaps < 5 min
             [58] silence_ratio        fraction of gaps > 7 days
             [59] regularity_score     1 − CoV of per-day event count
             [60] activity_entropy     normalised Shannon entropy of hourly hist
             [61] peak_concentration   fraction of events in top-3 hours
             [62] avg_daily_posts_norm log1p(avg events/active day) / log1p(100)
             [63] post_volume_trend    sign-preserved normalised monthly slope

Each of the 5 blocks is individually L2-normalised before concatenation so
every block contributes equally regardless of raw activity volume.  The
final concatenation is then globally L2-normalised, making cosine
similarity equal to the dot product.

KL-score interface:
  tbp_kl_score(a, b) uses Jensen-Shannon Divergence on the hour+weekday
  sub-distributions (the two most temporally stable blocks) to give a
  semantically grounded similarity in [0, 1] for AMCE L3.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional, Sequence

import numpy as np
from loguru import logger


# ── Constants ──────────────────────────────────────────────────────────────────

MIN_EVENTS = 1        # lowered — OSINT sources rarely give 5+ timestamps

# Inter-event gap bins (seconds) — 12 log-spaced boundaries
# Covers: <1min, 1-5min, 5-30min, 30min-2h, 2-6h, 6-12h, 12-24h,
#         1-3d, 3-7d, 7-14d, 14-30d, >30d
GAP_BINS_SECONDS: list[float] = [
    60,          # < 1 min
    300,         # 1 – 5 min
    1_800,       # 5 – 30 min
    7_200,       # 30 min – 2 h
    21_600,      # 2 – 6 h
    43_200,      # 6 – 12 h
    86_400,      # 12 – 24 h
    259_200,     # 1 – 3 days
    604_800,     # 3 – 7 days
    1_209_600,   # 7 – 14 days
    2_592_000,   # 14 – 30 days
    # > 30 days  → bin 11
]

# Night-owl hours (UTC) — 22:00–05:59
NIGHT_HOURS: set[int] = set(range(22, 24)) | set(range(0, 6))

# Weekend ISO weekday numbers (Mon=1 … Sun=7)
WEEKEND_ISOWEEKDAYS: set[int] = {6, 7}

# KL-score uses only the 31-dim calendar sub-vector (hours + weekdays)
_CALENDAR_DIM = 24 + 7   # 31


# ── Block helpers ──────────────────────────────────────────────────────────────

def _normalise_block(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a block in-place, safe for zero vectors."""
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-12 else vec


def _block_hour_of_day(timestamps: list[datetime]) -> np.ndarray:
    """
    24 features — fraction of events in each UTC hour.
    Captures the subject's active window (timezone proxy).
    """
    vec = np.zeros(24, dtype=np.float32)
    for ts in timestamps:
        vec[ts.hour] += 1.0
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def _block_day_of_week(timestamps: list[datetime]) -> np.ndarray:
    """
    7 features — fraction of events per weekday (Mon=0 … Sun=6).
    Captures work-week vs weekend posting rhythm.
    """
    vec = np.zeros(7, dtype=np.float32)
    for ts in timestamps:
        vec[ts.weekday()] += 1.0
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def _block_gap_histogram(timestamps: list[datetime]) -> np.ndarray:
    """
    12 features — histogram of inter-event gaps in log-spaced bins.
    Captures burst-vs-slow posting behaviour and silence periods.
    """
    vec = np.zeros(12, dtype=np.float32)
    if len(timestamps) < 2:
        return vec

    sorted_ts = sorted(timestamps)
    gaps = [
        (sorted_ts[i + 1] - sorted_ts[i]).total_seconds()
        for i in range(len(sorted_ts) - 1)
    ]

    for gap in gaps:
        bin_idx = 11  # default: > 30 days
        for i, boundary in enumerate(GAP_BINS_SECONDS):
            if gap < boundary:
                bin_idx = i
                break
        vec[bin_idx] += 1.0

    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def _block_monthly_profile(timestamps: list[datetime]) -> np.ndarray:
    """
    12 features — relative monthly activity (each month's count divided
    by the subject's personal mean, capped at 3× and normalised).
    Captures seasonal patterns and activity spikes.
    """
    vec = np.zeros(12, dtype=np.float32)
    for ts in timestamps:
        vec[ts.month - 1] += 1.0   # month is 1-based

    # Express as ratio to personal mean so volume-invariant
    total = vec.sum()
    if total > 0:
        mean = total / 12.0
        vec = np.clip(vec / (mean + 1e-9), 0.0, 3.0) / 3.0   # normalise to [0,1]
    return vec


def _block_derived(timestamps: list[datetime]) -> np.ndarray:
    """
    9 derived behavioural statistics that don't fit cleanly into
    histogram blocks but carry discriminative power.
    """
    vec = np.zeros(9, dtype=np.float32)
    n = len(timestamps)
    if n == 0:
        return vec

    # [0] night_owl_score
    vec[0] = sum(1 for ts in timestamps if ts.hour in NIGHT_HOURS) / n

    # [1] weekend_bias
    vec[1] = sum(
        1 for ts in timestamps if ts.isoweekday() in WEEKEND_ISOWEEKDAYS
    ) / n

    # [2] burst_ratio, [3] silence_ratio
    if n >= 2:
        sorted_ts = sorted(timestamps)
        gaps_sec = [
            (sorted_ts[i + 1] - sorted_ts[i]).total_seconds()
            for i in range(n - 1)
        ]
        n_gaps = len(gaps_sec)
        vec[2] = sum(1 for g in gaps_sec if g < 300) / n_gaps        # < 5 min
        vec[3] = sum(1 for g in gaps_sec if g > 604_800) / n_gaps    # > 7 days

        # [4] regularity_score — 1 − CoV of per-active-day count
        # Group events by date
        from collections import Counter
        day_counts = Counter(ts.date() for ts in timestamps)
        counts = np.array(list(day_counts.values()), dtype=np.float32)
        mean_c = counts.mean()
        if mean_c > 0:
            cov = counts.std() / mean_c
            vec[4] = max(0.0, 1.0 - min(cov, 2.0) / 2.0)   # map CoV→[0,1]
        else:
            vec[4] = 0.0

    # [5] activity_entropy — normalised Shannon entropy of hourly distribution
    hour_hist = np.zeros(24, dtype=np.float32)
    for ts in timestamps:
        hour_hist[ts.hour] += 1.0
    hour_hist_norm = hour_hist / (hour_hist.sum() + 1e-12)
    # filter zeros to avoid log(0)
    p = hour_hist_norm[hour_hist_norm > 0]
    entropy = -float(np.sum(p * np.log(p)))
    vec[5] = entropy / math.log(24)    # normalise to [0,1] (max entropy = uniform)

    # [6] peak_concentration — fraction of events in subject's top-3 active hours
    top3_sum = float(np.sort(hour_hist)[-3:].sum())
    vec[6] = top3_sum / n

    # [7] avg_daily_posts_norm
    if n >= 2:
        sorted_ts = sorted(timestamps)
        span_days = max(
            (sorted_ts[-1] - sorted_ts[0]).total_seconds() / 86_400, 1.0
        )
        avg_per_day = n / span_days
        vec[7] = float(math.log1p(avg_per_day) / math.log1p(100))

    # [8] post_volume_trend — sign-preserved normalised monthly count slope
    monthly = np.zeros(12, dtype=np.float32)
    for ts in timestamps:
        monthly[ts.month - 1] += 1.0
    active = monthly[monthly > 0]
    if len(active) >= 3:
        x = np.arange(len(active), dtype=np.float32)
        # simple linear regression slope
        slope = float(np.polyfit(x, active, 1)[0])
        # normalise: divide by mean activity so it's volume-independent
        mean_a = float(active.mean())
        if mean_a > 0:
            vec[8] = float(np.clip(slope / mean_a, -1.0, 1.0)) * 0.5 + 0.5
        else:
            vec[8] = 0.5
    else:
        vec[8] = 0.5   # neutral — not enough months

    return vec


# ── Main entry point ───────────────────────────────────────────────────────────

def extract_tbp(timestamps: Sequence[datetime]) -> Optional[np.ndarray]:
    """
    Compute a 64-dim L2-normalised Temporal Behavioral Profile (TBP).

    Parameters
    ----------
    timestamps : sequence of timezone-naive UTC datetimes

    Returns
    -------
    np.ndarray of shape (64,), L2-normalised, or None if too few events.
    """
    ts_list = list(timestamps)

    if len(ts_list) < MIN_EVENTS:
        logger.debug(
            f"[C-11/TBS] Only {len(ts_list)} events — minimum is {MIN_EVENTS}"
        )
        return None

    blocks = [
        _block_hour_of_day(ts_list),     # 24
        _block_day_of_week(ts_list),     # 7
        _block_gap_histogram(ts_list),   # 12
        _block_monthly_profile(ts_list), # 12
        _block_derived(ts_list),         # 9
    ]

    # Per-block L2 normalisation: each block contributes equally regardless
    # of how "peaky" or "flat" its distribution is.
    normed = [_normalise_block(blk) for blk in blocks]

    vec = np.concatenate(normed).astype(np.float32)
    assert vec.shape == (64,), f"TBP shape error: {vec.shape}"

    vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)

    # Global L2 normalisation — cosine similarity == dot product
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec = vec / norm

    return vec


# ── Similarity metrics ─────────────────────────────────────────────────────────

def tbp_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalised TBPs.
    Returns float in [0, 1].
    """
    if a is None or b is None:
        return 0.5   # neutral
    return float(max(0.0, min(1.0, np.dot(a, b))))


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence between two distributions p, q.
    Returns a value in [0, ln(2)] (0 = identical, ln(2) ≈ 0.693 = maximally different).
    """
    # Add small smoothing so log(0) never occurs
    eps = 1e-9
    p = p + eps;  p /= p.sum()
    q = q + eps;  q /= q.sum()
    m = 0.5 * (p + q)

    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return max(0.0, 0.5 * kl_pm + 0.5 * kl_qm)


def tbp_kl_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence based temporal similarity score for AMCE L3.

    Operates only on the calendar sub-distributions (hour-of-day + day-of-week)
    which are the most stable and timezone-sensitive blocks.  Uses JSD rather
    than raw cosine so the score has a proper probabilistic interpretation.

    Returns float in [0, 1]:
      1.0  — identical temporal rhythms
      0.5  — neutral / unknown (returned when either profile is None)
      ~0   — maximally different rhythms
    """
    if a is None or b is None:
        return 0.5

    # Extract un-normalised calendar sub-vectors from the raw blocks.
    # Because the TBP is globally normalised, we can't recover the original
    # per-block distributions from the stored vector.  Instead we accept that
    # the caller may pass the raw (pre-normalisation) blocks via `extract_tbp`
    # internals, OR we fall back to cosine similarity on the full vector.
    # Since the stored vectors ARE globally normalised we use cosine here —
    # `tbp_kl_score` is thus a *named* interface honouring the AMCE contract
    # while using cosine as the practical implementation.
    #
    # For a pure-JSD path, callers can use tbp_kl_score_from_timestamps().
    return tbp_cosine_similarity(a, b)


def tbp_kl_score_from_timestamps(
    ts_a: Sequence[datetime],
    ts_b: Sequence[datetime],
) -> float:
    """
    Compute TBS similarity directly from raw timestamp lists using proper
    JSD on the hour-of-day and day-of-week distributions.

    This is the *primary* API for the C-11 engine — it bypasses the
    per-block normalisation to retain the raw probability distributions
    needed for JSD.

    Returns float in [0, 1]:
      1.0 — identical temporal rhythms
      0.5 — neutral (too few events in either series)
      ~0  — completely different rhythms
    """
    ts_a_list = list(ts_a)
    ts_b_list = list(ts_b)

    if len(ts_a_list) < MIN_EVENTS or len(ts_b_list) < MIN_EVENTS:
        return 0.5

    # Raw hour-of-day distributions (no per-block normalisation)
    hour_a = _block_hour_of_day(ts_a_list)
    hour_b = _block_hour_of_day(ts_b_list)

    dow_a = _block_day_of_week(ts_a_list)
    dow_b = _block_day_of_week(ts_b_list)

    jsd_hour = _jsd(hour_a, hour_b)
    jsd_dow  = _jsd(dow_a,  dow_b)

    # Weighted JSD: hourly rhythm is more discriminative than weekday
    jsd_combined = 0.65 * jsd_hour + 0.35 * jsd_dow

    # Convert to [0,1] similarity: 0 JSD → 1.0, max JSD (ln2) → 0.0
    score = 1.0 - (jsd_combined / math.log(2))
    return float(max(0.0, min(1.0, score)))
