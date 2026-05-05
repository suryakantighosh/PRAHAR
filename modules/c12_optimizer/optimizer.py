"""
prahar/modules/c12_optimizer/optimizer.py
AMCE Weight Optimizer — pure math, no DB, no async.

Optimises two weight vectors via projected gradient descent with Adam moments:

  1. AMCE layer weights  w = (w_l1, w_l2, w_l3, w_l4)
       predicted = clip(w_l1*l1 + w_l2*l2 + w_l3*l3 - w_l4*l4, 0, 1)
       loss = MSE(predicted, analyst_score)

  2. CPIF signal weights  w = (w_bio, w_usr, w_tbs)
       Gradient estimated from L3 residuals — see cpif_gradient() docstring.

Constraints enforced via Euclidean projection onto the simplex:
  { w : sum(w) = 1.0, w_i >= W_FLOOR for all i }

This guarantees that no single layer/signal is ever silenced and the
weights remain a valid convex combination after every step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# ── Hyperparameters ────────────────────────────────────────────────────────────

# Minimum weight any single component may hold (prevents any layer from
# being zeroed out by a run of misleading feedback events).
W_FLOOR = 0.05

# Adam optimiser hyperparameters (standard defaults)
BETA1   = 0.9    # first-moment exponential decay
BETA2   = 0.999  # second-moment exponential decay
EPSILON = 1e-8   # numerical stability

# Learning rate
LR_DEFAULT = 0.01

# Number of gradient-descent steps per call to run_epoch()
STEPS_DEFAULT = 50

# Convergence threshold — stop early if mean absolute weight change < this
CONVERGENCE_TOL = 1e-5

# AMCE layer weight floors — w_l4 (penalty) is allowed to shrink more
W_FLOOR_L4 = 0.05   # same floor, but kept explicit for readability


# ── Weight state ───────────────────────────────────────────────────────────────

@dataclass
class AMCEWeights:
    """
    Current AMCE layer weight vector.
    Invariant: all weights >= W_FLOOR, sum(w_l1..w_l3) - w_l4 need not sum
    to 1 because w_l4 is a *penalty* subtracted, not a positive contribution.
    We therefore optimise w_pos = (w_l1, w_l2, w_l3) on the 3-simplex and
    treat w_l4 as a separate scalar in [W_FLOOR, 0.5].
    """
    w_l1: float = 0.35
    w_l2: float = 0.30
    w_l3: float = 0.20
    w_l4: float = 0.15   # penalty weight — NOT on the same simplex

    def as_array(self) -> np.ndarray:
        return np.array([self.w_l1, self.w_l2, self.w_l3, self.w_l4],
                        dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "AMCEWeights":
        return cls(w_l1=float(arr[0]), w_l2=float(arr[1]),
                   w_l3=float(arr[2]), w_l4=float(arr[3]))

    def is_valid(self) -> bool:
        arr = self.as_array()
        return (
            bool(np.all(arr >= W_FLOOR - 1e-9)) and
            abs(float(arr.sum()) - 1.0) < 1e-4
        )


@dataclass
class CPIFWeights:
    """CPIF signal weight vector — constrained to the 3-simplex."""
    w_bio: float = 0.40
    w_usr: float = 0.35
    w_tbs: float = 0.25

    def as_array(self) -> np.ndarray:
        return np.array([self.w_bio, self.w_usr, self.w_tbs], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "CPIFWeights":
        return cls(w_bio=float(arr[0]), w_usr=float(arr[1]),
                   w_tbs=float(arr[2]))

    def is_valid(self) -> bool:
        arr = self.as_array()
        return (
            bool(np.all(arr >= W_FLOOR - 1e-9)) and
            abs(float(arr.sum()) - 1.0) < 1e-4
        )


@dataclass
class AdamState:
    """Moment estimates for one weight vector of dimension `dim`."""
    dim:  int
    m:    np.ndarray = field(default=None)  # type: ignore[assignment]
    v:    np.ndarray = field(default=None)  # type: ignore[assignment]
    step: int = 0

    def __post_init__(self):
        if self.m is None:
            self.m = np.zeros(self.dim, dtype=np.float64)
        if self.v is None:
            self.v = np.zeros(self.dim, dtype=np.float64)


@dataclass
class OptimizerState:
    """Complete mutable state for one optimisation run."""
    amce:       AMCEWeights  = field(default_factory=AMCEWeights)
    cpif:       CPIFWeights  = field(default_factory=CPIFWeights)
    amce_adam:  AdamState    = field(default_factory=lambda: AdamState(dim=4))
    cpif_adam:  AdamState    = field(default_factory=lambda: AdamState(dim=3))
    epoch:      int = 0
    loss_history: list[float] = field(default_factory=list)


@dataclass
class FeedbackRecord:
    """
    One analyst correction event — all values needed by the optimizer.

    layer scores (l1–l4) come from the threat_score table;
    predicted_score / analyst_score come from feedback_event;
    cpif_components are optional — used for CPIF weight gradient.
    """
    predicted_score: float
    analyst_score:   float
    l1: float
    l2: float
    l3: float
    l4: float
    # Raw CPIF component scores (bio, usr, tbs) before weighting.
    # If None, CPIF gradient step is skipped for this record.
    cpif_bio: Optional[float] = None
    cpif_usr: Optional[float] = None
    cpif_tbs: Optional[float] = None


# ── Simplex projection ────────────────────────────────────────────────────────

def project_simplex(v: np.ndarray, floor: float = W_FLOOR) -> np.ndarray:
    """
    Euclidean projection of v onto:
        { w : sum(w) = 1.0, w_i >= floor for all i }

    Algorithm: Duchi et al. (2008) "Efficient Projections onto the ℓ1-Ball"
    adapted for the floored simplex by first subtracting the floor offset.

    Parameters
    ----------
    v     : unconstrained weight vector
    floor : minimum value each element may take (default W_FLOOR)

    Returns
    -------
    Projected weight vector with same shape as v.
    """
    n = len(v)
    budget = 1.0 - n * floor   # remaining mass after floors are paid

    if budget <= 0:
        # Degenerate: floor alone already fills the simplex — return uniform
        return np.full(n, 1.0 / n, dtype=np.float64)

    # Shift: project (v - floor) onto the budget-simplex, then shift back
    v_shifted = np.asarray(v, dtype=np.float64) - floor
    u = np.sort(v_shifted)[::-1]
    css = np.cumsum(u)
    rho_candidates = np.where(u * np.arange(1, n + 1) > (css - budget))[0]

    if len(rho_candidates) == 0:
        # All components need to be set to floor — shouldn't normally happen
        return np.full(n, floor, dtype=np.float64)

    rho = rho_candidates[-1]
    theta = (css[rho] - budget) / (rho + 1.0)
    return np.maximum(v_shifted - theta, 0.0) + floor


def project_l4(w_l4: float,
               lo: float = W_FLOOR_L4,
               hi: float = 0.50) -> float:
    """Clamp the penalty weight to [lo, hi] — simple box projection."""
    return float(np.clip(w_l4, lo, hi))


# ── Gradient computation ───────────────────────────────────────────────────────

def amce_predicted(
    l1: float, l2: float, l3: float, l4: float,
    w: np.ndarray,
) -> float:
    """Reconstruct predicted AMCE score from layer scores and weights."""
    raw = w[0] * l1 + w[1] * l2 + w[2] * l3 - w[3] * l4
    return float(np.clip(raw, 0.0, 1.0))


def amce_gradient(record: FeedbackRecord, w: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE loss w.r.t. AMCE weight vector w = (w_l1, w_l2, w_l3, w_l4).

    loss     = (analyst - predicted)^2
    δ        = analyst - predicted                  (positive → under-predicted)
    ∂loss/∂w_lk = -2δ * l_k   for k in {1,2,3}
    ∂loss/∂w_l4 = +2δ * l4    (penalty term: increasing w_l4 reduces predicted)
    """
    pred  = amce_predicted(record.l1, record.l2, record.l3, record.l4, w)
    delta = record.analyst_score - pred
    # Gradient points in the direction of steepest *increase* in loss,
    # so we negate it in the update step.
    grad = np.array([
        -2.0 * delta * record.l1,   # ∂loss/∂w_l1
        -2.0 * delta * record.l2,   # ∂loss/∂w_l2
        -2.0 * delta * record.l3,   # ∂loss/∂w_l3
        +2.0 * delta * record.l4,   # ∂loss/∂w_l4
    ], dtype=np.float64)
    return grad


def cpif_gradient(record: FeedbackRecord, w_cpif: np.ndarray,
                  w_amce: np.ndarray) -> Optional[np.ndarray]:
    """
    Gradient of MSE loss w.r.t. CPIF weight vector w_cpif = (w_bio, w_usr, w_tbs).

    The chain rule through the AMCE pipeline:
        predicted ≈ … + w_l3 * l3
        l3        = 0.40 * cpif_score + 0.30 * sif + 0.30 * tbs
        cpif_score = w_bio * bio + w_usr * usr + w_tbs * tbs_c

    So:
        ∂predicted/∂w_bio = w_l3 * 0.40 * bio   (holding l1, l2, l4 fixed)
        ∂loss/∂w_bio      = -2δ * w_l3 * 0.40 * bio

    Returns None if component scores are unavailable.
    """
    if record.cpif_bio is None or record.cpif_usr is None or record.cpif_tbs is None:
        return None

    pred  = amce_predicted(record.l1, record.l2, record.l3, record.l4, w_amce)
    delta = record.analyst_score - pred
    w_l3  = w_amce[2]

    # L3 internal CPIF weight is 0.40 (from amce.py: score_l3_behavioral)
    L3_CPIF_WEIGHT = 0.40

    factor = -2.0 * delta * w_l3 * L3_CPIF_WEIGHT
    grad = np.array([
        factor * record.cpif_bio,
        factor * record.cpif_usr,
        factor * record.cpif_tbs,
    ], dtype=np.float64)
    return grad


def batch_mse_loss(records: list[FeedbackRecord], w: np.ndarray) -> float:
    """MSE loss over a batch of feedback records given AMCE weight vector w."""
    if not records:
        return 0.0
    losses = [
        (r.analyst_score - amce_predicted(r.l1, r.l2, r.l3, r.l4, w)) ** 2
        for r in records
    ]
    return float(np.mean(losses))


# ── Adam update step ───────────────────────────────────────────────────────────

def adam_update(
    w: np.ndarray,
    grad: np.ndarray,
    state: AdamState,
    lr: float = LR_DEFAULT,
) -> np.ndarray:
    """
    One Adam parameter update.  Returns the updated (unconstrained) weight vector.
    The caller is responsible for projecting back onto the feasible set.
    """
    state.step += 1
    t = state.step

    state.m = BETA1 * state.m + (1.0 - BETA1) * grad
    state.v = BETA2 * state.v + (1.0 - BETA2) * (grad ** 2)

    m_hat = state.m / (1.0 - BETA1 ** t)
    v_hat = state.v / (1.0 - BETA2 ** t)

    return w - lr * m_hat / (np.sqrt(v_hat) + EPSILON)


# ── Epoch runner ───────────────────────────────────────────────────────────────

def run_epoch(
    records: list[FeedbackRecord],
    state: OptimizerState,
    lr: float = LR_DEFAULT,
    steps: int = STEPS_DEFAULT,
    shuffle: bool = True,
) -> dict:
    """
    Run one training epoch: `steps` stochastic gradient-descent mini-steps
    over `records`, updating both AMCE and CPIF weights in-place on `state`.

    Uses mini-batch SGD with batch_size=1 (online update) for responsiveness
    to individual analyst corrections — appropriate given that feedback events
    are sparse and each is a strong signal.

    Parameters
    ----------
    records : list of FeedbackRecord (analyst corrections)
    state   : mutable OptimizerState updated in-place
    lr      : learning rate
    steps   : maximum number of gradient steps
    shuffle : randomise record order each epoch

    Returns
    -------
    dict with loss_before, loss_after, steps_taken, converged flag
    """
    if not records:
        return {"loss_before": 0.0, "loss_after": 0.0,
                "steps_taken": 0, "converged": True}

    w_amce = state.amce.as_array()
    w_cpif = state.cpif.as_array()

    loss_before = batch_mse_loss(records, w_amce)

    indices = list(range(len(records)))
    if shuffle:
        rng = np.random.default_rng(seed=state.epoch)
        rng.shuffle(indices)

    prev_w_amce = w_amce.copy()
    steps_taken = 0

    for step in range(steps):
        rec = records[indices[step % len(indices)]]

        # ── AMCE weight update ─────────────────────────────────────────────
        grad_amce = amce_gradient(rec, w_amce)
        w_amce_new = adam_update(w_amce, grad_amce, state.amce_adam, lr)

        # Project all 4 weights onto the 4-simplex with floor.
        # Matches amce.py convention: all four weights sum to 1.0.
        w_amce = project_simplex(w_amce_new, floor=W_FLOOR)

        # ── CPIF weight update (when component scores available) ───────────
        grad_cpif = cpif_gradient(rec, w_cpif, w_amce)
        if grad_cpif is not None:
            w_cpif_new = adam_update(w_cpif, grad_cpif, state.cpif_adam, lr)
            w_cpif = project_simplex(w_cpif_new, floor=W_FLOOR)

        steps_taken += 1

        # Early stop: weight change negligible
        if step > 0 and step % 10 == 0:
            delta = float(np.max(np.abs(w_amce - prev_w_amce)))
            if delta < CONVERGENCE_TOL:
                break
            prev_w_amce = w_amce.copy()

    loss_after = batch_mse_loss(records, w_amce)

    # Write back
    state.amce = AMCEWeights.from_array(w_amce)
    state.cpif = CPIFWeights.from_array(w_cpif)
    state.epoch += 1
    state.loss_history.append(loss_after)

    converged = bool(np.max(np.abs(w_amce - prev_w_amce)) < CONVERGENCE_TOL)
    return {
        "loss_before":  round(loss_before, 6),
        "loss_after":   round(loss_after,  6),
        "steps_taken":  steps_taken,
        "converged":    converged,
        "epoch":        state.epoch,
    }


def run_until_convergence(
    records: list[FeedbackRecord],
    state: OptimizerState,
    lr: float = LR_DEFAULT,
    max_epochs: int = 200,
    patience: int = 10,
) -> dict:
    """
    Run epochs until convergence or max_epochs is reached.
    Implements early stopping with patience: stop if loss has not improved
    by more than CONVERGENCE_TOL for `patience` consecutive epochs.

    Returns final stats dict.
    """
    best_loss = float("inf")
    no_improve = 0
    stats = {}

    for _ in range(max_epochs):
        stats = run_epoch(records, state, lr=lr)
        loss  = stats["loss_after"]

        if best_loss - loss > CONVERGENCE_TOL:
            best_loss  = loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience or stats["converged"]:
            break

    stats["total_epochs"] = state.epoch
    return stats
