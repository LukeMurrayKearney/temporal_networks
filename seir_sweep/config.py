"""Grid definition, rep budgets and paths for the SEIR parameter sweep.

The sweep is the factorial

    {NB, log-GMM mean-match} x {ICC 0, 0.365, 0.95} x {n = 10k, 100k, 1M} x beta

where

* **model** fixes the community-contact marginal. Both arms match the survey
  degree distribution (see ``community_structural_comparison.ipynb``), so any
  epidemic difference between them is attributable to the *cross-cell
  correlation* the joint log-GMM keeps and the per-cell NB throws away -- not
  to a marginal mismatch.
* **ICC** fixes how persistent the one_time layer is across days:
  ``0`` = fresh redraw every night (no individual persistence),
  ``0.365`` = the real France 2015 (Beraud) 2-day panel value,
  ``0.95`` = the near-permanent-trait regime the default recipe produces.
  Realised via ``one_time_noise={'kind': 'partial_resample', 'rho': ...}``,
  the only procedure that hits a target ICC while leaving the daily degree
  distribution (mean, variance, skew) unchanged.
* **beta** is swept over one shared grid (identical for every model/ICC/size
  cell, so cells are comparable beta-for-beta) spanning R0 well under 1 to
  well over 10. See ``calibrate_beta.py``.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_DIR = HERE / "outputs"

# ---------------------------------------------------------------------------
# Epidemiology held fixed across the whole sweep
# ---------------------------------------------------------------------------

SIGMA = 1.0          # E1->E2->E3->I, mean 3 days incubation
GAMMA = 0.25         # I->R, mean 4 days infectious
N_SEED_INFECTED = 5  # index cases per replicate (matches the existing pipeline)

# ---------------------------------------------------------------------------
# Sweep axes
# ---------------------------------------------------------------------------

MODELS = ["nb", "loggmm_mm"]

MODEL_LABELS = {
    "nb": "Negative binomial (per-cell independent)",
    "loggmm_mm": "Log-GMM, per-cell mean-matched (joint/correlated)",
}

# Target one_time intraclass correlation (between-person share of one_time
# degree variance). 0.0 is the special "fresh daily redraw" arm.
ICC_TARGETS = [0.0, 0.365, 0.95]

SIZES = [10_000, 100_000, 1_000_000]

# Beta grid, shared by every (model, ICC, size) cell so cells are comparable
# beta-for-beta. Derived by ``calibrate_beta.py`` from a measured R0(beta) curve
# (n=30k, 400 index cases x 3 capped runs per beta; see
# ``outputs/beta_r0_calibration.csv``). Regenerate with:
#     python calibrate_beta.py --write-config
#
# NOTE on the top of the range: R0 *saturates* near ~11 in this model. Duration
# weighting means the shortest contact bin transmits at 0.00625*beta, so beta has
# to grow by orders of magnitude to recruit those contacts, and once every
# contact transmits with near-certainty R0 is capped by the number of *distinct*
# people met during a ~4-day infectious period. Measured: beta=800 -> R0 10.8,
# beta=2000 -> R0 10.9. So the grid reaches "above 10" but R0=13 is unreachable
# at any beta -- it is a property of the contact model, not of the grid.
BETAS = [0.05, 0.10, 0.17, 0.37, 0.59, 1.97, 15.3, 84.4, 800.0]

# Nominal R0 each beta targets (informational -- the *measured* R0 is recorded
# per replicate as ``r0_measured``).
BETA_R0_NOMINAL = {
    0.05: 0.4, 0.10: 0.7, 0.17: 1.0, 0.37: 1.5, 0.59: 2.0,
    1.97: 3.0, 15.3: 5.0, 84.4: 8.0, 800.0: 10.8,
}

# ---------------------------------------------------------------------------
# Replicate budget
#
# Cost scales super-linearly with n (the simulator rebuilds the full daily
# adjacency in Python every simulated day), so reps are traded down as n grows.
# net_reps  = independent network realisations (captures structural variability)
# sim_reps  = epidemic replicates per network per beta (captures seeding/stochastic
#             variability -- this is what takeoff probability is estimated from)
# ---------------------------------------------------------------------------

REP_BUDGET = {
    10_000:    {"net_reps": 5, "sim_reps": 10},
    100_000:   {"net_reps": 3, "sim_reps": 4},
    1_000_000: {"net_reps": 1, "sim_reps": 2},
}

# A replicate is called a "takeoff" when it infects more than this fraction of
# the population; used for takeoff probability. Recorded as a flag per run so
# the threshold can be changed at analysis time.
TAKEOFF_FRACTION = 0.01

# Structural probing: how many days of contacts to draw for degree/ICC stats.
# 2 is the minimum for the 2-day ICC estimator (matched to the France panel).
STRUCT_PROBE_DAYS = 2

# Base seed; every replicate derives a reproducible seed from this plus its
# grid coordinates (see run_sweep._seeds).
RNG_SEED = 20260825


def cell_id(model: str, icc: float, n_nodes: int, net_rep: int) -> str:
    """Stable directory-safe identifier for one network realisation."""
    return f"{model}__icc{int(round(icc * 1000)):04d}__n{n_nodes}__net{net_rep}"


def grid(models=None, iccs=None, sizes=None):
    """All (model, icc, n_nodes, net_rep) network-realisation tasks."""
    models = MODELS if models is None else models
    iccs = ICC_TARGETS if iccs is None else iccs
    sizes = SIZES if sizes is None else sizes
    out = []
    for n in sizes:
        for model in models:
            for icc in iccs:
                for rep in range(REP_BUDGET[n]["net_reps"]):
                    out.append((model, icc, n, rep))
    return out
