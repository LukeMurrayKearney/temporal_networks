"""Grid definition, rep budgets and paths for the *structure* sweep.

Where the main sweep (``config.py`` / ``run_sweep.py``) varies the community
contact model and the one_time persistence on a single network *architecture*,
this one varies the architecture itself:

    {temporal, static} x {households, no households} x {NB, log-GMM mean-match}

at ``n = 100_000`` over the **same beta grid** as the main sweep, so every cell
here is comparable beta-for-beta with every cell there.

The fourth combination -- **temporal x households** -- is deliberately *not*
run: it is exactly the main sweep, whose ``n=100000`` slice is already complete
for both models at all three ICCs. Splice it in at analysis time from
``outputs/runs_summary.csv``; this sweep supplies the other three corners.

    corner                 | run here | where it comes from
    -----------------------|----------|--------------------------------------
    temporal x households  | no       | outputs/runs_summary.csv (main sweep)
    temporal x no-hh       | yes      | arm ``nohh_temporal``
    static   x households  | yes      | arm ``hh_static``
    static   x no-hh       | yes      | arm ``nohh_static``

Axis definitions
----------------

**temporal vs static.** A temporal arm draws a fresh day of community contacts
every simulated day from the three frequency layers (everyday / few_times_week /
one_time). A static arm collapses the frequency axis -- the community fit is
made on counts *summed over frequency* -- into one fixed edge set that is on
every day. Same survey, same duration weighting; what changes is whether a
contact is a nightly draw or a permanent edge.

**households vs no households.** "No households" means a population of
singletons -- no household layer at all -- and nothing else: the people have the
same UK age structure, and community contact is drawn from the **same
age-structured fit** as the matching household arm (the summed fit for the
static arm, the main sweep's layered fit for the temporal arm), with stubs
matched by age group in the same way. So this axis removes the household layer
alone. Contacts with household members, which were taken out of the survey data
because the household layer represents them, are not put back.

(Until 2026-10-06 the ``nohh_*`` arms followed the older 2x2 in
``seir_comparison_common.py`` (``realization_no_household_*``) and drew contact
from the pooled, age-blind ``reconnect_reduced.json``, which removed age
structure together with households. Every arm is now age-structured.)

**ICC** (one_time persistence) applies only to the temporal arm -- a static
network has no daily one_time layer to persist. It is carried at the same three
targets as the main sweep so ``temporal x no-hh`` can be compared with
``temporal x households`` at matched persistence. Static arms carry ``icc=None``
and are tagged ``iccna`` in their cell id.
"""

from __future__ import annotations

from pathlib import Path

import config as C

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_DIR = HERE / "outputs_structure"

# ---------------------------------------------------------------------------
# Everything epidemiological is inherited from the main sweep, unchanged, so
# the two sets of outputs are directly comparable. Importing rather than
# re-declaring means they cannot drift apart.
# ---------------------------------------------------------------------------

SIGMA = C.SIGMA
GAMMA = C.GAMMA
N_SEED_INFECTED = C.N_SEED_INFECTED
BETAS = C.BETAS
BETA_R0_NOMINAL = C.BETA_R0_NOMINAL
TAKEOFF_FRACTION = C.TAKEOFF_FRACTION
STRUCT_PROBE_DAYS = C.STRUCT_PROBE_DAYS

# Distinct from ``C.RNG_SEED``: the cell coordinates differ (there is an arm
# axis), but a separate base seed also guarantees that no realisation here can
# collide with one from the main sweep.
RNG_SEED = 20260918

# ---------------------------------------------------------------------------
# Sweep axes
# ---------------------------------------------------------------------------

MODELS = ["nb", "loggmm_mm"]

# arm -> (temporal?, households?). The missing fourth corner is the main sweep.
ARMS: dict[str, tuple[bool, bool]] = {
    "hh_static":     (False, True),
    "nohh_static":   (False, False),
    "nohh_temporal": (True,  False),
}

ARM_LABELS = {
    "hh_static": "Static community network + households (age-structured)",
    "nohh_static": "Static community network, no households (age-structured)",
    "nohh_temporal": "Temporal community network, no households (age-structured)",
}

# The corner this sweep does not run, and where its rows live instead.
EXCLUDED_ARM = "hh_temporal"
EXCLUDED_ARM_SOURCE = C.OUT_DIR / "runs_summary.csv"

ICC_TARGETS = C.ICC_TARGETS          # temporal arms only
SIZES = [100_000]


def arm_is_temporal(arm: str) -> bool:
    return ARMS[arm][0]


def arm_has_households(arm: str) -> bool:
    return ARMS[arm][1]


def iccs_for_arm(arm: str, iccs=None) -> list[float | None]:
    """ICC values that apply to an arm.

    A static arm has no one_time layer, so persistence is undefined for it and
    it gets the single value ``None`` no matter what was asked for on the CLI --
    otherwise ``--iccs`` would silently fan a static cell out into three
    identical copies under three different names.
    """
    if not arm_is_temporal(arm):
        return [None]
    return list(ICC_TARGETS if iccs is None else iccs)


# ---------------------------------------------------------------------------
# Replicate budget
#
# Same shape as the main sweep at 100k, so the spliced-in temporal x household
# corner carries the same weight as the three corners run here.
# ---------------------------------------------------------------------------

REP_BUDGET = {
    100_000: {"net_reps": 10, "sim_reps": 10},
    10_000:  {"net_reps": 10, "sim_reps": 10},   # for smoke tests / cheap slices
}


def cell_id(arm: str, model: str, icc: float | None, n_nodes: int,
            net_rep: int) -> str:
    """Stable directory-safe identifier for one network realisation."""
    icc_tag = "na" if icc is None else f"{int(round(icc * 1000)):04d}"
    return f"{arm}__{model}__icc{icc_tag}__n{n_nodes}__net{net_rep}"


def grid(arms=None, models=None, iccs=None, sizes=None):
    """All ``(arm, model, icc, n_nodes, net_rep)`` network-realisation tasks."""
    arms = list(ARMS) if arms is None else arms
    models = MODELS if models is None else models
    sizes = SIZES if sizes is None else sizes
    out = []
    for n in sizes:
        for arm in arms:
            for model in models:
                for icc in iccs_for_arm(arm, iccs):
                    for rep in range(REP_BUDGET[n]["net_reps"]):
                        out.append((arm, model, icc, n, rep))
    return out
