"""Fit loading and network construction for the sweep's model x ICC arms.

Two marginal models:

* ``nb``        -- per-cell independent negative binomial (throws away
                   cross-(contact_age x duration) correlation).
* ``loggmm_mm`` -- joint log-GMM per age group with **per-cell mean-matching**,
                   which pins every cell's mean to the survey while keeping the
                   mixture's shape and cross-cell correlation. Both arms match
                   the survey marginal degree distribution, so differences
                   between them isolate the effect of the correlation.

One persistence axis, applied to the one_time layer:

* ``icc = 0``   -- no ``rate_sampler``: stubs are redrawn fresh every night, so
                   a node's one_time degree is independent day to day.
* ``icc > 0``   -- a persistent per-node rate is drawn once, then each day is
                   ``partial_resample``d with ``rho`` calibrated so the two-day
                   ICC hits the target. Partial resampling is used because it
                   is the only procedure that hits a target ICC while leaving
                   the daily degree distribution (mean/variance/skew) intact.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(REPO.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import seir_comparison_common as cc          # noqa: E402

# ``cc``'s data paths are relative to the CWD (they assume the script runs from
# ``temporal_networks/``). Pin them to absolute paths so the sweep can be run
# from anywhere, including a worker process with a different CWD.
cc.DATA_ROOT = REPO.parent / "analyse households"
cc.EGO_DATA = REPO.parent / "analytics" / "ego_data"

from network import N_AGE                    # noqa: E402
from temporal import (                       # noqa: E402
    TemporalContacts, fit_nb_mom, nb_raw_sampler, nb_rate_sampler,
    make_count_rate_sampler,
)
from gmm_contacts import (                   # noqa: E402
    loggmm_raw_sampler, add_cell_mean_match_layered,
)
from gillespie_seir import (                 # noqa: E402
    temporal_edges_fn, HOUSEHOLD_WEIGHT,
)

FITS_CACHE = HERE / "outputs" / "fits_cache.pkl"
_FITS = None  # per-process lazy cache


# ---------------------------------------------------------------------------
# Fits (computed once, reused by every worker)
# ---------------------------------------------------------------------------

def prepare_fits(force: bool = False) -> Path:
    """Fit/collect both models' parameters once and cache them to disk."""
    if FITS_CACHE.exists() and not force:
        return FITS_CACHE
    FITS_CACHE.parent.mkdir(parents=True, exist_ok=True)

    hh_ages, all_contacts, source_ages, _reduced = cc.load_survey_data()
    nb_layered = cc.fit_params_layered(all_contacts, source_ages, fit_nb_mom)

    gmm_cache = REPO / "outputs_loggmm" / "networks" / "loggmm_fits.pkl"
    if not gmm_cache.exists():
        raise FileNotFoundError(
            f"log-GMM fit cache not found at {gmm_cache}. Run the fitting step in "
            "community_structural_comparison.ipynb (or run_seir_comparison_loggmm.py) first."
        )
    with open(gmm_cache, "rb") as f:
        loggmm_layered = pickle.load(f)["params_layered"]

    loggmm_mm_layered = add_cell_mean_match_layered(loggmm_layered, all_contacts, source_ages)

    with open(FITS_CACHE, "wb") as f:
        pickle.dump({
            "hh_ages": hh_ages,
            "nb": nb_layered,
            "loggmm_mm": loggmm_mm_layered,
            "gmm_components": [
                {a: (g.n_components if g is not None else None)
                 for a, g in p["gmms"].items()}
                for p in loggmm_layered
            ],
        }, f)
    return FITS_CACHE


def get_fits() -> dict:
    global _FITS
    if _FITS is None:
        if not FITS_CACHE.exists():
            prepare_fits()
        with open(FITS_CACHE, "rb") as f:
            _FITS = pickle.load(f)
    return _FITS


def samplers_for(model: str):
    """(params, raw_sampler, rate_sampler_factory) for a model key."""
    fits = get_fits()
    if model == "nb":
        return fits["nb"], nb_raw_sampler, nb_rate_sampler
    if model == "loggmm_mm":
        return (fits["loggmm_mm"], loggmm_raw_sampler,
                make_count_rate_sampler(loggmm_raw_sampler))
    raise ValueError(f"unknown model {model!r}")


# ---------------------------------------------------------------------------
# ICC calibration
# ---------------------------------------------------------------------------

def calibrate_rho(one_time_rates: dict, nodes_by_age: dict, n_nodes: int,
                  target_icc: float) -> tuple[float, float, float]:
    """rho for ``partial_resample`` that lands the two-day ICC on ``target_icc``.

    With a persistent per-node total rate ``lam`` (mean ``m``, variance ``B``),
    daily counts ``Poisson(lam)`` have ICC ``B / (B + m)``. Partial resampling
    keeps a node's own rate with probability ``rho`` and otherwise swaps in a
    random other node's, which multiplies the two-day covariance by ``rho^2``:

        ICC(rho) = rho^2 * B / (B + m)   =>   rho^2 = target * (B + m) / B

    Returns ``(rho, icc_persistent, icc_achievable)``. When the target exceeds
    the persistent ceiling (``B/(B+m)``) rho clamps to 1 and the achievable ICC
    is that ceiling -- Poisson daily noise cannot be made *less* noisy, so the
    ceiling is reported rather than silently missed.
    """
    lam = np.zeros(n_nodes, dtype=np.float64)
    for (a, _b, _d), arr in one_time_rates.items():
        idx = np.asarray(nodes_by_age.get(a, []), dtype=np.int64)
        if idx.size:
            lam[idx] += arr
    m, B = float(lam.mean()), float(lam.var())
    if B <= 0 or m <= 0:
        return 1.0, float("nan"), float("nan")
    icc_persist = B / (B + m)
    rho_sq = target_icc * (B + m) / B
    if rho_sq >= 1.0:
        return 1.0, icc_persist, icc_persist
    rho = float(np.sqrt(max(rho_sq, 0.0)))
    return rho, icc_persist, float(rho ** 2 * icc_persist)


# ---------------------------------------------------------------------------
# Network realisation
# ---------------------------------------------------------------------------

class Realisation:
    """One built network realisation, ready to simulate on."""

    def __init__(self, tc, pop, nodes_by_age, hh_by_node, household_sizes,
                 household_edges, everyday_edges, meta):
        self.tc = tc
        self.pop = pop
        self.nodes_by_age = nodes_by_age
        self.hh_by_node = hh_by_node
        self.household_sizes = household_sizes
        self.household_edges = household_edges
        self.everyday_edges = everyday_edges
        self.meta = meta
        self.n_nodes = pop.n_nodes
        self.node_age = np.zeros(pop.n_nodes, dtype=np.int64)
        for nid, node in pop.nodes.items():
            self.node_age[nid] = node.age

    def edges_fn(self):
        return temporal_edges_fn(
            self.tc,
            static_edges=[(u, v, "household", HOUSEHOLD_WEIGHT)
                          for u, v in self.household_edges],
        )

    def household_degree(self) -> np.ndarray:
        deg = np.zeros(self.n_nodes)
        for u, v in self.household_edges:
            deg[u] += 1
            deg[v] += 1
        return deg

    def probe_degrees(self, n_days: int) -> dict[str, np.ndarray]:
        """Draw ``n_days`` of contacts and return per-node degree by layer.

        Advances the realisation's own RNG, so these days are consumed before
        the epidemic starts (days are exchangeable, so this is a harmless
        burn-in and keeps the structural probe from perturbing a rerun).
        """
        layers = ["everyday", "few_times_week", "one_time"]
        out = {L: np.zeros((n_days, self.n_nodes)) for L in layers}
        for d in range(n_days):
            for u, v, _dur, layer in self.tc.daily_contacts():
                out[layer][d, u] += 1
                out[layer][d, v] += 1
        return out


def build_realisation(model: str, target_icc: float, n_nodes: int,
                      seed: int) -> Realisation:
    """Build one population + community network for a (model, ICC, n) cell."""
    fits = get_fits()
    params, raw_sampler, rate_sampler = samplers_for(model)

    rng = np.random.default_rng(seed)
    cc.N_NODES = n_nodes
    pop, nba, hbn, hs, he = cc.build_population(fits["hh_ages"], rng)

    everyday = cc.build_everyday_edges_ud(
        params[0], nba, hbn, he, rng, raw_sampler=raw_sampler
    )

    meta = {
        "model": model, "target_icc": target_icc, "n_nodes_requested": n_nodes,
        "n_nodes": pop.n_nodes, "n_households": pop.n_households,
        "n_household_edges": len(he), "n_everyday_edges": len(everyday),
        "network_seed": seed,
    }

    if target_icc <= 0:
        # Fresh nightly redraw: no persistent rate at all.
        tc = TemporalContacts.build(
            pop=pop, nb_params=params, everyday_edges=everyday, rng=rng,
            raw_sampler=raw_sampler, rate_sampler=None,
        )
        meta.update({"noise_kind": "fresh_daily", "rho": float("nan"),
                     "icc_persistent": 0.0, "icc_target_achievable": 0.0})
        return Realisation(tc, pop, nba, hbn, hs, he, everyday, meta)

    # Persistent rates, then calibrate rho on the *realised* rate distribution.
    tc = TemporalContacts.build(
        pop=pop, nb_params=params, everyday_edges=everyday, rng=rng,
        raw_sampler=raw_sampler, rate_sampler=rate_sampler,
    )
    rho, icc_persist, icc_achievable = calibrate_rho(
        tc.one_time_rates, nba, pop.n_nodes, target_icc
    )
    tc.one_time_noise = {"kind": "partial_resample", "rho": rho}
    meta.update({
        "noise_kind": "partial_resample", "rho": float(rho),
        "icc_persistent": float(icc_persist),
        "icc_target_achievable": float(icc_achievable),
    })
    return Realisation(tc, pop, nba, hbn, hs, he, everyday, meta)
