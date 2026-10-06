"""Fits and network construction for the structure sweep's three arms.

The three architectures built here are the three corners of
``{temporal, static} x {households, no households}`` that the main sweep does
not cover (see ``config_structure.py``):

* ``hh_static``     -- real household age templates, household clique layer,
                       plus **one fixed community edge set** matched from the
                       age-structured fit to counts *summed over frequency*.
* ``nohh_static``   -- singleton population, no household layer, one fixed
                       community edge set from the **same** age-structured
                       summed fit as ``hh_static``.
* ``nohh_temporal`` -- singleton population, no household layer, the three
                       frequency layers redrawn nightly from the **same**
                       age-structured layered fit the main sweep
                       (``hh_temporal``) uses. Carries the same ICC (one_time
                       persistence) axis as the main sweep.

**Every arm is age-structured.** A no-household arm is its household
counterpart with the household layer taken away and nothing else changed: the
same fitted community-contact parameters, the same stub matching by age group,
on a population of singletons with the same UK age structure. (Until
2026-10-06 the ``nohh_*`` arms drew contact from a fit that pooled every age
into one bucket, so removing households also removed age structure; results
produced by that construction carry no ``nohh_contact_fit`` column and the
driver refuses to resume on top of them.)

Every arm draws community contacts through the same ``raw_sampler`` interface,
so the NB and log-GMM models differ only in *how a node's contact vector is
drawn* -- never in the architecture around it.

Log-GMM arms are mean-matched, as in the main sweep: ``add_cell_mean_match``
below is the summed counterpart of ``gmm_contacts.
add_cell_mean_match_layered``, pinning each (contact_age x duration) cell's mean
count to the survey so the two models' marginals agree and only the *joint*
structure differs.
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
    N_DUR, TemporalContacts, fit_nb_mom, nb_raw_sampler, nb_rate_sampler,
    make_count_rate_sampler,
)
from gmm_contacts import (                   # noqa: E402
    loggmm_raw_sampler, _cell_mean_scales,
)
from gillespie_seir import (                 # noqa: E402
    static_edges_fn, temporal_edges_fn, DURATION_WEIGHTS, HOUSEHOLD_WEIGHT,
)

import config_structure as CS                # noqa: E402
import builders as MB                        # noqa: E402  (the main sweep's)
from builders import calibrate_rho           # noqa: E402

FITS_CACHE = CS.OUT_DIR / "fits_cache.pkl"
LOGGMM_CACHE = REPO / "outputs_loggmm" / "networks" / "loggmm_fits.pkl"

TEMPORAL_LAYERS = ["everyday", "few_times_week", "one_time"]

# Written into the network metadata of every no-household realisation, so the
# results themselves record which construction produced them.
NOHH_CONTACT_FIT = "age_structured"

_FITS = None  # per-process lazy cache


# ---------------------------------------------------------------------------
# Log-GMM mean-matching for the summed parameter set
# ---------------------------------------------------------------------------

def add_cell_mean_match(params: dict, X_by_age: dict[int, np.ndarray],
                        seed: int = 0, n_mc: int = 200_000) -> dict:
    """Attach per-cell mean-match scales to a single fitted log-GMM param dict.

    ``gmm_contacts.add_cell_mean_match_layered`` does this for the three-layer
    fit the temporal arms use; the static arms need the same treatment for
    their ``params_summed`` fit, which is a plain dict rather than a per-layer
    list. ``X_by_age[a]`` is the
    ``(n_participants, n_cells)`` survey matrix that age group's mixture was fit
    on, flattened the same way -- the scales are read off the fitted mixture, so
    nothing is refitted.
    """
    rng = np.random.default_rng(seed)
    scales: dict[int, np.ndarray] = {}
    for a, gmm in params["gmms"].items():
        X = X_by_age.get(a)
        if gmm is None or X is None or X.shape[0] == 0:
            continue
        scales[a] = _cell_mean_scales(gmm, X, rng, n_mc=n_mc)
    return dict(params, mean_match=True, stochastic_round=True, cell_scales=scales)


def _X_summed_by_age(all_contacts: np.ndarray, source_ages: np.ndarray) -> dict:
    """Age-structured survey matrix, summed over frequency (static arms)."""
    summed = all_contacts.sum(axis=3)                      # (n, N_AGE, N_DUR)
    out = {}
    for a in range(N_AGE):
        mask = source_ages == a
        out[a] = summed[mask].reshape(int(mask.sum()), N_AGE * N_DUR)
    return out


# ---------------------------------------------------------------------------
# Fits (computed once, reused by every worker)
# ---------------------------------------------------------------------------

def prepare_fits(force: bool = False) -> Path:
    """Fit/collect the static arms' parameters once and cache them to disk.

    Both static arms (with and without households) use the age-structured fit
    to counts summed over frequency. The NB fit is a cheap moment match and is
    done here. The log-GMM fit is the expensive part and is **not** refitted: it
    is read from the same ``outputs_loggmm/networks/loggmm_fits.pkl`` the main
    sweep uses, so both sweeps rest on identical mixtures.

    The temporal no-household arm needs the *layered* fit, and takes it straight
    from the main sweep's own cache (``builders.samplers_for``) rather than
    from here, so it cannot differ from what ``hh_temporal`` was run with.

    A cache written before 2026-10-06 also holds ``reduced_*`` entries (the
    pooled, age-blind fits the no-household arms used to draw from). Nothing
    reads them any more; the file is otherwise identical and need not be
    regenerated.
    """
    if FITS_CACHE.exists() and not force:
        return FITS_CACHE
    FITS_CACHE.parent.mkdir(parents=True, exist_ok=True)

    hh_ages, all_contacts, source_ages, _reduced = cc.load_survey_data()

    nb = {
        "summed": cc.fit_params_summed(all_contacts, source_ages, fit_nb_mom),
    }

    if not LOGGMM_CACHE.exists():
        raise FileNotFoundError(
            f"log-GMM fit cache not found at {LOGGMM_CACHE}. Run the fitting step in "
            "community_structural_comparison.ipynb (or run_seir_comparison_loggmm.py) first."
        )
    with open(LOGGMM_CACHE, "rb") as f:
        gmm = pickle.load(f)

    loggmm_mm = {
        "summed": add_cell_mean_match(
            gmm["params_summed"], _X_summed_by_age(all_contacts, source_ages)),
    }

    with open(FITS_CACHE, "wb") as f:
        pickle.dump({
            "hh_ages": hh_ages,
            "singleton_hh_ages": cc.SINGLETON_HH_AGES,
            "nb": nb,
            "loggmm_mm": loggmm_mm,
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
    """(raw_sampler, rate_sampler) for a model key."""
    if model == "nb":
        return nb_raw_sampler, nb_rate_sampler
    if model == "loggmm_mm":
        return loggmm_raw_sampler, make_count_rate_sampler(loggmm_raw_sampler)
    raise ValueError(f"unknown model {model!r}")


def params_for(model: str, key: str = "summed"):
    """Fitted static-arm parameters for ``model`` (the age-structured summed fit)."""
    fits = get_fits()
    if model not in fits:
        raise ValueError(f"unknown model {model!r}")
    return fits[model][key]


# ---------------------------------------------------------------------------
# Network realisation
# ---------------------------------------------------------------------------

class Realisation:
    """One built network realisation, ready to simulate on.

    Mirrors ``builders.Realisation`` (same attribute and method names, so
    ``metrics.py`` and the driver treat both sweeps identically), but covers
    static edge sets and household-free populations as well. ``nodes_by_age``
    is always the population's real age grouping.
    """

    def __init__(self, pop, nodes_by_age, hh_by_node, household_sizes,
                 household_edges, meta, *, tc=None, static_edges=None,
                 community_degree=None):
        self.pop = pop
        self.nodes_by_age = nodes_by_age
        self.hh_by_node = hh_by_node
        self.household_sizes = household_sizes
        self.household_edges = household_edges
        self.meta = meta
        self.tc = tc
        self._static_edges = static_edges
        self._community_degree = community_degree
        self.n_nodes = pop.n_nodes
        self.node_age = np.zeros(pop.n_nodes, dtype=np.int64)
        for nid, node in pop.nodes.items():
            self.node_age[nid] = node.age

    @property
    def is_temporal(self) -> bool:
        return self.tc is not None

    def edges_fn(self):
        if self.is_temporal:
            return temporal_edges_fn(
                self.tc,
                static_edges=[(u, v, "household", HOUSEHOLD_WEIGHT)
                              for u, v in self.household_edges],
            )
        # One list object, returned every day: ``simulate_seir`` recognises it
        # by identity and skips the O(E) daily adjacency rebuild.
        return static_edges_fn(self._static_edges)

    def household_degree(self) -> np.ndarray:
        deg = np.zeros(self.n_nodes)
        for u, v in self.household_edges:
            deg[u] += 1
            deg[v] += 1
        return deg

    def probe_degrees(self, n_days: int) -> dict[str, np.ndarray]:
        """Per-node community degree by layer, as a ``(n_days, n_nodes)`` array.

        For a temporal arm this draws ``n_days`` fresh days (advancing the
        realisation's own RNG, so those days are consumed before the epidemic
        starts -- days are exchangeable, so it is a harmless burn-in). A static
        arm has one unchanging degree per node, so the single ``community``
        layer is reported once and the temporal layers are *absent* rather than
        zero-filled: a static network has no ``one_time`` degree, and writing
        zeros would read as "measured, and zero".
        """
        if not self.is_temporal:
            return {"community": self._community_degree[None, :]}
        out = {L: np.zeros((n_days, self.n_nodes)) for L in TEMPORAL_LAYERS}
        for d in range(n_days):
            for u, v, _dur, layer in self.tc.daily_contacts():
                out[layer][d, u] += 1
                out[layer][d, v] += 1
        return out


def _degree_from_edges(edges, n_nodes: int) -> np.ndarray:
    deg = np.zeros(n_nodes)
    for e in edges:
        deg[e[0]] += 1
        deg[e[1]] += 1
    return deg


def _build_static(model: str, n_nodes: int, seed: int, *, households: bool):
    """Static-community arm: ``hh_static`` (households) or ``nohh_static``.

    The two differ in the population only -- observed household compositions, or
    one person per household. The community layer is built the same way in
    both: the age-structured summed fit, stubs matched within and between the
    population's nine age groups.
    """
    fits = get_fits()
    raw_sampler, _ = samplers_for(model)
    rng = np.random.default_rng(seed)
    cc.N_NODES = n_nodes

    templates = fits["hh_ages"] if households else fits["singleton_hh_ages"]
    pop, nba, hbn, hs, he = cc.build_population(templates, rng)
    params = params_for(model, "summed")

    community = cc.build_static_community_edges(
        params, nba, hbn, he, rng, raw_sampler=raw_sampler
    )

    edges = ([(u, v, "household", HOUSEHOLD_WEIGHT) for u, v in he]
             + [(u, v, "community", w) for u, v, w in community])

    meta = {
        "model": model, "n_nodes_requested": n_nodes, "n_nodes": pop.n_nodes,
        "n_households": pop.n_households, "n_household_edges": len(he),
        "n_community_edges": len(community), "network_seed": seed,
        "noise_kind": "static", "rho": float("nan"),
        "icc_persistent": float("nan"), "icc_target_achievable": float("nan"),
    }
    return Realisation(
        pop, nba, hbn, hs, he, meta,
        static_edges=edges,
        community_degree=_degree_from_edges(community, pop.n_nodes),
    )


def _build_nohh_temporal(model: str, target_icc: float, n_nodes: int, seed: int):
    """Temporal community layers on a singleton (household-free) population.

    Step for step the main sweep's ``builders.build_realisation`` -- the same
    age-structured layered parameters, read from the main sweep's own fit cache,
    the same everyday matching, the same persistence calibration -- with the
    household templates swapped for singletons, so there is no household layer.
    """
    fits = get_fits()
    params, raw_sampler, rate_sampler = MB.samplers_for(model)
    rng = np.random.default_rng(seed)
    cc.N_NODES = n_nodes

    pop, nba, hbn, hs, he = cc.build_population(fits["singleton_hh_ages"], rng)

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
        return Realisation(pop, nba, hbn, hs, he, meta, tc=tc)

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
    return Realisation(pop, nba, hbn, hs, he, meta, tc=tc)


def build_realisation(arm: str, model: str, target_icc: float | None,
                      n_nodes: int, seed: int) -> Realisation:
    """Build one population + community network for an ``(arm, model, icc, n)`` cell."""
    if arm == "hh_static":
        real = _build_static(model, n_nodes, seed, households=True)
    elif arm == "nohh_static":
        real = _build_static(model, n_nodes, seed, households=False)
    elif arm == "nohh_temporal":
        if target_icc is None:
            raise ValueError("nohh_temporal needs an ICC target")
        real = _build_nohh_temporal(model, target_icc, n_nodes, seed)
    else:
        raise ValueError(f"unknown arm {arm!r} (expected one of {list(CS.ARMS)})")

    real.meta.update({
        "arm": arm,
        "temporal": CS.arm_is_temporal(arm),
        "households": CS.arm_has_households(arm),
        "target_icc": target_icc if target_icc is not None else float("nan"),
    })
    if not CS.arm_has_households(arm):
        # Recorded with the results: a no-household row without this value was
        # produced by the earlier pooled (age-blind) construction.
        real.meta["nohh_contact_fit"] = NOHH_CONTACT_FIT
    return real
