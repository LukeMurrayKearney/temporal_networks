"""Shared machinery for the family of `run_seir_comparison*.py` scripts.

`run_seir_comparison.py` (negative binomial, the original/baseline model) is
left untouched and does not import this module -- it keeps its own inline
copies of everything below. This module exists so the three *alternative*
community-contact scripts (`run_seir_comparison_lognormal.py`,
`run_seir_comparison_gamma.py`, `run_seir_comparison_egoresample.py`) share
one implementation of the parts that don't depend on the distributional
choice: population/household construction, static and everyday-layer
community-edge building, the 2x2 realization factories, and the
simulate-and-save loop. The only thing that varies between the four scripts
is *how per-node community contact counts are drawn* -- i.e. which
`raw_sampler` (see `temporal.py`) and which fitted/pooled `params` get
threaded through.

Every realization factory and edge-building helper here takes an explicit
`raw_sampler` argument (default: `temporal.nb_raw_sampler`) rather than
hard-coding negative-binomial sampling, so the exact same code path builds
NB, log-normal, gamma, or ego-resampled networks -- only the `params` values
and `raw_sampler` passed in differ.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from network import Population, N_AGE
from temporal import (
    TemporalContacts,
    nb_raw_sampler,
    build_stubs_with_padding,
    _match_between_dur,
    _match_within_dur,
)
from gillespie_seir import (
    SEIRParams,
    simulate_seir,
    static_edges_fn,
    temporal_edges_fn,
    stack_daily_snapshots,
    DURATION_WEIGHTS,
    HOUSEHOLD_WEIGHT,
)
import outbreak_metrics as om

# ---------------------------------------------------------------------------
# Configuration shared by every model in the 2x2 design (identical across
# the NB baseline and all three alternative-distribution scripts, so runs
# are comparable beta-for-beta and rep-for-rep).
# ---------------------------------------------------------------------------

N_DUR = 5
N_FREQ = 3
LAYER_NAMES = ["everyday", "few_times_week", "one_time"]

DATA_ROOT = Path("..") / "analyse households"
EGO_DATA = Path("..") / "analytics" / "ego_data"

RNG_SEED = 42
N_NODES = 50_000
SIGMA = 1.0
GAMMA = 0.25
N_SEED_INFECTED = 5

N_DAYS_STRUCTURE = 60

# Beta grid re-chosen for the duration-weighted model: because short contacts
# now transmit at a small fraction of beta, R0 saturates around ~2-2.5 for the
# household models (see calibration), so the old 0.01-0.11 grid would sit almost
# entirely below takeoff. These betas span R0 ~0.3 (sub-critical) to ~2
# (super-critical) for household_temporal.
BETAS = [0.03, 0.05, 0.09, 0.15, 0.25, 0.45, 0.80]
RUNS_STATIC = {_beta: 10 for _beta in BETAS}
RUNS_TEMPORAL = {_beta: 10 for _beta in BETAS}

SINGLETON_HH_AGES = [[2], [8], [15], [24], [35], [45], [55], [65], [80]]


class ZeroContactModel:
    def sample_contacts(self, age_group: int, rng=None) -> np.ndarray:
        return np.zeros(N_AGE, dtype=float)


# ---------------------------------------------------------------------------
# Survey loading + generic per-cell parametric fitting
#
# Fitting a 2-parameter family (NB, log-normal, gamma - anything with a
# `fit_cell_fn(counts) -> (param0, param1)` signature) one (a, b, d) cell at
# a time is identical code regardless of family; only `fit_cell_fn` changes.
# ---------------------------------------------------------------------------

def load_survey_data() -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (hh_ages, all_contacts, source_ages, reduced_arr)."""
    with open(DATA_ROOT / "hh_egos" / "uk_reconnect_hh_ages.json") as f:
        hh_ages = json.load(f)
    with open(EGO_DATA / "reconnect_for_temp.json") as f:
        raw_ego = json.load(f)
    with open(EGO_DATA / "reconnect_reduced.json") as f:
        raw_reduced = json.load(f)

    all_contacts = np.array([r["contacts"] for r in raw_ego], dtype=np.int32)
    source_ages = np.array([r["age"] for r in raw_ego], dtype=np.int32)
    reduced_arr = np.array(raw_reduced, dtype=np.float64)

    print(f"Household templates: {len(hh_ages):,}")
    print(f"Ego records (age-structured): {len(raw_ego):,}  shape {all_contacts.shape}")
    print(f"Ego records (reduced, age-blind): {reduced_arr.shape}")
    return hh_ages, all_contacts, source_ages, reduced_arr


def fit_params_layered(all_contacts: np.ndarray, source_ages: np.ndarray, fit_cell_fn) -> list[np.ndarray]:
    params_list = []
    for f in range(N_FREQ):
        params_f = np.zeros((N_AGE, N_AGE, N_DUR, 2), dtype=np.float64)
        counts_f = all_contacts[:, :, :, f]
        for a in range(N_AGE):
            mask_a = source_ages == a
            counts_af = counts_f[mask_a]
            for b in range(N_AGE):
                for d in range(N_DUR):
                    p0, p1 = fit_cell_fn(counts_af[:, b, d].astype(float))
                    params_f[a, b, d, 0] = p0
                    params_f[a, b, d, 1] = p1
        params_list.append(params_f)
    return params_list


def fit_params_summed(all_contacts: np.ndarray, source_ages: np.ndarray, fit_cell_fn) -> np.ndarray:
    counts_summed = all_contacts.sum(axis=3)
    params = np.zeros((N_AGE, N_AGE, N_DUR, 2), dtype=np.float64)
    for a in range(N_AGE):
        mask_a = source_ages == a
        counts_af = counts_summed[mask_a]
        for b in range(N_AGE):
            for d in range(N_DUR):
                p0, p1 = fit_cell_fn(counts_af[:, b, d].astype(float))
                params[a, b, d, 0] = p0
                params[a, b, d, 1] = p1
    return params


def fit_params_reduced_layered(reduced_arr: np.ndarray, fit_cell_fn) -> list[np.ndarray]:
    params_list = []
    for f in range(N_FREQ):
        params_f = np.zeros((1, 1, N_DUR, 2), dtype=np.float64)
        counts_f = reduced_arr[:, :, f]
        for d in range(N_DUR):
            p0, p1 = fit_cell_fn(counts_f[:, d].astype(float))
            params_f[0, 0, d, 0] = p0
            params_f[0, 0, d, 1] = p1
        params_list.append(params_f)
    return params_list


def fit_params_reduced_summed(reduced_arr: np.ndarray, fit_cell_fn) -> np.ndarray:
    counts_summed = reduced_arr.sum(axis=2)
    params = np.zeros((1, 1, N_DUR, 2), dtype=np.float64)
    for d in range(N_DUR):
        p0, p1 = fit_cell_fn(counts_summed[:, d].astype(float))
        params[0, 0, d, 0] = p0
        params[0, 0, d, 1] = p1
    return params


# ---------------------------------------------------------------------------
# Per-realization population + edge construction (distribution-agnostic:
# `raw_sampler` decides how per-node counts are drawn from `params`)
# ---------------------------------------------------------------------------

def build_population(hh_ages: list, rng_net: np.random.Generator):
    pop = Population.build(
        n_nodes=N_NODES, hh_ages=hh_ages, contact_model=ZeroContactModel(),
        layer_names=["community"], rng=rng_net,
    )
    nodes_by_age: dict[int, list[int]] = {a: [] for a in range(N_AGE)}
    for nid, node in pop.nodes.items():
        nodes_by_age[node.age].append(nid)
    hh_by_node = {m.node_id: h.hh_id for h in pop.households for m in h.members}
    household_sizes = {h.hh_id: h.size for h in pop.households}
    household_edges = [e for hh in pop.households for e in hh.edges()]
    return pop, nodes_by_age, hh_by_node, household_sizes, household_edges


def build_static_community_edges(
    params_single_layer: np.ndarray,
    nodes_by_age: dict[int, list[int]],
    hh_by_node: dict[int, int],
    household_edges: list[tuple[int, int]],
    rng_net: np.random.Generator,
    raw_sampler=nb_raw_sampler,
) -> list[tuple[int, int, float]]:
    """Ratio-padded stub matching for one (N_AGE, N_AGE, N_DUR, 2)-shaped param set.

    Returns ``(u, v, weight)`` triples, where ``weight = DURATION_WEIGHTS[d]``
    is the duration-derived transmissibility of that edge's duration bin.
    """
    existing = {(min(u, v), max(u, v)) for u, v in household_edges}
    stubs = build_stubs_with_padding(params_single_layer, nodes_by_age, rng_net, raw_sampler=raw_sampler)
    edges: list[tuple[int, int, float]] = []
    for a in range(N_AGE):
        for b in range(a, N_AGE):
            for d in range(N_DUR):
                if a == b:
                    pool = stubs.get((a, a, d), [])
                    if len(pool) >= 2:
                        edges.extend(
                            (u, v, DURATION_WEIGHTS[_d]) for u, v, _d in
                            _match_within_dur(pool, d, rng_net, existing, hh_by_node)
                        )
                else:
                    pool_a = stubs.get((a, b, d), [])
                    pool_b = stubs.get((b, a, d), [])
                    if pool_a and pool_b:
                        edges.extend(
                            (u, v, DURATION_WEIGHTS[_d]) for u, v, _d in
                            _match_between_dur(pool_a, pool_b, d, rng_net, existing, hh_by_node)
                        )
    return edges


def build_everyday_edges_ud(
    params_everyday: np.ndarray,
    nodes_by_age: dict[int, list[int]],
    hh_by_node: dict[int, int],
    household_edges: list[tuple[int, int]],
    rng_net: np.random.Generator,
    raw_sampler=nb_raw_sampler,
) -> list[tuple[int, int, int]]:
    """Everyday edges kept as (u, v, duration) triples for TemporalContacts."""
    existing = {(min(u, v), max(u, v)) for u, v in household_edges}
    stubs = build_stubs_with_padding(params_everyday, nodes_by_age, rng_net, raw_sampler=raw_sampler)
    edges: list[tuple[int, int, int]] = []
    for a in range(N_AGE):
        for b in range(a, N_AGE):
            for d in range(N_DUR):
                if a == b:
                    pool = stubs.get((a, a, d), [])
                    if len(pool) >= 2:
                        edges.extend(_match_within_dur(pool, d, rng_net, existing, hh_by_node))
                else:
                    pool_a = stubs.get((a, b, d), [])
                    pool_b = stubs.get((b, a, d), [])
                    if pool_a and pool_b:
                        edges.extend(
                            _match_between_dur(pool_a, pool_b, d, rng_net, existing, hh_by_node)
                        )
    return edges


def flat_nodes_by_age(pop: Population) -> dict[int, list[int]]:
    """Every node in a single age-blind group, for the reduced-data (age-blind) params."""
    return {0: list(pop.nodes.keys())}


def edges_to_csv(edges, path: Path, layer: str | None = None) -> None:
    if layer is None:
        df = pd.DataFrame(edges, columns=["u", "v"])
    else:
        df = pd.DataFrame([(u, v, layer) for u, v in edges], columns=["u", "v", "layer"])
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Realization factories: rng_net -> (edges_fn, n_nodes, node_household,
#                                     household_sizes, structural_stats)
# ---------------------------------------------------------------------------

def realization_household_temporal(rng_net, hh_ages, params, raw_sampler=nb_raw_sampler,
                                   rate_sampler=None):
    pop, nba, hbn, hs, he = build_population(hh_ages, rng_net)
    everyday = build_everyday_edges_ud(params[0], nba, hbn, he, rng_net, raw_sampler=raw_sampler)
    tc = TemporalContacts.build(
        pop=pop, nb_params=params, everyday_edges=everyday, rng=rng_net,
        raw_sampler=raw_sampler, rate_sampler=rate_sampler,
    )
    edges_fn = temporal_edges_fn(
        tc, static_edges=[(u, v, "household", HOUSEHOLD_WEIGHT) for u, v in he]
    )
    struct = {
        "n_nodes": pop.n_nodes, "n_households": pop.n_households,
        "n_household_edges": len(he), "n_everyday_edges": len(everyday),
    }
    return edges_fn, pop.n_nodes, hbn, hs, struct


def realization_household_static(rng_net, hh_ages, params_summed, raw_sampler=nb_raw_sampler):
    pop, nba, hbn, hs, he = build_population(hh_ages, rng_net)
    community = build_static_community_edges(params_summed, nba, hbn, he, rng_net, raw_sampler=raw_sampler)
    edges = (
        [(u, v, "household", HOUSEHOLD_WEIGHT) for u, v in he]
        + [(u, v, "community", w) for u, v, w in community]
    )
    struct = {
        "n_nodes": pop.n_nodes, "n_households": pop.n_households,
        "n_household_edges": len(he), "n_community_edges": len(community),
    }
    return static_edges_fn(edges), pop.n_nodes, hbn, hs, struct


def realization_no_household_static(rng_net, singleton_hh_ages, params_reduced_summed, raw_sampler=nb_raw_sampler):
    """Age-blind: community edges from `reconnect_reduced.json` (frequency-summed)."""
    pop, _nba, hbn, _hs, he = build_population(singleton_hh_ages, rng_net)
    flat_nba = flat_nodes_by_age(pop)
    community = build_static_community_edges(
        params_reduced_summed, flat_nba, hbn, he, rng_net, raw_sampler=raw_sampler,
    )
    edges = [(u, v, "community", w) for u, v, w in community]
    struct = {"n_nodes": pop.n_nodes, "n_community_edges": len(community)}
    return static_edges_fn(edges), pop.n_nodes, None, None, struct


def realization_no_household_temporal(rng_net, singleton_hh_ages, params_reduced_layered,
                                      raw_sampler=nb_raw_sampler, rate_sampler=None):
    """Age-blind: everyday/few_times_week/one_time layers from `reconnect_reduced.json`."""
    pop, _nba, hbn, _hs, he = build_population(singleton_hh_ages, rng_net)
    flat_nba = flat_nodes_by_age(pop)
    everyday = build_everyday_edges_ud(
        params_reduced_layered[0], flat_nba, hbn, he, rng_net, raw_sampler=raw_sampler,
    )
    tc = TemporalContacts.build(
        pop=pop, nb_params=params_reduced_layered, everyday_edges=everyday,
        rng=rng_net, nodes_by_age=flat_nba, raw_sampler=raw_sampler, rate_sampler=rate_sampler,
    )
    edges_fn = temporal_edges_fn(tc, static_edges=[])
    struct = {"n_nodes": pop.n_nodes, "n_everyday_edges": len(everyday)}
    return edges_fn, pop.n_nodes, None, None, struct


def build_model_runners(
    hh_ages: list,
    params_layered,
    params_summed,
    params_reduced_layered,
    params_reduced_summed,
    raw_sampler=nb_raw_sampler,
    rate_sampler=None,
    singleton_hh_ages: list = SINGLETON_HH_AGES,
    seed_offsets: tuple[int, int, int, int] = (1_000, 2_000, 3_000, 4_000),
) -> dict[str, tuple]:
    """(realization_fn, runs_grid, seed_offset) for each of the four models.

    ``rate_sampler`` (optional) makes the temporal models' one_time layer a
    persistent per-node Poisson rate (correlated with the individual across
    days) instead of a fresh nightly redraw; it is ignored by the static
    models, which have no daily one_time layer. See ``TemporalContacts.build``.
    """
    return {
        "household_temporal": (
            lambda rng_net: realization_household_temporal(
                rng_net, hh_ages, params_layered, raw_sampler, rate_sampler),
            RUNS_TEMPORAL, seed_offsets[0],
        ),
        "household_static": (
            lambda rng_net: realization_household_static(rng_net, hh_ages, params_summed, raw_sampler),
            RUNS_STATIC, seed_offsets[1],
        ),
        "no_household_static": (
            lambda rng_net: realization_no_household_static(
                rng_net, singleton_hh_ages, params_reduced_summed, raw_sampler),
            RUNS_STATIC, seed_offsets[2],
        ),
        "no_household_temporal": (
            lambda rng_net: realization_no_household_temporal(
                rng_net, singleton_hh_ages, params_reduced_layered, raw_sampler, rate_sampler),
            RUNS_TEMPORAL, seed_offsets[3],
        ),
    }


# ---------------------------------------------------------------------------
# Simulation + saving
# ---------------------------------------------------------------------------

def run_and_save_model(
    name: str,
    realization_fn,
    runs_grid: dict[float, int],
    seed_offset: int,
    net_dir: Path,
    outbreak_dir: Path,
) -> None:
    out_dir = outbreak_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    net_out_dir = net_dir / name
    net_out_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = out_dir / "daily_snapshots"
    snapshots_dir.mkdir(exist_ok=True)

    summary_rows = []
    transmission_frames = []
    example_saved = False
    trajectories_by_beta: dict[float, list] = {}

    t_model0 = time.perf_counter()
    for beta, n_reps in runs_grid.items():
        beta_trajectories = []
        for rep in range(n_reps):
            t0 = time.perf_counter()
            rng_net = np.random.default_rng((seed_offset, int(round(beta * 1000)), rep, 0))
            rng_sim = np.random.default_rng((seed_offset, int(round(beta * 1000)), rep, 1))

            edges_fn, n_nodes, node_household, household_sizes, struct = realization_fn(rng_net)

            if not example_saved:
                _save_example_realization(name, net_out_dir, edges_fn, n_nodes)
                example_saved = True

            initial_infected = rng_sim.choice(n_nodes, size=N_SEED_INFECTED, replace=False).tolist()
            params = SEIRParams(beta=beta, sigma=SIGMA, gamma=GAMMA)
            traj = simulate_seir(
                n_nodes=n_nodes, edges_fn=edges_fn, params=params,
                initial_infected=initial_infected, rng=rng_sim,
            )
            beta_trajectories.append(traj)

            tdf = pd.DataFrame([t.__dict__ for t in traj.transmissions])
            if tdf.empty:
                tdf = pd.DataFrame(columns=[
                    "time", "source", "target", "layer",
                    "source_generation", "target_generation",
                ])
            tdf.insert(0, "rep", rep)
            tdf.insert(0, "beta", beta)
            transmission_frames.append(tdf)

            peak_t, peak_i = traj.peak_infectious()
            row = {
                "beta": beta,
                "rep": rep,
                "final_size": traj.final_size(),
                "attack_rate": traj.final_size() / n_nodes,
                "peak_infectious": peak_i,
                "peak_time": peak_t,
                "duration_days": float(traj.times[-1]),
                "n_events": len(traj.times),
                "r0_estimate": om.r0_estimate(tdf, traj.seed_nodes),
                "seed_nodes": json.dumps(traj.seed_nodes),
                **struct,
            }
            if node_household is not None:
                infected = om.infected_node_set(tdf, traj.seed_nodes)
                row.update(om.household_infection_stats(node_household, household_sizes, infected))
            else:
                row.update({
                    "pct_households_infected": float("nan"),
                    "household_attack_rate": float("nan"),
                    "n_multi_person_households": 0,
                })
            summary_rows.append(row)

            print(f"  [{name}] beta={beta} rep={rep}: {time.perf_counter() - t0:.1f}s  "
                  f"final_size={row['final_size']}  duration_days={row['duration_days']:.0f}")

        trajectories_by_beta[beta] = beta_trajectories

    # Every run now stops on its own once it dies out (no fixed day cap), so
    # replicate durations vary. Stack every replicate onto one common grid --
    # the longest-running replicate across the whole model -- so every
    # daily_snapshots/beta_*.npz shares the same day axis (required by the
    # comparison notebook, which reads a single `sim_days` from params.json).
    sim_days = max(
        int(round(traj.times[-1]))
        for trajs in trajectories_by_beta.values()
        for traj in trajs
    )

    for beta, beta_trajectories in trajectories_by_beta.items():
        stacked = stack_daily_snapshots(beta_trajectories, sim_days)
        np.savez(snapshots_dir / f"beta_{int(round(beta * 1000)):04d}.npz", **stacked)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "runs_summary.csv", index=False)

    epidemic_cols = {
        "final_size", "attack_rate", "peak_infectious", "peak_time", "duration_days",
        "n_events", "r0_estimate", "seed_nodes", "pct_households_infected",
        "household_attack_rate", "n_multi_person_households",
    }
    struct_cols = [c for c in summary_df.columns if c not in epidemic_cols]
    summary_df[struct_cols].to_csv(net_out_dir / "network_realizations_summary.csv", index=False)

    transmission_log = pd.concat(transmission_frames, ignore_index=True)
    transmission_log.to_csv(out_dir / "transmission_log.csv", index=False)

    with open(out_dir / "params.json", "w") as f:
        json.dump({
            "sigma": SIGMA, "gamma": GAMMA, "n_seed_infected": N_SEED_INFECTED,
            "n_nodes_requested": N_NODES, "sim_days": sim_days,
            "betas": sorted(runs_grid.keys()),
            "runs_grid": runs_grid,
            "has_households": bool(summary_df["pct_households_infected"].notna().any()),
        }, f, indent=2)

    print(f"[{name}] done in {time.perf_counter() - t_model0:.1f}s  "
          f"({len(summary_rows)} realizations, {len(transmission_log):,} transmission events)")


def _save_example_realization(name: str, net_out_dir: Path, edges_fn, n_nodes: int) -> None:
    example_dir = net_out_dir / "example_realization"
    example_dir.mkdir(exist_ok=True)
    is_temporal = "temporal" in name
    n_days_full_dump = 3 if is_temporal else 1

    degree_rows = []
    n_days_total = N_DAYS_STRUCTURE if is_temporal else n_days_full_dump
    for day in range(n_days_total):
        day_edges = edges_fn(day)
        if day < n_days_full_dump:
            pd.DataFrame(day_edges, columns=["u", "v", "layer", "weight"]).to_csv(
                example_dir / f"day_{day:02d}.csv", index=False
            )
        if is_temporal:
            degree_rows.append(_degree_stats_for_day(day, day_edges, n_nodes))

    if is_temporal:
        pd.DataFrame(degree_rows).to_csv(example_dir / "degree_over_time.csv", index=False)

    with open(example_dir / "README.txt", "w") as f:
        f.write(
            "One example network realization out of this model's full ensemble "
            "(see network_realizations_summary.csv / runs_summary.csv for the "
            "structural stats of every realization). For static models day_00.csv "
            "is the whole (fixed) network; for temporal models each day_NN.csv is "
            "a different simulated day from the same realization (only the first "
            f"{n_days_full_dump} days are dumped in full — degree_over_time.csv "
            f"covers all {n_days_total} sampled days as summary statistics only, "
            "since community layers change daily and full edge lists for every "
            "day would be large).\n"
        )


def _degree_stats_for_day(day: int, day_edges: list[tuple], n_nodes: int) -> dict:
    degree_total = np.zeros(n_nodes, dtype=np.int64)
    degree_community = np.zeros(n_nodes, dtype=np.int64)
    for u, v, layer, _weight in day_edges:
        degree_total[u] += 1
        degree_total[v] += 1
        if layer != "household":
            degree_community[u] += 1
            degree_community[v] += 1
    return {
        "day": day,
        "mean_degree_total": float(degree_total.mean()),
        "std_degree_total": float(degree_total.std()),
        "mean_degree_community": float(degree_community.mean()),
        "std_degree_community": float(degree_community.std()),
    }
