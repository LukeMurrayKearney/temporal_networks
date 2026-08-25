"""Per-replicate epidemic metrics beyond the base ``runs_summary`` columns.

Everything here is derived from artefacts the simulator already produces (the
transmission log, the trajectory, the contact network), and is grouped to match
the five things worth discriminating models on once their *marginal* degree
distributions already agree:

A. threshold / takeoff        -- ``takeoff_metrics``
B. superspreading             -- ``offspring_metrics``
C. layer & age attribution    -- ``layer_metrics``, ``age_attack_metrics``
D. dynamics                   -- ``growth_metrics``, ``generation_interval_metrics``
E. structural covariates      -- ``network_structure_metrics``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

# ---------------------------------------------------------------------------
# A. Threshold / takeoff
# ---------------------------------------------------------------------------

def takeoff_metrics(final_size: int, n_nodes: int, n_seeds: int,
                    takeoff_fraction: float) -> dict:
    """Flag and size of a takeoff, for per-cell takeoff *probability*.

    The probability itself is an average over replicates, so what is stored per
    replicate is the indicator plus the raw sizes it is derived from.
    """
    beyond_seeds = int(final_size) - int(n_seeds)
    return {
        "final_size": int(final_size),
        "infections_beyond_seeds": max(beyond_seeds, 0),
        "attack_rate": float(final_size) / n_nodes,
        "took_off": bool(final_size >= takeoff_fraction * n_nodes),
        "takeoff_fraction_used": takeoff_fraction,
    }


# ---------------------------------------------------------------------------
# B. Superspreading / offspring distribution
# ---------------------------------------------------------------------------

def _fit_nb_dispersion(counts: np.ndarray) -> float:
    """MLE of the negative-binomial dispersion k for offspring counts.

    Mean is fixed at the sample mean (its own MLE under the NB), and the
    profile likelihood is maximised over k -- the Lloyd-Smith et al. (2005)
    superspreading parameter. Small k = a few individuals cause most
    transmission; k -> inf recovers Poisson (homogeneous) transmission.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size < 2:
        return float("nan")
    mean = counts.mean()
    if mean <= 0:
        return float("nan")
    var = counts.var(ddof=1)
    if var <= mean:                     # under-dispersed: NB has no valid k
        return float("inf")

    def neg_ll(log_k: float) -> float:
        k = np.exp(log_k)
        return -float(np.sum(stats.nbinom.logpmf(counts, k, k / (k + mean))))

    # method-of-moments start, then bounded 1-D search in log space
    k0 = mean ** 2 / (var - mean)
    try:
        res = optimize.minimize_scalar(
            neg_ll, bracket=None, bounds=(np.log(1e-4), np.log(1e4)), method="bounded"
        )
        return float(np.exp(res.x)) if res.success else float(k0)
    except Exception:
        return float(k0)


def offspring_metrics(tdf: pd.DataFrame, seed_nodes, n_nodes: int) -> dict:
    """Realised individual reproduction number and its dispersion.

    Built over every individual who was *infected and had the chance to
    transmit* (all infectors plus every infected node with zero offspring), so
    the zero class is included and the mean is unbiased.
    """
    empty = {
        "n_infectors_considered": 0, "R_mean_realised": float("nan"),
        "R_var_realised": float("nan"), "offspring_dispersion_k": float("nan"),
        "offspring_max": 0, "frac_zero_offspring": float("nan"),
        "top10pct_transmission_share": float("nan"),
        "top20pct_transmission_share": float("nan"),
    }
    if tdf is None or tdf.empty:
        return empty

    # every node that was ever infected (seeds + all targets) could transmit
    infected = {int(n) for n in seed_nodes}
    infected.update(int(t) for t in tdf["target"].to_numpy())
    src_counts = tdf["source"].value_counts()
    counts = np.zeros(len(infected), dtype=np.int64)
    for i, node in enumerate(infected):
        counts[i] = int(src_counts.get(node, 0))

    total = counts.sum()
    order = np.sort(counts)[::-1]
    n = len(order)
    top10 = order[: max(1, int(round(0.10 * n)))].sum()
    top20 = order[: max(1, int(round(0.20 * n)))].sum()

    return {
        "n_infectors_considered": int(n),
        "R_mean_realised": float(counts.mean()),
        "R_var_realised": float(counts.var(ddof=1)) if n > 1 else float("nan"),
        "offspring_dispersion_k": _fit_nb_dispersion(counts),
        "offspring_max": int(counts.max()),
        "frac_zero_offspring": float((counts == 0).mean()),
        "top10pct_transmission_share": float(top10 / total) if total else float("nan"),
        "top20pct_transmission_share": float(top20 / total) if total else float("nan"),
    }


# ---------------------------------------------------------------------------
# C. Layer and age attribution
# ---------------------------------------------------------------------------

def layer_metrics(tdf: pd.DataFrame, layers=("household", "everyday",
                                             "few_times_week", "one_time")) -> dict:
    """Share of transmission carried by each contact layer."""
    out = {f"frac_trans_{L}": float("nan") for L in layers}
    out["n_transmissions"] = 0 if tdf is None or tdf.empty else int(len(tdf))
    if tdf is None or tdf.empty:
        return out
    counts = tdf["layer"].value_counts()
    total = int(counts.sum())
    for L in layers:
        out[f"frac_trans_{L}"] = float(counts.get(L, 0)) / total if total else float("nan")
    return out


def age_attack_metrics(tdf: pd.DataFrame, seed_nodes, node_age: np.ndarray,
                       n_age: int = 9) -> dict:
    """Attack rate within each age group (infected / population of that group)."""
    out = {}
    infected = {int(n) for n in seed_nodes}
    if tdf is not None and not tdf.empty:
        infected.update(int(t) for t in tdf["target"].to_numpy())
    if not len(node_age):
        return {f"attack_rate_age{a}": float("nan") for a in range(n_age)}
    inf_arr = np.fromiter(infected, dtype=np.int64, count=len(infected))
    inf_ages = node_age[inf_arr]
    pop_counts = np.bincount(node_age, minlength=n_age).astype(float)
    inf_counts = np.bincount(inf_ages, minlength=n_age).astype(float)
    for a in range(n_age):
        out[f"attack_rate_age{a}"] = (
            float(inf_counts[a] / pop_counts[a]) if pop_counts[a] > 0 else float("nan")
        )
    return out


# ---------------------------------------------------------------------------
# D. Dynamics: growth rate, doubling time, generation interval
# ---------------------------------------------------------------------------

def growth_metrics(daily: dict, n_nodes: int, min_cases: int = 20,
                   max_frac: float = 0.05) -> dict:
    """Early exponential growth rate r and doubling time.

    Fits ``log(cumulative infections)`` against day over the window running
    from the first day with >= ``min_cases`` to the last day still below
    ``max_frac`` of the population -- i.e. after seeding noise but before
    susceptible depletion bends the curve.
    """
    cum = n_nodes - np.asarray(daily["S"], dtype=float)
    days = np.arange(len(cum), dtype=float)
    mask = (cum >= min_cases) & (cum <= max_frac * n_nodes)
    if mask.sum() < 3:
        return {"growth_rate_r": float("nan"), "doubling_time_days": float("nan"),
                "growth_fit_days": int(mask.sum())}
    slope, intercept = np.polyfit(days[mask], np.log(cum[mask]), 1)
    # A slope indistinguishable from zero (a fizzling / sub-critical run) makes
    # log(2)/slope explode to meaningless magnitudes and poisons any average, so
    # report no doubling time rather than a spurious one.
    doubling = float(np.log(2) / slope) if slope > 1e-4 else float("nan")
    return {
        "growth_rate_r": float(slope),
        "doubling_time_days": doubling,
        "growth_fit_days": int(mask.sum()),
    }


def generation_interval_metrics(tdf: pd.DataFrame, seed_nodes) -> dict:
    """Realised generation interval: infector -> infectee infection-time gaps."""
    if tdf is None or tdf.empty:
        return {"gen_interval_mean": float("nan"), "gen_interval_sd": float("nan"),
                "gen_interval_median": float("nan"), "n_gen_intervals": 0}
    inf_time = {int(n): 0.0 for n in seed_nodes}
    for tgt, t in zip(tdf["target"].to_numpy(), tdf["time"].to_numpy()):
        inf_time[int(tgt)] = float(t)
    gaps = [
        float(t) - inf_time[int(src)]
        for src, t in zip(tdf["source"].to_numpy(), tdf["time"].to_numpy())
        if int(src) in inf_time
    ]
    if not gaps:
        return {"gen_interval_mean": float("nan"), "gen_interval_sd": float("nan"),
                "gen_interval_median": float("nan"), "n_gen_intervals": 0}
    g = np.asarray(gaps)
    return {
        "gen_interval_mean": float(g.mean()),
        "gen_interval_sd": float(g.std(ddof=1)) if g.size > 1 else float("nan"),
        "gen_interval_median": float(np.median(g)),
        "n_gen_intervals": int(g.size),
    }


def peak_metrics(traj, n_nodes: int) -> dict:
    peak_t, peak_i = traj.peak_infectious()
    return {
        "peak_infectious": int(peak_i),
        "peak_prevalence": float(peak_i) / n_nodes,
        "peak_time": float(peak_t),
        "duration_days": float(traj.times[-1]),
        "n_events": int(len(traj.times)),
    }


# ---------------------------------------------------------------------------
# E. Structural covariates of the realised network
# ---------------------------------------------------------------------------

def _degree_stats(deg: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_mean": float(deg.mean()),
        f"{prefix}_var": float(deg.var(ddof=1)) if deg.size > 1 else float("nan"),
        f"{prefix}_skew": float(stats.skew(deg)) if deg.size > 2 else float("nan"),
        f"{prefix}_p99": float(np.percentile(deg, 99)),
        f"{prefix}_max": float(deg.max()),
    }


def network_structure_metrics(day_degrees: dict[str, np.ndarray],
                              hh_degree: np.ndarray) -> dict:
    """Degree distribution shape per layer + realised 2-day one_time ICC.

    ``day_degrees[layer]`` is a ``(n_probe_days, n_nodes)`` per-node degree
    array drawn from the realised network. The one_time ICC uses the same
    two-day estimator applied to the France panel
    (``Cov(day1, day2) / Var(pooled)``), so the model value and the 0.365
    empirical target are like-for-like.
    """
    out: dict[str, float] = {}
    for layer, arr in day_degrees.items():
        arr = np.asarray(arr, dtype=float)
        out.update(_degree_stats(arr.ravel(), f"deg_{layer}"))
        out[f"deg_{layer}_daily_mean"] = float(arr.mean())
    out.update(_degree_stats(np.asarray(hh_degree, dtype=float), "deg_household"))

    total = sum(np.asarray(a, dtype=float) for a in day_degrees.values())
    out.update(_degree_stats(total.ravel(), "deg_community_total"))

    ot = np.asarray(day_degrees.get("one_time"), dtype=float)
    if ot is not None and ot.ndim == 2 and ot.shape[0] >= 2:
        y1, y2 = ot[0], ot[1]
        pooled_var = np.concatenate([y1, y2]).var(ddof=1)
        out["one_time_icc_realised"] = (
            float(np.cov(y1, y2, ddof=1)[0, 1] / pooled_var) if pooled_var > 0 else float("nan")
        )
    else:
        out["one_time_icc_realised"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def daily_layer_incidence(tdf: pd.DataFrame, n_days: int,
                          layers=("household", "everyday", "few_times_week",
                                  "one_time")) -> pd.DataFrame:
    """Per-day transmission counts split by layer (long format)."""
    if tdf is None or tdf.empty:
        return pd.DataFrame(columns=["day", *layers])
    day = np.floor(tdf["time"].to_numpy()).astype(int)
    tmp = pd.DataFrame({"day": day, "layer": tdf["layer"].to_numpy()})
    wide = (tmp.groupby(["day", "layer"]).size().unstack(fill_value=0)
            .reindex(columns=list(layers), fill_value=0)
            .reindex(range(n_days + 1), fill_value=0))
    wide.index.name = "day"
    return wide.reset_index()
