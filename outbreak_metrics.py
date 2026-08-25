"""Post-hoc analysis helpers for Gillespie SEIR transmission logs.

These operate on the saved transmission log (one row per infection event:
time, source, target, layer, source_generation, target_generation) plus
lightweight population metadata, and are shared by ``run_seir_comparison.py``
(to compute summary statistics at save time) and the comparison notebook
(to reproduce and extend those analyses on demand).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def infected_node_set(transmissions_df: pd.DataFrame, seed_nodes: Sequence[int]) -> set[int]:
    """All nodes that were ever infected (seeds plus every transmission target)."""
    infected = {int(n) for n in seed_nodes}
    if not transmissions_df.empty:
        infected.update(int(n) for n in transmissions_df["target"])
    return infected


def secondary_case_distribution(
    transmissions_df: pd.DataFrame,
    seed_nodes: Sequence[int],
    max_generation: int = 1,
) -> pd.Series:
    """Secondary-case counts for every node that reached generation <= ``max_generation``.

    Includes nodes with zero secondary cases (dead-end infections) so the
    distribution — and its mean, used for R0 — is unbiased.
    """
    nodes = {int(n) for n in seed_nodes}
    if not transmissions_df.empty:
        early_targets = transmissions_df.loc[
            transmissions_df["target_generation"] <= max_generation, "target"
        ]
        nodes.update(int(n) for n in early_targets)

    counts = {n: 0 for n in nodes}
    if not transmissions_df.empty:
        src_counts = (
            transmissions_df.loc[transmissions_df["source"].isin(nodes), "source"]
            .value_counts()
        )
        for n, c in src_counts.items():
            counts[int(n)] = int(c)
    return pd.Series(counts, name="secondary_cases")


def r0_estimate(transmissions_df: pd.DataFrame, seed_nodes: Sequence[int]) -> float:
    """Mean secondary cases directly caused by the initial (generation-0) index cases.

    This is the operational R0: each index case is introduced into an
    otherwise fully susceptible population, so its direct offspring count is
    the cleanest available estimate of the basic reproduction number for
    that network/beta combination.
    """
    seed_set = {int(n) for n in seed_nodes}
    if not seed_set:
        return float("nan")
    dist = secondary_case_distribution(transmissions_df, seed_nodes, max_generation=0)
    return float(dist.loc[dist.index.isin(seed_set)].mean())


def infections_by_layer(transmissions_df: pd.DataFrame) -> pd.Series:
    """Total infection count attributable to each contact layer."""
    if transmissions_df.empty:
        return pd.Series(dtype=int)
    return transmissions_df["layer"].value_counts()


def daily_infections_by_layer(transmissions_df: pd.DataFrame) -> pd.DataFrame:
    """New-infection counts per (day, layer), for tracking layer importance over time."""
    if transmissions_df.empty:
        return pd.DataFrame(columns=["day", "layer", "count"])
    df = transmissions_df.copy()
    df["day"] = np.floor(df["time"]).astype(int)
    return df.groupby(["day", "layer"]).size().reset_index(name="count")


def household_infection_stats(
    node_household: dict[int, int],
    household_sizes: dict[int, int],
    infected_nodes: set[int],
) -> dict[str, float]:
    """Household-level outcomes, restricted to multi-person households.

    pct_households_infected : fraction of multi-person households with >=1 case.
    household_attack_rate   : mean, over infected multi-person households, of
                              (infected members - 1) / (household size - 1) —
                              i.e. the share of the *rest* of the household
                              that caught it after the first case.
    """
    hh_infected_members: dict[int, int] = {}
    for node, hh in node_household.items():
        if node in infected_nodes:
            hh_infected_members[hh] = hh_infected_members.get(hh, 0) + 1

    multi = {hh: size for hh, size in household_sizes.items() if size >= 2}
    if not multi:
        return {
            "pct_households_infected": float("nan"),
            "household_attack_rate": float("nan"),
            "n_multi_person_households": 0,
        }

    n_infected_hh = sum(1 for hh in multi if hh_infected_members.get(hh, 0) > 0)
    pct_infected = n_infected_hh / len(multi)

    sar_values = [
        (hh_infected_members.get(hh, 0) - 1) / (size - 1)
        for hh, size in multi.items()
        if hh_infected_members.get(hh, 0) >= 1
    ]
    sar = float(np.mean(sar_values)) if sar_values else float("nan")

    return {
        "pct_households_infected": pct_infected,
        "household_attack_rate": sar,
        "n_multi_person_households": len(multi),
    }


def growth_rate_series(cumulative_infected: np.ndarray) -> np.ndarray:
    """Instantaneous growth rate r(t) = d(ln cumulative_infected)/dt.

    Uses a centred finite difference (``np.gradient``) on a daily series of
    cumulative-ever-infected counts (``n_nodes - S(t)``).
    """
    log_cum = np.log(np.clip(cumulative_infected.astype(float), 1.0, None))
    return np.gradient(log_cum)
