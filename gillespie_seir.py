"""Exact network Gillespie SEIR simulation with an Erlang-distributed exposed period.

The exposed period is split into three sub-compartments E1, E2, E3, each an
independent exponential stage with mean stay time 1 day; their sum is a
Gamma(shape=3, rate=1) incubation period with mean 3 days (the "chain trick").
The infectious period is a single exponential compartment with mean 4 days.

Transmission is network-based: every realised S-I edge is an independent
Poisson process with rate ``beta * weight`` (``beta`` is the variable
transmission-rate parameter; ``weight`` is a per-edge multiplier in ``(0, 1]``
that scales transmissibility by contact duration -- see ``DURATION_WEIGHTS``).
Edges carry a **layer** label (e.g. "household", "everyday", "community") and
an optional weight; a node pair may have more than one simultaneous edge if it
appears in more than one layer on a given day, in which case each layer's
edge is an independent transmission channel with its own weight. Edges given
without a weight (plain ``(u, v, layer)`` triples) default to weight ``1.0``,
so unweighted callers behave exactly as before. The simulator is a
Doob-Gillespie continuous-time Markov chain, so it is exact given the
contact network for a period.

Time-varying contact structure is supported by swapping the edge set at
fixed "day" boundaries between events.  This is exact: exponential holding
times are memoryless, so stopping a draw at a boundary and re-drawing under
the new rates on the far side is equivalent to simulating the true
piecewise-constant-rate CTMC.

Every transmission event is logged (time, source, target, layer, source
generation) so that downstream analysis can reconstruct the transmission
tree: per-link-type infection counts, secondary case distributions, and
generation-based R0 estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

S, E1, E2, E3, I, R = range(6)
STATE_NAMES = ["S", "E1", "E2", "E3", "I", "R"]
_EXPOSED = (E1, E2, E3)

# An edge is ``(u, v, layer)`` or ``(u, v, layer, weight)``. A missing weight
# defaults to 1.0 (see ``_edge_weight``), so unweighted callers are unchanged.
Edge = tuple  # (u, v, layer[, weight])

# ---------------------------------------------------------------------------
# Duration-weighted transmissibility
#
# The survey records each contact's duration in one of five bins d=0..4
# (<5min, 5-15min, 15-60min, 1-4h, >4h). We scale a realised contact's
# transmission rate by the bin's interval-midpoint duration in minutes,
# [3, 10, 37.5, 150, 480], normalised by the longest (480) -- so a >4h
# contact transmits at the full ``beta`` and a <5min contact at ~0.6% of it.
# A contact of duration bin d thus fires at rate ``beta * DURATION_WEIGHTS[d]``.
# ---------------------------------------------------------------------------
DURATION_WEIGHTS = tuple(m / 480.0 for m in (3.0, 10.0, 37.5, 150.0, 480.0))

# Household edges carry no survey duration (co-residence is not a timed
# contact), so they are treated as the longest bin -- full-strength
# transmission. Kept as one named constant so scaling households down too is a
# one-line change.
HOUSEHOLD_WEIGHT = 1.0


def _edge_weight(edge) -> float:
    """Weight of an edge tuple; ``(u, v, layer)`` triples default to 1.0."""
    return float(edge[3]) if len(edge) > 3 else 1.0


@dataclass
class SEIRParams:
    """Rate parameters, in units of 1/day.

    beta  : per-edge S->E1 transmission rate (variable).
    sigma : rate of E1->E2, E2->E3, E3->I (mean stay time 1/sigma per stage;
            default 1.0 for a mean stay time of 1 day per exposed class).
    gamma : rate of I->R (mean infectious period 1/gamma; default 0.25 for a
            mean infectious period of 4 days).
    """

    beta: float
    sigma: float = 1.0
    gamma: float = 0.25

    @property
    def mean_incubation(self) -> float:
        return 3.0 / self.sigma

    @property
    def mean_infectious_period(self) -> float:
        return 1.0 / self.gamma


@dataclass
class Transmission:
    time: float
    source: int
    target: int
    layer: str
    source_generation: int
    target_generation: int


@dataclass
class SEIRTrajectory:
    """Event-driven trajectory of an SEIR run.

    ``times`` and each array in ``counts`` are aligned and record the state
    of the system immediately after every event (plus the initial condition
    and each day boundary). ``transmissions`` is the full transmission log,
    one entry per infection event, in time order — this is the raw material
    for transmission-tree analyses (secondary case counts, generations,
    per-layer infection counts, R0 estimates).
    """

    times: np.ndarray
    counts: dict[str, np.ndarray]
    day_boundaries: np.ndarray
    transmissions: list[Transmission]
    seed_nodes: list[int]

    def total_exposed(self) -> np.ndarray:
        return self.counts["E1"] + self.counts["E2"] + self.counts["E3"]

    def final_size(self) -> int:
        return int(self.counts["R"][-1])

    def peak_infectious(self) -> tuple[float, int]:
        idx = int(np.argmax(self.counts["I"]))
        return float(self.times[idx]), int(self.counts["I"][idx])

    def daily_snapshot(self, n_days: int) -> dict[str, np.ndarray]:
        """State counts at integer day marks ``0..n_days`` (step-function value)."""
        marks = np.arange(n_days + 1, dtype=float)
        idx = np.searchsorted(self.times, marks, side="right") - 1
        idx = np.clip(idx, 0, len(self.times) - 1)
        out = {name: arr[idx] for name, arr in self.counts.items()}
        out["E"] = out["E1"] + out["E2"] + out["E3"]
        return out


# ---------------------------------------------------------------------------
# Mutable book-keeping for one run
# ---------------------------------------------------------------------------

class _NetworkState:
    """Per-node state, per-state membership lists, and the live S-I edge set.

    All are maintained incrementally in O(degree) per state change so that a
    Gillespie step never has to rescan the whole population. Edges carry a
    layer label and a transmissibility weight, and a node pair may have
    parallel edges from different layers, each tracked as an independent SI
    process. Live S-I edges are held in per-weight *buckets* so the total
    transmission rate is ``beta * total_weight`` and the specific edge that
    fires is drawn in proportion to its weight in ``O(#distinct weights)``.
    """

    def __init__(self, n_nodes: int, initial_state: np.ndarray):
        self.n_nodes = n_nodes
        self.state = initial_state.copy()

        self.members: dict[int, list[int]] = {s: [] for s in range(6)}
        self.pos: dict[int, int] = {}  # node -> index within members[state[node]]
        for node in range(n_nodes):
            self._add_member(node, int(self.state[node]))

        # adjacency[u] = list of (neighbour, layer, weight)
        self.adjacency: list[list[tuple[int, str, float]]] = [[] for _ in range(n_nodes)]
        # live S-I edges bucketed by weight: weight -> list of (s_node, i_node, layer)
        self.si_buckets: dict[float, list[tuple[int, int, str]]] = {}
        # (s, i, layer) key -> (weight, index within that weight's bucket)
        self.si_pos: dict[tuple[int, int, str], tuple[float, int]] = {}

    @property
    def total_weight(self) -> float:
        """Sum of the weights of all live S-I edges (== the beta-free transmission
        rate). Computed fresh from the (few) buckets each read so it never drifts
        from repeated float add/subtract of tiny per-edge weights."""
        return sum(w * len(bucket) for w, bucket in self.si_buckets.items())

    @property
    def si_edges(self) -> list[tuple[int, int, str]]:
        """Flat list of all live S-I edges (order not stable across changes).

        Backwards-compatible accessor; the simulator itself uses the weighted
        buckets and ``total_weight`` directly.
        """
        return [e for bucket in self.si_buckets.values() for e in bucket]

    # -- per-state membership (for O(1) uniform random selection) ----------

    def _add_member(self, node: int, st: int) -> None:
        lst = self.members[st]
        self.pos[node] = len(lst)
        lst.append(node)

    def _remove_member(self, node: int, st: int) -> None:
        lst = self.members[st]
        idx = self.pos[node]
        last = lst[-1]
        lst[idx] = last
        self.pos[last] = idx
        lst.pop()

    def count(self, st: int) -> int:
        return len(self.members[st])

    # -- adjacency / SI-edge bookkeeping ------------------------------------

    def set_adjacency(self, edges: Iterable[Edge]) -> None:
        """Replace the active contact network and rebuild the S-I edge set."""
        adj: list[list[tuple[int, str, float]]] = [[] for _ in range(self.n_nodes)]
        for edge in edges:
            u, v, layer = edge[0], edge[1], edge[2]
            weight = _edge_weight(edge)
            adj[u].append((v, layer, weight))
            adj[v].append((u, layer, weight))
        self.adjacency = adj
        self._rebuild_si_edges()

    def _rebuild_si_edges(self) -> None:
        self.si_buckets = {}
        self.si_pos = {}
        state = self.state
        for u in self.members[S]:
            for v, layer, weight in self.adjacency[u]:
                if state[v] == I:
                    self._add_si_edge(u, v, layer, weight)

    @staticmethod
    def _key(u: int, v: int, layer: str) -> tuple[int, int, str]:
        return (u, v, layer) if u < v else (v, u, layer)

    def _add_si_edge(self, s_node: int, i_node: int, layer: str, weight: float) -> None:
        key = self._key(s_node, i_node, layer)
        if key in self.si_pos:
            return
        bucket = self.si_buckets.get(weight)
        if bucket is None:
            bucket = self.si_buckets[weight] = []
        self.si_pos[key] = (weight, len(bucket))
        bucket.append((s_node, i_node, layer))

    def _remove_si_edge(self, u: int, v: int, layer: str) -> None:
        key = self._key(u, v, layer)
        entry = self.si_pos.pop(key, None)
        if entry is None:
            return
        weight, idx = entry
        bucket = self.si_buckets[weight]
        last = bucket.pop()            # swap-remove within this weight's bucket
        if idx < len(bucket):          # removed edge was not already the last
            bucket[idx] = last
            self.si_pos[self._key(*last)] = (weight, idx)

    def sample_si_edge(self, rng: np.random.Generator) -> tuple[int, int, str]:
        """Draw one live S-I edge with probability proportional to its weight.

        Picks a weight bucket in proportion to ``weight * bucket_size`` (the
        bucket's share of the total transmission rate), then a uniform edge
        within it. There are only a handful of distinct weights, so this is
        effectively O(1).
        """
        items = [(w, b) for w, b in self.si_buckets.items() if b]
        if len(items) == 1:
            bucket = items[0][1]
            return bucket[int(rng.integers(len(bucket)))]
        total = sum(w * len(b) for w, b in items)
        threshold = rng.random() * total
        acc = 0.0
        for weight, bucket in items:
            acc += weight * len(bucket)
            if threshold < acc:
                return bucket[int(rng.integers(len(bucket)))]
        bucket = items[-1][1]          # float round-off guard
        return bucket[int(rng.integers(len(bucket)))]

    # -- state transitions ---------------------------------------------------

    def set_state(self, node: int, new_state: int) -> None:
        old_state = int(self.state[node])
        if old_state == new_state:
            return
        self._remove_member(node, old_state)
        self.state[node] = new_state
        self._add_member(node, new_state)

        # Only S<->I boundary changes can create/destroy S-I edges.
        if old_state not in (S, I) and new_state not in (S, I):
            return
        for nbr, layer, weight in self.adjacency[node]:
            nbr_state = int(self.state[nbr])
            was_si = (old_state == S and nbr_state == I) or (old_state == I and nbr_state == S)
            is_si = (new_state == S and nbr_state == I) or (new_state == I and nbr_state == S)
            if was_si and not is_si:
                self._remove_si_edge(node, nbr, layer)
            elif is_si and not was_si:
                s_node, i_node = (node, nbr) if new_state == S else (nbr, node)
                self._add_si_edge(s_node, i_node, layer, weight)


def _initial_state(n_nodes: int, initial_infected: Sequence[int]) -> np.ndarray:
    state = np.zeros(n_nodes, dtype=np.int8)
    for nid in initial_infected:
        state[nid] = I
    return state


def _fire_event(
    ns: _NetworkState,
    r_e1: float,
    r_e2: float,
    r_e3: float,
    r_recov: float,
    r_trans: float,
    rng: np.random.Generator,
    t: float,
    generation: dict[int, int],
    transmissions: list[Transmission],
) -> None:
    weights = np.array([r_e1, r_e2, r_e3, r_recov, r_trans])
    choice = rng.choice(5, p=weights / weights.sum())

    if choice == 0:
        members = ns.members[E1]
        node = members[int(rng.integers(len(members)))]
        ns.set_state(node, E2)
    elif choice == 1:
        members = ns.members[E2]
        node = members[int(rng.integers(len(members)))]
        ns.set_state(node, E3)
    elif choice == 2:
        members = ns.members[E3]
        node = members[int(rng.integers(len(members)))]
        ns.set_state(node, I)
    elif choice == 3:
        members = ns.members[I]
        node = members[int(rng.integers(len(members)))]
        ns.set_state(node, R)
    else:
        s_node, i_node, layer = ns.sample_si_edge(rng)
        src_gen = generation[i_node]
        tgt_gen = src_gen + 1
        generation[s_node] = tgt_gen
        transmissions.append(Transmission(t, i_node, s_node, layer, src_gen, tgt_gen))
        ns.set_state(s_node, E1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def simulate_seir(
    n_nodes: int,
    edges_fn: Callable[[int], Sequence[Edge]],
    params: SEIRParams,
    initial_infected: Sequence[int],
    rng=None,
    day_length: float = 1.0,
    max_days: int | None = None,
) -> SEIRTrajectory:
    """Exact Gillespie SEIR simulation on a (possibly time-varying) network.

    Parameters
    ----------
    n_nodes :
        Number of individuals; node ids must be ``0..n_nodes-1``.
    edges_fn :
        Called once per simulated day as ``edges_fn(day_index)``, returning
        the full contact edge list ``[(u, v, layer), ...]`` active that day
        (household + community layers already combined, each tagged with a
        layer name). For a static network, ignore ``day_index`` and always
        return the same edges. Must be able to supply edges for as many days
        as the epidemic actually takes to burn out -- there is no day limit.
    params :
        SEIRParams(beta, sigma, gamma).
    initial_infected :
        Node ids seeded in the infectious (I) compartment at t=0 (generation 0).
    rng :
        Seed or ``numpy.random.Generator``.
    day_length :
        Time units per simulated day (default 1.0, matching ``sigma``/
        ``gamma`` being specified per day).
    max_days :
        Optional hard cap on the number of simulated days. ``None`` (default)
        means run to burn-out with no limit, as in production. A finite cap is
        a calibration aid: R0 (the index cases' direct offspring) is complete
        within the seeds' infectious lifetime, so a short capped run estimates
        it far more cheaply than a full outbreak. Do not use a cap for
        production runs -- it truncates the epidemic.
    """
    rng = np.random.default_rng(rng)
    ns = _NetworkState(n_nodes, _initial_state(n_nodes, initial_infected))

    generation: dict[int, int] = {nid: 0 for nid in initial_infected}
    transmissions: list[Transmission] = []

    times = [0.0]
    counts = {name: [ns.count(code)] for code, name in enumerate(STATE_NAMES)}
    day_boundaries = [0.0]

    def _record(t: float) -> None:
        times.append(t)
        for code, name in enumerate(STATE_NAMES):
            counts[name].append(ns.count(code))

    t = 0.0
    day = 0
    while True:
        ns.set_adjacency(edges_fn(day))
        day_end = t + day_length
        day_boundaries.append(day_end)

        while True:
            r_e1 = params.sigma * ns.count(E1)
            r_e2 = params.sigma * ns.count(E2)
            r_e3 = params.sigma * ns.count(E3)
            r_recov = params.gamma * ns.count(I)
            r_trans = params.beta * ns.total_weight
            total_rate = r_e1 + r_e2 + r_e3 + r_recov + r_trans

            if total_rate <= 0.0:
                break

            dt = rng.exponential(1.0 / total_rate)
            if t + dt >= day_end:
                break

            t += dt
            _fire_event(ns, r_e1, r_e2, r_e3, r_recov, r_trans, rng, t, generation, transmissions)
            _record(t)

        t = day_end
        _record(t)

        if ns.count(E1) + ns.count(E2) + ns.count(E3) + ns.count(I) == 0:
            break

        if max_days is not None and (day + 1) >= max_days:
            break

        day += 1

    return SEIRTrajectory(
        times=np.array(times),
        counts={k: np.array(v) for k, v in counts.items()},
        day_boundaries=np.array(day_boundaries),
        transmissions=transmissions,
        seed_nodes=list(initial_infected),
    )


# ---------------------------------------------------------------------------
# edges_fn helpers
# ---------------------------------------------------------------------------

def tag_edges(edges: Sequence[tuple[int, int]], layer: str, weight: float = 1.0) -> list[Edge]:
    """Attach a constant layer label (and weight) to a plain ``(u, v)`` edge list."""
    return [(u, v, layer, weight) for u, v in edges]


def static_edges_fn(edges: Sequence[Edge]) -> Callable[[int], Sequence[Edge]]:
    """Wrap a fixed, layer-tagged edge list so it is used on every day."""

    def _fn(_day: int) -> Sequence[Edge]:
        return edges

    return _fn


def temporal_edges_fn(
    tc,
    static_edges: Sequence[Edge] = (),
) -> Callable[[int], list[Edge]]:
    """Wrap a ``TemporalContacts`` instance as a layer-tagged ``edges_fn``.

    Each call advances ``tc`` by one simulated day (``tc.daily_contacts()``
    draws a fresh day from its own RNG state), and prepends any always-on
    edges such as household cliques (already tagged, e.g. via ``tag_edges``).

    Each community contact's duration bin is mapped to a transmissibility
    weight via ``DURATION_WEIGHTS``, so the returned edges are 4-tuples
    ``(u, v, layer, weight)``. ``static_edges`` are passed through unchanged,
    so tag them with the weight you want (e.g. ``HOUSEHOLD_WEIGHT``).
    """
    static_list = list(static_edges)

    def _fn(_day: int) -> list[Edge]:
        today = [
            (u, v, layer, DURATION_WEIGHTS[dur])
            for u, v, dur, layer in tc.daily_contacts()
        ]
        return static_list + today

    return _fn


# ---------------------------------------------------------------------------
# Replicates
# ---------------------------------------------------------------------------

def run_replicates(
    n_reps: int,
    make_edges_fn: Callable[[np.random.Generator], Callable[[int], Sequence[Edge]]],
    n_nodes: int,
    params: SEIRParams,
    n_seed: int,
    seed: int = 0,
) -> list[SEIRTrajectory]:
    """Run ``n_reps`` independent stochastic realisations.

    ``make_edges_fn(rng)`` must build and return a fresh ``edges_fn`` for
    each replicate (important for temporal networks, whose ``TemporalContacts``
    object carries its own RNG state and must not be shared across replicates).
    Initial infecteds and the Gillespie draws use a child RNG derived from
    ``seed`` + replicate index; ``make_edges_fn`` receives its own independent
    child RNG so network construction and epidemic dynamics don't share a
    stream. Each replicate runs until it dies out on its own -- there is no
    day limit.
    """
    trajectories = []
    for rep in range(n_reps):
        rng_net = np.random.default_rng((seed, rep, 0))
        rng_sim = np.random.default_rng((seed, rep, 1))
        edges_fn = make_edges_fn(rng_net)
        initial_infected = rng_sim.choice(n_nodes, size=n_seed, replace=False).tolist()
        traj = simulate_seir(
            n_nodes=n_nodes,
            edges_fn=edges_fn,
            params=params,
            initial_infected=initial_infected,
            rng=rng_sim,
        )
        trajectories.append(traj)
    return trajectories


def stack_daily_snapshots(
    trajectories: Sequence[SEIRTrajectory],
    n_days: int,
) -> dict[str, np.ndarray]:
    """Align replicate trajectories onto a common daily grid.

    Returns ``{state_name: array of shape (n_reps, n_days + 1)}``.
    """
    per_state: dict[str, list[np.ndarray]] = {name: [] for name in STATE_NAMES + ["E"]}
    for traj in trajectories:
        snap = traj.daily_snapshot(n_days)
        for name in per_state:
            per_state[name].append(snap[name])
    return {name: np.stack(arrs) for name, arrs in per_state.items()}
