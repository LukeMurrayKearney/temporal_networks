"""Time-varying community contacts for outbreak simulation.

Three contact layers with distinct temporal dynamics:

everyday        Static edges built once; present every simulated day.
few_times_week  Each stub gets a pre-sampled panel of K candidate contacts.
                Each day a per-stub Bernoulli draw (p ~ Uniform(1/7, 3/7))
                decides whether the contact activates; if so a panel member
                is drawn uniformly.  The same pair can meet on multiple days.
one_time        Stubs are resampled fresh every day from the fitted NB
                parameters.  Different contacts occur each day while
                preserving the age-duration structure from the survey.
"""

from __future__ import annotations

import numpy as np

from network import Population, N_AGE

N_DUR = 5


# ---------------------------------------------------------------------------
# NB helpers  (importable so the build notebook can use the same functions)
# ---------------------------------------------------------------------------

def fit_nb_mom(counts: np.ndarray) -> tuple[float, float]:
    """Method-of-moments NB fit.  Returns (n, p) for numpy's negative_binomial."""
    mu = float(counts.mean())
    if mu < 1e-12:
        return (0.0, 0.5)
    var = float(counts.var())
    if var <= mu + 1e-12:
        n_par = 1e6
    else:
        n_par = mu ** 2 / (var - mu)
    p = n_par / (n_par + mu)
    return (float(n_par), float(p))


def sample_nb(
    n_par: float,
    p_par: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw samples from NB(n_par, p_par); falls back to Poisson for large n."""
    if n_par < 1e-12:
        return np.zeros(size, dtype=np.int64)
    if n_par > 1e4:
        mu = n_par * (1.0 - p_par) / p_par
        return rng.poisson(mu, size=size).astype(np.int64)
    return rng.negative_binomial(n_par, p_par, size=size).astype(np.int64)


# ---------------------------------------------------------------------------
# Pluggable raw-count samplers
#
# A ``raw_sampler(params, nodes_by_age, rng)`` draws, for every (source_age a,
# contact_age b, duration d) triplet, one per-node count array (aligned with
# ``nodes_by_age[a]``) — exactly what NB-specific "Step 1" used to do inline
# inside ``build_stubs_with_padding`` and inside ``TemporalContacts``.
# Everything downstream (ratio-padding, stub-list construction, ftw panels,
# daily one_time resampling) only ever consumes this dict, so it is entirely
# distribution-agnostic: swapping in a log-normal, gamma, or direct
# ego-resampling sampler (see ``contact_distributions.py`` / ``ego_resample.py``)
# requires no changes here. ``params`` is opaque to everything but the
# sampler itself — for the parametric families it is a
# ``(n_age, n_age, n_dur, 2)`` array; for ego-resampling it can be any
# per-age pool structure.
# ---------------------------------------------------------------------------

def nb_raw_sampler(
    params_f: np.ndarray,
    nodes_by_age: dict[int, list[int]],
    rng: np.random.Generator,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Default raw_sampler: independent NB(n, p) draw per (a, b, d) cell."""
    n_age, n_contact_age, n_dur, _ = params_f.shape
    raw: dict[tuple[int, int, int], np.ndarray] = {}
    for a in range(n_age):
        n_a = len(nodes_by_age.get(a, []))
        if n_a == 0:
            continue
        for b in range(n_contact_age):
            for d in range(n_dur):
                n_par = params_f[a, b, d, 0]
                p_par = params_f[a, b, d, 1]
                raw[(a, b, d)] = (
                    np.zeros(n_a, dtype=np.int64) if n_par < 1e-12
                    else sample_nb(n_par, p_par, n_a, rng)
                )
    return raw


# ---------------------------------------------------------------------------
# Rate samplers: a per-node *persistent* Poisson rate for the one_time layer
#
# The default one_time behaviour redraws every node's contact counts from
# scratch each simulated day, so a node's day-to-day one_time degree is
# independent noise. A rate_sampler instead draws each node a persistent
# per-cell rate ONCE (a node characteristic); the daily one_time degree is
# then Poisson(rate). A node with a high drawn rate is consistently more
# connected day after day -- the one_time layer becomes correlated with the
# individual rather than reshuffled nightly.
#
# A rate_sampler has the same ``(params, nodes_by_age, rng) -> {(a,b,d): arr}``
# signature as a raw_sampler, but each ``arr`` is a float Poisson *rate* per
# node (aligned with ``nodes_by_age[a]``), not an integer count.
# ---------------------------------------------------------------------------

def nb_rate_sampler(
    params_f: np.ndarray,
    nodes_by_age: dict[int, list[int]],
    rng: np.random.Generator,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Latent NB rate per node: NB(n, p) == Poisson(lambda), lambda ~ Gamma(n, (1-p)/p).

    Drawing ``lambda`` once per node (here) and ``Poisson(lambda)`` each day
    reproduces the NB marginal on any single day while making the daily counts
    for a node fluctuate around its own persistent rate -- exactly the
    Poisson-Gamma structure of the negative binomial, split across time.
    """
    n_age, n_contact_age, n_dur, _ = params_f.shape
    rates: dict[tuple[int, int, int], np.ndarray] = {}
    for a in range(n_age):
        n_a = len(nodes_by_age.get(a, []))
        if n_a == 0:
            continue
        for b in range(n_contact_age):
            for d in range(n_dur):
                n_par = params_f[a, b, d, 0]
                p_par = params_f[a, b, d, 1]
                if n_par < 1e-12 or p_par <= 0.0:
                    rates[(a, b, d)] = np.zeros(n_a, dtype=np.float64)
                else:
                    scale = (1.0 - p_par) / p_par
                    rates[(a, b, d)] = rng.gamma(n_par, scale, size=n_a)
    return rates


def make_count_rate_sampler(raw_sampler):
    """Turn any raw_sampler into a rate_sampler by drawing counts once.

    For non-NB families (gamma, log-normal, ego-resample, log-GMM) the
    "sample once, Poisson-resample daily" recipe is: draw one contact-count
    vector per node from the fitted distribution and use it (as a float) as
    that node's persistent Poisson rate. The daily one_time degree is then
    ``Poisson(count)`` -- fluctuating around the node's own drawn characteristic.
    """

    def _rate_sampler(
        params,
        nodes_by_age: dict[int, list[int]],
        rng: np.random.Generator,
    ) -> dict[tuple[int, int, int], np.ndarray]:
        raw = raw_sampler(params, nodes_by_age, rng)
        return {key: counts.astype(np.float64) for key, counts in raw.items()}

    return _rate_sampler


def make_parametric_raw_sampler(sample_fn):
    """Build a raw_sampler for any "independent-per-cell, 2 parameters" family.

    ``sample_fn(param0, param1, size, rng) -> np.ndarray[int]`` draws ``size``
    counts for one (a, b, d) cell.  By convention ``param0 < 1e-12`` marks a
    degenerate/zero-mean cell (mirroring ``fit_nb_mom``'s ``n_par`` sentinel)
    and short-circuits to zeros without calling ``sample_fn``.
    """

    def _raw_sampler(
        params_f: np.ndarray,
        nodes_by_age: dict[int, list[int]],
        rng: np.random.Generator,
    ) -> dict[tuple[int, int, int], np.ndarray]:
        n_age, n_contact_age, n_dur, _ = params_f.shape
        raw: dict[tuple[int, int, int], np.ndarray] = {}
        for a in range(n_age):
            n_a = len(nodes_by_age.get(a, []))
            if n_a == 0:
                continue
            for b in range(n_contact_age):
                for d in range(n_dur):
                    p0 = params_f[a, b, d, 0]
                    p1 = params_f[a, b, d, 1]
                    raw[(a, b, d)] = (
                        np.zeros(n_a, dtype=np.int64) if p0 < 1e-12
                        else sample_fn(p0, p1, n_a, rng)
                    )
        return raw

    return _raw_sampler


def _finalize_stubs(
    raw: dict[tuple[int, int, int], np.ndarray],
    nodes_by_age: dict[int, list[int]],
    n_age: int,
    n_dur: int,
    rng: np.random.Generator,
) -> dict[tuple[int, int, int], list[int]]:
    """Ratio-pad asymmetric pools, then expand counts into node-id stub lists.

    For each (source_age a, contact_age b, duration d) triplet, computes
    the total stubs flowing a→(b,d) and b→(a,d).  If the pools differ,
    the smaller pool is scaled up by the ratio r = N_larger / N_smaller.
    Each node's raw count is multiplied by r then **stochastically rounded**:
    a fractional part f gives floor(x)+1 with probability f, floor(x) with
    probability 1−f.  Zero counts are never inflated.
    """
    # ---- Step 1: balance pools for each unordered off-diagonal (a,b,d) ---
    padded: dict[tuple[int, int, int], np.ndarray] = {k: v.copy() for k, v in raw.items()}

    for a in range(n_age):
        for b in range(a + 1, n_age):
            for d in range(n_dur):
                c_ab = raw.get((a, b, d), np.zeros(0, dtype=np.int64))
                c_ba = raw.get((b, a, d), np.zeros(0, dtype=np.int64))
                N_ab = int(c_ab.sum())
                N_ba = int(c_ba.sum())

                if N_ab == 0 or N_ba == 0:
                    continue

                if N_ab < N_ba:
                    r = N_ba / N_ab
                    scaled = c_ab.astype(np.float64) * r
                    floored = np.floor(scaled).astype(np.int64)
                    frac = np.clip(scaled - floored, 0.0, 1.0)
                    padded[(a, b, d)] = floored + rng.binomial(1, frac)
                elif N_ba < N_ab:
                    r = N_ab / N_ba
                    scaled = c_ba.astype(np.float64) * r
                    floored = np.floor(scaled).astype(np.int64)
                    frac = np.clip(scaled - floored, 0.0, 1.0)
                    padded[(b, a, d)] = floored + rng.binomial(1, frac)

    # ---- Step 2: build stub lists from padded counts ----------------------
    stubs: dict[tuple[int, int, int], list[int]] = {}
    for a in range(n_age):
        node_ids_a = nodes_by_age.get(a, [])
        for b in range(n_age):
            for d in range(n_dur):
                counts = padded.get((a, b, d))
                if counts is None or counts.size == 0:
                    continue
                nonzero = np.nonzero(counts)[0]
                if nonzero.size == 0:
                    continue
                stub_list = stubs.setdefault((a, b, d), [])
                for idx in nonzero:
                    stub_list.extend([node_ids_a[idx]] * int(counts[idx]))

    return stubs


# ---------------------------------------------------------------------------
# Approach B: ratio-padded stub generation
# ---------------------------------------------------------------------------

def build_stubs_with_padding(
    params_f: np.ndarray,
    nodes_by_age: dict[int, list[int]],
    rng: np.random.Generator,
    raw_sampler=nb_raw_sampler,
) -> dict[tuple[int, int, int], list[int]]:
    """Sample community stubs and balance asymmetric pools by ratio scaling.

    Parameters
    ----------
    params_f :
        Parameter array of shape (N_AGE, N_AGE, N_DUR, 2) for one frequency
        layer, in whatever form ``raw_sampler`` expects (NB (n, p) by
        default; log-normal (mu, sigma), gamma (shape, scale), or an
        ego-pool structure for other samplers).
    nodes_by_age :
        Mapping age-group index → list of node IDs in that group.
    rng :
        numpy random Generator.
    raw_sampler :
        ``(params_f, nodes_by_age, rng) -> {(a, b, d): counts}``, called
        once to draw per-node counts for every (source_age, contact_age,
        duration) triplet. Defaults to independent NB draws; see
        ``contact_distributions.py`` and ``ego_resample.py`` for alternatives.

    Returns
    -------
    stubs :
        ``(src_age, contact_age, duration)`` → list of node IDs (with
        repetition for count > 1).
    """
    n_age = len(nodes_by_age)
    raw = raw_sampler(params_f, nodes_by_age, rng)
    return _finalize_stubs(raw, nodes_by_age, n_age, N_DUR, rng)


# ---------------------------------------------------------------------------
# Stub matching with duration attribute
# ---------------------------------------------------------------------------

def _match_between_dur(
    stubs_a: list[int],
    stubs_b: list[int],
    duration: int,
    rng: np.random.Generator,
    existing: set[tuple[int, int]],
    hh_by_node: dict[int, int],
) -> list[tuple[int, int, int]]:
    """Pair stubs_a (age a) against stubs_b (age b), returning (u, v, duration)."""
    arr_a = np.array(stubs_a, dtype=np.int64)
    arr_b = np.array(stubs_b, dtype=np.int64)
    rng.shuffle(arr_a)
    rng.shuffle(arr_b)
    n = min(len(arr_a), len(arr_b))
    arr_a = arr_a[:n]
    arr_b = arr_b[:n].copy()
    edges: list[tuple[int, int, int]] = []
    for i in range(n):
        u = int(arr_a[i])
        for j in range(i, n):
            v = int(arr_b[j])
            if u == v:
                continue
            if hh_by_node.get(u) == hh_by_node.get(v):
                continue
            key = (min(u, v), max(u, v))
            if key in existing:
                continue
            arr_b[i], arr_b[j] = arr_b[j], arr_b[i]
            existing.add(key)
            edges.append((u, v, duration))
            break
    return edges


def _match_within_dur(
    stubs: list[int],
    duration: int,
    rng: np.random.Generator,
    existing: set[tuple[int, int]],
    hh_by_node: dict[int, int],
) -> list[tuple[int, int, int]]:
    """Pair stubs within one pool, returning (u, v, duration)."""
    arr = np.array(stubs, dtype=np.int64)
    rng.shuffle(arr)
    n = len(arr)
    used = np.zeros(n, dtype=bool)
    edges: list[tuple[int, int, int]] = []
    for i in range(n):
        if used[i]:
            continue
        u = int(arr[i])
        for j in range(i + 1, n):
            if used[j]:
                continue
            v = int(arr[j])
            if u == v:
                continue
            if hh_by_node.get(u) == hh_by_node.get(v):
                continue
            key = (min(u, v), max(u, v))
            if key in existing:
                continue
            used[i] = used[j] = True
            existing.add(key)
            edges.append((u, v, duration))
            break
    return edges


# ---------------------------------------------------------------------------
# TemporalContacts
# ---------------------------------------------------------------------------

class TemporalContacts:
    """Time-varying community contacts for outbreak simulation.

    Usage
    -----
    Build once from a Population and fitted NB parameters, then call
    ``daily_contacts()`` or ``simulate(n_days)`` to generate daily edge lists.

    >>> tc = TemporalContacts.build(pop, nb_params, everyday_edges, rng=42)
    >>> contacts_day0 = tc.daily_contacts()  # list of (u, v, duration, layer)
    >>> all_days = tc.simulate(14)
    """

    def __init__(
        self,
        pop: Population,
        nb_params: list[np.ndarray],
        everyday_edges: list[tuple[int, int, int]],
        ftw_panels: dict[int, list[tuple]],
        nodes_by_age: dict[int, list[int]],
        hh_by_node: dict[int, int],
        rng: np.random.Generator,
        raw_sampler=nb_raw_sampler,
        one_time_rates: dict[tuple[int, int, int], np.ndarray] | None = None,
        one_time_noise: dict | None = None,
    ):
        self.pop = pop
        self.nb_params = nb_params
        self.everyday_edges = everyday_edges
        # node_id → list of (contact_age, duration, candidates, p_active)
        self.ftw_panels = ftw_panels
        self.nodes_by_age = nodes_by_age
        self.hh_by_node = hh_by_node
        self.rng = rng
        # (params, nodes_by_age, rng) -> {(a,b,d): counts}; called for the
        # few_times_week panel build (once) and the one_time layer (daily).
        # See ``nb_raw_sampler`` / ``contact_distributions.py`` / ``ego_resample.py``.
        self.raw_sampler = raw_sampler
        # {(a,b,d): per-node persistent Poisson rate}, aligned with
        # nodes_by_age[a]. When set (via a rate_sampler passed to ``build``),
        # the one_time layer is Poisson(rate) around each node's own drawn
        # rate every day instead of a fresh distribution redraw -- so a node's
        # one_time connectivity is correlated with the individual across days.
        # When None, the legacy fresh-redraw-every-day behaviour is used.
        self.one_time_rates = one_time_rates
        # How the persistent per-node rate is turned into a *daily* count -- the
        # "noise procedure". A dict ``{"kind": ...}`` (see _one_time_daily_counts).
        # Default plain ``Poisson(rate)`` reuses the same rate every day and so
        # is maximally day-to-day correlated (ICC ~0.85 vs the real French panel's
        # 0.365); other kinds inject within-individual variation to lower that.
        self.one_time_noise = one_time_noise or {"kind": "persistent"}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        pop: Population,
        nb_params: list[np.ndarray],
        everyday_edges: list[tuple[int, int, int]],
        rng=None,
        nodes_by_age: dict[int, list[int]] | None = None,
        raw_sampler=nb_raw_sampler,
        rate_sampler=None,
        one_time_noise: dict | None = None,
    ) -> "TemporalContacts":
        """Construct a TemporalContacts object from a built Population.

        Parameters
        ----------
        pop :
            Synthetic population (households already assembled).
        nb_params :
            Three per-layer parameter arrays/structures
            ``[everyday, few_times_week, one_time]``, in whatever form
            ``raw_sampler`` expects (NB ``(n_age, n_age, n_dur, 2)`` [n, p]
            by default). ``n_age``/``n_dur`` need not match
            ``network.N_AGE``/the survey's duration count — see ``nodes_by_age``.
        everyday_edges :
            Pre-matched static ``(u, v, duration)`` tuples for the everyday
            layer.  These are fixed and appear on every simulated day.
        rng :
            Seed or ``numpy.random.Generator``.
        nodes_by_age :
            Optional override mapping community-contact age-group index →
            node ids.  Defaults to grouping by each node's own ``age``
            (``network.N_AGE`` groups) — pass this explicitly when ``nb_params``
            was fit on a coarser age grouping than the population's household
            age bins (e.g. a handful of broad bands instead of all N_AGE), so
            stub sampling buckets nodes consistently with how the NB
            parameters were fit.
        raw_sampler :
            ``(params, nodes_by_age, rng) -> {(a, b, d): counts}``, used for
            both the few_times_week panel build (once, here) and the
            one_time layer (redrawn every simulated day, unless ``rate_sampler``
            makes it persistent). Defaults to independent NB draws; see
            ``contact_distributions.py`` and ``ego_resample.py`` for alternatives.
        rate_sampler :
            Optional ``(params, nodes_by_age, rng) -> {(a, b, d): rate}`` that
            draws each node a *persistent* per-cell Poisson rate for the
            one_time layer, ONCE, here. When provided, the one_time degree on
            each simulated day is ``Poisson(rate)`` around the node's own drawn
            rate -- so one_time connectivity is correlated with the individual
            across days -- instead of a fresh redraw every day. See
            ``nb_rate_sampler`` (NB Gamma latent) and ``make_count_rate_sampler``
            (sample-once for any other family). When ``None``, the legacy
            fresh-every-day behaviour is kept.
        one_time_noise :
            Optional ``{"kind": ...}`` selecting how each persistent one_time
            rate becomes a daily count (only used when ``rate_sampler`` is set).
            Controls day-to-day correlation without disturbing the single-day
            marginal beyond what each kind implies; see
            :meth:`_one_time_daily_counts`. Defaults to plain ``Poisson(rate)``.
        """
        rng = np.random.default_rng(rng)

        if nodes_by_age is None:
            nodes_by_age = {a: [] for a in range(N_AGE)}
            for nid, node in pop.nodes.items():
                nodes_by_age[node.age].append(nid)

        hh_by_node: dict[int, int] = {
            m.node_id: h.hh_id for h in pop.households for m in h.members
        }

        ftw_panels = cls._build_ftw_panels(
            nb_params[1], nodes_by_age, hh_by_node, rng, raw_sampler
        )

        # Draw each node's persistent one_time rate once (node characteristic),
        # if a rate_sampler was supplied. nb_params[2] is the one_time layer.
        one_time_rates = None
        if rate_sampler is not None:
            one_time_rates = rate_sampler(nb_params[2], nodes_by_age, rng)

        return cls(
            pop=pop,
            nb_params=nb_params,
            everyday_edges=everyday_edges,
            ftw_panels=ftw_panels,
            nodes_by_age=nodes_by_age,
            hh_by_node=hh_by_node,
            rng=rng,
            raw_sampler=raw_sampler,
            one_time_rates=one_time_rates,
            one_time_noise=one_time_noise,
        )

    @staticmethod
    def _build_ftw_panels(
        params_f: np.ndarray,
        nodes_by_age: dict[int, list[int]],
        hh_by_node: dict[int, int],
        rng: np.random.Generator,
        raw_sampler=nb_raw_sampler,
    ) -> dict[int, list[tuple]]:
        """Pre-sample contact panels for every few_times_week stub.

        For each node with a non-zero few_times_week stub count for some
        (contact_age b, duration d), a panel of candidate node IDs is drawn
        from ``nodes_by_age[b]`` (excluding the node's own household).
        Multiple stubs of the same type get non-overlapping candidate panels.

        Each panel has K candidates where K is 3 or 4 with equal probability.

        Each panel entry also stores ``p_active ~ Uniform(1/7, 3/7)``: the
        probability that this particular relationship is active on a given day,
        corresponding to 1–3 contacts per week.

        Returns
        -------
        dict : node_id → list of (contact_age, duration, candidates, p_active)
        """
        panels: dict[int, list[tuple]] = {}
        raw = raw_sampler(params_f, nodes_by_age, rng)

        for (a, b, d), counts in raw.items():
            node_ids_a = nodes_by_age.get(a, [])
            candidates_b = np.array(nodes_by_age.get(b, []), dtype=np.int64)
            if len(candidates_b) == 0:
                continue

            for idx in np.nonzero(counts)[0]:
                node_id = int(node_ids_a[idx])
                hh_id = hh_by_node.get(node_id, -1)
                count = int(counts[idx])

                # Draw panel size K ~ {3, 4} with equal probability
                # for each stub, then sample enough candidates without
                # replacement to fill all panels.
                k_vals = rng.choice([3, 4], size=count)
                k_total = int(k_vals.sum())
                sample_size = min(k_total + max(int(k_vals.max()), 15), len(candidates_b))
                sampled = rng.choice(candidates_b, size=sample_size, replace=False)
                eligible = [
                    int(c) for c in sampled
                    if int(c) != node_id
                    and hh_by_node.get(int(c), -2) != hh_id
                ]
                if not eligible:
                    continue

                if node_id not in panels:
                    panels[node_id] = []

                offset = 0
                for i in range(count):
                    k = int(k_vals[i])
                    panel = eligible[offset: offset + k]
                    offset += k
                    if not panel:
                        break
                    p_active = float(rng.uniform(1.0 / 7, 3.0 / 7))
                    panels[node_id].append((b, d, panel, p_active))

        return panels

    # ------------------------------------------------------------------
    # Daily contact generation
    # ------------------------------------------------------------------

    def daily_contacts(self) -> list[tuple[int, int, int, str]]:
        """Return all contacts for one simulated day.

        Returns
        -------
        list of ``(u, v, duration, layer)`` tuples.  Layer is one of
        ``"everyday"``, ``"few_times_week"``, ``"one_time"``.
        """
        contacts: list[tuple[int, int, int, str]] = []
        contacts += [(u, v, dur, "everyday") for u, v, dur in self.everyday_edges]
        contacts += self._sample_ftw_day()
        contacts += self._resample_one_time()
        return contacts

    def _sample_ftw_day(self) -> list[tuple[int, int, int, str]]:
        """Activate few_times_week panel relationships stochastically."""
        seen: set[tuple[int, int]] = set()
        edges: list[tuple[int, int, int, str]] = []
        for node_id, panel_list in self.ftw_panels.items():
            for _contact_age, duration, candidates, p_active in panel_list:
                if self.rng.random() >= p_active:
                    continue
                v = int(self.rng.choice(candidates))
                key = (min(node_id, v), max(node_id, v))
                if key not in seen:
                    seen.add(key)
                    edges.append((node_id, v, duration, "few_times_week"))
        return edges

    def _one_time_daily_counts(self) -> dict[tuple[int, int, int], np.ndarray]:
        """Turn the persistent per-node one_time rates into today's counts.

        The ``one_time_noise["kind"]`` selects the *noise procedure* -- how much
        of a node's daily one_time degree is a persistent trait vs day-to-day
        churn. All kinds keep the expected per-node count equal to the drawn
        rate (so the population mean is preserved); they differ in the day-to-day
        correlation they induce and, as a side effect, in the daily marginal's
        dispersion/skew. Randomness that represents "how active is this node
        today" is drawn **once per node** and shared across that node's
        (contact_age, duration) cells, so a node's cells stay coherent.

        Kinds (``self.one_time_noise``):

        * ``"persistent"`` (default) -- ``Poisson(rate)`` with the same rate every
          day. Maximal correlation: ICC = Var(rate)/(Var(rate)+E[rate]).
        * ``"blend"`` (``alpha``) -- shrink each rate toward its cell mean:
          ``Poisson(alpha*rate + (1-alpha)*mean_rate)``. Lowers the between-node
          variance to ``alpha^2 Var(rate)``, so it *reduces* the daily marginal's
          dispersion and skew. ``alpha=1`` recovers persistent.
        * ``"gamma_mult"`` (``theta``) -- multiply by a per-node/day Gamma(θ,1/θ)
          factor (mean 1): ``Poisson(rate*G)``. Injects within-node overdispersion
          and *increases* the marginal's variance/skew. θ→∞ recovers persistent.
        * ``"partial_resample"`` (``rho``) -- each day, each node keeps its own
          rate with probability ``rho``, else adopts another (uniformly random)
          node's rate. Leaves the daily marginal (and its skew) exactly unchanged
          while cutting the 2-day correlation to ``rho^2`` of the persistent ICC.
        """
        rates = self.one_time_rates
        rng = self.rng
        noise = self.one_time_noise
        kind = noise.get("kind", "persistent")
        n_age = len(self.nodes_by_age)

        if kind == "persistent":
            return {key: rng.poisson(rate) for key, rate in rates.items()}

        if kind == "blend":
            alpha = float(noise["alpha"])
            return {key: rng.poisson(alpha * rate + (1.0 - alpha) * rate.mean())
                    for key, rate in rates.items()}

        if kind == "gamma_mult":
            theta = float(noise["theta"])
            # one activity multiplier per node per day, shared across its cells
            g_by_age = {a: rng.gamma(theta, 1.0 / theta, size=len(self.nodes_by_age.get(a, [])))
                        for a in range(n_age)}
            return {(a, b, d): rng.poisson(rate * g_by_age[a])
                    for (a, b, d), rate in rates.items()}

        if kind == "partial_resample":
            rho = float(noise["rho"])
            # per-node keep decision + replacement index, shared across cells so a
            # replaced node adopts another node's *whole* rate vector (keeps the
            # cross-cell correlation of the fitted mixture intact)
            keep_by_age, perm_by_age = {}, {}
            for a in range(n_age):
                n_a = len(self.nodes_by_age.get(a, []))
                keep_by_age[a] = rng.random(n_a) < rho
                perm_by_age[a] = rng.integers(0, n_a, n_a) if n_a > 0 else np.zeros(0, dtype=int)
            out: dict[tuple[int, int, int], np.ndarray] = {}
            for (a, b, d), rate in rates.items():
                lam_today = np.where(keep_by_age[a], rate, rate[perm_by_age[a]])
                out[(a, b, d)] = rng.poisson(lam_today)
            return out

        raise ValueError(f"unknown one_time_noise kind: {kind!r}")

    def _resample_one_time(self) -> list[tuple[int, int, int, str]]:
        """Draw today's one_time contacts.

        If persistent per-node rates were drawn at build time
        (``self.one_time_rates``), today's per-node counts come from
        :meth:`_one_time_daily_counts` under the configured noise procedure
        (default ``Poisson(rate)`` around each node's own drawn rate -- so a
        node's one_time degree is correlated with the individual across days).
        Otherwise the legacy behaviour redraws counts from ``raw_sampler``
        afresh every day.
        """
        if self.one_time_rates is not None:
            raw = self._one_time_daily_counts()
        else:
            params_f = self.nb_params[2]
            raw = self.raw_sampler(params_f, self.nodes_by_age, self.rng)
        stubs: dict[tuple[int, int, int], list[int]] = {}

        for (a, b, d), counts in raw.items():
            nonzero = np.nonzero(counts)[0]
            if nonzero.size == 0:
                continue
            node_ids_a = self.nodes_by_age.get(a, [])
            stub_list = stubs.setdefault((a, b, d), [])
            for ix in nonzero:
                stub_list.extend([node_ids_a[ix]] * int(counts[ix]))

        # Fresh existing set each day: one_time contacts don't accumulate
        existing_today: set[tuple[int, int]] = set()
        raw_edges: list[tuple[int, int, int]] = []

        n_age = len(self.nodes_by_age)
        for a in range(n_age):
            for b in range(a, n_age):
                for d in range(N_DUR):
                    if a == b:
                        pool = stubs.get((a, a, d), [])
                        if len(pool) >= 2:
                            raw_edges.extend(
                                _match_within_dur(
                                    pool, d, self.rng, existing_today, self.hh_by_node
                                )
                            )
                    else:
                        pool_a = stubs.get((a, b, d), [])
                        pool_b = stubs.get((b, a, d), [])
                        if pool_a and pool_b:
                            raw_edges.extend(
                                _match_between_dur(
                                    pool_a, pool_b, d,
                                    self.rng, existing_today, self.hh_by_node,
                                )
                            )

        return [(u, v, dur, "one_time") for u, v, dur in raw_edges]

    # ------------------------------------------------------------------
    # Multi-day simulation
    # ------------------------------------------------------------------

    def simulate(self, n_days: int) -> list[list[tuple[int, int, int, str]]]:
        """Simulate ``n_days`` of contacts.

        Returns a list of length ``n_days``; each element is the contact list
        for that day (output of :meth:`daily_contacts`).
        """
        return [self.daily_contacts() for _ in range(n_days)]

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_ftw_panels(self) -> int:
        """Total number of pre-sampled few_times_week panel entries."""
        return sum(len(v) for v in self.ftw_panels.values())

    @property
    def n_everyday_edges(self) -> int:
        return len(self.everyday_edges)
