"""Joint log-Gaussian-Mixture-Model community-contact sampler.

A "more complex" contact model than the per-cell parametric fits
(``temporal.nb_raw_sampler`` / ``contact_distributions.py``) or the direct
ego-resampling (``ego_resample.py``). It follows the same recipe as the
Gaussian mixtures in ``network speed/popnet/fitting.py`` (``fit_log_gmm``):
fit a Gaussian mixture on ``log1p``-transformed contact vectors, per age
group, with a BIC-selected number of components, and back-transform samples
with ``expm1``.

The difference from popnet is the vector being modelled. popnet fits a joint
GMM over the 9 contact-age counts. Here each non-household layer's survey
record for a source-age group is the full **(contact_age x duration)** table,
so we fit a **joint GMM over that flattened 45-D vector** (9 contact ages x 5
durations). One draw therefore produces a whole node's contact vector across
every (contact_age, duration) cell at once, preserving the correlation the
real egos have across contact ages *and* durations -- the thing per-cell
independent fits throw away. Age-blind (``no_household_*``) layers have no
contact-age axis, so the vector is 5-D (durations only) in a single bucket.

Covariance is **diagonal** per component (the default). In 45-D with only a
few hundred to a couple thousand egos per age group, a *full* covariance costs
~1035 parameters per component, so the held-out BIC never supports more than a
single Gaussian -- however rigorous the search. A diagonal covariance costs
just 45 parameters per component, letting the BIC search actually grow the
mixture (typically 3-7 components on the dense one_time layer). Cross-cell
correlation is then represented by the **placement of the multiple components**
(each a distinct mean profile over the 45 cells) rather than within a single
component -- the standard way to model a correlated high-dimensional vector
when data is too thin for a full covariance.

Everything is exposed through the same ``raw_sampler(params, nodes_by_age,
rng) -> {(a, b, d): counts}`` contract used across ``temporal.py``, so the
log-GMM is a drop-in alongside the NB / gamma / log-normal / ego samplers.
``params`` here is the opaque per-layer structure returned by the ``fit_*``
functions below: a dict ``{"gmms": {age: GaussianMixture|None}, "n_contact_age",
"n_dur"}``.
"""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

N_AGE = 9
N_DUR = 5
N_FREQ = 3


# ---------------------------------------------------------------------------
# BIC-selected component count (compact form of popnet's _find_optimal_ng)
# ---------------------------------------------------------------------------

def _select_ng(
    X: np.ndarray,
    max_ng: int,
    n_splits: int,
    reg_covar: float,
    seed: int,
    covariance_type: str = "diag",
    test_frac: float = 0.2,
    patience: int = 5,
) -> int:
    """Choose the GMM component count by repeated held-out BIC (median rule).

    For ng = 1, 2, ...: fit on an 80% split, score ``gmm.bic`` on the 20% held
    out, and repeat over ``n_splits`` shuffles (default 30). The score for that
    ng is the **median** out-of-sample BIC over those shuffles -- robust to the
    occasional bad EM run, unlike the mean. The search stops once the median BIC
    has *risen* for ``patience`` consecutive ng values (default 5), and returns
    the ng with the **lowest median BIC seen so far** (the min over the explored
    range). Raising ``max_ng`` lets the mixture use as many components as the
    held-out BIC actually supports. Mirrors ``popnet.fitting._find_optimal_ng``
    but with a wider search and a median-of-many-splits criterion.
    """
    n = len(X)
    n_test = max(1, int(round(n * test_frac)))
    n_train = n - n_test
    if n_train < 2:
        return 1

    rng = np.random.default_rng(seed)
    bic_history: list[float] = []
    consecutive_increases = 0

    for ng in range(1, max_ng + 1):
        if ng > n_train:
            break
        trial_bics: list[float] = []
        for _ in range(n_splits):
            idx = rng.permutation(n)
            X_tr, X_te = X[idx[:n_train]], X[idx[n_train:]]
            try:
                gmm = GaussianMixture(
                    n_components=ng, covariance_type=covariance_type, reg_covar=reg_covar,
                    max_iter=200, n_init=1, random_state=int(rng.integers(0, 2**30)),
                )
                gmm.fit(X_tr)
                if gmm.converged_:
                    trial_bics.append(float(gmm.bic(X_te)))
            except Exception:
                pass
        if not trial_bics:
            break
        bic_history.append(float(np.median(trial_bics)))
        if len(bic_history) >= 2 and bic_history[-1] > bic_history[-2]:
            consecutive_increases += 1
            if consecutive_increases >= patience:
                break
        else:
            consecutive_increases = 0

    if not bic_history:
        return 1
    return int(np.argmin(bic_history)) + 1


def _fit_one_age(
    a: int,
    X: np.ndarray | None,
    max_ng: int,
    n_splits: int,
    reg_covar: float,
    covariance_type: str,
) -> tuple[int, "GaussianMixture | None", int]:
    """Select ng by held-out median BIC and fit the final GMM for one age group.

    Pure (no shared state) so it can run in a joblib worker. Returns
    ``(age, fitted_gmm_or_None, n_egos)``.
    """
    if X is None or len(X) < 2:
        return a, None, 0 if X is None else len(X)
    X_log = np.log1p(X.astype(np.float64))
    ng = _select_ng(X_log, min(max_ng, len(X_log) // 2 or 1), n_splits, reg_covar,
                    seed=a, covariance_type=covariance_type)
    gmm = GaussianMixture(
        n_components=max(ng, 1), covariance_type=covariance_type, reg_covar=reg_covar,
        max_iter=300, n_init=1, random_state=0,
    )
    gmm.fit(X_log)
    return a, gmm, len(X_log)


def _fit_gmm_by_age(
    X_by_age: dict[int, np.ndarray],
    n_contact_age: int,
    n_dur: int,
    max_ng: int,
    n_splits: int,
    reg_covar: float,
    verbose: bool,
    tag: str,
    covariance_type: str = "diag",
    n_jobs: int = -1,
) -> dict:
    """Fit one BIC-selected log-GMM per age group over log1p(counts) vectors.

    ``X_by_age[a]`` is the ``(n_egos_a, n_contact_age * n_dur)`` raw count
    matrix for source-age ``a`` (already flattened over contact_age x duration).
    Returns the opaque param dict consumed by ``loggmm_raw_sampler``.

    The rigorous BIC search (30 shuffles x up to ``max_ng`` components) is
    expensive, so age groups are fit in parallel across processes (``n_jobs``,
    default all cores). Each worker pins BLAS to a single thread; the tiny 45-D
    fits get no benefit from BLAS threading and only thrash when oversubscribed.
    """
    items = list(X_by_age.items())
    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_fit_one_age)(a, X, max_ng, n_splits, reg_covar, covariance_type)
            for a, X in items
        )
    except Exception:  # joblib missing / unavailable -> serial fallback
        results = [_fit_one_age(a, X, max_ng, n_splits, reg_covar, covariance_type)
                   for a, X in items]

    gmms: dict[int, GaussianMixture | None] = {}
    for a, gmm, n_egos in sorted(results):
        gmms[a] = gmm
        if verbose:
            ng = gmm.n_components if gmm is not None else 0
            print(f"    [{tag}] age {a}: n={n_egos:5d}  ng*={ng}", flush=True)
    return {"gmms": gmms, "n_contact_age": n_contact_age, "n_dur": n_dur,
            "covariance_type": covariance_type}


# ---------------------------------------------------------------------------
# Fitting entry points (mirror the fit_params_* shapes in the run scripts)
# ---------------------------------------------------------------------------

def fit_log_gmm_layered(
    all_contacts: np.ndarray,
    source_ages: np.ndarray,
    max_ng: int = 10,
    n_splits: int = 30,
    reg_covar: float = 1e-2,
    covariance_type: str = "diag",
    verbose: bool = False,
) -> list[dict]:
    """Age-structured, per-frequency-layer joint log-GMMs (household_* models).

    ``all_contacts``: (n_participants, N_AGE, N_DUR, N_FREQ). Returns one param
    dict per frequency layer ``[everyday, few_times_week, one_time]``, each a
    joint 45-D (contact_age x duration) log-GMM per source-age group.
    """
    params_list = []
    for f in range(N_FREQ):
        X_by_age = {}
        for a in range(N_AGE):
            mask_a = source_ages == a
            X_by_age[a] = all_contacts[mask_a, :, :, f].reshape(int(mask_a.sum()), N_AGE * N_DUR)
        params_list.append(
            _fit_gmm_by_age(X_by_age, N_AGE, N_DUR, max_ng, n_splits, reg_covar,
                            verbose, tag=f"layer{f}", covariance_type=covariance_type)
        )
    return params_list


def fit_log_gmm_summed(
    all_contacts: np.ndarray,
    source_ages: np.ndarray,
    max_ng: int = 10,
    n_splits: int = 30,
    reg_covar: float = 1e-2,
    covariance_type: str = "diag",
    verbose: bool = False,
) -> dict:
    """Age-structured joint log-GMM, summed over frequency (household_static)."""
    counts_summed = all_contacts.sum(axis=3)  # (n, N_AGE, N_DUR)
    X_by_age = {}
    for a in range(N_AGE):
        mask_a = source_ages == a
        X_by_age[a] = counts_summed[mask_a].reshape(int(mask_a.sum()), N_AGE * N_DUR)
    return _fit_gmm_by_age(X_by_age, N_AGE, N_DUR, max_ng, n_splits, reg_covar,
                           verbose, tag="summed", covariance_type=covariance_type)


def fit_log_gmm_reduced_layered(
    reduced_arr: np.ndarray,
    max_ng: int = 10,
    n_splits: int = 30,
    reg_covar: float = 1e-2,
    covariance_type: str = "diag",
    verbose: bool = False,
) -> list[dict]:
    """Age-blind, per-frequency-layer 5-D log-GMMs (no_household_temporal).

    ``reduced_arr``: (n_participants, N_DUR, N_FREQ) -- no contact-age axis.
    Everyone pooled into a single bucket (age 0), 5-D (durations only).
    """
    params_list = []
    for f in range(N_FREQ):
        X = reduced_arr[:, :, f]  # (n, N_DUR)
        params_list.append(
            _fit_gmm_by_age({0: X}, 1, N_DUR, max_ng, n_splits, reg_covar,
                            verbose, tag=f"reduced_layer{f}", covariance_type=covariance_type)
        )
    return params_list


def fit_log_gmm_reduced_summed(
    reduced_arr: np.ndarray,
    max_ng: int = 10,
    n_splits: int = 30,
    reg_covar: float = 1e-2,
    covariance_type: str = "diag",
    verbose: bool = False,
) -> dict:
    """Age-blind 5-D log-GMM, summed over frequency (no_household_static)."""
    counts_summed = reduced_arr.sum(axis=2)  # (n, N_DUR)
    return _fit_gmm_by_age({0: counts_summed}, 1, N_DUR, max_ng, n_splits, reg_covar,
                           verbose, tag="reduced_summed", covariance_type=covariance_type)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _component_cov(gmm: GaussianMixture, k: int, dim: int) -> np.ndarray:
    """Full covariance matrix of component k for any sklearn covariance_type."""
    ct = gmm.covariance_type
    if ct == "full":
        return gmm.covariances_[k]
    if ct == "tied":
        return gmm.covariances_
    if ct == "diag":
        return np.diag(gmm.covariances_[k])
    return np.eye(dim) * float(gmm.covariances_[k])  # spherical


def _gmm_sample_batch(gmm: GaussianMixture, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``size`` samples from a fitted GaussianMixture (any covariance type)."""
    dim = gmm.means_.shape[1]
    comps = rng.choice(len(gmm.weights_), size=size, p=gmm.weights_)
    out = np.empty((size, dim), dtype=np.float64)
    for k in np.unique(comps):
        sel = comps == k
        n_k = int(sel.sum())
        cov = _component_cov(gmm, int(k), dim)
        try:
            out[sel] = rng.multivariate_normal(gmm.means_[k], cov, size=n_k)
        except np.linalg.LinAlgError:
            out[sel] = rng.multivariate_normal(gmm.means_[k], cov + np.eye(dim) * 1e-4, size=n_k)
    return out


def loggmm_raw_sampler(
    params: dict,
    nodes_by_age: dict[int, list[int]],
    rng: np.random.Generator,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Draw one joint contact vector per node and slice it into (a, b, d) cells.

    Every (contact_age b, duration d) cell for a given node comes from the
    *same* joint GMM draw, so the resulting stub counts preserve the fitted
    cross-cell correlation, like the ego sampler but from a smooth generative
    model instead of the empirical rows.

    Discretisation of the back-transformed ``expm1`` draw is controlled by
    ``params.get("stochastic_round", False)``:

    * ``False`` (default) -- ``round``. Deterministic nearest-integer; every
      continuous draw below 0.5 collapses to 0, which on this sparse data
      systematically **loses** the ``1``-contact mass (a downward mean bias).
    * ``True`` -- **stochastic rounding**: a draw of ``x`` becomes
      ``floor(x)+1`` with probability ``x-floor(x)`` else ``floor(x)``, so the
      expected integer equals the continuous draw. This removes the round-down
      bias (``E[count] = expm1(sample)``) at the cost of the mild upward
      lognormal ``exp(sigma^2/2)`` bias of the back-transform itself.

    **Per-cell mean-matching** (``params.get("mean_match", False)`` with
    ``params["cell_scales"]``): each cell's continuous draw is multiplied by
    ``empirical_cell_mean / GMM_implied_cell_mean`` before rounding, pinning the
    first moment of every (contact_age, duration) cell to the survey while
    keeping the mixture's shape and cross-cell correlation. It corrects both the
    round-down clip *and* the ``exp(sigma^2/2)`` back-transform bias at once (see
    :func:`add_cell_mean_match_layered`). Because the scale acts on the
    continuous draw, it must be paired with stochastic rounding for the pinned
    mean to survive discretisation (``E[count] = E[scaled draw]``).
    """
    gmms = params["gmms"]
    n_contact_age = params["n_contact_age"]
    n_dur = params["n_dur"]
    stochastic = params.get("stochastic_round", False)
    mean_match = params.get("mean_match", False)
    cell_scales = params.get("cell_scales") if mean_match else None
    raw: dict[tuple[int, int, int], np.ndarray] = {}
    for a, node_ids in nodes_by_age.items():
        n_a = len(node_ids)
        if n_a == 0:
            continue
        gmm = gmms.get(a)
        if gmm is None:
            for b in range(n_contact_age):
                for d in range(n_dur):
                    raw[(a, b, d)] = np.zeros(n_a, dtype=np.int64)
            continue
        samples = _gmm_sample_batch(gmm, n_a, rng)  # (n_a, n_contact_age * n_dur)
        cont = np.maximum(0.0, np.expm1(samples))
        if cell_scales is not None and a in cell_scales:
            cont = cont * cell_scales[a]  # (n_a, n_cells) * (n_cells,), pins per-cell mean
        if stochastic:
            floor = np.floor(cont)
            counts = (floor + (rng.random(cont.shape) < (cont - floor))).astype(np.int64)
        else:
            counts = np.round(cont).astype(np.int64)
        counts = counts.reshape(n_a, n_contact_age, n_dur)
        for b in range(n_contact_age):
            for d in range(n_dur):
                raw[(a, b, d)] = counts[:, b, d]
    return raw


# ---------------------------------------------------------------------------
# Per-cell mean-matching (the clean undershoot fix)
# ---------------------------------------------------------------------------

def _cell_mean_scales(
    gmm: GaussianMixture,
    X: np.ndarray,
    rng: np.random.Generator,
    n_mc: int = 200_000,
) -> np.ndarray:
    """Per-cell multiplicative scale that pins the continuous draw's mean to data.

    Returns a length-``n_cells`` array ``s`` such that scaling this age group's
    back-transformed draws by ``s`` makes the expected count in every cell equal
    the empirical per-cell mean ``X.mean(0)``. The GMM-implied continuous mean is
    estimated by Monte Carlo through the *same* ``expm1`` / ``maximum(0, .)``
    pipeline the sampler uses (so the ``maximum(0, .)`` truncation is accounted
    for exactly, unlike the closed-form lognormal mean). Cells the GMM implies as
    ~0 get a scale of 1 (nothing to rescale).
    """
    target = X.astype(np.float64).mean(axis=0)                  # empirical per-cell mean
    implied = np.maximum(0.0, np.expm1(_gmm_sample_batch(gmm, n_mc, rng))).mean(axis=0)
    scale = np.ones_like(target)
    nz = implied > 1e-9
    scale[nz] = target[nz] / implied[nz]
    return scale


def add_cell_mean_match_layered(
    params_layered: list[dict],
    all_contacts: np.ndarray,
    source_ages: np.ndarray,
    seed: int = 0,
    n_mc: int = 200_000,
) -> list[dict]:
    """Attach per-cell mean-match scales to an already-fitted layered log-GMM.

    Takes the ``params_layered`` list from :func:`fit_log_gmm_layered` (one dict
    per frequency layer) and returns a new list with ``mean_match=True``,
    ``stochastic_round=True`` and ``cell_scales={age: array}`` added, so
    :func:`loggmm_raw_sampler` pins every (contact_age, duration) cell's mean
    count to the survey empirical mean. No refit is needed -- the scales are read
    off the fitted mixtures and the same survey vectors the fit consumed.
    """
    rng = np.random.default_rng(seed)
    out = []
    for f, params in enumerate(params_layered):
        scales: dict[int, np.ndarray] = {}
        for a, gmm in params["gmms"].items():
            if gmm is None:
                continue
            mask_a = source_ages == a
            if not mask_a.any():
                continue
            X = all_contacts[mask_a, :, :, f].reshape(int(mask_a.sum()), N_AGE * N_DUR)
            scales[a] = _cell_mean_scales(gmm, X, rng, n_mc=n_mc)
        out.append(dict(params, mean_match=True, stochastic_round=True, cell_scales=scales))
    return out
