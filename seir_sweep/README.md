# SEIR parameter sweep: model x one_time persistence x population size x beta

Parallel driver for large SEIR runs over the factorial

```
{NB, log-GMM mean-match}  x  {ICC 0, 0.365, 0.95}  x  {n = 10k, 100k, 1M}  x  beta grid
```

built on the structural findings in `../community_structural_comparison.ipynb`.

## Why this grid

Both **model** arms match the survey marginal degree distribution, so any
epidemic difference between them isolates the **cross-(contact_age x duration)
correlation** the joint log-GMM keeps and the per-cell NB throws away — not a
marginal mismatch. (They do differ in one_time degree *skewness*: NB ~3.8,
log-GMM ~6.5, against a survey value of ~12 — recorded per realisation.)

The **ICC** axis is the one_time layer's day-to-day persistence, which the
notebook showed is badly mis-calibrated by default (~0.95 against the real
France 2015 2-day panel value of **0.365**):

| arm | mechanism | meaning |
|---|---|---|
| `0.0` | no rate sampler — stubs redrawn nightly | no individual persistence |
| `0.365` | `partial_resample` with calibrated `rho` | the **real** empirical value |
| `0.95` | `partial_resample`, `rho -> 1` | near-permanent-trait regime |

See **`icc_construction_explainer.ipynb`** for the construction-level
walkthrough of this axis: how each arm is built, why the default recipe pins the
ICC at `B/(B+m)`, how `rho` is derived, and why the two obvious alternatives to
`partial_resample` are rejected. It runs standalone from the cached fits (~1 min).

`partial_resample` is used because it is the only noise procedure that hits a
target ICC while leaving the daily degree distribution (mean, variance, skew)
unchanged. `rho` is calibrated per realisation from the *realised* rate
distribution via `rho^2 = target * (B + m) / B`. When a target exceeds the
persistent ceiling `B/(B+m)` (NB tops out near 0.946) `rho` clamps to 1 and the
achieved value is recorded in `icc_target_achievable` — targets are never
silently missed.

## Usage

```bash
python calibrate_beta.py --write-config     # (re)derive the beta grid from R0
python run_sweep.py --estimate              # cost projection, runs nothing
python run_sweep.py --sizes 10000 -j 8      # run the 10k slice
python run_sweep.py --collect               # merge shards into final CSVs
```

Re-running skips completed **(realisation, beta)** units (done-markers in
`outputs/_done/`), so an interrupted sweep resumes at the beta it was on.

Because progress is tracked per beta, **betas can be added to a finished grid**
without re-running or overwriting the finished ones — they land on the same
network realisations, so the design stays paired:

```bash
# interleave the threshold region of an already-complete grid
python run_sweep.py --sizes 10000 100000 \
    --betas 0.0707 0.1304 0.2508 0.4672 1.0781
python run_sweep.py --collect
```

Re-running with betas that are already done is a no-op, so overlapping sets are
safe. `--force` re-runs the requested cells and betas — scoped to what you
asked for, never the whole grid.

## Feasibility

Measured on this machine (8 cores, ~4 GB RAM free, 27 GB disk):

| size | status |
|---|---|
| 10k | fine — minutes per realisation |
| 100k | heavy — the simulator rebuilds the whole daily adjacency in Python, so a supercritical run is tens of minutes |
| 1M | **not viable here** — needs several GB of adjacency per worker and many hours per supercritical run. Run that slice on a bigger machine. |

Use `--no-transmission-log` for the largest cells if disk is tight; the logs
are the single biggest output.

## Outputs (`outputs/`)

| file | grain | contents |
|---|---|---|
| `runs_summary.csv` | one row per (model, ICC, n, net_rep, beta, sim_rep) | every metric below |
| `network_summary.csv` | one row per network realisation | structural covariates |
| `transmission_logs/*.csv.gz` | one row per infection event | `time, source, target, layer, source_generation, target_generation` (the original `transmission_log.csv` schema) |
| `daily_series/*.csv.gz` | one row per day | `S, E1..E3, E, I, R` + per-layer daily incidence |
| `beta_r0_calibration.csv` | one row per beta | measured R0 |
| `manifest.json` | — | full run configuration |
| `fits_cache.pkl` | — | the two models' fitted parameters |

### Metrics in `runs_summary.csv`

**A. Threshold / takeoff** — `final_size`, `infections_beyond_seeds`,
`attack_rate`, `took_off`, `takeoff_fraction_used`. Takeoff *probability* is the
mean of `took_off` within a cell; raw per-replicate rows are kept so the
threshold can be changed at analysis time.

**B. Superspreading** — `R_mean_realised`, `R_var_realised`,
`offspring_dispersion_k` (Lloyd-Smith NB MLE, small k = superspreading),
`offspring_max`, `frac_zero_offspring`, `top10pct_transmission_share`,
`top20pct_transmission_share`, `n_infectors_considered`. Plus `r0_measured`
(seed-only offspring mean).

**C. Attribution** — `frac_trans_{household, everyday, few_times_week,
one_time}`, `attack_rate_age0..8`, `pct_households_infected`,
`household_attack_rate`.

**D. Dynamics** — `growth_rate_r`, `doubling_time_days`, `growth_fit_days`,
`gen_interval_{mean,sd,median}`, `n_gen_intervals`, `peak_infectious`,
`peak_prevalence`, `peak_time`, `duration_days`, `n_events`.

**E. Structure** (in `network_summary.csv`) — per-layer degree
`mean/var/skew/p99/max` for `everyday`, `few_times_week`, `one_time`,
`household`, `community_total`; `one_time_icc_realised` (2-day estimator,
directly comparable to the France 0.365); `rho`, `icc_persistent`,
`icc_target_achievable`, `noise_kind`; edge and household counts.

**F. Reproducibility** — `network_seed`, `sim_seed`, `seed_nodes`, `max_days`,
`sim_seconds`, `network_build_seconds`, plus `manifest.json`. Seeds derive
deterministically from grid coordinates, so any single replicate can be
reproduced in isolation.

## Files

- `config.py` — grid, beta values, replicate budget, fixed epidemiology
- `builders.py` — fits, ICC calibration, network realisation
- `metrics.py` — all derived metrics
- `run_sweep.py` — parallel driver / CLI
- `calibrate_beta.py` — empirical R0(beta) scan
- `icc_construction_explainer.ipynb` — how the ICC axis is constructed (explainer)
- `sweep_analysis.ipynb` — analysis of the sweep outputs
