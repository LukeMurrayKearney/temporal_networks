# SEIR structure sweep: temporality x households x model

Sibling of the main sweep (`README.md` / `run_sweep.py`). That one holds the
network *architecture* fixed — temporal, with households — and varies the
contact model and the one_time persistence. This one varies the architecture:

```
{temporal, static}  x  {households, none}  x  {NB, log-GMM mean-match}   at n = 100k
```

on the **same beta grid** as the main sweep, so every cell is comparable
beta-for-beta across both.

## The missing corner is deliberate

`temporal x households` **is** the main sweep, and its `n=100000` slice is
already complete (10 net_reps x 2 models x 3 ICCs x 21 betas x 10 sim_reps). It
is not re-run here.

| corner | run here | source |
|---|---|---|
| temporal x households | no | `outputs/runs_summary.csv` (main sweep) |
| temporal x no households | yes | arm `nohh_temporal` |
| static x households | yes | arm `hh_static` |
| static x no households | yes | arm `nohh_static` |

At analysis time, filter the main sweep's `runs_summary.csv` to `n_nodes ≈ 100k`
and tag those rows `arm="hh_temporal", temporal=True, households=True`, then
concatenate with `outputs_structure/runs_summary.csv`. The two tables share a
schema — this sweep's is the main sweep's columns plus `arm`, `temporal`,
`households` and `frac_trans_community` — so the concat needs no reshaping.
`--collect` prints this reminder and checks the main table is where it expects.

## What the axes mean

**temporal vs static.** A temporal arm redraws a day of community contacts every
simulated day from the three frequency layers (everyday / few_times_week /
one_time). A static arm fits the community marginal to counts **summed over
frequency** and matches them into one fixed edge set that is on every day. Same
survey, same duration weighting; what changes is whether a contact is a nightly
draw or a permanent edge. Measured at n=100k: static aggregate degree ≈ 8.6,
temporal daily total degree ≈ 7.1, so the two are comparable in daily
connectivity and the arms differ in *structure*, not in raw contact volume.

**households vs none.** Following the established 2x2 in
`seir_comparison_common.py` (`realization_no_household_*`), "no households" means
a population of singletons — no household layer at all — with community contacts
from the **age-blind reduced** survey (`reconnect_reduced.json`: 5 duration bins,
no contact-age axis). This axis therefore removes household structure *and* age
structure together, exactly as the published 2x2 does. It is not a
households-only ablation, and the age-attack columns are uninformative for the
`nohh_*` arms.

**model.** Unchanged from the main sweep: `nb` (per-cell independent negative
binomial) vs `loggmm_mm` (joint log-GMM with per-cell mean-matching). Both match
the survey marginal, so differences isolate the cross-cell correlation. The
log-GMM mixtures are **not** refitted — they are read from the same
`outputs_loggmm/networks/loggmm_fits.pkl` the main sweep uses, which already
holds the summed and reduced variants. `builders_structure.add_cell_mean_match`
is the summed/reduced counterpart of `gmm_contacts.add_cell_mean_match_layered`.

**ICC** is a temporal-only axis — a static network has no one_time layer to
persist. It is carried at the main sweep's three targets (0, 0.365, 0.95) so
`temporal x no-hh` can be matched to `temporal x households` at equal
persistence. `--iccs` is *ignored* for the static arms rather than fanning them
into three identical copies; their cell ids read `iccna` and their
`target_icc` is `NaN`.

## Grid size

```
hh_static     : 2 models x 1 (no ICC) x 10 net_reps = 20 realisations
nohh_static   : 2 models x 1 (no ICC) x 10 net_reps = 20 realisations
nohh_temporal : 2 models x 3 ICCs     x 10 net_reps = 60 realisations
                                              total  100 realisations
                          x 21 betas x 10 sim_reps = 21,000 epidemic runs
```

## Cost

Measured at n=100k, one NB realisation per arm, at betas 0.17 / 0.8 / 9:

| arm | per run | per realisation (210 runs) | realisations | total |
|---|---|---|---|---|
| `hh_static` | 4s / 11s / 14s | ~35 min | 20 | ~12 core-h |
| `nohh_static` | 1s / 5s / 8s | ~20 min | 20 | ~7 core-h |
| `nohh_temporal` | 18s / 249s / 123s | ~7 h | 60 | ~420 core-h |

The static arms are cheap because of the identity fast path below — the
adjacency is built once per run instead of once per simulated day. The temporal
arm rebuilds it daily and is essentially the whole wall-clock. On 48 cores that
is roughly two waves of long tasks, ~15 h, inside the 48 h the slurm script
requests. Peak RSS is ~0.7 GB per worker in every arm.

## Usage

```bash
python run_structure_sweep.py --estimate           # cost projection, runs nothing
python run_structure_sweep.py -j 8                 # the whole grid
python run_structure_sweep.py --arms hh_static     # one architecture only
python run_structure_sweep.py --sizes 10000 -j 6   # cheap smoke slice
python run_structure_sweep.py --collect            # merge shards into final CSVs
```

On the cluster:

```bash
mkdir -p slurm_logs && sbatch run_structure_sweep.slurm
```

Resuming and splitting work exactly as in the main sweep: progress is tracked per
`(realisation, beta)` in `outputs_structure/_done/`, so re-submitting skips
finished work, betas can be added to a finished grid, and jobs can be split along
`--arms`, `--models`, `--iccs` or `--betas` (every one of those is part of the
done-marker key). `--force` is scoped to the cells and betas requested.

## A note on `gillespie_seir.py`

The static arms needed one change to the shared simulator: `simulate_seir` now
skips the daily `set_adjacency` when `edges_fn` hands back the *same list object*
as the previous day, which is exactly what `static_edges_fn` does. Adjacency and
the live S-I edge set are already maintained exactly by `set_state` — audited
against a from-scratch rebuild at all 103 day boundaries of a 20k-node outbreak,
zero mismatches — so the skipped work is a genuine O(E) no-op. Measured speedup
on a static run: **~6.5x**.

Temporal runs are untouched: `temporal_edges_fn` builds a fresh list every call,
so it never hits the identity check and rebuilds exactly as before. The main
sweep's results are therefore unaffected.

One consequence worth knowing: the skipped rebuild leaves the *set* of live S-I
edges and the total transmission rate identical, but not the order of edges
within a weight bucket (swap-removal shuffles them; a rebuild re-inserts in node
order). `sample_si_edge` draws a uniform index within a bucket, so a **static**
run from a given seed now walks a different sample path from the same
distribution than it would have before. Stored results are unaffected, but
re-running `run_seir_comparison*.py`'s static models will not reproduce their old
CSVs replicate-for-replicate — only their distributions.

## Outputs (`outputs_structure/`)

Same layout and schema as the main sweep's `outputs/` (see `README.md` for the
full metric list), plus:

| column | meaning |
|---|---|
| `arm` | `hh_static`, `nohh_static`, `nohh_temporal` |
| `temporal` | bool — the first factor |
| `households` | bool — the second factor |
| `frac_trans_community` | share of transmission on the static community layer |

Layer columns are reported **only for layers the arm actually has**, so a static
arm has no `frac_trans_one_time` at all rather than `0.0`, and the temporal arm
has no `frac_trans_community`. Missing means "no such layer", not "measured, and
zero" — after concatenating the arms those cells are `NaN`. The same applies to
the per-layer degree columns in `network_summary.csv`, and to the household
columns for the `nohh_*` arms (`n_multi_person_households = 0`, the rest `NaN`).

## Files

- `config_structure.py` — arms, grid, replicate budget; inherits all
  epidemiology and the beta grid from `config.py` so the two sweeps cannot drift
- `builders_structure.py` — fits and network construction for the three arms
- `run_structure_sweep.py` — parallel driver / CLI
- `run_structure_sweep.slurm` — cluster submission script
- `metrics.py` — shared with the main sweep, unchanged
