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

**households vs none.** "No households" means a population of singletons — no
household layer at all — and nothing else. The people have the same UK age
structure, and community contact is drawn from the **same age-structured fit**
as the matching household arm: the summed fit for `nohh_static` (as
`hh_static`), the main sweep's layered fit for `nohh_temporal` (as
`hh_temporal`), with stubs matched by age group in the same way. A no-household
arm is therefore its household counterpart with the household layer taken away.
Contacts with household members, which were removed from the survey data
because the household layer represents them, are not put back.

Two differences from the household arms remain, both in the population rather
than the contact model: a singleton population has exactly the requested number
of people and the target UK age structure, whereas the household builder adds
top-up people (about 8%, mostly children).

> **Changed 2026-10-06.** Until then the `nohh_*` arms followed the older 2x2 in
> `seir_comparison_common.py` and drew contact from the pooled, age-blind
> `reconnect_reduced.json`, so removing households also removed age structure.
> Results from that construction are obsolete. See
> [Re-running the no-household arms](#re-running-the-no-household-arms).

**model.** Unchanged from the main sweep: `nb` (per-cell independent negative
binomial) vs `loggmm_mm` (joint log-GMM with per-cell mean-matching). Both match
the survey marginal, so differences isolate the cross-cell correlation. The
log-GMM mixtures are **not** refitted — the static arms read the summed fit from
the same `outputs_loggmm/networks/loggmm_fits.pkl` the main sweep uses, and
`nohh_temporal` reads the layered fit from the main sweep's own
`outputs/fits_cache.pkl`, so it uses exactly the parameters `hh_temporal` was
run with. `builders_structure.add_cell_mean_match` is the summed counterpart of
`gmm_contacts.add_cell_mean_match_layered`.

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

## Re-running the no-household arms

The no-household arms changed construction on 2026-10-06 (age-structured
contact, see above), so every earlier `nohh_*` result has to be replaced. Three
steps, all from `seir_sweep/`:

**1. Delete the old no-household results** — wherever the rerun will happen, and
on this machine too if it holds a copy:

```bash
find outputs_structure/_done outputs_structure/shards \
     outputs_structure/transmission_logs outputs_structure/daily_series \
     outputs/_analysis_cache \
     -maxdepth 1 -type f -name '*nohh_*' -delete
python run_structure_sweep.py --collect      # rebuild the tables without them
```

`*nohh_*` matches only no-household files: `hh_static` files do not contain it,
and the main sweep's files carry no arm prefix at all. Swap `-delete` for
`-print | wc -l` to count first (locally: 3,554 files).

**2. Run them again**

```bash
bash rerun_nohh.sh --no-transmission-log             # here (see the file header)
mkdir -p slurm_logs && sbatch rerun_nohh.slurm       # on the cluster
```

80 realisations, 16,800 epidemics, about 650 core-hours. Interrupted runs resume
when the same command is given again.

**3. Refresh the per-run files and the notebooks** — `regen_perrun.py` for the
no-household arms if the logs were skipped or left on the cluster, then
`rerun_notebooks.sh`.

The driver enforces the order. A realisation built since the change records
`nohh_contact_fit = age_structured` in its network shard; a no-household cell
with files on disk but no such record is from the old construction, and
`run_structure_sweep.py` refuses to run on top of it (and `--collect` warns),
because the done-markers would skip the old betas and the write-once network
shard would keep the old structure. `--force` does not get round this — it only
clears done-markers.

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
| `nohh_contact_fit` | `network_summary.csv` only: `age_structured` for a no-household realisation built since 2026-10-06; empty for household arms (not applicable) and for obsolete no-household rows |

Layer columns are reported **only for layers the arm actually has**, so a static
arm has no `frac_trans_one_time` at all rather than `0.0`, and the temporal arm
has no `frac_trans_community`. Missing means "no such layer", not "measured, and
zero" — after concatenating the arms those cells are `NaN`. The same applies to
the per-layer degree columns in `network_summary.csv`, and to the household
columns for the `nohh_*` arms (`n_multi_person_households = 0`, the rest `NaN`).

## Getting the results back off the cluster

`./fetch_results.sh` pulls down only what is new. rsync compares size and mtime
first, so after the first pull each later run moves only the files that appeared
since — which is what the write-once shards, done-markers and per-run `.csv.gz`
files are built for.

```bash
REMOTE=you@login.cluster ./fetch_results.sh summary   # the two CSVs + manifest  (~30 MB)
REMOTE=you@login.cluster ./fetch_results.sh shards    # + per-(cell,beta) shards + done markers
REMOTE=you@login.cluster ./fetch_results.sh series    # + daily_series/          (~100 MB)
REMOTE=you@login.cluster ./fetch_results.sh logs      # + transmission_logs/     (TENS OF GB)
```

`RPATH` defaults to `temporal_networks/seir_sweep` **relative to your remote
home**, so it normally needs no setting. Do not write `RPATH=~/...`: an unquoted
tilde is expanded by your *local* shell, so the cluster is then asked for
`/home/<your-local-user>/...`, which does not exist there. The script detects
that case and rewrites it, but a bare relative path is cleaner.

One SSH connection is shared between all the transfers (`ControlMaster`), so a
2FA cluster prompts once rather than once per directory, and `ssh` calls in the
next ten minutes are free too. `--dry-run` passes straight through to rsync.
Failed transfers are counted and the script exits non-zero — a directory that
simply does not exist on the cluster yet is listed separately as skipped, not as
an error. There is an sftp/lftp fallback in the footer for a cluster without
rsync, and a `logs` pull refuses to start with under 40 GB free.

**Disk.** The main sweep's transmission logs are already 12 GB and this sweep
adds ~21,000 more runs. Pull `summary` and `shards` first — they carry every
metric in `runs_summary.csv` — and fetch logs selectively (they are only needed
for the transmission-tree and household sections of the analysis):

```bash
rsync -avhP --ignore-existing --include='*hh_static*' --exclude='*'     you@cluster:~/temporal_networks/seir_sweep/outputs_structure/transmission_logs/     ./outputs_structure/transmission_logs/
```

Progress can be checked without pulling anything:

```bash
ssh you@cluster "ls ~/temporal_networks/seir_sweep/outputs_structure/_done | wc -l"
# 2100 when complete (100 cells x 21 betas)
```

## Analysis

The analysis is split by **comparison regime**, one notebook per question, all
at **n = 100,000**. Two shared modules sit underneath:

- **`unified_common.py`** -- palette and encoding contract, loads and unifies
  the two sweeps, fits and inverts the measured $\beta \to R_0$ curve, per-run
  readers, and the deterministic population rebuild.
- **`figure_library.py`** -- one function per figure, parameterised by an
  *encoding mode* so the same panel can be drawn in whichever comparison a
  notebook is making. It also owns the two benchmarks every figure is measured
  against: the **survey** (structure) and the **homogeneous Erlang-3 SEIR ODE**
  (epidemiology).

| notebook | varies | held fixed | encoding mode |
|---|---|---|---|
| `analysis_1_architecture.ipynb` | the 2x2 `{temporal, static}` x `{households, none}` | contact model (one figure each), ICC 0.365 | `arch` -- colour = architecture only |
| `analysis_2_contact_model.ipynb` | NB vs log-GMM | `hh_temporal`, ICC 0.365 | `model` -- colour = contact model |
| `analysis_3_persistence.ipynb` | ICC 0 / 0.365 / 0.95 | `hh_temporal`, both models shown | `icc` -- colour = ICC ramp, linestyle = model |

**Notebook 1 draws each contact model in its own figure.** Because the model is
constant within a figure it needs no encoding, which frees colour to carry
**structure and nothing else** -- so the NB figure and the log-GMM figure use
identical colours for identical architecture and can be read side by side as
one comparison.

**Notebooks 2 and 3 are confounder-free by restriction.** Both are pinned to
temporal + households, so the only thing moving is the axis named in the title.
Notebook 2 additionally repeats its headline metrics at all three ICC values to
show the model contrast is not an artefact of one persistence setting.

Every notebook carries the full figure suite: contact-count distributions, mean
degree and Earth-mover distance by age, age-mixing matrices and their residuals
against the survey, the duration mix and its transmission weighting, the
$\beta \to R_0$ map, final size against the well-mixed limit, growth rate
against the homogeneous SEIR root, takeoff probability, dispersion $k$ and the
top-decile share, peak height against peak timing, matched-$R_0$ outcome
distributions, trajectories against the ODE, generation structure, degree and
age of the infected and of the infectors, the offspring distribution against
Poisson, layer attribution, the epidemic course by age, and -- for the
architectures that have households -- which households are reached, which drive
spread onwards, and how infection first enters them.

Three principles hold throughout:

- **Every axis is measured $R_0$, never the nominal target.** A static and a
  temporal network at the same beta are *not* at the same $R_0$: `hh_static`
  needs beta 0.59 to reach $R_0 \approx 2$ where `nohh_static` needs 3.25.
  Architectures are compared at matched $R_0$ via a fitted, inverted Hill curve,
  and the achieved value is reported because the beta grid is finite.
- **Comparisons are against a benchmark, not only against each other.** An arm
  can be called wrong against the survey or against the homogeneous ODE; against
  another arm it is only different.
- **Household outcomes are recovered, not summarised.** The sweeps store node
  ids, not populations, but `network_summary.csv` records each realisation's
  `network_seed` and the builders are deterministic, so the exact population is
  rebuilt and joined to the transmission log by node id.

### The no-household arms and age structure

Results produced before 2026-10-06 came from `nohh_*` arms built from `reduced`
parameters of shape `(1, 1, 5, 2)` -- a single age bucket -- so in those results
removing households also removed age-assortative mixing, and the notebooks'
prose (and `build_nb1.py`) describes that confound. The arms are now
age-structured. Once they have been re-run, the notebooks have to be executed
again and that prose rewritten: the flat age profile and the wide contact
distribution it explains belong to the old construction.

### Per-run data and `regen_perrun.py`

`outputs_structure/` was pulled from the cluster with summary CSVs only, so its
`transmission_logs/` and `daily_series/` were empty and every tree-level figure
would have covered the `hh_temporal` corner alone.

They are reproducible rather than lost: `run_structure_sweep.py` derives both
the network seed and the simulation seed deterministically from the grid
coordinates and records them, so **`regen_perrun.py`** rebuilds the realisation
from `network_seed`, re-runs with `sim_seed`, and writes the two per-run files.
It touches nothing else -- no shards, no done-markers, no summary CSVs -- and
cross-checks each regenerated run's final size against the recorded summary
row, discarding and reporting any that disagree rather than letting a stale
result be believed.

```bash
python regen_perrun.py --r0 1.5 2.0 4.0 --net-reps 2 --sim-reps 2 -j 5 --iccs 0.365
```

Verified bit-for-bit on a `nohh_static` n=100k replicate: same seed nodes, same
`r0_measured` 1.4, same final size 55440. Two details are load-bearing --
`probe_degrees` must be called before the epidemic (it advances the
realisation's RNG, as `run_task` does), and `edges_fn` is built once per
realisation and shared across betas.

Figures that still lack their per-run files print what is missing and which
`fetch_results.sh` target supplies it, rather than drawing a partial figure as
if it were complete.

`unified_analysis_SUPERSEDED.ipynb.bak` is the previous single notebook that
answered all three questions at once, kept for reference.

## Files

- `config_structure.py` — arms, grid, replicate budget; inherits all
  epidemiology and the beta grid from `config.py` so the two sweeps cannot drift
- `builders_structure.py` — fits and network construction for the three arms
- `run_structure_sweep.py` — parallel driver / CLI
- `run_structure_sweep.slurm` — cluster submission script
- `rerun_nohh.sh`, `rerun_nohh.slurm` — re-run every no-household simulation
  (local / cluster), after the old ones have been deleted
- `fetch_results.sh` — incremental download from the cluster
- `regen_perrun.py` — rebuild missing `transmission_logs/` and `daily_series/`
  from the recorded seeds, when only the summary CSVs were pulled
- `metrics.py` — shared with the main sweep, unchanged

Analysis layer:

- `unified_common.py` — data loading, palette, matched-$R_0$ machinery,
  per-run readers, deterministic population rebuild
- `figure_library.py` — the figures, plus the survey and homogeneous-ODE
  benchmarks they are measured against
- `analysis_1_architecture.ipynb` — architecture, one figure per contact model
- `analysis_2_contact_model.ipynb` — NB vs log-GMM, temporal + households only
- `analysis_3_persistence.ipynb` — ICC 0 / 0.365 / 0.95, temporal + households only
- `build_nb1.py`, `build_nb2.py`, `build_nb3.py` — regenerate each notebook from
  source, so a notebook is never hand-edited in place
