"""Parallel SEIR sweep over network *architecture*: {temporal, static} x
{households, no households} x {NB, log-GMM} at n=100k, on the main sweep's beta grid.

This is the sibling of ``run_sweep.py``. That one holds the architecture fixed
(temporal, with households) and varies the contact model and the one_time
persistence; this one varies the architecture. The **temporal x households**
corner is deliberately not run here -- it *is* the main sweep, whose n=100000
slice is already complete -- so this sweep covers the other three corners and
the fourth is spliced in from ``outputs/runs_summary.csv`` at analysis time.
See ``config_structure.py`` for what each axis means.

Each *task* is one network realisation ``(arm, model, icc, n_nodes, net_rep)``:
the population and community network are built once, probed for structural
covariates, and then every ``beta`` x ``sim_rep`` epidemic is run on that same
realisation (a paired design that also amortises the expensive network build).
Tasks are distributed over a process pool.

Outputs land in ``outputs_structure/``, in exactly the schema ``run_sweep.py``
writes to ``outputs/`` (plus ``arm``/``temporal``/``households`` columns), so
the two runs_summary tables concatenate directly. Writing is
shard-per-(task, beta) so parallel workers never contend, and completed
``(realisation, beta)`` units are skipped on re-run, so an interrupted sweep
resumes where it stopped. Betas can be added to a finished grid: re-run with the
new values and only those are simulated, on the same network realisations as the
originals (seeds derive from the grid coordinates, not from the beta list).

Examples
--------
    python run_structure_sweep.py --estimate            # cost projection, runs nothing
    python run_structure_sweep.py -j 8                  # the whole grid, 8 workers
    python run_structure_sweep.py --arms hh_static      # one architecture only
    python run_structure_sweep.py --sizes 10000 -j 4    # cheap smoke slice
    python run_structure_sweep.py --collect             # merge shards into final CSVs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zlib
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config as C                                      # noqa: E402
import config_structure as CS                           # noqa: E402
import metrics as M                                     # noqa: E402
from builders_structure import (                        # noqa: E402
    NOHH_CONTACT_FIT, build_realisation, prepare_fits,
)

sys.path.insert(0, str(C.REPO))
from gillespie_seir import SEIRParams, simulate_seir     # noqa: E402
import outbreak_metrics as om                            # noqa: E402

SHARD_DIR = CS.OUT_DIR / "shards"
DONE_DIR = CS.OUT_DIR / "_done"
TLOG_DIR = CS.OUT_DIR / "transmission_logs"
SERIES_DIR = CS.OUT_DIR / "daily_series"

TRANSMISSION_COLUMNS = ["time", "source", "target", "layer",
                        "source_generation", "target_generation"]

# Which contact layers each arm actually has. Only these are reported, so a
# layer an arm does not possess comes out *missing* rather than 0.0 -- a static
# network has no one_time layer at all, and a zero would read as "measured, and
# nothing went through it".
ARM_LAYERS = {
    "hh_static": ("household", "community"),
    "nohh_static": ("community",),
    "nohh_temporal": ("everyday", "few_times_week", "one_time"),
}


def _ensure_dirs() -> None:
    for d in (CS.OUT_DIR, SHARD_DIR, DONE_DIR, TLOG_DIR, SERIES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _code(text: str) -> int:
    """Stable integer for a name (``hash()`` is salted per process)."""
    return zlib.crc32(text.encode()) % 100_000


# Seed coordinate for a static arm, which has no ICC. ``SeedSequence`` rejects
# negative entropy, so the sentinel is a large positive value instead of -1;
# real targets are ``int(icc * 1000)`` in [0, 1000], so it cannot collide.
ICC_COORD_NONE = 999_999


def _icc_coord(icc: float | None) -> int:
    """ICC as a seed coordinate; static arms (no ICC) get a distinct sentinel."""
    return ICC_COORD_NONE if icc is None else int(icc * 1000)


def _seed(*coords) -> int:
    """Reproducible per-replicate seed from grid coordinates."""
    ss = np.random.SeedSequence([CS.RNG_SEED, *(int(c) for c in coords)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _btag(beta: float) -> str:
    """Filename-safe key for a beta value (same convention as the run stems)."""
    return f"{int(round(beta * 1e6)):08d}"


# ---------------------------------------------------------------------------
# Results left over from the age-blind no-household construction
# ---------------------------------------------------------------------------

_NOHH_ARMS = [a for a in CS.ARMS if not CS.arm_has_households(a)]
_NOHH_CELL_RE = re.compile(
    "((?:" + "|".join(re.escape(a) for a in _NOHH_ARMS) + r")__.+?__net\d+)")

# What to run to clear them, printed wherever stale results are reported.
STALE_NOHH_HELP = f"""\
  Delete the old no-household results, then rebuild the summary tables:

      cd {HERE}
      find outputs_structure/_done outputs_structure/shards \\
           outputs_structure/transmission_logs outputs_structure/daily_series \\
           outputs/_analysis_cache \\
           -maxdepth 1 -type f -name '*nohh_*' -delete
      python run_structure_sweep.py --collect
"""


def stale_nohh_cells(cells=None) -> list[str]:
    """No-household cells on disk that predate age-structured contact.

    Until 2026-10-06 the ``nohh_*`` arms drew community contact from a pooled,
    age-blind fit. A realisation built since then records ``nohh_contact_fit``
    in its network shard, which is written before any epidemic is run on it. So
    a no-household cell that has *any* file on disk but no such record is left
    over from the old construction, and resuming on top of it would silently
    mix the two: the done-markers would skip the old betas, and the write-once
    network shard would keep the old structure.

    ``cells`` restricts the check to those cell ids; ``None`` checks every
    no-household cell found on disk.
    """
    on_disk: set[str] = set()
    for d in (SHARD_DIR, DONE_DIR, TLOG_DIR, SERIES_DIR):
        if not d.exists():
            continue
        with os.scandir(d) as it:
            for entry in it:
                m = _NOHH_CELL_RE.search(entry.name)
                if m:
                    on_disk.add(m.group(1))
    if cells is not None:
        on_disk &= set(cells)

    stale = []
    for cell in sorted(on_disk):
        shard = SHARD_DIR / f"network_{cell}.csv"
        fit = None
        if shard.exists():
            head = pd.read_csv(shard, nrows=1)
            if "nohh_contact_fit" in head.columns and len(head):
                fit = head["nohh_contact_fit"].iloc[0]
        if fit != NOHH_CONTACT_FIT:
            stale.append(cell)
    return stale


# ---------------------------------------------------------------------------
# One task = one network realisation, all betas and sim reps on it
# ---------------------------------------------------------------------------

def run_task(arm: str, model: str, icc: float | None, n_nodes: int, net_rep: int,
             betas: list[float], sim_reps: int, max_days: int | None,
             save_transmission_log: bool, save_daily_series: bool) -> dict:
    cell = CS.cell_id(arm, model, icc, n_nodes, net_rep)
    # Betas are tracked one at a time, so a cell that already holds some betas
    # only runs the ones it is missing. The network is rebuilt from the same
    # ``net_seed``, so added betas land on exactly the same realisation as the
    # betas that were run before -- the design stays paired.
    todo = [b for b in betas
            if not (DONE_DIR / f"{cell}__beta{_btag(b)}.done").exists()]
    if not todo:
        return {"cell": cell, "status": "skipped", "seconds": 0.0,
                "n_runs": 0, "n_betas": 0}

    layers = ARM_LAYERS[arm]

    t_start = time.perf_counter()
    net_seed = _seed(1, _code(arm), _code(model), _icc_coord(icc), n_nodes, net_rep)
    real = build_realisation(arm, model, icc, n_nodes, net_seed)
    build_s = time.perf_counter() - t_start

    # ---- structural covariates (E) -------------------------------------
    probe = real.probe_degrees(CS.STRUCT_PROBE_DAYS)
    struct = M.network_structure_metrics(probe, real.household_degree())
    struct.update(real.meta)
    struct.update({"cell": cell, "net_rep": net_rep,
                   "network_build_seconds": build_s})
    del probe
    net_shard = SHARD_DIR / f"network_{cell}.csv"
    if not net_shard.exists():      # same seed -> same network; write once
        pd.DataFrame([struct]).to_csv(net_shard, index=False)

    edges_fn = real.edges_fn()
    n_runs = 0
    for beta in todo:
        rows = []
        for rep in range(sim_reps):
            sim_seed = _seed(2, _code(arm), _code(model), _icc_coord(icc), n_nodes,
                             net_rep, int(round(beta * 1e6)), rep)
            rng_sim = np.random.default_rng(sim_seed)
            seeds = rng_sim.choice(real.n_nodes, size=CS.N_SEED_INFECTED,
                                   replace=False).tolist()

            t0 = time.perf_counter()
            traj = simulate_seir(
                n_nodes=real.n_nodes, edges_fn=edges_fn,
                params=SEIRParams(beta=beta, sigma=CS.SIGMA, gamma=CS.GAMMA),
                initial_infected=seeds, rng=rng_sim, max_days=max_days,
            )
            sim_s = time.perf_counter() - t0

            tdf = pd.DataFrame([t.__dict__ for t in traj.transmissions])
            if tdf.empty:
                tdf = pd.DataFrame(columns=TRANSMISSION_COLUMNS)

            n_days = int(np.ceil(traj.times[-1]))
            daily = traj.daily_snapshot(n_days)

            row = {
                "cell": cell, "arm": arm, "temporal": CS.arm_is_temporal(arm),
                "households": CS.arm_has_households(arm), "model": model,
                "target_icc": icc if icc is not None else float("nan"),
                "n_nodes": real.n_nodes, "net_rep": net_rep,
                "beta": beta, "sim_rep": rep,
                "r0_measured": om.r0_estimate(tdf, traj.seed_nodes),
                "n_seeds": CS.N_SEED_INFECTED, "seed_nodes": json.dumps(seeds),
                "sim_seconds": sim_s, "network_seed": net_seed,
                "sim_seed": sim_seed, "max_days": max_days if max_days else -1,
            }
            row.update(M.takeoff_metrics(traj.final_size(), real.n_nodes,
                                         CS.N_SEED_INFECTED, CS.TAKEOFF_FRACTION))
            row.update(M.peak_metrics(traj, real.n_nodes))
            row.update(M.offspring_metrics(tdf, traj.seed_nodes, real.n_nodes))
            row.update(M.layer_metrics(tdf, layers=layers))
            row.update(M.age_attack_metrics(tdf, traj.seed_nodes, real.node_age))
            row.update(M.growth_metrics(daily, real.n_nodes))
            row.update(M.generation_interval_metrics(tdf, traj.seed_nodes))
            row.update(om.household_infection_stats(
                real.hh_by_node, real.household_sizes,
                om.infected_node_set(tdf, traj.seed_nodes),
            ))
            rows.append(row)

            stem = f"{cell}__beta{int(round(beta * 1e6)):08d}__rep{rep}"
            if save_transmission_log and not tdf.empty:
                tdf.to_csv(TLOG_DIR / f"{stem}.csv.gz", index=False,
                           compression="gzip")
            if save_daily_series:
                series = pd.DataFrame({k: v for k, v in daily.items()})
                series.insert(0, "day", np.arange(len(series)))
                inc = M.daily_layer_incidence(tdf, n_days, layers=layers)
                series = series.merge(inc, on="day", how="left").fillna(0)
                series.to_csv(SERIES_DIR / f"{stem}.csv.gz", index=False,
                              compression="gzip")
            del tdf, traj, daily

        # Commit this beta on its own, so an interrupted task keeps the betas it
        # finished instead of losing the whole realisation.
        pd.DataFrame(rows).to_csv(
            SHARD_DIR / f"runs_{cell}__beta{_btag(beta)}.csv", index=False)
        (DONE_DIR / f"{cell}__beta{_btag(beta)}.done").write_text(json.dumps({
            "cell": cell, "beta": beta, "n_runs": len(rows),
            "seconds": time.perf_counter() - t_start,
        }))
        n_runs += len(rows)

    return {"cell": cell, "status": "ok", "n_runs": n_runs,
            "n_betas": len(todo), "seconds": time.perf_counter() - t_start}


def _worker(args) -> dict:
    """Process-pool entry point: never let one task kill the sweep."""
    try:
        # keep BLAS single-threaded so N workers do not oversubscribe the cores
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = "1"
        return run_task(*args)
    except Exception as exc:  # noqa: BLE001
        return {"cell": CS.cell_id(args[0], args[1], args[2], args[3], args[4]),
                "status": "error", "error": f"{exc}",
                "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect() -> None:
    """Merge per-task shards into the two top-level tables."""
    for prefix, out_name in (("runs_", "runs_summary.csv"),
                             ("network_", "network_summary.csv")):
        shards = sorted(SHARD_DIR.glob(f"{prefix}*.csv"))
        if not shards:
            print(f"  no {prefix}* shards yet")
            continue
        df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
        # Shards from different invocations can overlap if the same beta was
        # requested twice; keep one row per replicate rather than double-count.
        key = [k for k in ("cell", "beta", "sim_rep") if k in df.columns]
        if key:
            n_before = len(df)
            df = df.drop_duplicates(subset=key, keep="first")
            if len(df) < n_before:
                print(f"  {out_name}: dropped {n_before - len(df):,} duplicate rows")
        df.to_csv(CS.OUT_DIR / out_name, index=False)
        print(f"  {out_name}: {len(df):,} rows from {len(shards)} shards")

    stale = stale_nohh_cells()
    if stale:
        print(f"\n  WARNING: {len(stale)} no-household realisations on disk were "
              "built with the old age-blind\n  contact fit, and their rows are in "
              "the tables just written.")
        print(STALE_NOHH_HELP)

    src = CS.EXCLUDED_ARM_SOURCE
    print(f"\n  reminder: the {CS.EXCLUDED_ARM} corner is not in these tables.")
    print(f"  {'found' if src.exists() else 'MISSING'}: {src}")
    print("  Splice it in at analysis time, tagging those rows "
          f"arm={CS.EXCLUDED_ARM}, temporal=True, households=True.")


def estimate(tasks, betas, args) -> None:
    """Cost projection from measured per-realisation throughput."""
    print(f"\nGrid: {len(tasks)} network realisations x {len(betas)} betas "
          f"x sim_reps -> total epidemic runs:")
    total_runs = 0
    for arm in CS.ARMS:
        arm_tasks = [t for t in tasks if t[0] == arm]
        if not arm_tasks:
            continue
        runs = sum(len(betas) * CS.REP_BUDGET[t[3]]["sim_reps"] for t in arm_tasks)
        total_runs += runs
        print(f"  {arm:>14}: {len(arm_tasks):>3} realisations, {runs:>6} epidemic runs")
    print(f"  TOTAL: {total_runs:,} epidemic runs\n")

    print("Cost notes, measured on one n=100k realisation per arm (NB):")
    print("  *_static     : the edge set never changes, so simulate_seir builds the")
    print("                 adjacency once per run (identity fast path) and every")
    print("                 later day is pure event simulation. Cost then tracks the")
    print("                 outbreak size, not the day count:")
    print("                   hh_static    build 2s;  4s @b=0.17, 11s @b=0.8, 14s @b=9")
    print("                   nohh_static  build 2s;  1s @b=0.17,  5s @b=0.8,  8s @b=9")
    print("                 ~10 s/run x 210 runs => well under an hour per realisation.")
    print("  nohh_temporal: no fast path -- a fresh day of contacts means a genuine")
    print("                 O(E) adjacency rebuild every simulated day, which")
    print("                 dominates. Budget it like an n=100k temporal realisation")
    print("                 in the main sweep (tens of minutes per supercritical")
    print("                 run), minus the household layer. This arm is ~60 of the")
    print("                 100 realisations and essentially all of the wall-clock.")
    print("")
    print("  Note the two arms are NOT far apart in connectivity -- measured degree")
    print("  is 8.6 (static, aggregated) vs 7.1 (temporal, per day) -- so the cost")
    print("  gap is the rebuild, not a denser network.")
    print("  Use --no-transmission-log if disk is tight; the logs dominate it.")


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=list(CS.ARMS), choices=list(CS.ARMS))
    p.add_argument("--models", nargs="+", default=CS.MODELS, choices=CS.MODELS)
    p.add_argument("--iccs", nargs="+", type=float, default=CS.ICC_TARGETS,
                   help="one_time persistence targets; temporal arms only")
    p.add_argument("--sizes", nargs="+", type=int, default=CS.SIZES)
    p.add_argument("--betas", nargs="+", type=float, default=CS.BETAS)
    p.add_argument("-j", "--workers", type=int, default=min(8, os.cpu_count() or 4))
    p.add_argument("--max-days", type=int, default=None,
                   help="cap simulated days (default: run to burn-out)")
    p.add_argument("--no-transmission-log", action="store_true",
                   help="skip per-run transmission logs (saves a lot of disk)")
    p.add_argument("--no-daily-series", action="store_true")
    p.add_argument("--estimate", action="store_true", help="project cost, run nothing")
    p.add_argument("--collect", action="store_true", help="merge shards and exit")
    p.add_argument("--force", action="store_true", help="ignore done-markers")
    args = p.parse_args()

    _ensure_dirs()

    if args.collect:
        collect()
        return

    tasks_grid = CS.grid(args.arms, args.models, args.iccs, args.sizes)
    if args.estimate:
        estimate(tasks_grid, args.betas, args)
        return

    # --force clears done-markers only; it does not remove a write-once network
    # shard or a transmission log, so it cannot make old no-household results
    # safe to build on. They have to be deleted.
    stale = stale_nohh_cells({CS.cell_id(*t) for t in tasks_grid})
    if stale:
        print(f"REFUSING TO RUN: {len(stale)} of the requested no-household "
              "realisations already have results\non disk from the old "
              "age-blind contact fit (e.g. " + stale[0] + ").\n"
              "Running now would mix old and new results.\n")
        print(STALE_NOHH_HELP)
        sys.exit(2)

    if args.force:
        # Scoped to the cells and betas actually requested, so --force on a
        # slice can never discard finished work outside that slice.
        wanted = {CS.cell_id(*t) for t in tasks_grid}
        n_cleared = 0
        for b in args.betas:
            for cell in wanted:
                marker = DONE_DIR / f"{cell}__beta{_btag(b)}.done"
                if marker.exists():
                    marker.unlink()
                    n_cleared += 1
        print(f"  --force: cleared {n_cleared} done-markers "
              f"({len(wanted)} cells x {len(args.betas)} betas)")

    # What is already on disk for the requested slice, so an accidental re-run
    # of finished betas is visible before it costs anything.
    n_cells = len(tasks_grid)
    pending = {b: sum(1 for t in tasks_grid
                      if not (DONE_DIR / f"{CS.cell_id(*t)}__beta{_btag(b)}.done").exists())
               for b in args.betas}
    n_todo = sum(pending.values())
    print(f"Arms: {', '.join(args.arms)}  (not run: {CS.EXCLUDED_ARM} -- "
          f"it is the main sweep)")
    print(f"\nRequested {len(args.betas)} betas x {n_cells} realisations "
          f"= {n_cells * len(args.betas)} (cell, beta) units")
    print(f"  already done : {n_cells * len(args.betas) - n_todo}")
    print(f"  to run       : {n_todo}")
    if n_todo:
        fresh = [f"{b:g}" for b, k in pending.items() if k]
        print(f"  betas with work: {', '.join(fresh)}")
    else:
        print("  nothing to do -- every requested beta is already complete.")

    print("\nPreparing fits (NB moment fits + mean-matched log-GMM)...")
    prepare_fits()

    tasks = [
        (arm, model, icc, n, rep, args.betas, CS.REP_BUDGET[n]["sim_reps"],
         args.max_days, not args.no_transmission_log, not args.no_daily_series)
        for (arm, model, icc, n, rep) in tasks_grid
    ]
    # biggest first: better load balance on a fixed pool
    tasks.sort(key=lambda t: -t[3])

    manifest = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arms": args.arms, "excluded_arm": CS.EXCLUDED_ARM,
        "excluded_arm_source": str(CS.EXCLUDED_ARM_SOURCE),
        "models": args.models, "iccs": args.iccs, "sizes": args.sizes,
        "betas": args.betas, "beta_r0_nominal": CS.BETA_R0_NOMINAL,
        "rep_budget": {str(k): v for k, v in CS.REP_BUDGET.items()},
        "sigma": CS.SIGMA, "gamma": CS.GAMMA, "n_seed_infected": CS.N_SEED_INFECTED,
        "takeoff_fraction": CS.TAKEOFF_FRACTION, "rng_seed": CS.RNG_SEED,
        "max_days": args.max_days, "workers": args.workers,
        "n_tasks": len(tasks),
    }
    (CS.OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Running {len(tasks)} network realisations on {args.workers} workers\n")
    t0 = time.perf_counter()
    n_done = n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            n_done += 1
            if res.get("status") == "error":
                n_err += 1
                print(f"[{n_done}/{len(tasks)}] ERROR {res['cell']}: {res['error']}")
                print(res.get("traceback", ""))
            else:
                print(f"[{n_done}/{len(tasks)}] {res['status']:7s} {res['cell']} "
                      f"({res.get('n_betas', 0)} betas, {res.get('n_runs', 0)} runs, "
                      f"{res['seconds']:.0f}s)")

    print(f"\nAll tasks finished in {time.perf_counter() - t0:.0f}s ({n_err} errors)")
    print("Collecting shards...")
    collect()


if __name__ == "__main__":
    main()
