"""Parallel SEIR sweep over {model} x {one_time ICC} x {population size} x {beta}.

Each *task* is one network realisation ``(model, icc, n_nodes, net_rep)``: the
population and community network are built once, probed for structural
covariates, and then every ``beta`` x ``sim_rep`` epidemic is run on that same
realisation (a paired design that also amortises the expensive network build).
Tasks are distributed over a process pool.

Outputs land in ``outputs/`` -- see ``--help`` and the module docstring of
``metrics.py`` for what is recorded. Writing is shard-per-(task, beta) so
parallel workers never contend, and completed ``(realisation, beta)`` units are
skipped on re-run, so an interrupted sweep resumes where it stopped.

Because progress is tracked per beta, **betas can be added to a finished grid**:
re-run with the new values and only those are simulated, on the same network
realisations as the originals (the seeds derive from the grid coordinates, not
from the beta list). Overlapping beta sets are free -- already-finished ones are
skipped.

Examples
--------
    python run_sweep.py --estimate                  # cost projection, runs nothing
    python run_sweep.py --sizes 10000               # the 10k slice of the grid
    python run_sweep.py --sizes 10000 100000 -j 8   # bigger slice, 8 workers
    python run_sweep.py --collect                   # merge shards into final CSVs
    python run_sweep.py --sizes 10000 \
        --betas 0.071 0.130 0.251 0.467          # add betas to a finished grid
"""

from __future__ import annotations

import argparse
import json
import os
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

import config as C                                    # noqa: E402
import metrics as M                                   # noqa: E402
from builders import build_realisation, prepare_fits  # noqa: E402

sys.path.insert(0, str(C.REPO))
from gillespie_seir import SEIRParams, simulate_seir   # noqa: E402
import outbreak_metrics as om                          # noqa: E402

SHARD_DIR = C.OUT_DIR / "shards"
DONE_DIR = C.OUT_DIR / "_done"
TLOG_DIR = C.OUT_DIR / "transmission_logs"
SERIES_DIR = C.OUT_DIR / "daily_series"

TRANSMISSION_COLUMNS = ["time", "source", "target", "layer",
                        "source_generation", "target_generation"]


def _ensure_dirs() -> None:
    for d in (C.OUT_DIR, SHARD_DIR, DONE_DIR, TLOG_DIR, SERIES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _model_code(model: str) -> int:
    """Stable integer for a model name (``hash()`` is salted per process)."""
    return zlib.crc32(model.encode()) % 100_000


def _seed(*coords) -> int:
    """Reproducible per-replicate seed from grid coordinates."""
    ss = np.random.SeedSequence([C.RNG_SEED, *(int(c) for c in coords)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _btag(beta: float) -> str:
    """Filename-safe key for a beta value (same convention as the run stems)."""
    return f"{int(round(beta * 1e6)):08d}"


def _completed_betas(cell: str) -> list[float] | None:
    """Betas a legacy whole-realisation marker actually covered.

    Read from that cell's own shard, which records a ``beta`` per row -- ground
    truth, and immune to ``config.BETAS`` having been edited since the run.
    """
    shard = SHARD_DIR / f"runs_{cell}.csv"
    if not shard.exists():
        return None
    try:
        betas = pd.read_csv(shard, usecols=["beta"])["beta"].unique()
    except (ValueError, OSError, pd.errors.ParserError):
        return None
    return sorted(float(b) for b in betas) or None


def _migrate_done_markers() -> int:
    """Upgrade legacy whole-realisation markers to per-beta markers.

    Before the sweep tracked betas individually a completed realisation left a
    single ``{cell}.done``. That carried no beta information, so adding betas to
    an existing grid was impossible: re-running skipped every cell, and
    ``--force`` re-ran everything and overwrote the shard. Each legacy marker is
    expanded here into one marker per beta that the run actually produced, after
    which new betas can be added to a finished grid without touching the
    finished ones.

    The beta list comes from the cell's own shard, never from ``config.BETAS``:
    reading it from config would mean that editing ``BETAS`` before the first
    migrating run marked the *new* betas as already done and skipped them
    forever. A cell whose betas cannot be determined is left alone and reported,
    so it fails loudly rather than silently losing work.
    """
    legacy = [m for m in DONE_DIR.glob("*.done") if "__beta" not in m.name]
    if not legacy:
        return 0
    n_ok, unresolved = 0, []
    for m in legacy:
        cell = m.stem
        betas = _completed_betas(cell)
        if betas is None:
            unresolved.append(cell)
            continue
        try:
            info = json.loads(m.read_text())
        except (json.JSONDecodeError, OSError):
            info = {}
        for b in betas:
            marker = DONE_DIR / f"{cell}__beta{_btag(b)}.done"
            if not marker.exists():
                marker.write_text(json.dumps({
                    "cell": cell, "beta": b, "migrated_from": m.name,
                    "n_runs": info.get("n_runs"), "seconds": info.get("seconds"),
                }))
        m.unlink()
        n_ok += 1
    if n_ok:
        print(f"  migrated {n_ok} legacy done-markers -> per-beta markers "
              f"(betas read from each cell's shard)")
    if unresolved:
        print(f"  WARNING: {len(unresolved)} legacy markers left as-is -- no "
              f"readable shard to tell which betas they covered:")
        for cell in unresolved[:5]:
            print(f"    {cell}")
        print("  These cells will be re-run in full. Delete their markers to "
              "force that, or restore the shards.")
    return n_ok


# ---------------------------------------------------------------------------
# One task = one network realisation, all betas and sim reps on it
# ---------------------------------------------------------------------------

def run_task(model: str, icc: float, n_nodes: int, net_rep: int,
             betas: list[float], sim_reps: int, max_days: int | None,
             save_transmission_log: bool, save_daily_series: bool) -> dict:
    cell = C.cell_id(model, icc, n_nodes, net_rep)
    # Betas are tracked one at a time, so a cell that already holds some betas
    # only runs the ones it is missing. The network is rebuilt from the same
    # ``net_seed``, so added betas land on exactly the same realisation as the
    # betas that were run before -- the design stays paired.
    todo = [b for b in betas
            if not (DONE_DIR / f"{cell}__beta{_btag(b)}.done").exists()]
    if not todo:
        return {"cell": cell, "status": "skipped", "seconds": 0.0,
                "n_runs": 0, "n_betas": 0}

    t_start = time.perf_counter()
    net_seed = _seed(1, _model_code(model), int(icc * 1000), n_nodes, net_rep)
    real = build_realisation(model, icc, n_nodes, net_seed)
    build_s = time.perf_counter() - t_start

    # ---- structural covariates (E) -------------------------------------
    probe = real.probe_degrees(C.STRUCT_PROBE_DAYS)
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
            sim_seed = _seed(2, _model_code(model), int(icc * 1000), n_nodes,
                             net_rep, int(round(beta * 1e6)), rep)
            rng_sim = np.random.default_rng(sim_seed)
            seeds = rng_sim.choice(real.n_nodes, size=C.N_SEED_INFECTED,
                                   replace=False).tolist()

            t0 = time.perf_counter()
            traj = simulate_seir(
                n_nodes=real.n_nodes, edges_fn=edges_fn,
                params=SEIRParams(beta=beta, sigma=C.SIGMA, gamma=C.GAMMA),
                initial_infected=seeds, rng=rng_sim, max_days=max_days,
            )
            sim_s = time.perf_counter() - t0

            tdf = pd.DataFrame([t.__dict__ for t in traj.transmissions])
            if tdf.empty:
                tdf = pd.DataFrame(columns=TRANSMISSION_COLUMNS)

            n_days = int(np.ceil(traj.times[-1]))
            daily = traj.daily_snapshot(n_days)

            row = {
                "cell": cell, "model": model, "target_icc": icc,
                "n_nodes": real.n_nodes, "net_rep": net_rep,
                "beta": beta, "sim_rep": rep,
                "r0_measured": om.r0_estimate(tdf, traj.seed_nodes),
                "n_seeds": C.N_SEED_INFECTED, "seed_nodes": json.dumps(seeds),
                "sim_seconds": sim_s, "network_seed": net_seed,
                "sim_seed": sim_seed, "max_days": max_days if max_days else -1,
            }
            row.update(M.takeoff_metrics(traj.final_size(), real.n_nodes,
                                         C.N_SEED_INFECTED, C.TAKEOFF_FRACTION))
            row.update(M.peak_metrics(traj, real.n_nodes))
            row.update(M.offspring_metrics(tdf, traj.seed_nodes, real.n_nodes))
            row.update(M.layer_metrics(tdf))
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
                inc = M.daily_layer_incidence(tdf, n_days)
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
        return {"cell": C.cell_id(args[0], args[1], args[2], args[3]),
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
        df.to_csv(C.OUT_DIR / out_name, index=False)
        print(f"  {out_name}: {len(df):,} rows from {len(shards)} shards")


def estimate(tasks, betas, args) -> None:
    """Rough cost projection, anchored on measured throughput at 50k nodes."""
    # ~1 us per (node-day) of adjacency rebuild + ~1 ms per infection event,
    # calibrated from run_logs of the existing 50k-node pipeline.
    print(f"\nGrid: {len(tasks)} network realisations x {len(betas)} betas "
          f"x sim_reps -> total epidemic runs:")
    total_runs = 0
    for n in sorted({t[2] for t in tasks}):
        n_tasks = sum(1 for t in tasks if t[2] == n)
        runs = n_tasks * len(betas) * C.REP_BUDGET[n]["sim_reps"]
        total_runs += runs
        print(f"  n={n:>9,}: {n_tasks:>3} realisations, {runs:>5} epidemic runs")
    print(f"  TOTAL: {total_runs:,} epidemic runs\n")

    print("Feasibility on this machine (8 cores, ~4 GB RAM free, 27 GB disk):")
    print("  n=10k     : fine -- minutes per realisation.")
    print("  n=100k    : heavy -- the simulator rebuilds the full daily adjacency")
    print("              in Python, so a supercritical run is ~10-40 min each.")
    print("  n=1,000,000: NOT viable here. One realisation needs several GB of")
    print("              adjacency alone and a supercritical epidemic would take")
    print("              many hours; 4 GB free RAM will thrash or OOM. Run that")
    print("              slice on a larger machine (--sizes 1000000).")


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=C.MODELS, choices=C.MODELS)
    p.add_argument("--iccs", nargs="+", type=float, default=C.ICC_TARGETS)
    p.add_argument("--sizes", nargs="+", type=int, default=C.SIZES)
    p.add_argument("--betas", nargs="+", type=float, default=C.BETAS)
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

    tasks_grid = C.grid(args.models, args.iccs, args.sizes)
    if args.estimate:
        estimate(tasks_grid, args.betas, args)
        return

    print("Checking done-markers...")
    _migrate_done_markers()

    if args.force:
        # Scoped to the cells and betas actually requested, so --force on a
        # slice can never discard finished work outside that slice.
        wanted = {C.cell_id(m, i, n, r) for (m, i, n, r) in tasks_grid}
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
    pending = {b: sum(1 for (m, i, n, r) in tasks_grid
                      if not (DONE_DIR / f"{C.cell_id(m, i, n, r)}__beta{_btag(b)}.done").exists())
               for b in args.betas}
    n_todo = sum(pending.values())
    print(f"\nRequested {len(args.betas)} betas x {n_cells} realisations "
          f"= {n_cells * len(args.betas)} (cell, beta) units")
    print(f"  already done : {n_cells * len(args.betas) - n_todo}")
    print(f"  to run       : {n_todo}")
    if n_todo:
        fresh = [f"{b:g}" for b, k in pending.items() if k]
        print(f"  betas with work: {', '.join(fresh)}")
    else:
        print("  nothing to do -- every requested beta is already complete.")

    print("\nPreparing fits (NB + mean-matched log-GMM)...")
    prepare_fits()

    tasks = [
        (model, icc, n, rep, args.betas, C.REP_BUDGET[n]["sim_reps"],
         args.max_days, not args.no_transmission_log, not args.no_daily_series)
        for (model, icc, n, rep) in tasks_grid
    ]
    # biggest first: better load balance on a fixed pool
    tasks.sort(key=lambda t: -t[2])

    manifest = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": args.models, "iccs": args.iccs, "sizes": args.sizes,
        "betas": args.betas, "beta_r0_nominal": C.BETA_R0_NOMINAL,
        "rep_budget": {str(k): v for k, v in C.REP_BUDGET.items()},
        "sigma": C.SIGMA, "gamma": C.GAMMA, "n_seed_infected": C.N_SEED_INFECTED,
        "takeoff_fraction": C.TAKEOFF_FRACTION, "rng_seed": C.RNG_SEED,
        "max_days": args.max_days, "workers": args.workers,
        "n_tasks": len(tasks),
    }
    (C.OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

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
