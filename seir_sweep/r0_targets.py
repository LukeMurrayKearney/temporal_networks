"""The beta at which each variant reaches a target R0, read off a curve fitted to
the R0 measured in the simulations, and a set of runs made at exactly that beta.

"At R0 = 2" in the analysis means: at the beta where the variant's beta -> R0
curve, fitted to the R0 measured in its runs, reaches 2 (generations.py,
fit_beta_r0; R0 is generation-2 over generation-1 infections). Scalar outcomes
are interpolated to that beta from the simulated betas either side of it. The
transmission-tree and household figures need real runs with transmission logs,
and the nearest simulated beta that has logs can sit 0.1-0.2 off the target on
the curve. This script places 100 runs (every network realisation x 10 epidemics,
with transmission logs and daily series) at the fitted beta itself, per variant.

On the cluster, from seir_sweep/, once the generation counts are complete
(backfill_generations_logs.slurm and GENERATIONS=1 bash submit_dense.sh):

    python r0_targets.py                 # fit, write r0_targets.json, print the plan
    DRY_RUN=1 bash submit_r0_targets.sh  # the sbatch lines
    bash submit_r0_targets.sh            # one job per variant (16), ~120 CPU-h

Here, afterwards:

    python3 r0_targets.py --fetch        # check the jobs; copy their runs down
                                         # and rebuild the summary tables

The fit that chose a beta is kept in r0_targets.json and is not redone: running
the script again prints the plan, and a fresh fit for comparison, without
changing it (``--refit`` replaces it, after which the runs must be made again).
The betas listed there are left out of that comparison fit, so the runs made at
them cannot move their own target. The analysis fits every beta it has,
including these, so its fitted beta can differ from the one here by a fraction
of a percent; the runs here are still the nearest simulated beta to it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import config_structure as CS
import generations as G

HERE = Path(__file__).resolve().parent
PLAN_FILE = HERE / "r0_targets.json"
SIZE = 100_000
HOURS = {"temporal": 10, "static": 3}       # per job; runs at R0 ~ 2 take <= ~1.5 h


def icc_tag(icc) -> str:
    return "na" if icc is None else f"{icc:g}"


def variants() -> list[dict]:
    """Every variant at n = 100k, with its arm, model, ICC and cells."""
    out = []
    for m in C.MODELS:
        for i in C.ICC_TARGETS:
            out.append({"arm": "hh_temporal", "model": m, "icc": i, "out": C.OUT_DIR,
                        "cells": [C.cell_id(m, i, SIZE, r)
                                  for r in range(C.REP_BUDGET[SIZE]["net_reps"])]})
    for arm in CS.ARMS:
        for m in CS.MODELS:
            for i in CS.iccs_for_arm(arm, None):
                out.append({"arm": arm, "model": m, "icc": i, "out": CS.OUT_DIR,
                            "cells": [CS.cell_id(arm, m, i, SIZE, r)
                                      for r in range(CS.REP_BUDGET[SIZE]["net_reps"])]})
    for v in out:
        v["key"] = f"{v['arm']}|{v['model']}|{icc_tag(v['icc'])}"
        v["temporal"] = v["arm"].endswith("temporal")
    return out


def load_runs(v: dict) -> pd.DataFrame:
    """Every recorded run of one variant, from its shards, with generation counts."""
    want = {"cell", "beta", "sim_rep", "n_gen1", "n_gen2"}
    frames = []
    for cell in v["cells"]:
        for f in (v["out"] / "shards").glob(f"runs_{cell}__beta*.csv"):
            frames.append(pd.read_csv(f, usecols=lambda c: c in want))
    if not frames:
        return pd.DataFrame(columns=sorted(want))
    return G.attach(pd.concat(frames, ignore_index=True), (C.OUT_DIR, CS.OUT_DIR))


def fit_variant(v: dict, exclude_tags: set[str]) -> dict:
    d = load_runs(v)
    if not len(d):
        return {"status": "no runs"}
    d = d[~d["beta"].map(G.btag).isin(exclude_tags)]
    per_beta = (d["n_gen1"].notna() & d["n_gen2"].notna()).groupby(d["beta"]).all()
    pts = G.r0_points(d)
    fit = G.fit_beta_r0(pts["beta"], pts["r0"])
    res = {"betas": int(len(per_beta)), "betas_with_counts": int(per_beta.sum()),
           "points_fitted": 0 if fit is None else fit.n_points}
    if fit is None:
        return {**res, "status": "no fit"}
    return {**res, "status": "ok", "fit": fit,
            "params_a_b_c": list(fit.params), "rms": fit.rms,
            "beta_range": [fit.beta_lo, fit.beta_hi]}


def round_beta(b: float) -> float:
    return float(f"{b:.4g}")


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan(targets: list[float], refit: bool, allow_partial: bool) -> None:
    old = json.loads(PLAN_FILE.read_text()) if PLAN_FILE.exists() else None
    if old and not refit:
        print(f"{PLAN_FILE.name} exists (written {old['written']}); it is not changed "
              "without --refit. Its plan, against a fresh fit that leaves its own "
              "betas out:\n")
    rows, jobs, incomplete = [], [], []
    for v in variants():
        excl = set()
        if old and v["key"] in old["variants"]:
            excl = {G.btag(j["beta"]) for j in old["variants"][v["key"]]["jobs"]}
        r = fit_variant(v, excl)
        row = {"variant": v["key"], "betas with counts": f"{r.get('betas_with_counts', 0)}"
               f"/{r.get('betas', 0)}", "fitted": r.get("points_fitted", 0)}
        if r.get("betas_with_counts", 0) < r.get("betas", 0):
            incomplete.append(v["key"])
        if r["status"] == "ok":
            row["rms"] = round(r["rms"], 3)
            for t in targets:
                b = r["fit"].beta_at(t)
                row[f"beta @ R0={t:g}"] = round_beta(b) if np.isfinite(b) else np.nan
                if old and v["key"] in old["variants"]:
                    was = [j["beta"] for j in old["variants"][v["key"]]["jobs"]
                           if j["target_r0"] == t]
                    row[f"planned @ {t:g}"] = was[0] if was else np.nan
        else:
            row["rms"] = r["status"]
        rows.append(row)
        if r["status"] == "ok":
            for t in targets:
                b = r["fit"].beta_at(t)
                if np.isfinite(b):
                    jobs.append({"variant": v, "target_r0": t, "beta": round_beta(b),
                                 "fit": r})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nfit: R0 - 1 = c + a log1p(b beta) over the betas whose measured R0 is in "
          f"[{G.FIT_R0_MIN}, {G.FIT_R0_MAX}); rms = scatter of those points about it.")

    if old and not refit:
        return
    if incomplete and not allow_partial:
        sys.exit(f"\n{len(incomplete)} variant(s) have betas without generation counts "
                 f"({', '.join(incomplete[:4])}{' ...' if len(incomplete) > 4 else ''}). "
                 "Finish the backfill first (README_structure.md, 'How R0 is measured'),"
                 " or pass --allow-partial. Nothing written.")
    out = {"written": time.strftime("%Y-%m-%d %H:%M"), "size": SIZE,
           "fit": "R0 - 1 = c + a log1p(b beta), floored at R0 = 1; betas with measured "
                  f"R0 in [{G.FIT_R0_MIN}, {G.FIT_R0_MAX})",
           "r0": "generation-2 over generation-1 infections, pooled over the runs at a beta",
           "variants": {}}
    for j in jobs:
        v, r = j["variant"], j["fit"]
        e = out["variants"].setdefault(v["key"], {
            "arm": v["arm"], "model": v["model"], "icc": v["icc"],
            "temporal": v["temporal"], "params_a_b_c": r["params_a_b_c"],
            "rms": r["rms"], "points_fitted": r["points_fitted"],
            "beta_range": r["beta_range"], "jobs": []})
        e["jobs"].append({
            "target_r0": j["target_r0"], "beta": j["beta"],
            "name": f"r0t_{v['arm']}_{v['model']}_icc{icc_tag(v['icc'])}_R{j['target_r0']:g}",
            "cpus": len(v["cells"]), "hours": HOURS["temporal" if v["temporal"] else "static"]})
    PLAN_FILE.write_text(json.dumps(out, indent=1) + "\n")
    n = sum(len(e["jobs"]) for e in out["variants"].values())
    print(f"\nwrote {PLAN_FILE.name}: {n} job(s). Next: DRY_RUN=1 bash submit_r0_targets.sh")


# ---------------------------------------------------------------------------
# Checking and fetching (run here, not on the cluster)
# ---------------------------------------------------------------------------

REMOTE_SCRIPT = r'''
cd "$1" || { echo "##NODIR"; exit 0; }
echo "##QUEUE"
squeue -u "$USER" -h -o "%i|%j|%T|%M|%l" 2>/dev/null | grep '|r0t_' || true
echo "##DONE"
ls outputs/_done outputs_structure/_done 2>/dev/null | grep -E "$2" || true
echo "##END"
'''


def expected(plan_: dict) -> dict:
    """{job name: set of done-marker names}."""
    out = {}
    for v in variants():
        e = plan_["variants"].get(v["key"])
        for j in (e["jobs"] if e else []):
            out[j["name"]] = {f"{c}__beta{G.btag(j['beta'])}.done" for c in v["cells"]}
    return out


def fetch(force: bool, check_only: bool) -> None:
    import fetch_dense as FD                  # the cluster login, path and SSH socket
    FD.CTL_DIR.mkdir(parents=True, exist_ok=True)
    FD.CTL_DIR.chmod(0o700)
    rsh = " ".join(FD.SSH)
    src = f"{FD.REMOTE}:{FD.RPATH}"
    print(f"cluster: {src}\n")
    r = subprocess.run(["rsync", "-a", "-e", rsh, f"{src}/{PLAN_FILE.name}", str(PLAN_FILE)])
    if r.returncode != 0:
        sys.exit(f"could not copy {PLAN_FILE.name} from the cluster -- has "
                 "`python r0_targets.py` been run there?")
    plan_ = json.loads(PLAN_FILE.read_text())
    want = expected(plan_)
    tags = sorted({G.btag(j["beta"]) for e in plan_["variants"].values() for j in e["jobs"]})
    pattern = "__beta(" + "|".join(tags) + r")\.done$"
    out = subprocess.run(FD.SSH + [FD.REMOTE, "bash", "-l", "-s", "--", FD.RPATH, pattern],
                         input=REMOTE_SCRIPT, capture_output=True, text=True)
    if out.returncode != 0 or "##END" not in out.stdout:
        sys.exit(f"could not read the cluster's state over ssh:\n{out.stderr.strip()}")
    section, queue, done = None, {}, set()
    for line in out.stdout.splitlines():
        if line.startswith("##"):
            section = line[2:]
        elif section == "QUEUE" and line.strip():
            jid, name, state, used, limit = (line.split("|") + [""] * 5)[:5]
            queue[name] = f"{state.lower()} (job {jid}, {used} of {limit})"
        elif section == "DONE" and line.strip():
            done.add(line.strip())

    print(f"{'job':44s} {'units done':>11s}   queue")
    missing = []
    for name, units in want.items():
        k = len(units & done)
        print(f"{name:44s} {k:5d}/{len(units):<5d}   {queue.get(name, '-')}"
              + ("" if k == len(units) else "  <-- incomplete"))
        if k < len(units) and name not in queue:
            missing.append(name)
    complete = all(units <= done for units in want.values())
    if missing:
        print(f"\n{len(missing)} incomplete job(s) are no longer queued (out of time or "
              "failed). Resubmit on the cluster -- finished units are skipped:\n"
              "  JOBS=\"" + " ".join(missing) + "\" bash submit_r0_targets.sh")
    if check_only:
        return
    if not complete and not force:
        print("\nNot finished; nothing copied (--force copies what has finished).")
        return

    inc = [f"--include=*__beta{t}[._]*" for t in tags]
    failed = 0
    for d in ("outputs", "outputs_structure"):
        for sub in ("shards", "_done", "daily_series", "transmission_logs"):
            (HERE / d / sub).mkdir(parents=True, exist_ok=True)
            print(f"\n== {d}/{sub}")
            r = subprocess.run(["rsync", "-ahP", "--ignore-existing", "-e", rsh, *inc,
                                "--exclude=*", f"{src}/{d}/{sub}/", str(HERE / d / sub) + "/"])
            failed += r.returncode != 0
    if failed:
        sys.exit(f"{failed} transfer(s) failed; run again to retry (only missing files move).")
    for script in ("run_sweep.py", "run_structure_sweep.py"):
        print(f"\n== rebuilding the summary tables: python3 {script} --collect")
        if subprocess.run([sys.executable, script, "--collect"], cwd=HERE).returncode:
            sys.exit(f"{script} --collect failed")
    print("\nruns at the fitted betas in the local summary tables:")
    for v in variants():
        e = plan_["variants"].get(v["key"])
        if not e:
            continue
        r = pd.read_csv(v["out"] / "runs_summary.csv", usecols=["cell", "beta"],
                        low_memory=False)
        for j in e["jobs"]:
            got = int((r["cell"].isin(v["cells"]) & (r["beta"].map(G.btag)
                                                     == G.btag(j["beta"]))).sum())
            want_n = len(v["cells"]) * C.REP_BUDGET[SIZE]["sim_reps"]
            print(f"  {j['name']:44s} beta {j['beta']:<8g} {got:4d}/{want_n} "
                  + ("ok" if got == want_n else "<-- missing"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=float, nargs="+", default=[2.0],
                    help="target R0 values (default: 2)")
    ap.add_argument("--refit", action="store_true",
                    help="replace an existing r0_targets.json")
    ap.add_argument("--allow-partial", action="store_true",
                    help="plan even if some betas still lack generation counts")
    ap.add_argument("--fetch", action="store_true",
                    help="(here) check the r0t_ jobs and copy their runs down")
    ap.add_argument("--check-only", action="store_true", help="with --fetch: copy nothing")
    ap.add_argument("--force", action="store_true",
                    help="with --fetch: copy what has finished even if jobs remain")
    a = ap.parse_args()
    if a.fetch or a.check_only:
        fetch(a.force, a.check_only)
    else:
        plan(a.targets, a.refit, a.allow_partial)


if __name__ == "__main__":
    main()
