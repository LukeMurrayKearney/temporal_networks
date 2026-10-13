"""Choose extra beta values that make R0 in [0.2, 6] five times as densely
sampled, for every architecture, and plan the cluster jobs that run them.

    python dense_betas.py            # writes dense_betas.json and prints the plan

Needs the completed n=100k results of both sweeps (runs_summary.csv), because
the new betas are placed using each architecture's measured R0(beta) curve.

How the betas are chosen
------------------------
The four architectures turn beta into R0 very differently (a factor of about
eight at R0 = 2), so one shared list would put most new points outside some
arm's 0.2-6 window. Each architecture therefore gets its own list, shared by
every contact model and persistence level within it, so the within-arm
comparisons (NB vs log-GMM, ICC 0 / 0.365 / 0.95) stay paired beta-for-beta.

For an arm with ``n`` existing betas whose mean measured R0 lies in
[R0_LO, R0_HI]:

1. lay ``DENSITY * n`` targets evenly in R0 across [R0_LO, R0_HI];
2. remove, for each existing beta, the target nearest its R0, which leaves
   ``(DENSITY - 1) * n`` new targets filling the gaps between the old points;
3. invert each variant's measured R0(beta) curve at every target and take the
   geometric mean over the arm's variants. The curve is the mean measured R0 at
   each grid beta, made non-decreasing and interpolated linearly in log beta and
   log R0; below the smallest grid beta it is extended with R0 proportional to
   beta, which is what transmission does when every contact transmits with a
   small probability. (The Hill fits used elsewhere are poor at small beta and
   are not used here.)
4. round to three significant figures, and drop anything that collides with an
   existing beta or with another new one.

The achieved R0 of a new beta will differ a little between the arm's variants
and from its target (the spread across variants at a fixed beta is about
10-20%), exactly as it does for the existing grid.

Every new beta is run with the existing replication: all of the arm's network
realisations (10 per model and ICC) on the same seeds as before, and 10
epidemics each. Seeds derive from the grid coordinates and the beta value, so
the new points are paired with the old ones on the same networks.

Cost and job plan
-----------------
The cost of a run is read off the recorded ``sim_seconds`` of the existing grid
(the cluster's own timings), interpolated in log beta per arm. Jobs are split
so that every job's network realisations run in a single wave (one CPU per
realisation) and finish well inside the 48 h limit: by ICC for the temporal
arms, and by interleaved chunks of the beta list (betas i, i+k, i+2k, ...),
which balances cost because cost varies smoothly with beta.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import unified_common as U   # loads both sweeps' runs_summary.csv

HERE = Path(__file__).resolve().parent
OUT = HERE / "dense_betas.json"

R0_LO, R0_HI = 0.2, 6.0
DENSITY = 5                  # points in [R0_LO, R0_HI] after / before
SIZE = 100_000
SIM_REPS = 10
SIG_FIGS = 3
MAX_JOB_HOURS = 30.0         # target per job, leaving margin under the 48 h limit
TIME_SAFETY = 1.35           # requested time = estimate x this, rounded up


def round_sig(x: float, sig: int = SIG_FIGS) -> float:
    return float(f"{x:.{sig}g}")


def btag(b: float) -> int:
    """The integer the runners key a beta by (done markers, seeds, file names)."""
    return int(round(b * 1e6))


class MeasuredCurve:
    """Monotone R0(beta) for one variant from its measured grid means."""

    def __init__(self, beta, r0):
        o = np.argsort(beta)
        self.lb = np.log(np.asarray(beta, float)[o])
        r = np.maximum.accumulate(np.asarray(r0, float)[o])
        # strictly increasing for the inverse; ties come from the running max
        self.lr = np.log(r) + np.arange(r.size) * 1e-9

    def predict(self, b):
        lb = np.log(b)
        if lb < self.lb[0]:                     # R0 proportional to beta below the grid
            return float(np.exp(self.lr[0] + lb - self.lb[0]))
        return float(np.exp(np.interp(lb, self.lb, self.lr)))

    def beta_at(self, r0):
        lr = np.log(r0)
        if lr < self.lr[0]:
            return float(np.exp(self.lb[0] + lr - self.lr[0]))
        if lr > self.lr[-1]:
            return float("nan")
        return float(np.exp(np.interp(lr, self.lr, self.lb)))


def plan_arm(arm: str) -> dict:
    r = U.runs[(U.runs["n_req"] == SIZE) & (U.runs["arm"] == arm)]
    variants = sorted(r["variant"].unique())
    fits = []
    for v in variants:
        g = r[r["variant"] == v].groupby("beta")["r0_measured"].mean()
        fits.append(MeasuredCurve(g.index.to_numpy(float), g.to_numpy()))

    grid = r.groupby("beta")["r0_measured"].mean()
    existing = grid[(grid >= R0_LO) & (grid <= R0_HI)]
    n_old = len(existing)

    targets = list(np.linspace(R0_LO, R0_HI, DENSITY * n_old))
    for r0 in existing.to_numpy():
        targets.pop(int(np.argmin(np.abs(np.array(targets) - r0))))

    taken = {btag(b) for b in grid.index}
    new = []
    for t in targets:
        betas = [f.beta_at(t) for f in fits]
        if not all(np.isfinite(betas)) or min(betas) <= 0:
            raise RuntimeError(f"{arm}: cannot invert R0(beta) at R0 = {t:.3f}")
        b = round_sig(float(np.exp(np.mean(np.log(betas)))))
        if btag(b) in taken:
            continue
        taken.add(btag(b))
        new.append({"beta": b, "target_r0": float(t),
                    "fit_r0_range": [float(min(f.predict(b) for f in fits)),
                                     float(max(f.predict(b) for f in fits))]})
    new.sort(key=lambda d: d["beta"])

    # cost: recorded mean seconds per run, interpolated in log beta
    sec = r.groupby("beta")["sim_seconds"].mean()
    lb = np.log(sec.index.to_numpy(float))
    per_run = {d["beta"]: float(np.interp(np.log(d["beta"]), lb, sec.to_numpy()))
               for d in new}
    cells = r["cell"].unique()
    iccs = sorted({U.vparts(v)[2] for v in variants}, key=lambda x: (x is None, x))
    temporal = arm.endswith("temporal")
    cells_per_icc = len(cells) // max(len(iccs), 1)

    # per-cell hours for the whole new list, then chunks so a job fits
    per_cell_h = sum(per_run.values()) * SIM_REPS / 3600
    n_chunks = max(1, math.ceil(per_cell_h / MAX_JOB_HOURS))
    job_cells = cells_per_icc if temporal else len(cells)
    chunk_h = max(sum(per_run[b] for b in sorted(per_run)[i::n_chunks]) * SIM_REPS / 3600
                  for i in range(n_chunks))
    hours = max(1, math.ceil(chunk_h * TIME_SAFETY + 0.5))
    return {
        "variants": variants,
        "existing_betas_in_window": [float(b) for b in existing.index],
        "existing_r0_in_window": [float(x) for x in existing.to_numpy()],
        "n_existing_in_window": n_old,
        "new": new,
        "betas": [d["beta"] for d in new],
        "cost": {
            "cpu_hours": sum(per_run.values()) * SIM_REPS * len(cells) / 3600,
            "hours_per_cell_all_new_betas": per_cell_h,
        },
        "jobs": {
            "split_by_icc": temporal,
            "iccs": [i for i in iccs if i is not None],
            "n_chunks": n_chunks,
            "cpus": job_cells,
            "hours": hours,
            "n_jobs": (len(iccs) if temporal else 1) * n_chunks,
        },
    }


def main() -> None:
    plan = {"r0_window": [R0_LO, R0_HI], "density": DENSITY, "size": SIZE,
            "sim_reps": SIM_REPS, "arms": {}}
    for arm in U.ARM_ORDER:
        plan["arms"][arm] = plan_arm(arm)
    OUT.write_text(json.dumps(plan, indent=1))

    tot_cpu = tot_jobs = 0
    for arm, p in plan["arms"].items():
        j, c = p["jobs"], p["cost"]
        tot_cpu += c["cpu_hours"]; tot_jobs += j["n_jobs"]
        print(f"{arm:14s} in-window betas {p['n_existing_in_window']:3d} -> "
              f"+{len(p['betas']):3d} new  ({p['betas'][0]:g} .. {p['betas'][-1]:g})"
              f" | {c['cpu_hours']:7.0f} CPU-h | {j['n_jobs']} job(s) x {j['cpus']} CPUs"
              f" x {j['hours']} h ({c['hours_per_cell_all_new_betas']:.1f} h per"
              f" realisation over {j['n_chunks']} chunk(s))")
    print(f"{'total':14s} {tot_cpu:7.0f} CPU-h in {tot_jobs} jobs; wrote {OUT.name}")


if __name__ == "__main__":
    main()
