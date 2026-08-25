"""Pick the shared beta grid by measuring R0(beta) empirically.

R0 here is the operational one used everywhere else in this codebase: the mean
number of direct offspring of the generation-0 index cases, each introduced
into an otherwise fully susceptible population. Because that quantity is
complete within the seeds' own infectious lifetime, it can be measured with a
short **day-capped** run (``simulate_seir(..., max_days=...)``) rather than a
full outbreak -- orders of magnitude cheaper.

Many index cases are seeded at once (they are independent at this stage, since
almost nothing is infected yet), so one capped run yields many offspring
observations and R0 is estimated tightly.

    python calibrate_beta.py                 # scan and report
    python calibrate_beta.py --write-config  # also rewrite BETAS in config.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config as C                                     # noqa: E402
from builders import build_realisation, prepare_fits   # noqa: E402

sys.path.insert(0, str(C.REPO))
from gillespie_seir import SEIRParams, simulate_seir    # noqa: E402
import outbreak_metrics as om                           # noqa: E402

R0_TARGETS = [0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0]


def measure_r0(real, beta: float, n_seeds: int, max_days: int,
               seed: int) -> float:
    rng = np.random.default_rng(seed)
    seeds = rng.choice(real.n_nodes, size=n_seeds, replace=False).tolist()
    traj = simulate_seir(
        n_nodes=real.n_nodes, edges_fn=real.edges_fn(),
        params=SEIRParams(beta=beta, sigma=C.SIGMA, gamma=C.GAMMA),
        initial_infected=seeds, rng=rng, max_days=max_days,
    )
    tdf = pd.DataFrame([t.__dict__ for t in traj.transmissions])
    if tdf.empty:
        return 0.0
    return om.r0_estimate(tdf, traj.seed_nodes)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-nodes", type=int, default=20_000)
    p.add_argument("--n-seeds", type=int, default=300)
    p.add_argument("--max-days", type=int, default=25,
                   help="enough for the seeds' infectious period to finish")
    p.add_argument("--model", default="loggmm_mm", choices=C.MODELS)
    p.add_argument("--icc", type=float, default=0.365)
    p.add_argument("--betas", nargs="+", type=float,
                   default=[0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.6, 1.2, 2.4])
    p.add_argument("--reps", type=int, default=3,
                   help="independent capped runs averaged per beta")
    p.add_argument("--write-config", action="store_true")
    args = p.parse_args()

    prepare_fits()
    print(f"Building calibration network: {args.model}, ICC {args.icc}, "
          f"n={args.n_nodes:,}")
    real = build_realisation(args.model, args.icc, args.n_nodes, seed=99)
    print(f"  built: {real.n_nodes:,} nodes, rho={real.meta.get('rho')}\n")

    print(f"{'beta':>10}  {'R0':>7}  {'sd':>6}  (n_obs = n_seeds x reps)")
    rows = []
    for i, beta in enumerate(args.betas):
        vals = [measure_r0(real, beta, args.n_seeds, args.max_days,
                           seed=1000 + i * 100 + r) for r in range(args.reps)]
        r0 = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        rows.append({"beta": beta, "r0": r0, "r0_sd_across_reps": sd,
                     "reps": args.reps, "n_seeds": args.n_seeds})
        print(f"{beta:>10.4f}  {r0:>7.3f}  {sd:>6.3f}")

    df = pd.DataFrame(rows)
    # enforce monotonicity: R0 is theoretically non-decreasing in beta, so any
    # dip is estimator noise -- take the running maximum before interpolating.
    df["r0_monotone"] = np.maximum.accumulate(df["r0"].to_numpy())
    df.to_csv(C.OUT_DIR / "beta_r0_calibration.csv", index=False)

    # log-log interpolation of beta for each target R0 (R0 is monotone in beta
    # and saturates, so interpolate on the measured support only)
    lo, hi = df["r0_monotone"].min(), df["r0_monotone"].max()
    print(f"\nMeasured R0 range: {lo:.2f} .. {hi:.2f}")
    chosen = {}
    for target in R0_TARGETS:
        if target < lo or target > hi:
            print(f"  R0={target}: OUT OF RANGE of the scan")
            continue
        beta = float(np.interp(np.log(target), np.log(df["r0_monotone"].to_numpy()),
                               np.log(df["beta"].to_numpy())))
        chosen[target] = float(np.exp(beta))
    print("\nSuggested grid (R0 target -> beta):")
    for t, b in chosen.items():
        print(f"  R0 ~ {t:>5}  ->  beta = {b:.4g}")

    if args.write_config and chosen:
        betas = sorted(round(b, 5) for b in chosen.values())
        nominal = {round(b, 5): t for t, b in chosen.items()}
        cfg = (C.HERE / "config.py").read_text()
        import re
        cfg = re.sub(r"BETAS = \[[^\]]*\]", f"BETAS = {betas}", cfg, count=1)
        cfg = re.sub(r"BETA_R0_NOMINAL = \{[^}]*\}",
                     "BETA_R0_NOMINAL = " + repr(nominal), cfg, count=1)
        (C.HERE / "config.py").write_text(cfg)
        print(f"\nWrote BETAS/BETA_R0_NOMINAL into config.py")


if __name__ == "__main__":
    main()
