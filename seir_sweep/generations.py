"""Generation counts per run, and the reproduction number built from them.

R0 in this project is the average number of secondary cases caused by the
people the index cases infect. Index cases are generation 0, the people they
infect generation 1, and so on, so

    R0 = (infections in generation 2) / (infections in generation 1),

pooled over every run at a given beta (the data-driven-networks paper's
sum(I3)/sum(I2), whose numbering starts the seeds at generation 1). Index cases
are chosen at random; the people they infect were reached along contacts, so
their offspring is the reproduction number the epidemic actually runs at.

Every run now records ``n_gen1`` and ``n_gen2``. Runs made before that are
backfilled into side files, one per (network realisation, beta), next to the
shards:

    <outputs dir>/gen_counts/<cell>__beta<tag>.csv
        cell, beta, beta_tag, sim_rep, n_gen1, n_gen2, method,
        final_size_recorded, final_size_replay, exact

``method`` is ``log`` when the counts were read from the run's transmission log,
``record`` when the run had no transmissions at all (so both counts are zero),
and ``replay`` when the run was simulated again from its recorded seeds. A
replayed run is checked against the recorded final size: ``exact`` says whether
it is the same epidemic. Static arms replay exactly; temporal arms do when the
network realisation's betas are replayed in the order they were first run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SIDE_COLUMNS = ["cell", "beta", "beta_tag", "sim_rep", "n_gen1", "n_gen2", "method",
                "final_size_recorded", "final_size_replay", "exact"]


def btag(beta: float) -> str:
    return f"{int(round(float(beta) * 1e6)):08d}"


def gen_counts(tdf: pd.DataFrame) -> tuple[int, int]:
    """(infections in generation 1, infections in generation 2) of one run."""
    if tdf is None or tdf.empty:
        return 0, 0
    g = tdf["target_generation"].to_numpy()
    return int((g == 1).sum()), int((g == 2).sum())


def gen_row(tdf: pd.DataFrame) -> dict:
    """The per-run columns every run records."""
    n1, n2 = gen_counts(tdf)
    return {"n_gen1": n1, "n_gen2": n2,
            "r0_gen": (n2 / n1) if n1 > 0 else float("nan")}


def side_path(out_dir: Path, cell: str, beta: float) -> Path:
    return Path(out_dir) / "gen_counts" / f"{cell}__beta{btag(beta)}.csv"


def recorded_rows(out_dir: Path, cell: str, beta: float) -> pd.DataFrame | None:
    """The recorded runs of one (cell, beta), from its shard."""
    f = Path(out_dir) / "shards" / f"runs_{cell}__beta{btag(beta)}.csv"
    if not f.exists():
        return None
    return pd.read_csv(f, usecols=["sim_rep", "final_size", "n_seeds"])


def write_side(out_dir: Path, cell: str, beta: float, records: list[dict]) -> Path:
    f = side_path(out_dir, cell, beta)
    f.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.insert(0, "beta_tag", btag(beta))
    df.insert(0, "beta", beta)
    df.insert(0, "cell", cell)
    tmp = f.with_suffix(".tmp")
    df[SIDE_COLUMNS].to_csv(tmp, index=False)
    tmp.replace(f)                       # never leave a half-written side file
    return f


def from_logs(out_dir: Path, cell: str, beta: float) -> list[dict] | None:
    """Counts for every recorded run of one (cell, beta) from its logs, or None
    if any run that transmitted has no log on disk."""
    rec = recorded_rows(out_dir, cell, beta)
    if rec is None or not len(rec):
        return None
    tlog = Path(out_dir) / "transmission_logs"
    out = []
    for r in rec.itertuples():
        f = tlog / f"{cell}__beta{btag(beta)}__rep{int(r.sim_rep)}.csv.gz"
        if f.exists():
            n1, n2 = gen_counts(pd.read_csv(f, usecols=["target_generation"]))
            method = "log"
        elif int(r.final_size) <= int(r.n_seeds):
            n1, n2, method = 0, 0, "record"      # nobody was infected
        else:
            return None
        out.append({"sim_rep": int(r.sim_rep), "n_gen1": n1, "n_gen2": n2,
                    "method": method, "final_size_recorded": int(r.final_size),
                    "final_size_replay": np.nan, "exact": True})
    return out


def backfill_from_logs(out_dir: Path, cell: str, betas: list[float]) -> dict:
    """Write side files for every beta of one cell whose logs are complete."""
    done = skipped = present = 0
    for b in betas:
        if side_path(out_dir, cell, b).exists():
            present += 1
            continue
        recs = from_logs(out_dir, cell, b)
        if recs is None:
            skipped += 1
            continue
        write_side(out_dir, cell, b, recs)
        done += 1
    return {"cell": cell, "status": "ok", "written": done, "no_logs": skipped,
            "already": present}


def load_side_files(out_dir: Path) -> pd.DataFrame:
    files = sorted((Path(out_dir) / "gen_counts").glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=SIDE_COLUMNS)
    return pd.concat([pd.read_csv(f, dtype={"beta_tag": str}) for f in files],
                     ignore_index=True)


def pooled_r0(n1: np.ndarray, n2: np.ndarray) -> tuple[float, float]:
    """Pooled ratio sum(n2)/sum(n1) and its delta-method standard error."""
    n1, n2 = np.asarray(n1, float), np.asarray(n2, float)
    s1 = n1.sum()
    if s1 <= 0:
        return float("nan"), float("nan")
    r = n2.sum() / s1
    se = float(np.sqrt(((n2 - r * n1) ** 2).sum()) / s1) if n1.size > 1 else float("nan")
    return float(r), se


def attach(runs: pd.DataFrame, out_dirs) -> pd.DataFrame:
    """Add ``n_gen1``, ``n_gen2`` and ``r0_gen`` to a runs table.

    Runs that recorded the counts themselves keep them; the rest are filled from
    the side files in each of ``out_dirs``. Runs with neither stay NaN.
    """
    side = pd.concat([load_side_files(d) for d in out_dirs], ignore_index=True)
    key = (runs["cell"] + "|" + runs["beta"].map(btag) + "|"
           + runs["sim_rep"].astype(int).astype(str))
    for c in ("n_gen1", "n_gen2"):
        if c not in runs:
            runs[c] = np.nan
    if len(side):
        skey = (side["cell"] + "|" + side["beta_tag"].astype(str).str.zfill(8) + "|"
                + side["sim_rep"].astype(int).astype(str))
        s = side.set_index(skey)[["n_gen1", "n_gen2"]]
        s = s[~s.index.duplicated(keep="last")]
        for c in ("n_gen1", "n_gen2"):
            runs[c] = runs[c].fillna(key.map(s[c]))
    for c in ("n_gen1", "n_gen2"):
        runs[c] = pd.to_numeric(runs[c], errors="coerce").astype(float)
    n1 = runs["n_gen1"].where(runs["n_gen1"] > 0)       # 0 -> NaN, no 0/0
    runs["r0_gen"] = runs["n_gen2"] / n1
    return runs


def r0_points(d: pd.DataFrame, complete_only: bool = True) -> pd.DataFrame:
    """R0 at each beta of a runs table: pooled n_gen2 / n_gen1 over its runs.

    With ``complete_only`` a beta is left out unless every one of its runs has
    generation counts, so a partly backfilled beta never stands in for the whole.
    """
    rows = []
    for b, g in d.groupby("beta"):
        ok = g["n_gen1"].notna() & g["n_gen2"].notna()
        if not ok.any() or (complete_only and not ok.all()):
            continue
        r, se = pooled_r0(g.loc[ok, "n_gen1"], g.loc[ok, "n_gen2"])
        if np.isfinite(r):
            rows.append((float(b), r, se, int(ok.sum())))
    return pd.DataFrame(rows, columns=["beta", "r0", "sem", "n_runs"])
