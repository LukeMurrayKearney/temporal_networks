"""Check whether the dense-beta jobs have finished on the cluster, and if they
have, copy the results down and rebuild the summary tables here.

    python3 fetch_dense.py                 # check; fetch if everything is done
    python3 fetch_dense.py --check-only    # report progress, copy nothing
    python3 fetch_dense.py --force         # copy whatever has finished so far
    python3 fetch_dense.py --generations   # the same for the generation-count
                                           # backfill (gen_counts/ side files)

The cluster login and path are those of fetch_results.sh:

    REMOTE=user@cluster  (default: the login in fetch_results.sh's header)
    RPATH=temporal_networks/seir_sweep   (relative to the remote home)

What "finished" means
---------------------
Every new (network realisation, beta) unit writes a done-marker when its ten
epidemics are complete. The full list of units is known here, from
``dense_betas.json`` and the grid in ``config.py`` / ``config_structure.py``:
for each architecture, every network realisation of every model (and ICC)
times every new beta. The check lists the cluster's ``_done`` directories
over one SSH connection, compares them with that list, and asks slurm which
``dense_*`` jobs are still queued or running. The extension is finished when
every unit has its marker and no dense job is left in the queue.

If no job is queued but markers are missing, a job ran out of time or failed;
the script prints the ``sbatch`` line that resumes each incomplete job (finished
units are skipped, so resubmitting costs only what is missing).

What is copied
--------------
``./fetch_results.sh series``: the summary tables, the per-(cell, beta) shards,
the done-markers and the daily series of both sweeps, new files only. The dense
runs write no transmission logs (see run_dense.slurm), so there are none to
fetch. The jobs ran with ``--no-collect``, so the summary tables on the cluster
may not include them; they are rebuilt here from the shards with ``--collect``,
and the number of runs at the new betas is then checked against what was
expected.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import config as C
import config_structure as CS

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "dense_betas.json").read_text())
SIZE = PLAN["size"]
SIM_REPS = PLAN["sim_reps"]

REMOTE = os.environ.get("REMOTE", "matbvh@avon.scrtp.warwick.ac.uk")
RPATH = os.environ.get("RPATH", "temporal_networks/seir_sweep")
# The same control socket as fetch_results.sh, so its transfers reuse this
# script's authenticated connection and a 2FA cluster prompts only once.
CTL_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / ".fetch_results_ssh"
SSH = ["ssh", "-o", "ControlMaster=auto",
       "-o", f"ControlPath={CTL_DIR}/%r@%h:%p", "-o", "ControlPersist=10m"]


def btag(b: float) -> str:
    """The key the runners give a beta in done-marker and shard names."""
    return f"{int(round(b * 1e6)):08d}"


GEN = False          # set by --generations: check gen_counts/ side files instead


def unit_name(cell: str, b: float) -> str:
    return f"{cell}__beta{btag(b)}" + (".csv" if GEN else ".done")


def job_name(arm, icc, chunk) -> str:
    return (("gen_" if GEN else "") + f"dense_{arm}"
            + ("" if icc is None else f"_icc{icc!r}") + f"_c{chunk}")


def expected_units() -> dict:
    """{(arm, icc, chunk): set of unit file names} for every dense job."""
    jobs = {}
    for arm, p in PLAN["arms"].items():
        j = p["jobs"]
        temporal = j["split_by_icc"]
        for icc in (j["iccs"] if temporal else [None]):
            if arm == "hh_temporal":
                cells = [C.cell_id(m, icc, SIZE, r) for m in C.MODELS
                         for r in range(C.REP_BUDGET[SIZE]["net_reps"])]
            else:
                cells = [CS.cell_id(*t) for t in CS.grid([arm], None, [icc], [SIZE])
                         if t[2] == icc]
            for c in range(j["n_chunks"]):
                betas = p["betas"][c::j["n_chunks"]]
                jobs[(arm, icc, c)] = {unit_name(cell, b)
                                       for cell in cells for b in betas}
    return jobs


REMOTE_SCRIPT = r'''
cd "$1" || { echo "##NODIR"; exit 0; }
echo "##QUEUE"
squeue -u "$USER" -h -o "%i|%j|%T|%M|%l" 2>/dev/null | grep -E '\|(gen_)?dense_|\|gen_logs' || true
echo "##DONE_MAIN"
ls "outputs/$2" 2>/dev/null
echo "##DONE_STRUCT"
ls "outputs_structure/$2" 2>/dev/null
echo "##SHARDS"
[ "$2" = gen_counts ] && ls outputs/shards outputs_structure/shards 2>/dev/null | grep '^runs_.*__n100000__'
echo "##END"
'''


def remote_state() -> tuple[list[str], set[str], set[str]]:
    CTL_DIR.mkdir(parents=True, exist_ok=True)
    CTL_DIR.chmod(0o700)
    out = subprocess.run(SSH + [REMOTE, "bash", "-l", "-s", "--", RPATH,
                                "gen_counts" if GEN else "_done"],
                         input=REMOTE_SCRIPT, capture_output=True, text=True)
    if out.returncode != 0 or "##END" not in out.stdout:
        if "##NODIR" in out.stdout:
            sys.exit(f"cannot find {RPATH} on {REMOTE}")
        sys.exit(f"could not read the cluster's state over ssh ({REMOTE}):\n"
                 f"{out.stderr.strip()}")
    section, queue, done, shards = None, [], set(), set()
    for line in out.stdout.splitlines():
        if line.startswith("##"):
            section = line[2:]
            continue
        if section == "QUEUE" and line.strip():
            queue.append(line.strip())
        elif section in ("DONE_MAIN", "DONE_STRUCT") and line.strip():
            done.add(line.strip())
        elif section == "SHARDS" and line.strip():
            shards.add(line.strip()[len("runs_"):])     # -> "<cell>__beta<tag>.csv"
    return queue, done, shards


def report(jobs: dict, queue: list[str], done: set[str]) -> list[tuple]:
    """Print progress per job; return the incomplete jobs."""
    queued = defaultdict(list)
    for q in queue:
        jid, name, state, used, limit = (q.split("|") + [""] * 5)[:5]
        queued[name].append(f"{state.lower()} (job {jid}, {used} of {limit})")

    print(f"{'job':38s} {'units done':>14s}   queue")
    incomplete, tot_done, tot = [], 0, 0
    for (arm, icc, chunk), units in jobs.items():
        name = job_name(arm, icc, chunk)
        n_done = len(units & done)
        tot_done += n_done; tot += len(units)
        q = "; ".join(queued.get(name, [])) or "-"
        flag = "" if n_done == len(units) else "  <-- incomplete"
        print(f"{name:38s} {n_done:6d}/{len(units):<7d}   {q}{flag}")
        if n_done < len(units):
            incomplete.append((arm, icc, chunk, name, bool(queued.get(name))))
    print(f"{'total':38s} {tot_done:6d}/{tot:<7d}   "
          f"({100 * tot_done / tot:.1f}% of (network, beta) units; "
          f"{len(queue)} dense job(s) in the queue)")
    return incomplete


def resubmit_lines(incomplete: list[tuple]) -> list[str]:
    lines = []
    for arm, icc, chunk, name, in_queue in incomplete:
        if in_queue:
            continue
        j = PLAN["arms"][arm]["jobs"]
        lines.append(
            f"sbatch --job-name={name} --cpus-per-task={j['cpus']} "
            f"--time={j['hours']}:00:00 --export=ALL,DENSE_ARM={arm},"
            f"DENSE_ICC={'' if icc is None else repr(icc)},DENSE_CHUNK={chunk},"
            f"DENSE_NCHUNKS={j['n_chunks']},DENSE_LOGS=0,"
            f"DENSE_GENERATIONS={1 if GEN else 0} run_dense.slurm")
    return lines


def fetch_and_collect(rsync_extra: list[str]) -> None:
    env = dict(os.environ, REMOTE=REMOTE, RPATH=RPATH)
    print("\n== copying from the cluster: ./fetch_results.sh series")
    r = subprocess.run(["bash", str(HERE / "fetch_results.sh"), "series", *rsync_extra],
                       cwd=HERE, env=env)
    if r.returncode != 0:
        sys.exit("fetch_results.sh reported failed transfers; nothing rebuilt. "
                 "Run this script again to retry (only missing files are copied).")
    if "--dry-run" in rsync_extra:
        print("--dry-run: nothing copied, summary tables not rebuilt")
        return
    if GEN:
        return                       # side files need no collecting
    for script in ("run_sweep.py", "run_structure_sweep.py"):
        print(f"\n== rebuilding the summary tables: python3 {script} --collect")
        r = subprocess.run([sys.executable, script, "--collect"], cwd=HERE)
        if r.returncode != 0:
            sys.exit(f"{script} --collect failed")


def verify_local() -> bool:
    """Runs at the new betas in the local summary tables, against what was expected."""
    import pandas as pd
    jobs = expected_units()
    ok = True
    print("\n== runs at the new betas in the local summary tables")
    for arm, p in PLAN["arms"].items():
        path = (C.OUT_DIR if arm == "hh_temporal" else CS.OUT_DIR) / "runs_summary.csv"
        r = pd.read_csv(path, usecols=["cell", "beta"], low_memory=False)
        r = r[r["cell"].str.contains(f"__n{SIZE}__")]
        if arm != "hh_temporal":
            r = r[r["cell"].str.startswith(arm + "__")]
        new = {btag(b) for b in p["betas"]}
        got = int(r["beta"].map(lambda b: btag(b) in new).sum())
        want = SIM_REPS * sum(len(u) for (a, _i, _c), u in jobs.items() if a == arm)
        ok &= got == want
        print(f"  {arm:14s} {got:7d} of {want:7d} runs  "
              f"{'ok' if got == want else '<-- missing'}")
    return ok


def verify_generations(jobs: dict) -> bool:
    """Dense units with side files here, and how many replays matched the record."""
    import pandas as pd
    have = set()
    for d in (C.OUT_DIR, CS.OUT_DIR):
        if (d / "gen_counts").exists():
            have |= {f.name for f in (d / "gen_counts").iterdir()}
    want = set().union(*jobs.values())
    print(f"\n== generation counts here: {len(want & have)} of {len(want)} dense "
          "(network, beta) units")
    exact = tot = 0
    for d in (C.OUT_DIR, CS.OUT_DIR):
        for f in sorted((d / "gen_counts").glob("*.csv")) if (d / "gen_counts").exists() else []:
            if f.name in want:
                t = pd.read_csv(f, usecols=["exact"])
                exact += int(t["exact"].sum()); tot += len(t)
    if tot:
        print(f"   replayed runs identical to the recorded run: {exact} of {tot} "
              f"({100 * exact / tot:.1f}%); the rest are fresh replicates on the same "
              "network, still valid for R0")
    return want <= have


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-only", action="store_true", help="report, copy nothing")
    ap.add_argument("--force", action="store_true",
                    help="copy whatever has finished, even if jobs remain")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what fetch_results.sh would copy, copy nothing")
    ap.add_argument("--generations", action="store_true",
                    help="check and fetch the generation-count backfill instead")
    args = ap.parse_args()
    global GEN
    GEN = args.generations

    jobs = expected_units()
    print(f"cluster: {REMOTE}:{RPATH}\n")
    queue, done, shards = remote_state()
    incomplete = report(jobs, queue, done)
    if GEN:
        dense = set().union(*jobs.values())
        orig = shards - dense
        n_orig = len(orig & done)
        print(f"original grid (backfilled from logs by backfill_generations_logs.slurm): "
              f"{n_orig} of {len(orig)} (network, beta) units have generation counts")
        if n_orig < len(orig) and not any("gen_logs" in q for q in queue):
            print("   -> on the cluster: sbatch backfill_generations_logs.slurm "
                  "(units without logs, if any remain, are listed in its output)")

    if incomplete:
        running = [x for x in incomplete if x[4]]
        stopped = resubmit_lines(incomplete)
        if running:
            print(f"\nNot finished: {len(running)} incomplete job(s) still queued or running.")
        if stopped:
            print(f"\n{len(stopped)} incomplete job(s) are no longer in the queue -- they ran "
                  "out of time or failed. Resume them on the cluster, from seir_sweep/:")
            for line in stopped:
                print("  " + line)
        if not args.force:
            print("\nNothing copied. Run again later, or pass --force to copy what has "
                  "finished so far.")
            return
        print("\n--force: copying the units that have finished so far.")
    else:
        print("\nAll dense jobs have finished.")

    if args.check_only:
        print("--check-only: nothing copied.")
        return
    fetch_and_collect(["--dry-run"] if args.dry_run else [])
    if not args.dry_run:
        complete = verify_generations(jobs) if GEN else verify_local()
        print("\nDone." if complete else
              "\nSome runs are missing locally (expected if --force was used); run this "
              "script again once the cluster has finished.")


if __name__ == "__main__":
    main()
