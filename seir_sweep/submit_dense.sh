#!/bin/bash
# ---------------------------------------------------------------------------
# Submit every job of the dense-beta extension (see dense_betas.py and
# run_dense.slurm). Run on the cluster, from seir_sweep/:
#
#     bash submit_dense.sh                  # all four architectures
#     DRY_RUN=1 bash submit_dense.sh        # print the sbatch commands only
#     ARMS="hh_static nohh_static" bash submit_dense.sh   # a subset
#     DENSE_LOGS=1 bash submit_dense.sh     # keep transmission logs (~120 GB)
#
# The plan -- betas, how many chunks, CPUs and time per job -- is read from
# dense_betas.json, so it cannot drift from the betas themselves:
#
#     temporal arm : one job per ICC x chunk, 20 CPUs (one per network realisation)
#     static arm   : one job, 20 CPUs
#
# 14 jobs in all, about 5,500 CPU-hours; the longest jobs request 37 h.
#
# When every job has finished, rebuild the two summary tables once:
#     python run_sweep.py --collect
#     python run_structure_sweep.py --collect
# Jobs run with --no-collect so that two finishing together cannot both rewrite
# runs_summary.csv. A job that ran out of time is resumed by submitting it again.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
mkdir -p slurm_logs

PLAN=$(python - "${ARMS:-}" <<'PY'
import json, sys
plan = json.load(open("dense_betas.json"))["arms"]
want = sys.argv[1].split() or list(plan)
for arm in want:
    j = plan[arm]["jobs"]
    iccs = [repr(i) for i in j["iccs"]] if j["split_by_icc"] else [""]
    for icc in iccs:
        for c in range(j["n_chunks"]):
            print(arm, icc or "-", c, j["n_chunks"], j["cpus"], j["hours"])
PY
)

n=0
while read -r arm icc chunk nchunks cpus hours; do
    [ "$icc" = "-" ] && icc=""
    name="dense_${arm}${icc:+_icc$icc}_c${chunk}"
    cmd=(sbatch --job-name="$name" --cpus-per-task="$cpus" --time="${hours}:00:00"
         --export="ALL,DENSE_ARM=$arm,DENSE_ICC=$icc,DENSE_CHUNK=$chunk,DENSE_NCHUNKS=$nchunks,DENSE_LOGS=${DENSE_LOGS:-0}"
         run_dense.slurm)
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
    n=$((n + 1))
done <<< "$PLAN"
if [ "${DRY_RUN:-0}" = 1 ]; then echo "$n job(s) listed, none submitted"; else echo "$n job(s) submitted"; fi
