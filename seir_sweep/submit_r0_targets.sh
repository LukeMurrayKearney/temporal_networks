#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the runs at each variant's fitted beta for R0 = 2 (r0_targets.py;
# run_r0_target.slurm). Run on the cluster, from seir_sweep/, after
# `python r0_targets.py` has written r0_targets.json:
#
#     DRY_RUN=1 bash submit_r0_targets.sh      # print the sbatch commands only
#     bash submit_r0_targets.sh                # submit all of them (16 jobs)
#     JOBS="r0t_hh_static_nb_iccna_R2" bash submit_r0_targets.sh   # a subset
#
# About 120 CPU-hours in all; the temporal jobs take 1-2 h on 10 CPUs.
# When they have finished, here: python3 r0_targets.py --fetch
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
mkdir -p slurm_logs
[ -f r0_targets.json ] || { echo "r0_targets.json not found: run python r0_targets.py first" >&2; exit 1; }

PLAN=$(python - "${JOBS:-}" <<'PY'
import json, sys
want = set(sys.argv[1].split())
for key, e in json.load(open("r0_targets.json"))["variants"].items():
    for j in e["jobs"]:
        if want and j["name"] not in want:
            continue
        icc = "-" if e["icc"] is None else repr(e["icc"])
        print(j["name"], e["arm"], e["model"], icc, repr(j["beta"]), j["cpus"], j["hours"])
PY
)

n=0
while read -r name arm model icc beta cpus hours; do
    [ -n "$name" ] || continue
    [ "$icc" = "-" ] && icc=""
    cmd=(sbatch --job-name="$name" --cpus-per-task="$cpus" --time="${hours}:00:00"
         --export="ALL,R0T_ARM=$arm,R0T_MODEL=$model,R0T_ICC=$icc,R0T_BETA=$beta"
         run_r0_target.slurm)
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
    n=$((n + 1))
done <<< "$PLAN"
if [ "${DRY_RUN:-0}" = 1 ]; then echo "$n job(s) listed, none submitted"; else echo "$n job(s) submitted"; fi
