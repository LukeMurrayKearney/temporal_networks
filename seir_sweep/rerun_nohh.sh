#!/bin/bash
# ---------------------------------------------------------------------------
# Re-run EVERY no-household simulation of the structure sweep.
#
# Since 2026-10-06 the no-household arms draw community contact from the same
# age-structured fit as the household arms (builders_structure.py); before that
# they used a fit that pooled all ages. This script repeats the whole
# no-household part of the grid with the new construction:
#
#     nohh_static   : 2 models x 1 (no ICC) x 10 networks = 20 realisations
#     nohh_temporal : 2 models x 3 ICCs     x 10 networks = 60 realisations
#     each x 21 betas x 10 epidemics  =  16,800 epidemics at n = 100,000
#
# The household arms (hh_static here, hh_temporal in the main sweep) are not
# touched and not re-run.
#
# BEFORE THE FIRST RUN, delete the old no-household results. The driver refuses
# to start while they are on disk, and prints these same commands:
#
#     cd <this directory>
#     find outputs_structure/_done outputs_structure/shards \
#          outputs_structure/transmission_logs outputs_structure/daily_series \
#          outputs/_analysis_cache \
#          -maxdepth 1 -type f -name '*nohh_*' -delete
#     python3 run_structure_sweep.py --collect
#
# USAGE, from anywhere:
#     bash rerun_nohh.sh                           # every core, keeps the logs
#     JOBS=6 bash rerun_nohh.sh                    # six workers
#     bash rerun_nohh.sh --no-transmission-log     # see DISK below
#     nohup bash rerun_nohh.sh --no-transmission-log > rerun_nohh.log 2>&1 &
# On the cluster:
#     mkdir -p slurm_logs && sbatch rerun_nohh.slurm
#
# Anything after the script name is passed straight to run_structure_sweep.py,
# so a slice can be run on its own, e.g. "--iccs 0.365" or "--models nb".
#
# RESUMING: progress is recorded per (realisation, beta). If the run is
# interrupted, run this script again and it carries on from where it stopped.
# Do NOT delete anything in between.
#
# COST, from the recorded run times of the previous no-household sweep:
#     nohh_temporal ~640 core-hours, nohh_static ~10 core-hours.
#     48 cores: about a day.   8 cores: three to four days.
#     Memory: about 0.75 GB per worker.
#
# DISK: each epidemic writes a transmission log, about 10 GB over the whole
# rerun. The logs are needed only for the transmission-tree and trajectory
# figures, and regen_perrun.py can rebuild the few that those figures use from
# the recorded seeds. Pass --no-transmission-log to skip them; the script
# insists on it when less than MIN_FREE_GB (default 25) is free.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"
PY="${PYTHON:-$(command -v python3 || command -v python)}"

# One BLAS thread per worker: the driver forks JOBS processes, and a threaded
# BLAS in each would oversubscribe the cores. Must be set before python starts.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

keep_logs=1
for a in "$@"; do
    [ "$a" = "--no-transmission-log" ] && keep_logs=0
done
free_gb=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
if [ "$keep_logs" = 1 ] && [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    echo "Only ${free_gb} GB free here; the transmission logs need about 10 GB." >&2
    echo "Run with --no-transmission-log, or set MIN_FREE_GB=0 to go ahead anyway." >&2
    exit 1
fi

echo "host=$(hostname)  workers=${JOBS}  free=${free_gb}GB  python=${PY}"
echo "extra arguments: ${*:-(none)}"
echo "start $(date)"

# The driver checks for leftover age-blind results itself and exits with the
# delete commands if it finds any; it merges the shards into
# outputs_structure/{runs,network}_summary.csv when the pool drains.
"$PY" run_structure_sweep.py \
    --arms nohh_static nohh_temporal \
    --sizes 100000 \
    -j "${JOBS}" \
    "$@"

echo "end $(date)"

n_done=$(find outputs_structure/_done -maxdepth 1 -type f -name 'nohh_*.done' | wc -l)
echo
echo "no-household (realisation, beta) units finished: ${n_done} of 1680 for the full grid"
if [ "$n_done" -lt 1680 ]; then
    echo "Not complete (or a slice was requested). Run this script again to resume."
else
    echo "Complete. Next:"
    echo "  $PY regen_perrun.py --arms nohh_static nohh_temporal --iccs 0.365   # only if the logs were skipped"
    echo "  bash rerun_notebooks.sh                                             # refresh the three analysis notebooks"
fi
