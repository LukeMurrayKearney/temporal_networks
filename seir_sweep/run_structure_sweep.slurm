#!/bin/bash
#SBATCH --job-name=seir_structure
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700

#SBATCH --time=48:00:00

#SBATCH --output=slurm_logs/seir_structure_%j.out
#SBATCH --error=slurm_logs/seir_structure_%j.err

# ---------------------------------------------------------------------------
# SEIR structure sweep:
#   {temporal, static} x {households, none} x {NB, log-GMM}  at n=100k, same betas
#
# The temporal x households corner is NOT run here -- it is the main sweep
# (run_sweep.slurm), whose n=100000 slice is already complete. This job runs the
# other three corners; splice the fourth in from outputs/runs_summary.csv at
# analysis time. See config_structure.py.
#
# SUBMIT WITH, from seir_sweep/:
#     mkdir -p slurm_logs && sbatch run_structure_sweep.slurm
# or from temporal_networks/:
#     mkdir -p slurm_logs && sbatch seir_sweep/run_structure_sweep.slurm
#
# WHERE THE LOGS GO: slurm resolves the --output/--error paths above relative to
# THE DIRECTORY YOU RAN sbatch FROM -- not to where this script lives. So the
# logs land in <dir you submitted from>/slurm_logs/seir_structure_<jobid>.{out,err}:
#     submitted from seir_sweep/         -> seir_sweep/slurm_logs/
#     submitted from temporal_networks/  -> temporal_networks/slurm_logs/
#
# The mkdir is required the FIRST time, in whichever directory you submit from:
# slurm opens those files before the script body runs, so if the directory is
# missing the job dies immediately with "No such file or directory" and the
# mkdir further down never executes. (It only covers later runs.)
#
# Either submit directory works: the script locates run_structure_sweep.py itself.
#
# run_structure_sweep.py parallelises with a multiprocessing pool INSIDE a single
# task, so this is one task with 48 CPUs (not 48 MPI tasks). Do not wrap the
# python call in `srun -n $SLURM_CPUS_PER_TASK` -- that would launch 48
# independent copies of the whole sweep, all racing on the same output files.
# ---------------------------------------------------------------------------

module purge
module restore p1
source ~/p1/bin/activate

# ---------------------------------------------------------------------------
# CRITICAL: threads per worker must be 1, NOT $SLURM_CPUS_PER_TASK.
#
# The template's `export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` is right for a
# single-process threaded code, but here we fork 48 worker PROCESSES. Giving
# each one a 48-thread BLAS pool would ask for 48 x 48 = 2304 threads on 48
# cores -- the machine would spend its time context-switching.
#
# This must be set in the environment BEFORE python starts.
# run_structure_sweep._worker() also sets it, but the pool forks after
# numpy/sklearn are already imported, so by then each child has inherited an
# already-initialised BLAS thread pool and the in-process assignment is too late
# to shrink it.
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Keep the log directory alive for subsequent runs (too late for THIS job's own
# log files -- slurm opened those before the script started; see header).
mkdir -p "${SLURM_SUBMIT_DIR}/slurm_logs"

# ---------------------------------------------------------------------------
# Locate run_structure_sweep.py, so submitting from either seir_sweep/ or
# temporal_networks/ works.
#
# $0 / ${BASH_SOURCE[0]} are NOT usable here: slurm copies the batch script to a
# node-local spool path (e.g. /var/spool/slurmd/job12345/slurm_script), so they
# do not point at the real location in the repo. $SLURM_SUBMIT_DIR is the
# directory sbatch was invoked from, which is the reliable anchor.
# ---------------------------------------------------------------------------
if   [ -f "${SLURM_SUBMIT_DIR}/run_structure_sweep.py" ];            then SWEEP_DIR="${SLURM_SUBMIT_DIR}"
elif [ -f "${SLURM_SUBMIT_DIR}/seir_sweep/run_structure_sweep.py" ]; then SWEEP_DIR="${SLURM_SUBMIT_DIR}/seir_sweep"
else
    echo "ERROR: cannot find run_structure_sweep.py from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
    echo "       submit from seir_sweep/ or from temporal_networks/." >&2
    exit 1
fi
cd "$SWEEP_DIR" || exit 1

# ---------------------------------------------------------------------------
# Which slice of the grid to run.
#
# One realisation = one pool task. The default grid at n=100k is 100 tasks:
#     hh_static     : 2 models x 1 (no ICC)  x 10 net_reps = 20
#     nohh_static   : 2 models x 1 (no ICC)  x 10 net_reps = 20
#     nohh_temporal : 2 models x 3 ICCs      x 10 net_reps = 60
# ICC is a temporal-only axis -- a static network has no one_time layer to
# persist -- so --iccs is ignored for the two static arms rather than fanning
# them out into three identical copies.
#
# Measured per-run cost at n=100k (one NB realisation per arm, betas 0.17/0.8/9):
#     hh_static      4s / 11s / 14s     -> ~10 s x 210 runs  ~= 35 min per realisation
#     nohh_static    1s /  5s /  8s     -> ~5  s x 210 runs  ~= 20 min per realisation
#     nohh_temporal 18s / 249s / 123s   -> ~2 min x 210 runs ~= 7 h per realisation
# The static arms have the identity fast path (the edge set never changes, so
# the adjacency is built once per run); the temporal arm rebuilds it every
# simulated day, which is the whole difference. So the 40 static realisations
# are ~25 core-hours between them and the 60 temporal ones are ~420 core-hours.
#
# 100 tasks on 48 cores is roughly two waves of the long temporal tasks, i.e.
# ~15 h wall-clock, comfortably inside the 48 h requested. run_structure_sweep.py
# sorts biggest-first, so the long tasks start immediately and the cheap static
# ones backfill behind them.
# ---------------------------------------------------------------------------
# All four are overridable at submit time without editing this file, e.g.
#     sbatch --export=ALL,SWEEP_ARMS="nohh_temporal",SWEEP_JOBS=48 run_structure_sweep.slurm
ARMS="${SWEEP_ARMS:-hh_static nohh_static nohh_temporal}"
SIZES="${SWEEP_SIZES:-100000}"
JOBS="${SWEEP_JOBS:-$SLURM_CPUS_PER_TASK}"
EXTRA="${SWEEP_EXTRA:-}"

echo "host=$(hostname)  cpus=${SLURM_CPUS_PER_TASK}  arms=${ARMS}  sizes=${SIZES}  jobs=${JOBS}"
echo "start $(date)"

# Measured peak RSS per worker at n=100k: ~0.7 GB in every arm (0.69 GB static,
# 0.74 GB temporal). 48 workers x 0.75 GB ~= 36 GB against the 177 GB this job
# requests, so memory is comfortable at the full 48 workers.
# shellcheck disable=SC2086  # ARMS/SIZES/EXTRA are intentionally word-split
python run_structure_sweep.py \
    --arms ${ARMS} \
    --sizes ${SIZES} \
    -j "${JOBS}" \
    ${EXTRA}

echo "end $(date)"

# run_structure_sweep.py merges the per-task shards into
# outputs_structure/runs_summary.csv and outputs_structure/network_summary.csv
# when the pool drains, so no extra step is needed. If the job is killed mid-way,
# re-running --collect rebuilds those two files from whatever shards exist:
#     python run_structure_sweep.py --collect

# ---------------------------------------------------------------------------
# Resuming and splitting
# ---------------------------------------------------------------------------
# Progress is tracked per (realisation, beta) in outputs_structure/_done/, so
# re-submitting this exact script skips finished work and a timed-out job
# resumes at the beta it was on. Splitting across jobs is safe along every axis
# that is part of the done-marker key -- --arms, --models, --iccs, --betas:
#
#     # the long temporal arm on its own, one job per ICC
#     for I in 0.0 0.365 0.95; do
#       sbatch --export=ALL,SWEEP_ARMS="nohh_temporal",SWEEP_EXTRA="--iccs $I" \
#              run_structure_sweep.slurm
#     done
#
#     # the two cheap static arms together
#     sbatch --export=ALL,SWEEP_ARMS="hh_static nohh_static" run_structure_sweep.slurm
#
# Betas can also be ADDED to a finished grid: re-submit with the new values and
# only those are simulated, on the same network realisations as the originals,
# so the design stays paired. Overlapping beta sets cost nothing.
#
# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------
# 100 realisations x 21 betas x 10 sim_reps = 21,000 runs, each writing a
# transmission log and a daily series. The transmission logs are by far the
# largest output and the static arms produce the biggest outbreaks. If disk is
# tight:
#     SWEEP_EXTRA="--no-transmission-log"
# The per-run summary metrics in runs_summary.csv are computed before the log is
# written, so dropping the logs loses nothing in the main table -- only the
# ability to re-derive transmission-tree quantities afterwards.
