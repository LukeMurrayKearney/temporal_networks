#!/bin/bash
#SBATCH --job-name=seir_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700

#SBATCH --time=48:00:00

#SBATCH --output=slurm_logs/seir_sweep_%j.out
#SBATCH --error=slurm_logs/seir_sweep_%j.err

# ---------------------------------------------------------------------------
# SEIR sweep: {NB, log-GMM mean-match} x {ICC 0, 0.365, 0.95} x {n} x beta
#
# SUBMIT WITH, from seir_sweep/:
#     mkdir -p slurm_logs && sbatch run_sweep.slurm
# or from temporal_networks/:
#     mkdir -p slurm_logs && sbatch seir_sweep/run_sweep.slurm
#
# WHERE THE LOGS GO: slurm resolves the --output/--error paths above relative to
# THE DIRECTORY YOU RAN sbatch FROM -- not to where this script lives. So the
# logs land in <dir you submitted from>/slurm_logs/seir_sweep_<jobid>.{out,err}:
#     submitted from seir_sweep/         -> seir_sweep/slurm_logs/
#     submitted from temporal_networks/  -> temporal_networks/slurm_logs/
#
# The mkdir is required the FIRST time, in whichever directory you submit from:
# slurm opens those files before the script body runs, so if the directory is
# missing the job dies immediately with "No such file or directory" and the
# mkdir further down never executes. (It only covers later runs.)
#
# Either submit directory works: the script locates run_sweep.py itself below.
#
# run_sweep.py parallelises with a multiprocessing pool INSIDE a single task,
# so this is one task with 48 CPUs (not 48 MPI tasks). Do not wrap the python
# call in `srun -n $SLURM_CPUS_PER_TASK` -- that would launch 48 independent
# copies of the whole sweep, all racing on the same output files.
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
# This must be set in the environment BEFORE python starts. run_sweep._worker()
# also sets it, but the pool forks after numpy/sklearn are already imported, so
# by then each child has inherited an already-initialised BLAS thread pool and
# the in-process assignment is too late to shrink it.
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
# Locate run_sweep.py, so submitting from either seir_sweep/ or
# temporal_networks/ works.
#
# $0 / ${BASH_SOURCE[0]} are NOT usable here: slurm copies the batch script to a
# node-local spool path (e.g. /var/spool/slurmd/job12345/slurm_script), so they
# do not point at the real location in the repo. $SLURM_SUBMIT_DIR is the
# directory sbatch was invoked from, which is the reliable anchor.
# ---------------------------------------------------------------------------
if   [ -f "${SLURM_SUBMIT_DIR}/run_sweep.py" ];             then SWEEP_DIR="${SLURM_SUBMIT_DIR}"
elif [ -f "${SLURM_SUBMIT_DIR}/seir_sweep/run_sweep.py" ];  then SWEEP_DIR="${SLURM_SUBMIT_DIR}/seir_sweep"
else
    echo "ERROR: cannot find run_sweep.py from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
    echo "       submit from seir_sweep/ or from temporal_networks/." >&2
    exit 1
fi
cd "$SWEEP_DIR" || exit 1

# ---------------------------------------------------------------------------
# Which slice of the grid to run.
#
# One realisation = one pool task, so the useful worker count is capped by the
# number of realisations, not by cores:
#     n=10k  -> 30 realisations (5 net_reps x 2 models x 3 ICCs)
#     n=100k -> 18 realisations (3 net_reps x ...)
#     n=1M   ->  6 realisations (1 net_rep  x ...)
# 10k + 100k together is exactly 48 tasks for 48 cores. run_sweep.py sorts
# biggest-first, so the long 100k tasks start immediately and the short 10k
# ones backfill behind them.
#
# n=1,000,000 is NOT included here on purpose -- see the note at the bottom.
# ---------------------------------------------------------------------------
# All three are overridable at submit time without editing this file, e.g.
#     sbatch --export=ALL,SWEEP_SIZES=1000000,SWEEP_JOBS=6 run_sweep.slurm
SIZES="${SWEEP_SIZES:-10000 100000}"
JOBS="${SWEEP_JOBS:-$SLURM_CPUS_PER_TASK}"
EXTRA="${SWEEP_EXTRA:-}"

echo "host=$(hostname)  cpus=${SLURM_CPUS_PER_TASK}  sizes=${SIZES}  jobs=${JOBS}"
echo "start $(date)"

# Measured peak RSS per worker: 0.21 GB at n=10k, 0.59 GB at n=100k.
# 48 workers x 0.59 GB ~= 29 GB against the 177 GB this job requests, so
# memory is comfortable and the full 48 workers can be used.
# shellcheck disable=SC2086  # SIZES/EXTRA are intentionally word-split
python run_sweep.py \
    --sizes ${SIZES} \
    -j "${JOBS}" \
    ${EXTRA}

echo "end $(date)"

# run_sweep.py merges the per-task shards into outputs/runs_summary.csv and
# outputs/network_summary.csv when the pool drains, so no extra step is needed.
# If the job is killed mid-way, re-running `--collect` rebuilds those two files
# from whatever shards exist:
#     python run_sweep.py --collect

# ---------------------------------------------------------------------------
# Resuming
# ---------------------------------------------------------------------------
# Completed realisations leave a marker in outputs/_done/, and re-submitting
# this exact script skips them, so a timed-out job resumes where it stopped.
#
# CAVEAT -- do not split a size across jobs by --betas. The done-marker is keyed
# on (model, ICC, n, net_rep) and carries no beta information, so a second job
# running the remaining betas would see the marker, skip the realisation, and
# silently leave those betas unrun (it would also overwrite that cell's shard).
# Split only along --models / --iccs / --sizes, which are part of the key.
#
# ---------------------------------------------------------------------------
# n = 1,000,000
# ---------------------------------------------------------------------------
# Left out of SIZES because it does not fit in a 48 h wall-clock. The simulator
# rebuilds the whole daily adjacency in Python every simulated day; that costs
# ~4 s/day at 100k, so ~40 s/day at 1M, and a supercritical outbreak runs for
# ~200 days => roughly 2-6 h for a SINGLE run. Each 1M realisation is 9 betas
# x 2 sim_reps = 18 such runs, i.e. well over 48 h, and because the done-marker
# is written only after all 18 finish, a timeout loses the whole realisation.
#
# Options, in increasing order of effort:
#   1. Submit the 1M slice as 6 independent jobs, one per (model, ICC), each
#      carrying a single realisation and a long wall-clock:
#          for M in nb loggmm_mm; do for I in 0.0 0.365 0.95; do
#            sbatch --time=7-00:00:00 \
#                   --export=ALL,SWEEP_SIZES=1000000,SWEEP_JOBS=1,SWEEP_EXTRA="--models $M --iccs $I" \
#                   run_sweep.slurm
#          done; done
#      Splitting on --models/--iccs is safe: both are part of the done-marker key.
#   2. Cut the per-realisation cost first, via SWEEP_EXTRA: drop the saturating
#      high-R0 betas (also the slowest), e.g.
#          SWEEP_EXTRA="--betas 0.05 0.10 0.17 0.37 0.59 1.97 --no-transmission-log"
#      The transmission logs are by far the largest output at this size.
#   3. Make the done-marker beta-aware so 1M can be split by beta across an
#      array job -- a small change to run_sweep.run_task(), needed before
#      option 1 can be parallelised any further.
