#!/bin/bash
#SBATCH --job-name=seir_nohh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700

#SBATCH --time=48:00:00

#SBATCH --output=slurm_logs/seir_nohh_%j.out
#SBATCH --error=slurm_logs/seir_nohh_%j.err

# ---------------------------------------------------------------------------
# Re-run every no-household simulation of the structure sweep with the
# age-structured contact fit. Cluster wrapper around rerun_nohh.sh, which holds
# the details; the settings here are those of run_structure_sweep.slurm.
#
#     nohh_static   : 2 models x 1 (no ICC) x 10 networks = 20 realisations
#     nohh_temporal : 2 models x 3 ICCs     x 10 networks = 60 realisations
#
# BEFORE SUBMITTING, on the cluster, delete the old no-household results (the
# job stops at once, with these commands in its log, if they are still there):
#
#     cd <seir_sweep>
#     find outputs_structure/_done outputs_structure/shards \
#          outputs_structure/transmission_logs outputs_structure/daily_series \
#          outputs/_analysis_cache \
#          -maxdepth 1 -type f -name '*nohh_*' -delete
#     python run_structure_sweep.py --collect
#
# SUBMIT WITH, from seir_sweep/:
#     mkdir -p slurm_logs && sbatch rerun_nohh.slurm
# or from temporal_networks/:
#     mkdir -p slurm_logs && sbatch seir_sweep/rerun_nohh.slurm
#
# The logs go to <directory you submitted from>/slurm_logs/, which must exist
# before the first submission.
#
# 60 temporal realisations of about 11 core-hours each on 48 cores is two waves,
# about a day of wall-clock; the 20 static ones backfill in minutes. A job that
# times out is resumed by submitting this file again -- do not delete anything
# in between.
#
# Extra driver arguments go in NOHH_EXTRA, e.g. one job per ICC:
#     for I in 0.0 0.365 0.95; do
#       sbatch --export=ALL,NOHH_EXTRA="--arms nohh_temporal --iccs $I" rerun_nohh.slurm
#     done
#     sbatch --export=ALL,NOHH_EXTRA="--arms nohh_static" rerun_nohh.slurm
# (a later --arms overrides the script's own, so these select one arm each.)
# ---------------------------------------------------------------------------

module purge
module restore p1
source ~/p1/bin/activate

mkdir -p "${SLURM_SUBMIT_DIR}/slurm_logs"

# slurm runs a spooled copy of this file, so locate the sweep directory from
# the directory the job was submitted from.
if   [ -f "${SLURM_SUBMIT_DIR}/rerun_nohh.sh" ];            then SWEEP_DIR="${SLURM_SUBMIT_DIR}"
elif [ -f "${SLURM_SUBMIT_DIR}/seir_sweep/rerun_nohh.sh" ]; then SWEEP_DIR="${SLURM_SUBMIT_DIR}/seir_sweep"
else
    echo "ERROR: cannot find rerun_nohh.sh from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
    echo "       submit from seir_sweep/ or from temporal_networks/." >&2
    exit 1
fi

# One task with 48 CPUs: the driver parallelises with a process pool inside it.
# Do not wrap this in srun -n.
# shellcheck disable=SC2086  # NOHH_EXTRA is intentionally word-split
JOBS="${NOHH_JOBS:-$SLURM_CPUS_PER_TASK}" bash "${SWEEP_DIR}/rerun_nohh.sh" ${NOHH_EXTRA:-}
