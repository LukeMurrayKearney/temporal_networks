#!/bin/bash
#SBATCH --job-name=r0t
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3700

#SBATCH --time=10:00:00

#SBATCH --output=slurm_logs/r0t_%j.out
#SBATCH --error=slurm_logs/r0t_%j.err

# ---------------------------------------------------------------------------
# Runs for one variant at the beta where its fitted beta -> R0 curve reaches a
# target R0 (r0_targets.py, which writes r0_targets.json): every network
# realisation x 10 epidemics, WITH transmission logs and daily series, so the
# transmission-tree and household figures can be read at that R0 exactly.
#
# Normally submitted by submit_r0_targets.sh, which sets:
#
#     R0T_ARM    hh_temporal | hh_static | nohh_temporal | nohh_static
#     R0T_MODEL  nb | loggmm_mm
#     R0T_ICC    0.0, 0.365 or 0.95 for a temporal arm; empty for a static arm
#     R0T_BETA   the fitted beta
#
# Resuming: progress is recorded per (network realisation, beta); a job that
# times out is resumed by submitting it again with the same variables.
# ---------------------------------------------------------------------------

module purge
module restore p1
source ~/p1/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

mkdir -p "${SLURM_SUBMIT_DIR}/slurm_logs"
if   [ -f "${SLURM_SUBMIT_DIR}/r0_targets.json" ];            then SWEEP_DIR="${SLURM_SUBMIT_DIR}"
elif [ -f "${SLURM_SUBMIT_DIR}/seir_sweep/r0_targets.json" ]; then SWEEP_DIR="${SLURM_SUBMIT_DIR}/seir_sweep"
else
    echo "ERROR: cannot find r0_targets.json from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
    exit 1
fi
cd "$SWEEP_DIR" || exit 1

: "${R0T_ARM:?set R0T_ARM}" "${R0T_MODEL:?set R0T_MODEL}" "${R0T_BETA:?set R0T_BETA}"
ARGS=(--sizes 100000 --models "$R0T_MODEL" --betas "$R0T_BETA"
      -j "${R0T_JOBS:-$SLURM_CPUS_PER_TASK}" --no-collect)
[ -n "${R0T_ICC:-}" ] && ARGS+=(--iccs "$R0T_ICC")

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} arm=${R0T_ARM} model=${R0T_MODEL}" \
     "icc=${R0T_ICC:-n/a} beta=${R0T_BETA}"
echo "start $(date)"
if [ "$R0T_ARM" = hh_temporal ]; then
    python run_sweep.py "${ARGS[@]}"
else
    python run_structure_sweep.py --arms "$R0T_ARM" "${ARGS[@]}"
fi
echo "end $(date)"
