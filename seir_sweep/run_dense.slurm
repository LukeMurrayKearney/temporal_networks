#!/bin/bash
#SBATCH --job-name=seir_dense
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=3700

#SBATCH --time=48:00:00

#SBATCH --output=slurm_logs/seir_dense_%j.out
#SBATCH --error=slurm_logs/seir_dense_%j.err

# ---------------------------------------------------------------------------
# One job of the dense-beta extension: extra betas between R0 = 0.2 and 6, five
# times the existing density, for one architecture (and, for a temporal arm, one
# ICC), on the existing network realisations with the existing replication
# (10 networks per model x ICC, 10 epidemics each). The betas come from
# dense_betas.json, written by dense_betas.py; see that file for how they were
# chosen.
#
# Normally submitted by submit_dense.sh, which sets the variables below and the
# CPU count and time limit for each job from the plan in dense_betas.json:
#
#     DENSE_ARM      hh_temporal | hh_static | nohh_temporal | nohh_static
#     DENSE_ICC      one ICC target for a temporal arm (0.0, 0.365 or 0.95);
#                    empty for a static arm
#     DENSE_CHUNK    which chunk of the beta list this job runs (0-based) ...
#     DENSE_NCHUNKS  ... out of this many; chunk i is betas i, i+k, i+2k, ...
#     DENSE_LOGS     1 to keep per-run transmission logs (default 0: see below)
#     DENSE_EXTRA    anything else for the driver, e.g. "--models nb"
#
# Transmission logs are off by default. Every number in runs_summary.csv is
# computed before a log is written, so they lose nothing in the summary tables,
# and the extra points serve curves against R0; the transmission-tree analyses
# use the runs at R0 ~ 2 on the original grid, whose logs already exist (the
# analysis matches those figures among betas that have logs; see
# unified_common.matched(..., logged=True)). Keeping them would add roughly
# 120 GB on the cluster.
#
# Resuming: progress is recorded per (network realisation, beta). A job that
# times out is resumed by submitting it again with the same variables.
# ---------------------------------------------------------------------------

module purge
module restore p1
source ~/p1/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "${SLURM_SUBMIT_DIR}/slurm_logs"

if   [ -f "${SLURM_SUBMIT_DIR}/dense_betas.json" ];            then SWEEP_DIR="${SLURM_SUBMIT_DIR}"
elif [ -f "${SLURM_SUBMIT_DIR}/seir_sweep/dense_betas.json" ]; then SWEEP_DIR="${SLURM_SUBMIT_DIR}/seir_sweep"
else
    echo "ERROR: cannot find dense_betas.json from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
    exit 1
fi
cd "$SWEEP_DIR" || exit 1

: "${DENSE_ARM:?set DENSE_ARM}"
CHUNK="${DENSE_CHUNK:-0}"
NCHUNKS="${DENSE_NCHUNKS:-1}"
JOBS="${DENSE_JOBS:-$SLURM_CPUS_PER_TASK}"

BETAS=$(python - "$DENSE_ARM" "$CHUNK" "$NCHUNKS" <<'PY'
import json, sys
arm, i, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
betas = json.load(open("dense_betas.json"))["arms"][arm]["betas"]
print(" ".join(repr(b) for b in betas[i::k]))
PY
) || exit 1
[ -n "$BETAS" ] || { echo "no betas for $DENSE_ARM chunk $CHUNK/$NCHUNKS" >&2; exit 1; }

ARGS=(--sizes 100000 -j "$JOBS" --no-collect)
[ -n "${DENSE_ICC:-}" ] && ARGS+=(--iccs "$DENSE_ICC")
[ "${DENSE_LOGS:-0}" = 1 ] || ARGS+=(--no-transmission-log)

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} arm=${DENSE_ARM} icc=${DENSE_ICC:-n/a} chunk=${CHUNK}/${NCHUNKS}"
echo "betas: ${BETAS}"
echo "start $(date)"

# shellcheck disable=SC2086  # BETAS and DENSE_EXTRA are intentionally word-split
if [ "$DENSE_ARM" = hh_temporal ]; then
    python run_sweep.py "${ARGS[@]}" --betas ${BETAS} ${DENSE_EXTRA:-}
else
    python run_structure_sweep.py --arms "$DENSE_ARM" "${ARGS[@]}" --betas ${BETAS} ${DENSE_EXTRA:-}
fi

echo "end $(date)"
