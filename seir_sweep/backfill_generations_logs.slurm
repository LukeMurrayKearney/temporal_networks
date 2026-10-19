#!/bin/bash
#SBATCH --job-name=gen_logs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=3700

#SBATCH --time=06:00:00

#SBATCH --output=slurm_logs/gen_logs_%j.out
#SBATCH --error=slurm_logs/gen_logs_%j.err

# ---------------------------------------------------------------------------
# Backfill generation counts (generations.py) for every run at n = 100,000 that
# has a transmission log: the original grid of both sweeps. Each run's
# infections in generations 1 and 2 are read from its log and written to
# <outputs dir>/gen_counts/ side files; nothing else is touched. A run that
# infected nobody has no log and is recorded with zero counts.
#
# The dense runs have no logs; they are backfilled by replay instead:
#     GENERATIONS=1 bash submit_dense.sh
#
# SUBMIT WITH, from seir_sweep/:
#     mkdir -p slurm_logs && sbatch backfill_generations_logs.slurm
# Re-submitting is safe: (cell, beta) units that already have a side file are
# skipped.
# ---------------------------------------------------------------------------

module purge
module restore p1
source ~/p1/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

mkdir -p "${SLURM_SUBMIT_DIR}/slurm_logs"
if   [ -f "${SLURM_SUBMIT_DIR}/generations.py" ];            then SWEEP_DIR="${SLURM_SUBMIT_DIR}"
elif [ -f "${SLURM_SUBMIT_DIR}/seir_sweep/generations.py" ]; then SWEEP_DIR="${SLURM_SUBMIT_DIR}/seir_sweep"
else
    echo "ERROR: cannot find generations.py from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
    exit 1
fi
cd "$SWEEP_DIR" || exit 1

echo "start $(date)"
python run_sweep.py --sizes 100000 --generations logs -j "$SLURM_CPUS_PER_TASK"
python run_structure_sweep.py --sizes 100000 --generations logs -j "$SLURM_CPUS_PER_TASK"
echo "end $(date)"
