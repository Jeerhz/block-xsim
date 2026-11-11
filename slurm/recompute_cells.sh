#!/usr/bin/env bash
# Recompute the cells of result files from the stored per-candidate scores (CPU, no scoring).
#   sbatch slurm/recompute_cells.sh
#   RESULTS="data/rerun_mx/ntrex_xsim_*.json" OUTPUT=results/metricx_bleu/ntrex_xsim.json sbatch slurm/recompute_cells.sh
#SBATCH --job-name=recompute-cells
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cpu_devel
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
source slurm/env.sh

srun python -m block_xsim.recompute_cells ${RESULTS:-results/flores_*.json results/ntrex_*.json results/metricx_bleu/flores_*.json} \
  ${OUTPUT:+--output "$OUTPUT"}
