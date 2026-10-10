#!/usr/bin/env bash
# The blocks of one corpus and edit set (CPU).
#   CORPUS=ntrex EDITS=minor sbatch slurm/build_blocks.sh
#SBATCH --job-name=build-blocks
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
source slurm/env.sh

srun python -m block_xsim.build_blocks --corpus "${CORPUS:-flores}" --edits "${EDITS:-xsim}" \
  --langs ${LANGS:-de es fr ru}
