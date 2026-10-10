#!/usr/bin/env bash
# Blocks kept per length and sentences each edit applies to, both corpora and edit sets (CPU).
#   sbatch slurm/corpus_statistics.sh
#SBATCH --job-name=corpus-statistics
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail
source slurm/env.sh

srun python -m block_xsim.corpus_statistics --langs ${LANGS:-de es fr ru}
