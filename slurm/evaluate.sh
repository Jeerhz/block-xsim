#!/usr/bin/env bash
# The evaluation of one corpus and edit set (GPU). Split a long run over jobs with SCORERS,
# REGIMES and OUTPUT, then join the files with `python -m block_xsim.merge_results`.
#   CORPUS=ntrex EDITS=xsim sbatch slurm/evaluate.sh
#   CORPUS=ntrex EDITS=xsim SCORERS=me5 REGIMES=every_2nd OUTPUT=results/parts/ntrex_xsim_me5_2nd.json sbatch slurm/evaluate.sh
#SBATCH --job-name=evaluate
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
source slurm/env.sh

srun python -m block_xsim.evaluate --corpus "${CORPUS:-flores}" --edits "${EDITS:-xsim}" \
  --langs ${LANGS:-de es fr ru} --scorers ${SCORERS:-cometkiwi comet-da xlm-r labse me5} \
  --regimes ${REGIMES:-every_sentence every_2nd every_3rd one_sentence} \
  --batch_size "${BATCH_SIZE:-32}" ${OUTPUT:+--output "$OUTPUT"}
