# Sourced by every job right after its #SBATCH header: repository root as working
# directory, then the Python environment (VENV_PATH, else the conda env CONDA_ENV).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
else
  source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-block-xsim}"
fi
