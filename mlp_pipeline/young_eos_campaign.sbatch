#!/bin/bash -l
#SBATCH --job-name=liqeos
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
manifest=${1:?manifest required}
code="$(dirname "$manifest")/code"
exec "${YOUNG_PY:-$HOME/grace-venv/bin/python}" "$code/young_eos_campaign.py" run "$manifest" "${SLURM_ARRAY_TASK_ID:?}"
