#!/bin/bash -l
#SBATCH --job-name=h4_eos
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
set -euo pipefail
campaign=${1:?campaign snapshot directory required}
cd "${SLURM_SUBMIT_DIR:?}"
export PY="$HOME/grace-venv/bin/python"
export EOS_REPO_ROOT="$PWD"
export EOS_CODE_DIR="$campaign/mlp_pipeline"
export OMP_NUM_THREADS=1
exec "$PY" "$EOS_CODE_DIR/eos_queue.py" \
  "$campaign/data/final_model/eos_queue_h4o_then_missing.json" \
  --gpus 1 --per-gpu 1 \
  --model grace-ft:runs/grace/ft_mfinkc_cont --run
