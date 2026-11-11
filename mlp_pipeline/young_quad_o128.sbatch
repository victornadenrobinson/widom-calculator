#!/bin/bash -l
#SBATCH --job-name=quad128
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=12:00:00
set -euo pipefail
cd /lustre/scratch/mmm0037/H3O
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
printf 'Commit at startup: '
git rev-parse HEAD
printf 'O128 liquid H2O: GK9/GK21/GK31, 350 GPa, 7000 K, 6.25 ps per node\n'
printf 'GK17 is covered by separately submitted array task 131371_0; compare settings and volumes before combining.\n'
exec "$HOME/grace-venv/bin/python" mlp_pipeline/young_quad_sweep.py \
  --oxygen 128 --hper 2 --P 350 --T 7000 --ps 6.25 --per-gpu 4 \
  --root runs/quad384_350GPa_7000K --families 4 10 15
