#!/bin/bash -l
#SBATCH --job-name=o64_gl4rep
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
set -eo pipefail
cd /lustre/scratch/mmm0037/H3O
export PY="$HOME/grace-venv/bin/python"
case "${SLURM_ARRAY_TASK_ID:?}" in
  0) export SEED=101 ;;
  1) export SEED=201 ;;
  *) echo 'Unexpected replica index' >&2; exit 1 ;;
esac
export OUTDIR="runs/tdi_o64_gl4_replica_seed${SEED}"
if [ -e "$OUTDIR" ]; then
  echo "Output directory already exists: $OUTDIR; inspect before resuming or submitting again." >&2
  exit 1
fi
export VPA
VPA=$("$PY" - <<'PY'
import json
from pathlib import Path
p=Path('runs/quad_O64_H2_350GPa_7000K/gl4_gk9')
f=list(p.glob('*_stage0.json'))
if len(f)!=1:raise SystemExit('Expected one O64 GK9 stage0 file')
s=json.loads(f[0].read_text());a=s['args']
if s['N']!=192 or s['T']!=7000 or s['P_target']!=350 or s['phase']!='liquid' or a.get('hper',2)!=2:
    raise SystemExit('O64 GK9 stage0 does not match requested state')
if a['model']!='grace-ft:runs/grace/ft_mfinkc_cont/seed/1/final_model':
    raise SystemExit('Unexpected O64 model path')
print(format(s['V']/s['N'],'.17g'))
PY
)
export P=350 T=7000 HPER=2 MODEL='grace-ft:runs/grace/ft_mfinkc_cont/seed/1/final_model'
export NREP=4 LIQUID_START=sc RULE=gl NODES=4 POWER=4 PS=6.25 PS1=0.5
export DT=0.5 CHUNK=0.125 EQUIL=0.5 HOT=0.5 HOTT=12000 K=0 NPAR=4
export OMP_NUM_THREADS=1 TF_USE_LEGACY_KERAS=1
unset FRESH INIT ONLYNODES GRACE_MODE HHWALL GRACE_UQ
printf 'O64 H2O GL4 replica: seed %s, volume/atom %s, output %s\n' "$SEED" "$VPA" "$OUTDIR"
exec bash mlp_pipeline/young_tdi_par.sh liquid
