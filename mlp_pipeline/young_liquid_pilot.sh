#!/bin/bash
# Small GPU TDI pilot, not a converged hull point. Run inside a Young GPU allocation.
set -e
cd "$(dirname "$0")/.."
[ -n "${SLURM_JOB_ID:-}" ] || { echo 'Run inside an interactive GPU allocation, not on the login node.'; exit 1; }
nvidia-smi -L >/dev/null || exit 1
export PY=${PY:-$HOME/grace-venv/bin/python}
export PYTHONPATH="$PWD/mlp_pipeline${PYTHONPATH:+:$PYTHONPATH}"
export TF_USE_LEGACY_KERAS=1 TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
[ -n "$CUDA_VISIBLE_DEVICES" ] || { echo 'Slurm did not set CUDA_VISIBLE_DEVICES; check the allocation.'; exit 1; }
export MODEL=${MODEL:-grace-ft:runs/grace/ft_mfinkc_cont/seed/1/final_model}
export OUTDIR=${OUTDIR:-runs/tdi_mfinkc_pilot_350GPa_7000K}
# Clear inherited settings that would change this deliberately small pilot or delete old files.
unset FRESH ONLY0 ONLYNODES INIT VPA BULK K PREPHOLD PREPFREE ENGINE TAG RULE NGAUSS UQ GRACE_UQ
PLAN=$("$PY" -m tdi.quadrature_plan "$OUTDIR" "${QUAD:-auto}")
read -r RULE NODES NGAUSS SAVED_PS <<< "$PLAN"
export NODES
if [ "$RULE" = gk ]; then export RULE NGAUSS; else unset RULE NGAUSS; fi
export P=350 T=7000 HPER=2 NREP=2 NPAR=4 POWER=4 DT=0.5
export PS=${PS:-$SAVED_PS} EQUIL=0.2 HOT=0.2 HOTT=12000 CHUNK=0.125 PS1=0.2
export OMP_NUM_THREADS=1
printf 'Liquid H2O pilot: 96 atoms, 350 GPa, 7000 K; %s lambda values; GPU %s\n' "$NODES" "$CUDA_VISIBLE_DEVICES"
echo "Output: $OUTDIR; target $PS ps production per lambda (total, including saved samples)."
mkdir -p "$OUTDIR"
"$PY" mlp_pipeline/tdi/run_diary.py "$OUTDIR" start "$PS"
trap 'rc=$?; "$PY" mlp_pipeline/tdi/run_diary.py "$OUTDIR" finish "$PS" "$rc"' EXIT
bash mlp_pipeline/young_tdi_par.sh liquid
"$PY" mlp_pipeline/tdi/liquid_check.py "$OUTDIR"
echo 'Pilot finished. Assess time convergence and quadrature before interpreting free energies.'
