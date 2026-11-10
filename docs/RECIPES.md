# Recipes: how to run each calculation

One entry point: **`bash run.sh <task> key=value ...`** from the repo root on Perlmutter. It translates the keys to the underlying script and prints the exact command (`+ ...`) before running it; **`dry=1`** only prints. `bash run.sh help` lists the tasks. Defaults below are what you get when a key is left out.

Where to run: tasks marked **GPU** need an interactive GPU node (`salloc -A m3349_g -C gpu -q interactive -N 1 -G 4 -c 128 -t 02:00:00`); tasks marked **login** submit batch jobs and must run on a login node (submitting from inside a GPU allocation gives "Job request does not match any supported policy").

Common keys: `model=` (a run name like `mfin_f_lr1e-2_e50` or a full `grace-ft:...` spec; default the current best model), `dry=1`.

## Tasks

### eos — liquid / fluid EOS scan with live dashboard (GPU, ~15-25 min per composition at 4 GPUs)
`bash run.sh eos comp="h2o h3o h" T=7000`
- keys: `comp` (h2o h2.25 h3o h5o h7o h; several = one job queue), `T` (7000), `nv` (16 volumes), `vmax` / `vmin` (A^3/atom; default per composition), `ps` (0.75; pure H 0.5), `eq` (0.25; pure H 0.2), `per_gpu` (4).
- batch (login node): `bash run.sh eos comp="h2o h2.25 h3o h5o h7o h" T="1000 2000 3000 4000 5000 6000 7000 8000 9000" where=batch hours=3` = one job per T, `gpus=4` (default: full node, regular queue, ~1-1.5 h per T for all 6 compositions) or `gpus=1` / `2` (shared queue, same charge, ~4x / 2x longer: give hours=8 / 4); log logs_pw/eos_<model>_<T>K.*.out, dashboard every 10 min, no plot).
- cells ~384 atoms (pure H 729); mixtures premelted at 9000 K for 0.25 ps; output `runs/eos_live/<model>_<comp>_<T>K_<N>at/`; finished volumes are skipped on rerun; flags FROZE / COLLAPSE / CLOSE-HH / T-off / P-drift; per volume `rdf.csv`, coordination, H2 fraction.

### eos_plot — all scans vs DFT (and French 2009 for H2O), P-T maps (login or GPU node, CPU only)
`bash run.sh eos_plot T=7000` (add `loglog=1` for log-log axes)
- keys: `T`, `pmin` / `pmax`, `frames` (DFT-MD frames for the energy comparison; default train_mfin + valid_mfin).
- writes `runs/eos_live/eos_all_<T>K_<model>.png`, `runs/eos_live/eos_maps_<model>.png` and `eos_fits_<T>K_<model>.csv` (Vinet and BM3 fits per composition, least squares in log P, liquid points only; checks: monotonic, max residual > 3 %, K1 outside 2-10; over 10-1000 GPa Vinet is the form that works, BM3 misfits by ~2.5 % rms on synthetic Vinet data).

### tdi — full TDI free energy of one phase (GPU here, or one shared-GPU batch job)
`bash run.sh tdi phase=fcc P=500 T=5000 hper=2 quad=gk17 npar=6 ps=2`
`bash run.sh tdi phase=liquid P=500 T=5000 hper=2.25 quad=gl10 where=batch hours=8`
`bash run.sh tdi phase=H P=500 T=5000 quad=gl10 npar=4`
- `phase`: fcc | bcc | liquid | H (pure-H fluid, 729 atoms) | insert (`nghost=` protons into fcc H2O).
- `quad`: `gl<N>` = Gauss-Legendre in lambda = x^4 with N values (gl10, gl12); `gk<2n+1>` = Gauss-Kronrod: the n Gauss values plus n+1 added (gk17 = 8 + 9, gk21 = 10 + 11). A gk run gives G_Kronrod and G_Gauss(n) from the same data: their difference is the quadrature error. An existing gl run can be extended to gk later without redoing it (see `young_tdi_par.sh` header, ONLYNODES).
- keys: `P` (500), `T` (5000), `hper` (2; 2.25 = H2.25O), `npar` (4: lambda processes at a time on the GPU), `ps` (4 ps per lambda), `ps1` (1.5), `equil` (1), `power` (4), `gpu` (0), `where` (here | batch), `hours` (8, batch), `natoms`, `nghost`, `hot` / `hott` (liquid melt; default 0.5 ps at 12000 K), `k` (Einstein constant), `nrep`, `prephold`, `chunk` (ps between G printouts, 0.5), `tag`, `outdir`.
- volumes from the DFT EOS (`inventory_binned.csv`, linked from runs/ if needed); restartable (rerun the same command); output `runs/tdi/`; results `python3 mlp_pipeline/tdi/summarize.py runs/tdi`.
- cost (fcc 324 atoms, measured 2026-11-06 with the F10 model): **npar 4 is the best** (2/4/6/8 processes per GPU = 12.3/23.8/22.2/22.4 steps/s per GPU); one full TDI on one GPU at npar 4: 10 lambda x 2 ps 0.8 h, 10 x 4 ps 1.4 h, 17 x 2 ps 1.4 h, 17 x 4 ps 2.3 h, 21 x 2 ps 1.7 h. Four TDIs side by side: gpu=0..3. Large cells (> 400 atoms) need npar <= 8 on a 40 GB card, 729-atom H npar 4.

### tdi_bench — TDI throughput, different numbers of lambda processes per GPU at once (GPU, ~10 min)
`bash run.sh tdi_bench system=h2o layout="2 4 6 8"`
- `system`: h2o (fcc crystal, 324 atoms) | H (pure H 250) | ins | insH; `layout` (processes on GPU0..3), `steps` (200), `vpa`.
- prints steps/s per process and per GPU and the full-TDI wall time for 10 / 17 / 21 lambda values.

### finetune — one GRACE fine-tune as a batch job (login)
`bash run.sh finetune name=mfin_f_lr1e-2_e50_S10 train=mfin epochs=50 wE=10 wF=10 wS=10 lr=0.01`
- keys: `name` (required), `train` (mfin), `valid` (= train), `epochs` (50) or `updates`, `wE` / `wF` / `wS` (10 / 10 / 1), `mode` (frozen | naive), `lr` (0.01), `batch` (1), `gpus` (2: the 4.3k-frame tier needs the host RAM of 2 shared GPUs), `time` (24:00:00).
- always starts from the foundation model (ft_one.sh); output `runs/grace/ft_<name>/`; status `python3 mlp_pipeline/ft_status.py`.

### continue a fine-tune on new labels (e.g. k-point corrected; login, batch)
`echo "bash mlp_pipeline/ft_continue.sh mfinkc_cont mfin_f_lr1e-2_e50 mfinkc mfinkc 21450 0.001" > runs/ft_cont.cmd; GPUS=2 TIME=04:00:00 bash mlp_pipeline/ft_submit.sh runs/ft_cont.cmd`
- loads the base run's architecture (`-p seed/1/model.yaml`) and weights (`-cn checkpoints/checkpoint.best_test_loss`), resets epoch/step, base input.yaml with new data / updates / lr (optional wE wF wS after lr). Check: the INITIAL test error in `runs/grace/ft_<name>/seed/1/log.txt` must be ~the base model's (not ~150 meV/atom).

### score — compare fine-tunes (GPU, minutes per model)
`bash run.sh score only="mfin_f_lr1e-2_e50"`
- held-out frames (`compare_D` table), DFT-MD composition test (`composition_check`), sealed pair test (pair c, pass |c| <= 50 meV/H; only reference models that did not train on the pairs are a real test).

### dft — VASP single points for any set of geometries (login)
`bash run.sh dft frames=runs/x/frames.extxyz outd=runs/x_dft nodes=8 submit=1`
- keys: `frames`, `outd` (required), `template` (vasp_inputs; a k-mesh template for k-point labels), `exe` (vasp_gam; vasp_std for k-meshes), `nodes` (8 jobs), `minutes` (170), `submit` (0 = prepare only).
- 4 lanes x 32 ranks per node; a resubmission resumes unfinished points; collect with `points_collect.py` (skips unconverged SCF, prints a health table).

### VASP cost per snapshot (measured 2026-11-06, 32 MPI ranks per point, 4 points per CPU node)
| cell | Gamma-only label (vasp_gam) | 2x2x2 Gamma-centred (8 k, vasp_std) | 2x2x2 Monkhorst-Pack (4 k, vasp_std) |
|---|---|---|---|
| <= 260 atoms | ~1 min | 9 min | 3 min |
| 261-450 atoms | ~1.5 min (87 s for 324-352) | 24 min (median 22) | 12 min (median 11.5) |
| 451-600 atoms | ~3 min (est.) | 33 min | 14.5 min |
| > 600 atoms (pure H 686-729) | ~3-4 min | 35 min | 13-30 min (3 points) |
The 300-frame relabel cost 28.5 CPU node-hours (225 Gamma-centred + 75 MP points). A full MP 2x2x2 relabel of the 5143 training + validation frames: ~290 node-hours (scaled ~atoms^2).

### kpt — k-point relabelling of a stratified frame selection (login)
`bash run.sh kpt n=300 outd=runs/kpt222mp submit=1`
- MP 2x2x2 (4 k-points, 1.8x faster than Gamma-centred, as accurate), KPAR 1; keys `mesh` (mp222 | g222), `kpar`, `n`, `outd`, `nodes` (20), `minutes` (120), `submit`.
- check before submitting: `python3 mlp_pipeline/kpt_sanity.py runs/kpt222mp runs/kpt222mp2 --ref runs/kpt222/frames.extxyz`.
- after: collect `$PY mlp_pipeline/points_collect.py runs/kpt222mp/points_* runs/kpt222mp2/points_* --out runs/kpt222mp/labelled_mp222.extxyz --db-set K222`;
  fit the correction `$PY mlp_pipeline/kpt_correct.py fit runs/kpt222mp/labelled_mp222.extxyz --gamma runs/kpt222/frames.extxyz` (residuals by group / composition, leave-one-group-out CV, dP consistency, coverage);
  apply to tiers `$PY mlp_pipeline/kpt_correct.py apply runs/kpt222mp/kcorr.json runs/train_mfin.extxyz runs/valid_mfin.extxyz` (energy + stress shifted, forces unchanged, extrapolated frames counted).

## Not (yet) behind run.sh
- PWidom chemical potentials: `mlp_pipeline/pwidom_states.sh` / `perlmutter_pwidom.sh` (headers and RUNNING.md).
- DFT labelling campaigns of snapshot databases: `snapdb.py`, `lowp_quick.sh`.
- Training tiers: `build_tier.py` (exact commands in docs/HANDOFF.md section 2.1), `grace_prep_pkl.sh`.
- T-x phase diagram from free energies: `tdi/tx_diagram.py` (header).

## Earlier quadrature findings (old ft_m2 model; redo with the final model)
From docs/CURRENT_ISSUES.md (2026-10-27/28): Gauss-Kronrod 10 -> 21 lambda moved the H2.25O crystal G by -14.1 meV/atom but H2O by only -1.4 (x13 amplification per added H: ~85 meV/H quadrature uncertainty of the crystal mixing value at 500 GPa / 6000 K); crystals that melt in free MD but are held by the springs show huge Kronrod - Gauss (-327 / -451 meV/atom: not valid solids); MBAR over the saved lambda series (`tdi/bar_lambda.py`) is a quadrature-free cross-check; the insertion runs' quadrature error (P3) was never measured. Suggested test with the final model: gk17 for fcc H2O, fcc H2.25O, liquid H2O and pure H at 500-600 GPa / 5000 K, one GPU each.

## Young: current potential and first GPU check

Young simulation launchers default to the explicit SavedModel export
`grace-ft:runs/grace/ft_mfinkc_cont/seed/1/final_model`; `MODEL` overrides remain
available. Historical analysis/replay scripts retain the model in their source
metadata, and training/foundation-model comparison scripts retain their purpose.
PWidom and proxy-fit wrappers explicitly pass the selected model (even with
`--like`) and default to new `*_mfinkc_cont` output directories.
For new TDI campaigns explicitly set `OUTDIR=runs/tdi_mfinkc_cont` (the fcc/bcc
array uses `FBOUT`); old output directories are for continuing their original runs.

On a Young GPU node, from the repo root:

```bash
PY="$HOME/grace-venv/bin/python"
CUDA_VISIBLE_DEVICES=0 "$PY" mlp_pipeline/young_model_check.py
```

This reads the first `runs/valid_m.extxyz` frame and evaluates energy, forces and
stress; it performs no MD and writes no results. Override `--frames` if needed.
A passing check establishes model/environment compatibility, not accuracy or
TDI convergence. Model weights are on the cluster and cannot be tested in the
cloud checkout.

### Young ten-minute liquid pilot

Inside an interactive Young GPU allocation, run
`bash mlp_pipeline/young_liquid_pilot.sh`. Uses the first Slurm-assigned GPU,
96-atom H2O at 350 GPa / 7000 K, four Gauss–Legendre lambda values, 0.2 ps
at 12000 K, 0.2 ps equilibration and 0.25 ps production per lambda, dt 0.5 fs.
DFT hull volume lookup gives 2.52062075 A^3/atom. Approximate ten-minute target
includes model startup; no runtime guarantee. Output defaults to
`runs/tdi_mfinkc_pilot_350GPa_7000K`; repeat resumes that pilot. Finishes with
`liquid_check.py` (oxygen MSD/diffusion); short trajectories may not establish
phase. Four lambda values and this sampling length are for execution testing,
not production hull accuracy. Use a new OUTDIR for changed pilot settings.

Extend this pilot in the same directory with
`PS=1 bash mlp_pipeline/young_liquid_pilot.sh`: PS is the **total** production
length per lambda, so a completed 0.25 ps pilot adds 0.75 ps. `PS=1.25` adds
1 ps to the original pilot. Settings, tag and directory stay fixed. Saved
positions/momenta and samples are reused without reheating or equilibration.
`RUN_DIARY.md` and `RUN_DIARY.jsonl` record each invocation, target, sample counts
and exit code; finish checks the old sample prefixes are unchanged. An exit code
alone does not establish convergence or that every lambda reached its target;
inspect the final collection table. Do not run two extensions concurrently.
The legacy TDI state does not store the thermostat RNG state: continuation is
not bit-identical to an uninterrupted trajectory (unlike PWidom checkpoints).
Restart observer cadence was fixed to avoid a duplicate checkpoint sample;
CPU toy integration verified 20 samples extend to 40 with the old prefix intact.

### Saved-data liquid hull dashboard (CPU only; Young or Perlmutter)

```bash
"$PY" mlp_pipeline/tdi/hull_dashboard.py runs/tdi_mfinkc_pilot_350GPa_7000K --P 350 --T 7000 --out runs/plots/liquid_pilot
```

Produces PNG, PDF and a numerical CSV. One column per composition: integrand
<dU/dlambda> (log lambda), cumulative G with five-block statistical SEM, and
high-lambda temperature/full-MLP pressure history. No trajectory reads, MSD,
model evaluations or MD. An incomplete/single-system report still plots available
diagnostics; mixing needs completed liquid H2O and pure-H references. Mixed
model specifications, duplicate compositions and unequal quadrature grids in
mixing are refused. This is a provisional diagnostic, not automated certification:
model paths do not establish identical weights, saved metadata must be reliable,
and phase, finite-size, timestep and sampling errors need independent checks.

The top row shows mixing free energies and the sampled lower convex hull versus
**added-H atom fraction** x=(H/O−2)/(H/O+1), matching the per-atom energy basis.
An atomic O/H ideal compositional-entropy reference is overlaid, with a second
panel showing TDI minus that reference. This is the convention in tx_diagram.py,
not an ideal mixture of intact H2O molecules and H2. It is a chosen baseline,
not an additional entropy correction to the TDI. Different density/association
models can define different ideal references. Endpoint differences are exactly
zero by construction; mixture errors share correlated endmember uncertainties.
Error bars show marginal statistical errors only, not independent hull-node
uncertainties. Cumulative prefixes are correlated. Lines between integrand
samples guide the eye; the numerical integral uses the saved quadrature weights.
For lambda^power integration, the existing plot_tdi.py additionally shows the
transformed coordinate and individual weighted contributions. Run it when
diagnosing low-lambda quadrature. Phase checks remain optional separate work
with liquid_check.py; high-lambda diagnostics do not replace full-potential MD.
