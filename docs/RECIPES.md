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

### Quadrature extension of the Young pilot

After the time extension has finished, use
`QUAD=gk9 PS=6.25 bash mlp_pipeline/young_liquid_pilot.sh` to retain the four
GL4 lambda trajectories and add five Kronrod trajectories at the same target
length. Completed old lambdas are skipped; incomplete ones continue. Keep the
same output directory and do not run concurrent extensions. New trajectories
need their own hot preparation/equilibration and full requested production.
`run_history/` archives prior stage0/collected JSON metadata before each launch;
the diary records planned quadrature and checks old sample prefixes afterward.
The pilot now infers the existing rule and time target when QUAD/PS are omitted,
so a later time extension preserves GK9 rather than reverting to GL4.

Supported reuse is GL(n) -> GK(2n+1): **4 -> 9**, **8 -> 17**, **10 -> 21**,
**15 -> 31**. These are separate families, not a reusable sequence from 4 to 31.
The present code is not a Gauss–Kronrod–Patterson hierarchy: GK9 -> GK19/21
changes existing nodes and is refused. A higher family requires a fresh output
directory and new sampling. A genuinely repeatedly nested rule would need a
separate validated implementation. Embedded Kronrod−Gauss differences estimate
quadrature sensitivity; they also contain sampling noise and are not rigorous
error bounds. Compare at equal production cutoffs; missing new lambdas must not
be presented as a completed Kronrod integral.

Actual quadrature overlap is plotted in `docs/figures/quadrature_overlap.png`
(and PDF), with all values/node indices in `docs/data/quadrature_lambda_values.csv`.
Recreate with `python scripts/plot_quadrature_overlap.py`. The comparison uses
the implementation's power-four lambda mapping and relative tolerance 1e-12.
Different GK9/GK17/GK21/GK31 families share **one** midpoint at lambda=1/16
(x=1/2 before mapping); GL15 shares that midpoint too. The even GL4/8/10/32
rules otherwise do not share nodes across the listed families. Therefore saying
there is no reuse *between* pairs is too strong: one trajectory is physically
reusable across these GK rules, but the current same-directory continuation
requires preservation of the entire existing node prefix and does not remap
that isolated shared trajectory into a new family. Do not manually copy node
files by their integer indices; those indices denote different lambda values.

### Adaptive multi-GPU Young H2O quadrature study, 384 atoms

On the Young login node, request an interactive GPU allocation (check local
partition availability):

```bash
srun -p intergpu --gres=gpu:4 -c 32 --mem=120G --time=2:00:00 --pty bash
```

Inside it:

```bash
cd /lustre/scratch/mmm0037/H3O &&
git pull --ff-only origin gpt/youthful-jolly-review &&
"$HOME/grace-venv/bin/python" mlp_pipeline/young_quad_sweep.py --ps 6.25
```

A shared task queue covers GL4→GK9, GL8→GK17, GL10→GK21 and GL15→GK31.
The runner discovers all Slurm-assigned CUDA_VISIBLE_DEVICES (one, two, four or
more GPUs). Each free slot receives a waiting lambda from any ready family;
GPUs are not reserved per family. Default four workers per GPU; --per-gpu
overrides the cap. --families 4 10 selects only GL4/GK9 and GL10/GK21. Default 350 GPa / 7000 K H2O: **128 O + 256 H =
384 atoms**, bcc NREP=4 initial lattice, heated at 12000 K for 0.5 ps and
equilibrated at 7000 K for 0.5 ps before production at each lambda. This initial
lattice is for liquid preparation, not a bcc solid free-energy calculation.
`--P`, `--T`, `--root`, `--model`, `--ps` are available. Choose a distinct --root
for different conditions; a saved campaign configuration guards against mixing.

Interactive stdout is an updating dashboard with per-GPU task assignments and
per-family progress bars (no whole-screen clearing);
non-terminal output emits periodic snapshots. Complete launcher/node logs and
RUN_DIARY files live under `runs/quad384_350GPa_7000K/gl<N>_gk<M>/`.
Slurm errors injected by the allocation can still disrupt terminal rendering;
fix/filter the allocation's stderr separately rather than hiding model errors.
Each child receives one ID
from CUDA_VISIBLE_DEVICES, including UUID identifiers.

Rerun the same command to resume interrupted sampling, or increase --ps to extend
it. Existing samples at matching lambda indices/settings are kept; after a
family reaches GK, restart skips its GL stage. Completed lambda processes still
may load a model to verify/recover state but do not repeat the production MD.
Nothing from the 96-atom pilot can be reused as 384-atom MD. Separate families
repeat non-overlapping nodes; isolated common midpoints are not remapped.
Every stage0 is validated before scheduling its lambdas; existing complete
sample/checkpoint pairs are skipped only when model, lambda, timestep and
temperature match. Full model identity still requires immutable weights. The
code is snapshotted once per invocation, preserving a running campaign across
pulls. Do not share this root with a separately running pilot/launcher or run
other work on the same GPUs while using this scheduler.
All 31-node work is not guaranteed to finish within two hours. Restart inside a
new allocation if needed. Ctrl-C terminates child process groups; saved completed
chunks can be resumed. Check exit status/logs and completion, then perform
phase/convergence analysis. This new four-GPU runner has local geometry,
quadrature and syntax checks; actual four-GPU GRACE throughput awaits Young.

The adaptive quadrature runner accepts `--oxygen 128 --hper 2.25` for a
128 O + 288 H = 416-atom H2.25O liquid. `--oxygen` counts oxygen atoms (and
therefore H_xO formula units), not total atoms. The cubic starting-cell options include 16, 32, 54, 64, 108, 128 and 250 O atoms. H/O times the
oxygen count must give a whole number; otherwise the runner rejects it rather
than silently rounding the composition. The available DFT hull EOS lookup
covers H/O 2–7; intermediate compositions use interpolated volumes. Pure H
has no oxygen and needs a separate hydrogen-cell interface, not --oxygen.
The default H2O campaign keeps its original directory; other compositions and
sizes get automatically named roots. Explicit --root remains available and
existing campaign metadata refuses incompatible settings. Example (inside a
Young GPU allocation):

```bash
"$HOME/grace-venv/bin/python" mlp_pipeline/young_quad_sweep.py --oxygen 128 --hper 2.25 --ps 6.25
```

The runner prints O, H and total atom counts. The default is `--oxygen 128
--hper 2` (384 atoms), so existing H2O invocations and campaign configuration
remain compatible. For a systematic finite-size comparison, compare more than
one oxygen count and consider the differing initial-cell geometry; equal O
counts give different total atom counts as H/O changes.

To monitor a sweep that is already running, from a **separate Young login-node
terminal** (read-only; Ctrl-C stops only the viewer):

```bash
cd /lustre/scratch/mmm0037/H3O && git pull --ff-only origin gpt/youthful-jolly-review && "$HOME/grace-venv/bin/python" mlp_pipeline/tdi/watch_quad.py runs/quad384_350GPa_7000K
```

The viewer reads stage metadata and progress JSONs: it shows family/lambda,
percent, steps/s and ms/step when available, and marks stale files. Without a
progress file, "waiting or loading" covers both queued and model compilation;
it cannot infer GPU assignment from these files. Progress appears only after a
worker loads/compiles the model; updates are approximately every 30 seconds.
`--once` prints one snapshot. This viewer does not start, stop or modify jobs.
For future runner invocations the main dashboard also shows per-GPU worker
families, startup times, progress and rates. Already-running runner processes
continue to use the code they started with; pulling does not alter those workers.

**Size update:** `--oxygen 64` uses a 4×4×4 simple-cubic oxygen start (64 O, 192 atoms
for H2O); `--oxygen 108` uses 3×3×3 FCC (108 O, 324 atoms for H2O), and
`--oxygen 32` uses 2×2×2 FCC. The 64-O simple-cubic, 128-O BCC and 108-O FCC starting cells are cubic;
account for different starting lattices and test melting and equilibration. These are initial structures that must melt. Existing 128-O
BCC campaign paths and tags are unchanged. New run directories get a colored
new/resume/already-complete message in an interactive terminal; no automatic
cleanup is performed. The existing stage0 and collected metadata are archived
in run_history/ before updates. Model, size, composition and settings must
match for restart. Only saved completed chunks are resumed.

Compare the completed 32-O quadratures using a fresh figure prefix:

```bash
"$HOME/grace-venv/bin/python" mlp_pipeline/tdi/plot_quad_compare.py \
  runs/tdi_mfinkc_pilot_350GPa_7000K runs/tdi_qtest_gl* \
  --out runs/plots/quad_O32_comparison
```

The tool uses the common production-time cutoff for the summary, graphs the
full available cumulative G(t), derives embedded GL4 from GK9 (and analogues
for other families), and skips missing samples. It refuses mismatched recorded
model paths, P, T, cell size, volume and composition. If only the GL4/GK9 pilot
exists, only those two curves appear. Statistical errors and the differences
between related quadratures are correlated; error bars are not independent
confidence intervals. The figure does not establish phase or convergence.

The adaptive scheduler defaults to **four workers per GPU**, staggered by
three seconds between starts across the allocation to reduce simultaneous
model-loading/compilation pressure. Override with `--per-gpu N` and
`--launch-delay SECONDS` (0 disables staggering). Startup spacing changes job
scheduling only; it does not change or clear saved samples. More than four
workers may improve throughput on small cells, or may only increase memory use
and contention: measure steps/s and GPU memory before changing production runs.

For a quadrature convergence discrepancy, use `tdi/quad_diagnostics.py` with
explicit run directories. It checks each checkpoint's lambda, recorded model
path and sample series; reports each node's weighted first-to-second-half shift,
full and last-half free energies, and GK−GL with shared-node covariance included
(assuming independently sampled lambdas). Its five-block SEM may underestimate
correlation, and this is not an independent proof of equilibration, identical
weights, phase, or absence of systematic quadrature error.
