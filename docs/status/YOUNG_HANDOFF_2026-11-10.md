# Young migration and TDI campaign handoff — 2026-11-10

## Sources and scope

The user supplied the closing excerpt of their [Claude session](https://claude.ai/code/cshk_01RSfmJQJ1YsEG9jGzMhLXHJSZQv4bhL31kf1yVNVgXzNi), beginning “the training and the DFT labels are done”. The link itself was inaccessible from the cloud environment. This note incorporates the supplied excerpt, subsequent user-pasted cluster output and current repository code. It does not claim to summarize the full Claude conversation. Older proposed work in `docs/HANDOFF.md` may already be superseded by its later updates or this note.

## User preference: liquid cell geometry

Use cubic simulation boxes for liquid work, including finite-size comparisons. For 64 O, use a 4×4×4 simple-cubic oxygen starting grid with well-separated hydrogen positions, then premelt and verify liquid equilibration. FCC 3×3×3 gives 108 O; BCC 4×4×4 gives 128 O. Avoid rectangular liquid starting cells unless the user requests them.

## Goal and working arrangement

Prioritize liquid TDI across composition for the convex hull, with H2O melting and fcc–bcc TDI on Young GPUs. Use the k-point-corrected GRACE potential `mfinkc_cont`. Develop, test and push on `gpt/youthful-jolly-review` here; the user pulls and executes on the clusters. Give short, self-contained commands with the cluster/node, repository path and Python interpreter specified. No cluster access exists from this cloud checkout.

- Young repository: `/lustre/scratch/mmm0037/H3O`; GPU Python: `$HOME/grace-venv/bin/python` (user `mmm0037`). GPU scheduler is **Slurm**, with existing `gpu` / `allusers` / `freegpu` batch headers. The user uses interactive `srun -p intergpu --gres=gpu:3 -c 8 --mem=60G --time=2:00:00 --pty bash`.
- Young CPU nodes use **SGE** and a separate compatible environment. The Claude excerpt's blanket statement that Young uses SGE is incorrect for the GPU workflow.
- Perlmutter repository: `/pscratch/sd/v/victornr/H3O`; Python: `$SCRATCH/grace-env/bin/python` (user `victornr`).
- GitHub repository is `victornadenrobinson/widom-calculator`; the old `H3O` remote redirects. The local folder name need not change.

## Confirmed migration state

The user transferred a 23 MB archive of the whole `runs/grace/ft_mfinkc_cont` directory from Perlmutter to Young using SSH streaming (the existing `send_bundle.sh` documents this direction and Perlmutter's `~/.ssh/id_young`). Young's shell emits startup text, which is why the established transfer path avoids SCP. Young SHA-256 reported:

`d96cdeee6b960020c221046ffe9757d15ac03293b21334212916ce095dbe740b`

The source checksum was not pasted; end-to-end equality has not been independently established here. Extraction and placement succeeded. The export is at `runs/grace/ft_mfinkc_cont/seed/1/final_model/saved_model.pb`.

Young successfully switched from local `proxyfit` to `gpt/youthful-jolly-review`. The eight untracked scripts blocking checkout were preserved in `runs/young_scripts_before_switch_iPZHPK`. Six matched the incoming files; `mdview_lit.py` and `toy_live.py` were older versions lacking later visualization controls. No unique proxyfit patches were reported by the cherry-pick-aware comparison. No script recovery is needed.

Commit `c7c7ded` updates Young simulation launcher defaults to the explicit final-model export, preserves MODEL overrides, and adds `young_model_check.py`. It has been pushed; the user's pull and GPU model-check result are **not yet reported**. Shell/Python syntax were checked locally; no real GRACE evaluation was performed here. Training and historical replay retain their own intended models.

## Transfer priorities from Claude, with corrections

For inference, copy the complete SavedModel directory, including variables/assets. The whole model run was copied here. For further training, `model.yaml` alone is insufficient: also retain the TensorFlow checkpoint files and training configuration, with compatible training data/environment.

Useful additional files, subject to checking their actual presence and versions:

- `runs/inventory_binned.csv`: TDI volume lookup; launchers may require the root `inventory_binned.csv` link.
- `runs/kpt222mp/kcorr.json`: correction provenance and application to new labels; not an extra correction to apply during inference with mfinkc_cont.
- `runs/eos_live/`, `runs/tdi_final/`, and `runs/tdi_qtest/`: existing results to analyse before duplicating calculations.
- `runs/dft_md_{400,500,600}.extxyz`, validation sets and appropriate labelled pair sets for scoring. Claude named `runs/dgcheck/labelled.extxyz`; newer training notes use `labelled_v2.extxyz`, so check which version the desired score used.

Raw VASP folders, training pickles and older potentials need not all move to Young for inference. Preserve anything needed for reproducibility in durable storage before Perlmutter scratch expiry; the excerpt proposed HPSS archiving but did not establish that any archive was made.

## Existing Perlmutter TDI: user-reported analysis

All four stage-0 files record `grace-ft:runs/grace/ft_mfinkc_cont`, dated 2026-11-06, code `50f641388af7cc116a03ad1178231a946373e444+local`, Python 3.11.18, ASE 3.29.0 and NumPy 2.4.6. The exact local code changes are unknown; the launcher normally saves code under `runs/code/`. The recorded model path is not a checksum of the transferred weights.

All are 500 GPa, 5000 K, dt 0.5 fs, 17 lambda values, 2 ps per lambda, reported done:

| system | atoms | G (meV/atom) | statistical error | drift (meV/atom) | Kronrod−Gauss (meV/atom) | P_hi (GPa) |
|---|---:|---:|---:|---:|---:|---:|
| pure H | 729 | 1060.71 | 0.73 | 0.33 | -1.09 | 501.93 |
| fcc H2O | 324 | 2296.32 | 1.86 | -0.31 | 1.48 | 503.90 |
| fcc H2.25O | 351 | 2248.20 | 1.61 | 1.31 | 1.07 | 503.08 |
| liquid H2.25O | 104 | 2234.51 | 6.23 | 4.55 | 8.97 | 496.73 |

The summary reports crystal mixing **+610.0 ± 30.55 meV/added H**, versus fcc H2O plus fluid H; second-order pressure adjustment +0.25 meV/H, not applied. This is an average at H/O=2.25, not a dilute chemical potential or proof of zero solubility. No liquid H2O or bcc run appears in the supplied `tdi_final` listing. The liquid–fcc difference (~−13.7 meV/atom) needs phase checks, convergence/quadrature and finite-size assessment (104 vs 351 atoms) before a melting inference.

Analyse with `tdi/summarize.py runs/tdi_final -o <fresh-output>/tdi_results.csv`, using the GRACE Python. Avoid a trailing slash: the current summary uses basename and otherwise emits an empty dir field. Follow with phase checks, convergence and reports scoped to these runs; do not silently combine different potentials or old campaigns. The `tdi_qtest` directory was shown by shell completion, but its contents have not been supplied.

## EOS jobs and performance

Claude introduced `run.sh eos ... where=batch gpus=1|2|4` in commit `2c143c3`, already in our branch. Perlmutter TDI batch mode requests one shared GPU. EOS defaults to four GPUs per temperature; one/two GPU jobs use the shared queue. The excerpt discussed equal idealized allocation charges for four GPUs × one hour versus one GPU × four hours, citing NERSC policy. Treat these as historical policy/scaling statements, not a verified current accounting guarantee: idle time, CPU/memory allocation and achieved scaling can change actual cost.

The user intended to cancel only pending Perlmutter `eos_` jobs and continue on Young. Claude provided a cancellation command, but no cancellation output or current queue was supplied. Running EOS jobs and TDI were intended to continue. Check existing results and current queue before resubmitting. EOS completion skipping is **per recorded volume in the matching result CSV**, not a universal per-temperature guarantee; transferred results, model tags and settings must match.

Young A100 benchmarks already favour **four lambda processes per GPU** (old ft_m2: pure-H ~22.66 steps/s, insertion ~11.48 steps/s). Later Perlmutter F10 benchmarks agree on four processes (~23.8 steps/s for fcc H2O). Start from this layout; validate compatibility and timing with the final model rather than repeating a full tuning sweep.

## Next actions

1. Pull the branch on Young and run `young_model_check.py` inside a GPU allocation; await actual output.
2. Complete phase/convergence analysis of `tdi_final` and inspect `tdi_qtest`; inventory finished EOS results and pending jobs.
3. Adapt cluster submission deliberately: **run.sh batch submission still hardcodes Perlmutter**. Updating Young launcher model defaults did not make run.sh cluster-aware. Several old Young arrays still hardcode cell sizes and parallelism; review them before a new campaign.
4. Use explicit model-specific output directories for new TDI (OUTDIR, or FBOUT in the fcc/bcc array), matched reference conditions and suitable liquid cell sizes. Start with the missing liquid H2O/reference checks at 500 GPa / 5000 K, then expand the composition grid.

## Further user-reported observations

Perlmutter `runs/eos_live/` lists both parent-model `ft_mfin_f_lr1e-2_e50`
and final-model `ft_mfinkc_cont` folders. For the final model the listing
includes H2O/H3O/H at 7000 K and several compositions at 1000/2000 K; the
listing alone does not establish how many volume rows finished or which curve
produced the user-provided image. The closing Claude excerpt records the user's
intent to cancel pending EOS jobs, not confirmation that it happened.

Young's 64-O/192-atom TDI run later reported ~7.4 MD steps/s per worker × 4
workers = ~30 aggregate steps/s **per GPU** (reported live), after warming up.
This is comparable to the documented fcc H2O 324-atom/F10 ~23.8 steps/s per
GPU with four workers; earlier low reported speeds were affected by the old
heating/progress timer. These are different models, cells and conditions, so
not a controlled scaling comparison.
