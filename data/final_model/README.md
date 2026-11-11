# Final-potential results

This directory holds small, curated tables and their provenance. The current
potential is **GRACE-kc1** (a suggested working name; `mfinkc_cont` remains the
actual run/model identifier). It is the continued fine-tune on k-point-corrected
labels. Do not change model paths or identifiers in existing run metadata.

- Model run: `runs/grace/ft_mfinkc_cont`
- Export used in the recorded logs: `runs/grace/ft_mfinkc_cont/seed/1/final_model`
- The Perlmutter-to-Young transfer archive had SHA-256
  `d96cdeee6b960020c221046ffe9757d15ac03293b21334212916ce095dbe740b`.
  This is an **archive** checksum, not a checksum recorded by each run.
- The model name/path is not proof that two runs used identical weights. Keep
  original checkpoint fingerprints, commands and software provenance where available.

## Tables

`perlmutter_eos_7000K_reported.csv` is a **partial, rounded transcription**
of the user's printed Perlmutter EOS timing table on 2026-11-11. It has 16
reported volume points each for H2O (384 atoms), H3O (512) and H (729).
Sources are the same-named CSVs in Perlmutter's
`runs/eos_live/ft_mfinkc_cont_{h2o,h3o,h}_7000K_*at/`. Pressure is rounded
to 0.1 GPa, volume to 0.0001 A^3/atom, and timing to 0.1 ms/step in the
printed source; the oxygen diffusivity is as printed. Do not treat this as a
full-precision replacement for the original tables. `O_struct` is blank for H;
`other` does not by itself establish a liquid. The timing can include analysis
overhead, and production trajectories were short (0.75 ps for mixtures,
0.5 ps for H). Phase, equilibration and systematic errors remain to be checked.

The reported Perlmutter inventory also lists 1000 K and 2000 K EOS tables for
H2O, H2.25O, H3O, H5O, H7O and H, but their data have not been copied here.
The independently generated 11-row screened starting-volume CSV remains on
Perlmutter; it is not included because its numeric contents were not shared.
Young results and the remaining EOS grids have not yet been curated here.

When adding a dataset, preserve source precision and record the source path,
model identity, system, thermodynamic conditions, method, units, collection date,
uncertainty and known limitations. Store large trajectories, model weights and
checkpoints outside Git; keep these files as compact, reproducible summaries.
