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

## Purpose and immediate priorities

This is the curated numerical record for calculations with the final potential,
not an assertion that every saved result is converged or publication-ready.
**First priority: liquid EOS and liquid composition convex hulls.** Solid and
superionic EOS, transport and structure support phase identification and later
solid/liquid comparisons. Raw trajectories remain in the cluster run directories.

The initial composition grid is H2O, H2.25O, H3O, H5O, H7O and pure H.
Add compositions where hull curvature, phase separation or the experimental
sample compositions require them; this is not a fixed exhaustive grid.
Target approximately 10–500 GPa, up to 9000 K, above the user-drawn lower
boundary. That outline is a sampling target, not a measured melting line.
Prioritize 300–400 GPa and 5000–7000 K for the draft's experimental planning
window, with 350 GPa as the initial mixing comparison. Track missing pressure
brackets explicitly rather than extrapolating an EOS to fill gaps.

## Planned datasets

The paths below are a proposed organization; only files in the inventory below
exist so far. Preserve the current reported tables until full source tables are
imported with their provenance.

| Proposed dataset | Main quantities | Purpose / priority |
|---|---|---|
| `eos/liquid_points.csv` | Composition, N_H/N_O, N, V, density, target and measured T/P, pressure error and drift, potential and total internal energy per atom | **Now:** liquid EOS, volume selection for TDI, energy EOS for later Hugoniots |
| `eos/solid_superionic_points.csv` | Same thermodynamic fields, oxygen lattice, mobile species, phase diagnostics | Distinguish crystalline, superionic and liquid branches; retain metastable branches separately |
| `eos/coverage.csv` | Model, system, temperature, pressure range, sampling counts, phase status, source | Track present and missing coverage across both clusters |
| `eos/fits/` | Fit parameters, valid P/T domain, residuals, uncertainty and source-point IDs | Interpolate volumes; retain points as the primary data and avoid unsupported extrapolation |
| `free_energy/points.csv` | F and G(P,T,composition), normalization, reference, phase, volume, target/measured pressure, corrections and errors | **Now:** comparable thermodynamic values for liquid mixtures and endpoints |
| `free_energy/integrands.csv` | Lambda, quadrature weights, mean dU/dlambda, errors, equilibration/production lengths | Reproduce each integral and diagnose sampling/quadrature sensitivity |
| `free_energy/convergence.csv` | G versus sampling time, quadrature order, cell size, timestep and independent seed | Quantify numerical convergence separately from model error |
| `hulls/liquid_mixing.csv` | Composition coordinate, delta G_mix, uncertainty, endpoint IDs, ideal-reference/excess values where defined | **Now:** liquid mixing curves at common P,T |
| `hulls/liquid_hulls.csv` | Sampled lower hull, supporting vertices/tie lines, distance above hull, uncertainty analysis | Locate candidate demixing intervals; add compositions near uncertain curvature |
| `transport/diffusion.csv` | D_H, D_O, MSD fitting windows, errors, run length, thermostat and system size | Liquid/superionic discrimination and transport trends |
| `structure/rdf.csv` | g_HH(r), g_OH(r), g_OO(r), binning, normalization, sampling windows | Pair structure and reproducible coordination analysis |
| `structure/coordination.csv` | Coordination-number distributions and means, species pairs and cutoff definitions | C_N, oxygen order and composition-dependent local environments |
| `structure/speciation.csv` | Molecular/association fractions, bond definitions and optionally lifetimes | Track H2-like pairs and O–H environments without assuming stable molecules |
| `structure/concentration.csv` | Partial/concentration structure factors or spatial composition correlations | Detect segregation; quantify finite-size and sampling limitations |
| `validation/metrics.csv` | Energy/force/stress MAE and RMSE, bias, dataset/phase/composition/P/T breakdown | Model accuracy and transferability; distinguish training/validation/test sets |
| `validation/mixing_errors.csv` | Relative energy, composition/pair tests, k-point correction residuals and EOS deviations from DFT | Errors relevant to small mixing free energies, not only aggregate accuracy |
| `validation/stability.csv` | Temperature/energy drift, close contacts, timestep and out-of-domain checks | Document reliable simulation conditions |
| `performance/timings.csv` | Hardware, GPU count, workers/GPU, size, density, MD rate and timing scope | Plan allocations with comparable measurements |

Optional later derivatives include excess mixing volume, compressibility/bulk
modulus, thermal expansion and heat capacity. Derive these only with enough
P/T sampling and propagated uncertainty. Hugoniots require consistent energy
zeros and explicitly documented initial states. Electronic conductivity and
reflectivity require separate electronic calculations; ordinary classical MLP
trajectories alone do not provide them.

## Definitions and acceptance information

- Store N_H and N_O, not only a composition nickname. For the H2O–H hull use
  `x = (N_H - 2*N_O)/(N_H + N_O)`, the added-H atom fraction, and G per atom.
  Then `delta_G_mix = G - (1-x)*G_H2O - x*G_H` at common P,T. Pure H has x=1.
  A molecular fraction or per-formula-unit normalization needs its own explicit
  conversion; do not combine them in one hull.
- Negative mixing free energy relative to endpoints does not establish complete
  miscibility. Inspect convexity and possible intermediate tie lines. A hull of
  sampled compositions is provisional; correlated endpoint errors matter.
- Define any ideal-mixing entropy reference explicitly. Atomic, molecular and
  pseudocomponent references differ. Keep total mixing and excess values separate.
- Record intended and observed phase separately. Mobile H with an ordered,
  non-diffusive O sublattice is consistent with superionic behaviour; neither an
  input `liquid` flag nor an `other` structure label proves a liquid.
- Keep statistical, quadrature, finite-size and model uncertainties distinct.
  Record discarded equilibration, sampling windows, autocorrelation/blocking
  method and checkpoint consistency. Report corrections both before and after
  application, never silently fold them into G.
- Each row needs a stable run/state ID and source path; record model fingerprint
  when available, code commit (including local changes), units, extraction script,
  date, and status (`reported`, `screened`, `convergence_checked`, or `excluded`
  with a reason). Missing values must not become zeros. Keep literature and DFT
  comparison points identified by their own method and source.

## Existing validation sources to curate

Start with [the model-development handoff](../../docs/HANDOFF.md),
[reported model accuracy](../../docs/data/model_accuracy_reported.csv), and the
scoring/transferability scripts listed there. Preserve each metric's reference
labels: Gamma-point and k-point-corrected DFT comparisons are different tests.
The handoff reports continuation from `mfin_f_lr1e-2_e50` for 21450 updates,
and corrected-label test E RMSE 9.7 meV/atom, force MAE 0.201 eV/angstrom,
and stress RMSE 2.2 GPa. These are reported historical values, not independently
recomputed here. Curate composition/pair tests and correction transfer alongside
the aggregate errors, with their original dataset definitions.

## Connection to the paper draft

The supplied 4 November 2026 draft, *Hydrogen–water mixing under shock
compression of pre-compressed clathrates: laser-driven experiments and ab initio
free energies*, needs consistent free-energy normalization, uncertainties and
additional H5O/H7O compositions for convexity, plus EOS deviations from linear
mixing. These map directly to the first-priority tables above. Retaining internal
energy alongside P and V supports its later Hugoniot work. Keep MLP results,
AIMD data and experimental/model-derived shock states separately identified;
these new calculations do not by themselves resolve experimental state assignments.

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

`perlmutter_tdi_500GPa_5000K_reported.csv` transcribes the four rows
of the user's `tdi/summarize.py runs/tdi_final/` output. Statistical errors,
drift and pressure diagnostics are as printed; pressure corrections are not
applied. These are short, preliminary runs and the listed errors are not total
uncertainty. The stage0 metadata recorded `grace-ft:runs/grace/ft_mfinkc_cont`.

## Evidence on the six-composition 7000 K EOS figure

The figure shared in chat has no model identifier. The repository handoff
(`docs/HANDOFF.md`, EOS section) reports six-composition 7000 K EOS work with
`mfin_f_lr1e-2_e50`, followed by a plan to rerun with `mfinkc_cont`. This makes
the older model a plausible source, but the figure is not uniquely identified.
The user's Perlmutter inventory separately found final-model 7000 K directories
for H2O, H3O and H and logs naming the final model. Its final-model 1000/2000 K
directories have all six compositions, but names alone do not establish
completed pressure points or liquid phase. The shared Claude conversation says
that pending EOS jobs were to be cancelled and moved to Young; it does not prove
which specific jobs completed. Figure-only points must not be relabelled as
final-model results or used as precise numerical EOS data.

## Explicit liquid EOS target grid

`eos_liquid_targets.csv` lists 468 planning targets across all six compositions.
Temperatures are spaced by 1000 K and pressures use 10, 25, 50, 75, 100, 150,
200, 250, 300, 350, 400, 450, 500 GPa, restricted by the approximate drawn lower
boundary. All rows are targets, not completion claims. Existing volume scans
can provide bracketing points; this is not an instruction to submit 468 new MD
runs. Refine near melting, demixing and the paper window as needed.

Reproduce the plan plot and overlay locally available final-model summary CSVs:

```bash
python mlp_pipeline/eos_state_plan.py --root runs/eos_live --out runs/plots/eos_state_plan
```

Open circles show planned P,T; filled blue circles show reported EOS results at
measured P and set T, red crosses show ordered oxygen labels. The plot includes
the rounded Perlmutter table, and local roots add full EOS tables. Observations
are not snapped to targets or automatically certified as converged liquids.
Young results and Perlmutter results not yet copied into the repository are
otherwise absent. Known 1000/2000 K directories have no plotted points until
numerical pressure rows are supplied.

The command also writes an energy CSV/plot when full local EOS tables are
available. Existing summary tables save `epot_per_atom`; per-volume `thermo.csv`
saves whole-cell `epot` and `ekin` versus time. The energy report recovers mean
nuclear kinetic energy using the writer's temperature convention
`K/N = 3*(N-1)*kB*T_mean/(2*N)` and reports `epot + K` per atom. This is the
model potential plus classical nuclear kinetic energy. Since the model's DFT
labels can be electronic free energies (TOTEN), an electronic entropy correction
may be needed to interpret it as thermodynamic internal energy for Hugoniots.
No such correction or energy-reference alignment is silently applied. Missing
energy columns are omitted, never imputed from the rounded timing transcription.

`young_eos_7000K_reported.csv` transcribes the 48 completed H2.25O/H5O/H7O
rows printed by the user's Young dashboard on 2026-11-11. All values retain the
printed (rounded) precision; flags are preserved, including CLOSE-HH and T-off.
No potential-energy values were printed in this table. Source paths are
`runs/eos_live/ft_mfinkc_cont_<composition>_7000K_<N>at/<tag>.csv`.

`eos_7000K_refine_volumes.json` proposes 12 additional volumes at 7000 K:
H2.25O near 300/350/450 GPa, H5O near 300/350/450/500 GPa, H7O near
300/350/400/450/500 GPa. These initial guesses interpolate V linearly in log P
from the rounded reported rows; actual measured pressure can differ. Flagged
rows are permitted only as volume guesses, not certified EOS or free-energy
inputs. These are additional points, not extensions or replacements of flagged
runs. The original short H2O high-pressure runs retain bcc oxygen order and do
not establish liquid coverage; their bcc starting lattice may not have melted.
