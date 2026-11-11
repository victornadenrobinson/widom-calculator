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
