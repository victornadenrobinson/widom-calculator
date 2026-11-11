# EOS coverage snapshots (11 November 2026)

These are historical snapshots generated in the Codex workspace, not live Young
results and not publication-ready phase diagrams. The `eos_rillo` PNG/PDF is
the latest and most complete here. See `../README.md` for underlying data,
provenance and instructions for regeneration with Young `runs/eos_live`.

- `perlmutter_eos_shared_7000K.png`: 48 rounded Perlmutter points shared in chat.
- `eos_state_plan`: initial target grid and Perlmutter points.
- `eos_state_plan_updated`: target grid plus reported Young 7000 K mixture points.
- `eos_literature`: preliminary literature overlays.
- `eos_dft_coverage`: project DFT EOS points and literature context.
- `eos_literature_hydrogen`: DFT and hydrogen reference estimates.
- `eos_rillo`: latest: above plus approximate visual trace of Rillo CEIMC H line.

Open circles are planning targets, not missing-run flags; filled points are
reported calculations, not proof of liquid phase. The blue Rillo line is a
manual figure trace with estimated reading uncertainty >=5 GPa and >=50 K, not
original numerical data. Reference models and their uncertainty differ. These
snapshots predate the current 2000/3000 K Young calculations. Companion PDFs
exist for the state-plan figures. Future high-volume live plots and trajectories
belong under `runs/`, not automatically in Git.
