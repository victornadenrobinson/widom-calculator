# Young EOS table export, 2026-11-12

Original full-precision CSV copies from the user-supplied Young archive and its
manifest. All 31 table SHA-256 hashes in the manifest were verified on import.
The manifest names original Young paths; run trajectories remain on Young.
Table rows are reported results, not automatically validated liquid phases,
converged calculations, or proof of identical checkpoint weights.

The separate energy-window file was supplied by the user; it assumes 250 fs
initial equilibration for selected mixture runs and must be checked against
run settings. Model potential plus classical nuclear kinetic energy is not
necessarily a DFT thermodynamic internal energy. Store raw trajectories and
DFT labels outside Git. Beware duplicates with rounded transcriptions.
