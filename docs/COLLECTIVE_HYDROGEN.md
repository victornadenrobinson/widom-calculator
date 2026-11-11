# Collective pure-H trajectory analysis

This CPU analysis reads existing extxyz trajectories; it launches no MD and loads no GRACE model. Use the Python environment used for EOS runs, containing ASE, NumPy, SciPy, and Matplotlib.

From the repository root on Young:

```bash
python -u mlp_pipeline/collective_hydrogen.py --root runs/eos_live --model ft_mfinkc_cont --temperature 2000 --out runs/collective_h_2000K
```

Use an allocated CPU session for a full campaign. Calibration is performed separately at each measured density and cached under the output directory. Defaults use 720 reference configurations per density (64 atoms each). The run reads trajectories one frame at a time, retaining only structural features. CPU/memory cost per frame is quadratic in atom count.

Outputs: summary.csv, collective.png, collective.pdf, settings.json, per-run *_fit.npz arrays, and cache/*.npz. Pressure comes from the matching EOS summary row by volume per atom; missing or ambiguous rows are reported and excluded from pressure plots. Existing output files in the selected output directory are replaced; trajectory inputs are read only.

The estimator uses pair counts per atom in 24 distance bins, plus ten bins of nearest/second-nearest distance ratios. It jointly fits molecular fraction and the nominal bond-length and width parameters of synthetic reference mixtures. No neural network or individual-atom classifier is used. Interpolation across reference fractions and bond lengths is allowed; width is selected from three reference widths. Fitting is weighted least squares with a diagonal noise scale, not a calibrated Bayesian posterior or a molecular partition function.

References are random sequential insertions with a nonpartner exclusion distance of 0.54 Å. Planted bond lengths follow clipped Gaussian distributions. The references are not equilibrated hydrogen. Reference atom count differs from the production system, which may introduce finite-size bias. Density is matched, but the packing model and electronic bonding are not validated. A small residual is necessary but not sufficient for a meaningful molecular fraction. Standardized mismatch is a diagnostic relative to synthetic single-frame variability, not a p-value or reduced chi-square; it is not an acceptance threshold.

The first half of saved snapshots is discarded by default (--discard). Error bars are standard deviations of fractions fitted to contiguous blocks of 20 saved frames (--block-frames), not confidence intervals. At 20 fs spacing this is 400 fs per block. Trajectory autocorrelation may extend beyond a block. Remainder frames contribute to the overall estimate but not to complete-block statistics. Fewer than two blocks yields no block SD. No real-time bond survival is inferred.

An exploratory sensitivity run can change --exclusion (e.g. 0.60), --reference-atoms (e.g. 128), and --discard using separate output directories. The script deliberately rejects nonorthorhombic cells, changing densities, and non-pure-H inputs. Compare fits to observed feature distributions in the NPZ outputs before interpreting pressure trends. Do not rescale endpoints or impose monotonicity to make a transition sharper.
