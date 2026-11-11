# Hydrogen and water phase references for EOS coverage

The markers on the P–T coverage maps are literature context, not model results
or definitive phase boundaries. Verify the original papers and methods before
using a curve in a publication.

- [Cheng et al., Nature 585, 217–220 (2020)](https://doi.org/10.1038/s41586-020-2677-y),
  *Evidence for supercritical behaviour of high-pressure liquid hydrogen*.
  Their MLP study reports a continuous molecular-to-atomic transition in the
  liquid and no first-order transition above their melting line. This is a
  different conclusion from some other calculations; it is not plotted as a
  sharp LLPT boundary. See also the
  [published comment](https://www.nature.com/articles/s41586-021-04078-x)
  and [reply](https://www.nature.com/articles/s41586-021-04079-w).
- [The liquid-liquid phase transition of hydrogen and its critical point](https://arxiv.org/abs/2412.14953),
  PBE/NequIP finite-size analysis, estimates a critical region of 155–160 GPa,
  1200–1300 K. The H panel shows that stated interval as a diamond and error
  bars; these are the authors' reported bounds, not statistical confidence
  intervals. The paper notes uncertainty in the melting line and possible
  metastability against the molecular solid.
- [Morales et al., PNAS (2010)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2919906/)
  report approximate melting/metallization crossings at 220 GPa / 700 K
  (DFT) and 290 GPa / 550 K (QMC); crosses indicate those two estimates only,
  not traced melting lines.
- [Pierleoni et al., PNAS (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4983859/)
  present CEIMC hydrogen transition predictions and experimental comparisons.
  The curves have not yet been digitized or independently tabulated here.
- [Millot et al., Nature Physics (2018)](https://doi.org/10.1038/s41567-017-0017-4)
  provide shock evidence and simulations for superionic water. The supplied
  figure's water-melting curve is explicitly described as traced; its original
  coordinate file has not been located, so it has not been silently promoted
  into a precise curve. Existing Cheng et al. (2021) water melting tables are
  plotted within their tabulated domain only.

The user's H2/D2 shock Hugoniots are paths through P–T space, not phase
boundaries. The unidentified `H melt P/T` columns in the project CSV are not
used until their provenance and units are confirmed. Published predictions
differ by electronic method, nuclear quantum treatment, size and sampling.
