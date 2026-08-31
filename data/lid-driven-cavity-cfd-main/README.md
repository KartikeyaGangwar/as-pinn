# Lid-Driven Cavity Flow Solver (2D)

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18312938.svg)](https://doi.org/10.5281/zenodo.18312938)

High-resolution finite-difference solver for the classical 2D incompressible
lid-driven cavity problem, implemented using the streamfunction–vorticity formulation.

![Showcase](figures/lid_driven_cavity_showcase_Re1000.png)

## Features
- Streamfunction–vorticity formulation
- ADI scheme for vorticity transport
- Red–Black SOR for the streamfunction Poisson equation
- Physically consistent coordinate system
- Instant post-processing from saved solutions (`.npz`)

## Usage

Run a simulation:
```bash
python lid_driven_cavity_fdm.py
```
Generate plots from a saved solution:
```bash
python plot_from_npz.py
```
## Output

    Flow fields (ψ, ω, u, v, p) saved as NumPy archives (.npz)

    Publication-quality diagnostic and showcase plots

    Centerline velocity profiles for validation

## Notes

This implementation is intended for educational and reference purposes.
It emphasizes numerical clarity, physical correctness, and reproducibility.
It does not claim methodological novelty.

**Reference**  
Ghia, U., Ghia, K. N., & Shin, C. T. (1982). *High-Re solutions for incompressible flow using the Navier–Stokes equations and a multigrid method*. Journal of Computational Physics, 48(3), 387–411.

## Related Work

- Physics-Informed Neural Network (PINN) solver for the same problem:  
  [Fluid-Dynamics-PINNs](https://github.com/KartikeyGangwar/Fluid-Dynamics-PINNs)

## Citation

If you use this solver, code, or generated data in your research, please cite:

**Kartikey Singh (2026).**  
*A Reference Finite-Difference Solver for the 2D Lid-Driven Cavity Flow.*  
Zenodo. https://doi.org/10.5281/zenodo.18312938

### BibTeX
```bibtex
@software{gangwar2026solver,
  author       = {Singh, Kartikey},
  title        = {A Reference Finite-Difference Solver for the 2D Lid-Driven Cavity Flow},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18312938},
  url          = {https://doi.org/10.5281/zenodo.18312938}
}
