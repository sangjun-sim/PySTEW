<p align="center">
  <img src="assets/pystew-logo.png" alt="PySTEW logo: d and p orbitals with exchange and basis-rotation arrows" width="480">
</p>

# PySTEW

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/sangjun-sim/PySTEW?style=flat)](https://github.com/sangjun-sim/PySTEW/stargazers)
[![Issues](https://img.shields.io/github/issues/sangjun-sim/PySTEW)](https://github.com/sangjun-sim/PySTEW/issues)

PySTEW (**Py**thon package for **S**ensitivity **T**ests of **E**xchange parameters in a **W**annier basis) tests the basis dependence of exchange parameters $J_{ij}$ using Wannier90 Hamiltonians and TB2J. For more details, see the [PySTEW note](https://sangjun-sim.github.io/posts/physics-pystew/). It mixes magnetic $d$ and ligand $p$ orbital Wannier functions across selected angles, applying the same transformation to both spin channels. The mixing automatically includes all $p$ Wannier functions belonging to each magnetic atom's nearest ligand-atom shell, using atomic positions and `tolerance` $\AA$. No `bond` setting is needed. An optional `bond` value retains the manual distance cutoff between Wannier centers. 

It computes $J$ and the average Wannier electron population $q$ at each angle $\theta$, then prints only $dJ/dq$ at $\theta=0$. The `thetas` must include at least three angles, including zero. $J_1$, $J_2$ and $J_3$ are selected automatically from the first three magnetic-pair distance shells within `rcut`. Distances within `tolerance` $\AA$ of a shell's shortest distance are grouped together. The output reports each shell's distance, pair count and mean derivative; if fewer than three shells are available, only those are printed. No `distance` setting is needed. An optional scalar or list of distances overrides automatic selection; `null` also uses automatic selection.

Install the dependencies with 

```
python -m pip install -r requirements.txt
```

Set the input paths and calculation settings in `example.json`, then run

```
python main.py example.json
```

## Test environment

Input and parser validation were performed with the following versions.

| Component | Version |
| --- | --- |
| Python | 3.9.6 |
| NumPy | 2.0.2 |
| SciPy | 1.13.1 |
| ASE | 3.26.0 |
