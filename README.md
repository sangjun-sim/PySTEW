<p align="center">
  <img src="assets/pystew-logo.png" alt="PySTEW logo: d and p orbitals with exchange and basis-rotation arrows" width="480">
</p>

# PySTEW

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/sangjun-sim/PySTEW?style=flat)](https://github.com/sangjun-sim/PySTEW/stargazers)
[![Issues](https://img.shields.io/github/issues/sangjun-sim/PySTEW)](https://github.com/sangjun-sim/PySTEW/issues)

PySTEW (**Py**thon package for **S**ensitivity **T**ests of **E**xchange parameters in a **W**annier basis) tests the basis dependence of exchange parameters $J_{ij}$ using Wannier90 Hamiltonians and TB2J.
It mixes magnetic $d$ and ligand $p$ orbital Wannier functions across selected angles, applying the same transformation to both spin channels.
It computes $J$ and the mean Wannier electron population $q$ at each angle, then prints only $dJ/dq$ at $\theta=0$ for each selected pair and for the mean $J$ and $q$ of those pairs. The `thetas` setting must contain at least three angles, including zero.
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
