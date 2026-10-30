#!/usr/bin/env python3
"""Mix d/p Wannier functions, then print total J, charge and finite-difference dJ/dq.

Usage:
  python main.py example.json

See example.json for configuration (theta in degrees, energies in eV).
All fields are required, with no option defaults. Paths are relative to the JSON file.
Lines beginning with // are comments.

SRC contains UP/DOWN .win, _hr.dat and _centres.xyz files. Transformed models
are written to OUT/theta=VALUE. The same U(k) is applied to both spins.
The hopping convention c=H_reference[d,p]/norm is retained from mix_dp.py.
Weights select H_up, H_down, or (H_up+H_down)/2 BEFORE normalization.

q is the mean Wannier electron population of the two magnetic atoms, not an
ionic charge. dJ/dq is Delta J / Delta q along the chosen mixing path, printed
for adjacent sorted angles and for the full endpoint interval. No orbital J,
C or S is calculated. Final WF order must follow the .win projections.
Both spins must use the same per-WF integer home-cell convention for mixing.
Band preservation is checked on the input .win k mesh.

Requires NumPy, SciPy, ASE, HamiltonIO and TB2J (tested with TB2J 0.9.20).
"""

from contextlib import redirect_stdout
import os
import shutil
import sys

import numpy as np
from scipy.linalg import expm

import parser


def build_generator(Hup, Hdn, R, centres, cell, d_indices, p_indices, bond, weight):
    reference = {"up": Hup, "down": Hdn}.get(weight)
    if weight == "average":
        reference = (Hup + Hdn) / 2.0
    
    Rindex = {tuple(r): i for i, r in enumerate(R)}
    images = np.array([[i, j, l] for i in (-1, 0, 1)
                       for j in (-1, 0, 1) for l in (-1, 0, 1)])
    X = np.zeros_like(Hup)
    
    for d in d_indices:
        candidates = []
        for p in p_indices:
            for r in images:
                if np.linalg.norm(centres[p] + r @ cell - centres[d]) < bond:
                    candidates.append((p, r, reference[Rindex[tuple(r)], d, p]))
        norm = np.sqrt(sum(abs(h) ** 2 for _, _, h in candidates))
        for p, r, h in candidates:
            c = h / norm
            X[Rindex[tuple(r)], d, p] -= np.conj(c)
            X[Rindex[tuple(-r)], p, d] += c
    
    return X


def prepare_transform(k, R, deg, Hup, Hdn, X):
    ph = np.exp(2j * np.pi * k @ R.T) / deg
    phinv = np.exp(-2j * np.pi * k @ R.T) / len(k)
    Xk = np.einsum("kr,rmn->kmn", ph, X, optimize=True)
    Hk = tuple(np.einsum("kr,rmn->kmn", ph, H, optimize=True) for H in (Hup, Hdn))
    
    return ph, phinv, Hk, Xk


def transform_prepared(prepared, theta):
    ph, phinv, Hk, Xk = prepared
    U = np.array([expm(np.deg2rad(theta) * x) for x in Xk])
    output = []
    
    for spin, H in zip(("up", "down"), Hk):
        rotated = U.conj().transpose(0, 2, 1) @ H @ U
        HR = np.einsum("kr,kmn->rmn", phinv, rotated, optimize=True)
        back = np.einsum("kr,rmn->kmn", ph, HR, optimize=True)
        band_error = np.max(np.abs(np.linalg.eigvalsh(rotated) - np.linalg.eigvalsh(H)))
        roundtrip_error = np.max(np.abs(back - rotated))
        print(f"theta={theta:g} {spin}: max band change="
              f"{band_error:.3e} eV; mesh round trip={roundtrip_error:.3e} eV", file=sys.stderr)
        output.append(HR)
    
    return tuple(output)


def shift_cells(data, shifts):
    if not np.any(shifts):
        return data
    
    groups, inverse = np.unique(shifts, axis=0, return_inverse=True)
    indices = [np.flatnonzero(inverse == i) for i in range(len(groups))]
    shifted = {}
    
    for R, matrix in data.items():
        for left, rows in zip(groups, indices):
            for right, columns in zip(groups, indices):
                new_R = tuple(np.asarray(R) - left + right)
                if new_R not in shifted:
                    shifted[new_R] = np.zeros_like(matrix)
                ix = np.ix_(rows, columns)
                shifted[new_R][ix] += matrix[ix]
    
    return shifted


def total_exchange_class():
    from TB2J.exchangeCL2 import ExchangeCL2

    class TotalExchangeCL2(ExchangeCL2):
        # TB2J 0.9.20 computes orbital matrices even when its flag is False.
        def get_all_A(self, Gup, Gdn):
            totals = {}
            for iR, pairs in self.R_ijatom_dict.items():
                iRm = self.R_negative_index[iR]
                for iatom, jatom in pairs:
                    Gu = self.GR_atom(Gup[iR], iatom, jatom)
                    Gd = self.GR_atom(Gdn[iRm], jatom, iatom)
                    totals[(self.short_Rlist[iR], iatom, jatom)] = np.trace(
                        self.get_Delta(iatom) @ Gu @ self.get_Delta(jatom) @ Gd
                    ) / (4.0 * np.pi)
            return {}, totals

        get_all_A_vectorized = get_all_A

        def A_to_Jtensor(self):
            for (R, iatom, jatom), value in self.JJ.items():
                if R == (0, 0, 0) and iatom == jatom:
                    continue
                alignment = np.sign(np.dot(self.spinat[iatom], self.spinat[jatom]))
                key = (R, self.ispin(iatom), self.ispin(jatom))
                self.exchange_Jdict[key] = np.imag(value) / alignment

    return TotalExchangeCL2


def calculate(seeds, atoms, basis, args):
    from HamiltonIO.wannier import WannierHam

    labels = parser.atom_labels(atoms)
    atom_index = {label: i for i, label in enumerate(labels)}
    positions = atoms.get_scaled_positions(wrap=False)[[atom_index[label.split("|")[0]] for label in basis]]
    models = []

    for seed in seeds:
        centres = parser.read_centres(f"{seed}_centres.xyz", len(basis))
        shifts = np.rint(atoms.cell.scaled_positions(centres) - positions).astype(int)
        nw, R, deg, H = parser.read_hr(f"{seed}_hr.dat")
        data = shift_cells({tuple(r): matrix / g for r, g, matrix in zip(R, deg, H)}, shifts)
        model = WannierHam(nbasis=nw, data=data, positions=positions,
                           R_degens=np.ones(len(data), dtype=int))
        model.set_atoms(atoms)
        models.append(model)

    exchange = total_exchange_class()(
        tbmodels=tuple(models), 
        atoms=atoms, 
        basis=basis,
        magnetic_elements=args.elements, 
        efermi=args.efermi,
        kmesh=args.kmesh, 
        Rcut=args.rcut, 
        smearing=args.smearing,
        nz=args.nz, 
        nproc=1, 
        use_cache=False, 
        orb_decomposition=False,
    )

    try:
        exchange.calculate_all()
        result = {}
        for key, value in exchange.exchange_Jdict.items():
            distance = exchange.distance_dict[key][1]
            if args.distance is not None and abs(distance - args.distance) > args.tolerance:
                continue
            R, i, j = key
            ia, ja = exchange.iatom(i), exchange.iatom(j)
            charge = 0.5 * (exchange.charges[ia] + exchange.charges[ja])
            result[(labels[ia], labels[ja], tuple(R))] = (distance, float(value) * 1000.0, float(charge))
        return result, exchange.kmesh
    finally:
        exchange.finalize()


def print_results(thetas, results, distance, tolerance):
    keys = set(results[0])
    keys = sorted(keys, key=lambda key: (results[0][key][0], key))
    print("# J in meV; q = mean Wannier electron population of the pair, in electrons")
    print("theta i j Rx Ry Rz distance J q")

    for theta, result in zip(thetas, results):
        for i, j, R in keys:
            dist, J, q = result[(i, j, R)]
            print(str(theta), i, j, *R, f"{dist:.6f}", f"{J:.9g}", f"{q:.9g}")
    
    intervals = [("adjacent", i, i + 1) for i in range(len(thetas) - 1)]
    if len(thetas) > 2:
        intervals.append(("overall", 0, len(thetas) - 1))
    
    print("# dJdq = (J_to-J_from)/(q_to-q_from), in meV/e")
    print("scope theta_from theta_to i j Rx Ry Rz distance dJdq")
    
    for scope, left, right in intervals:
        for i, j, R in keys:
            dist, jl, ql = results[left][(i, j, R)]
            _, jr, qr = results[right][(i, j, R)]
            dj, dq = jr - jl, qr - ql
            derivative = dj / dq if dq else (float("inf") if dj else float("nan"))
            print(scope, str(thetas[left]), str(thetas[right]), i, j, *R,
                  f"{dist:.6f}", f"{derivative:.9g}")
    
    if distance is not None:
        means = [np.mean([result[key][1:] for key in keys], axis=0) for result in results]
        print(f"# mean over {len(keys)} directed pairs within {distance} +/- {tolerance} Angstrom")
        for theta, (J, q) in zip(thetas, means):
            print(f"# mean theta={theta} J={J:.9g} q={q:.9g}")
        for scope, left, right in intervals:
            jl, ql = means[left]
            jr, qr = means[right]
            dj, dq = jr - jl, qr - ql
            derivative = dj / dq if dq else (float("inf") if dj else float("nan"))
            print(f"# mean {scope} theta_from={thetas[left]} "
                  f"theta_to={thetas[right]} "
                  f"dJdq={derivative:.9g}")


def main(argv=None):
    args = parser.parse_args(argv, description=__doc__)
    prefixes, thetas = args.prefixes, args.thetas
    reference, hamiltonians, spin_centres = None, [], []
    
    for prefix in prefixes:
        seed = args.src / prefix
        k, atoms, basis = parser.read_win(f"{seed}.win")
        nw, R, deg, H = parser.read_hr(f"{seed}_hr.dat")
        spin_centres.append(parser.read_centres(f"{seed}_centres.xyz", nw))
        if reference is None:
            reference = k, atoms, basis, R, deg
        hamiltonians.append(H)
    
    k, atoms, basis, R, deg = reference
    centres = spin_centres[0]
    images = np.array([[i, j, l] for i in (-1, 0, 1)
                       for j in (-1, 0, 1) for l in (-1, 0, 1)])
    shifts = images @ atoms.cell
    nearest = [min(range(len(atoms)), key=lambda i:
                   np.linalg.norm(centre - atoms.positions[i] - shifts, axis=1).min())
               for centre in centres]
    
    d_indices = [m for m, i in enumerate(nearest) if atoms[i].symbol == args.mag]
    p_indices = [m for m, i in enumerate(nearest) if atoms[i].symbol == args.lig]

    # X transforms the Hamiltonians and one writes new .hr.dat files for each theta
    X = build_generator(*hamiltonians, R, centres, atoms.cell,
                        d_indices, p_indices, args.bond, args.weight)
    prepared = prepare_transform(k, R, deg, *hamiltonians, X)
    
    print(f"# weight={args.weight}; thetas_deg={thetas}; efermi={args.efermi}; "
          f"smearing={args.smearing}; nz={args.nz}; rcut={args.rcut}")
    print("# WF order follows win projections; centres only correct integer home-cell shifts for TB2J")
    
    results = []
    
    for theta in thetas:
        directory = args.out / f"theta={theta}"
        transformed = transform_prepared(prepared, theta)
        directory.mkdir(parents=True, exist_ok=True)

        for prefix, H in zip(prefixes, transformed):
            parser.write_hr(directory / f"{prefix}_hr.dat", R, deg, H, f"mix_dp theta={theta:g} deg weight={args.weight}")
            for suffix in (".win", ".eig", "_centres.xyz"):
                source = args.src / f"{prefix}{suffix}"
                if source.exists():
                    shutil.copy(source, directory / source.name)
        
        if (args.src / "POSCAR").exists():
            shutil.copy(args.src / "POSCAR", directory / "POSCAR")
        
        print(f"Calculating total J and charge at theta={theta:g}", file=sys.stderr, flush=True)
        
        with redirect_stdout(sys.stderr):
            result, mesh = calculate([directory / prefix for prefix in prefixes], atoms, basis, args)
        
        results.append(result)
    
    print(f"# TB2J kmesh={mesh}")
    print_results(thetas, results, args.distance, args.tolerance)


if __name__ == "__main__":
    main()
