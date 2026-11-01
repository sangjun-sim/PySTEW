#!/usr/bin/env python3
"""Mix d/p Wannier functions directly in k space, then print dJ/dq at theta=0.

Usage:
  python main.py example.json

See example.json for configuration (theta in degrees, energies in eV).
All fields except distance and bond are required. Paths are relative to the JSON file.
Lines beginning with // are comments.

SRC contains UP/DOWN .win, _hr.dat and _centres.xyz files. H_theta(k) is evaluated
directly at the TB2J k points; no transformed Hamiltonian files are written.
OUT is the TB2J output directory. The same U(k) is applied to both spins.
The hopping convention c=H_reference[d,p]/norm is retained from mix_dp.py.
Weights select H_up, H_down, or (H_up+H_down)/2 BEFORE normalization.
Mixing includes all p WFs on the nearest ligand-atom shell, grouped using
tolerance. Optional bond overrides this with a WF-centre distance cutoff in Angstrom.

q is the mean Wannier electron population of the two magnetic atoms, not an
ionic charge. dJ/dq is np.gradient(J, q, edge_order=2) along the chosen mixing
path, using at least three angles including zero. Only dJ/dq at theta=0 is
printed for each pair and for the mean J and q of J1, J2 and J3. These are the
first three magnetic-pair distance shells within rcut, grouped using tolerance.
Optional distance overrides automatic selection with a scalar or a list.
No orbital J, C or S is calculated. Final WF order must follow the .win projections.
Both spins must use the same per-WF integer home-cell convention for mixing.
No Hamiltonian k-to-R reconstruction or subsequent interpolation is performed.

Requires NumPy, SciPy, ASE, HamiltonIO and TB2J (tested with TB2J 0.9.20).
"""

from contextlib import redirect_stdout
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm
from HamiltonIO.wannier import WannierHam

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import parser


def build_generator(Hup, Hdn, R, centres, cell, d_indices, p_indices, bond, weight, tolerance):
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
                distance = np.linalg.norm(centres[p] + r @ cell - centres[d])
                if bond is None or distance < bond:
                    candidates.append((p, r, distance))
        if bond is None and candidates:
            closest = min(distance for _, _, distance in candidates)
            candidates = [candidate for candidate in candidates
                          if candidate[2] <= closest + tolerance]
        candidates = [(p, r, reference[Rindex[tuple(r)], d, p])
                      for p, r, _ in candidates]
        norm = np.sqrt(sum(abs(h) ** 2 for _, _, h in candidates))
        for p, r, h in candidates:
            c = h / norm
            X[Rindex[tuple(r)], d, p] -= np.conj(c)
            X[Rindex[tuple(-r)], p, d] += c
    
    return X


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


def print_results(thetas, results, distance, tolerance):
    index = thetas.index(0.0)
    reference = results[index]
    keys = sorted(reference, key=lambda key: (reference[key][0], key))

    if distance is None:
        shells = []
        for key in keys:
            dist = reference[key][0]
            if not shells or dist - shells[-1][0] > tolerance:
                if len(shells) == 3:
                    break
                shells.append((dist, []))
            shells[-1][1].append(key)
    else:
        shells = [(target, [key for key in keys
                            if abs(reference[key][0] - target) <= tolerance])
                  for target in np.atleast_1d(distance)]
    selected = {key for _, shell_keys in shells for key in shell_keys}
    keys = [key for key in keys if key in selected]

    print("# theta=0; dJdq in meV/e, evaluated with np.gradient(J, q, edge_order=2)")
    print("i j Rx Ry Rz distance dJdq")

    for i, j, R in keys:
        dist = reference[(i, j, R)][0]
        J, q = np.array([result[(i, j, R)][1:] for result in results]).T
        derivative = np.gradient(J, q, edge_order=2)[index]
        print(i, j, *R, f"{dist:.6f}", f"{derivative:.9g}")

    for shell, (target, shell_keys) in enumerate(shells, 1):
        means = np.array([np.mean([result[key][1:] for key in shell_keys], axis=0)
                          for result in results])
        derivative = np.gradient(means[:, 0], means[:, 1], edge_order=2)[index]
        selection = f" within {target} +/- {tolerance} Angstrom"
        label = f"J{shell} " if distance is None or np.ndim(distance) else ""
        print(f"# {label}mean over {len(shell_keys)} directed pairs{selection}")
        print(f"# {label}mean theta=0 dJdq={derivative:.9g}")


class RotatedWannierHam(WannierHam):
    def __init__(self, R, deg, H, X, positions, shifts, theta):
        data = {tuple(r): h / g for r, g, h in zip(R, deg, H)}
        super().__init__(nbasis=H.shape[-1], data=data, positions=positions)
        self.rotation_R = np.asarray(R)
        self.generator = X / np.asarray(deg)[:, None, None]
        self.shifts = np.asarray(shifts)
        self.theta = np.deg2rad(theta)

    def gen_ham(self, k, convention=2):
        H = super().gen_ham(k, convention=convention)
        phase = np.exp(self.R2kfactor * (self.rotation_R @ k))
        Xk = np.einsum("r,rij->ij", phase, self.generator, optimize=True)
        U = expm(self.theta * Xk)
        rotated = U.conj().T @ H @ U

        # Rotate in the original WF gauge, then shift each WF to its atom's cell.
        cell_phase = np.exp(self.R2kfactor * (self.shifts @ k))

        return cell_phase.conj()[:, None] * rotated * cell_phase[None, :]


def calculate(models, atoms, basis, args):
    labels = parser.atom_labels(atoms)
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
        output_path=str(args.out),
    )
    try:
        exchange.calculate_all()
        result = {}
        for key, value in exchange.exchange_Jdict.items():
            distance = exchange.distance_dict[key][1]
            if args.distance is not None and not np.any(
                np.abs(distance - np.asarray(args.distance)) <= args.tolerance
            ):
                continue
            R, i, j = key
            ia, ja = exchange.iatom(i), exchange.iatom(j)
            charge = 0.5 * (exchange.charges[ia] + exchange.charges[ja])
            result[(labels[ia], labels[ja], tuple(R))] = (
                distance, float(value) * 1000.0, float(charge)
            )
        return result, exchange.kmesh
    finally:
        exchange.finalize()


def main(argv=None):
    args = parser.parse_args(argv, description=__doc__)
    hamiltonians, spin_centres = [], []
    reference = None

    for prefix in args.prefixes:
        seed = args.src / prefix
        k, atoms, basis = parser.read_win(f"{seed}.win")
        nw, R, deg, H = parser.read_hr(f"{seed}_hr.dat")
        spin_centres.append(parser.read_centres(f"{seed}_centres.xyz", nw))
        if reference is None:
            reference = atoms, basis, R, deg
        hamiltonians.append(H)

    atoms, basis, R, deg = reference
    centres = spin_centres[0]
    images = np.array([[i, j, l] for i in (-1, 0, 1)
                       for j in (-1, 0, 1) for l in (-1, 0, 1)])
    images_cart = images @ atoms.cell
    nearest = [min(range(len(atoms)), key=lambda i:
                   np.linalg.norm(centre - atoms.positions[i] - images_cart, axis=1).min())
               for centre in centres]
    d_indices = [m for m, i in enumerate(nearest) if atoms[i].symbol == args.mag]
    p_indices = [m for m, i in enumerate(nearest) if atoms[i].symbol == args.lig]
    mixing_centres = centres
    if args.bond is None:
        # Keep the assigned atoms in each WF's original home cell.
        mixing_centres = np.array([
            atoms.positions[i] + images_cart[np.argmin(
                np.linalg.norm(centre - atoms.positions[i] - images_cart, axis=1)
            )] for centre, i in zip(centres, nearest)
        ])
    X = build_generator(*hamiltonians, R, mixing_centres, atoms.cell,
                        d_indices, p_indices, args.bond, args.weight, args.tolerance)

    labels = parser.atom_labels(atoms)
    atom_index = {label: i for i, label in enumerate(labels)}
    positions = atoms.get_scaled_positions(wrap=False)[
        [atom_index[label.split("|")[0]] for label in basis]
    ]
    spin_shifts = [np.rint(atoms.cell.scaled_positions(c) - positions).astype(int)
                   for c in spin_centres]
    print(f"# weight={args.weight}; thetas_deg={args.thetas}; efermi={args.efermi}; "
          f"smearing={args.smearing}; nz={args.nz}; rcut={args.rcut}")

    results = []
    for theta in args.thetas:
        models = []
        for H, shifts in zip(hamiltonians, spin_shifts):
            model = RotatedWannierHam(R, deg, H, X, positions, shifts, theta)
            model.set_atoms(atoms)
            models.append(model)
        print(f"Calculating total J and charge at theta={theta:g}",
              file=sys.stderr, flush=True)
        with redirect_stdout(sys.stderr):
            result, mesh = calculate(models, atoms, basis, args)
        results.append(result)

    print(f"# TB2J kmesh={mesh}")
    print_results(args.thetas, results, args.distance, args.tolerance)


if __name__ == "__main__":
    main()
