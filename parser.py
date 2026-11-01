import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from ase import Atoms
from ase.io import read as read_atoms
from ase.units import Bohr


def atom_labels(atoms):
    counts, labels = Counter(), []
    
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] += 1
        labels.append(f"{symbol}{counts[symbol]}")

    return labels


def block(text, name):
    match = re.search(rf"^\s*begin\s+{name}\s*$(.*?)^\s*end\s+{name}\s*$", text, re.I | re.M | re.S)
    
    return [line.strip() for line in match[1].splitlines() if line.strip()]


def read_win(path):
    text = re.sub(r"[!#].*", "", Path(path).read_text())
    cell_lines = block(text, "unit_cell_cart")
    unit = cell_lines.pop(0).lower() if len(cell_lines) == 4 else "ang"
    cell = np.array([line.split() for line in cell_lines], dtype=float)
    cell *= Bohr if unit == "bohr" else 1.0
    fractional = bool(re.search(r"^\s*begin\s+atoms_frac\s*$", text, re.I | re.M))
    lines = block(text, "atoms_frac" if fractional else "atoms_cart")
    unit = lines.pop(0).lower() if lines[0].lower() in ("ang", "bohr") else "ang"
    symbols = [line.split()[0] for line in lines]
    coordinates = np.array([line.split()[1:] for line in lines], dtype=float)
    
    if not fractional:
        coordinates *= Bohr if unit == "bohr" else 1.0
    
    atoms = Atoms(symbols, cell=cell, pbc=True, **{"scaled_positions" if fractional else "positions": coordinates,})
    counts = {
        "s": 1, "p": 3, "d": 5, "f": 7,
        "sp": 2, "sp2": 3, "sp3": 4, "sp3d": 5, "sp3d2": 6,
        "pz": 1, "px": 1, "py": 1, "dz2": 1, "dxz": 1,
        "dyz": 1, "dx2-y2": 1, "dxy": 1,
    }

    labels, assigned, basis = atom_labels(atoms), Counter(), []
    for line in block(text, "projections"):
        if line.lower() in ("ang", "bohr"):
            continue
        match = re.fullmatch(r"([A-Z][a-z]?)\s*:\s*([^:]+)", line, re.I)
        species, orbitals = match.groups()
        orbitals = [orb.strip().lower() for orb in re.split(r"[;,]", orbitals)]
        for iatom, symbol in enumerate(symbols):
            if symbol.lower() == species.lower():
                for _ in range(sum(counts[orb] for orb in orbitals)):
                    assigned[iatom] += 1
                    basis.append(f"{labels[iatom]}|orb_{assigned[iatom]}")

    k = np.array([line.split()[:3] for line in block(text, "kpoints")], dtype=float)
    
    return k, atoms, basis


def read_centres(path, nw):
    entries = read_atoms(path, format="xyz")
    centres = entries.positions[np.array(entries.get_chemical_symbols()) == "X"]

    return centres


def read_hr(path):
    with Path(path).open() as stream:
        stream.readline()
        nw, nr = int(stream.readline()), int(stream.readline())
        deg = []
        while len(deg) < nr:
            line = next(stream)
            deg.extend(map(int, line.split()))
        rows = np.loadtxt(stream, ndmin=2)

    rows = rows.reshape(nr, nw * nw, 7)
    
    R = rows[:, 0, :3].astype(int)
    indices = rows[:, :, 3:5].astype(int) - 1
    
    H = np.empty((nr, nw, nw), dtype=complex)
    H[np.arange(nr)[:, None], indices[:, :, 0], indices[:, :, 1]] = (
        rows[:, :, 5] + 1j * rows[:, :, 6]
    )

    return nw, R, np.array(deg), H


def write_hr(path, R, deg, H, comment):
    nr, nw = H.shape[:2]
    rows = np.empty((nr * nw * nw, 7))
    rows[:, :3] = np.repeat(R, nw * nw, axis=0)
    ii, jj = np.meshgrid(np.arange(1, nw + 1), np.arange(1, nw + 1), indexing="ij")
    rows[:, 3] = np.tile(jj.ravel(), nr)
    rows[:, 4] = np.tile(ii.ravel(), nr)
    values = H.transpose(0, 2, 1).reshape(-1)
    rows[:, 5], rows[:, 6] = values.real, values.imag

    with Path(path).open("w") as stream:
        stream.write(f" {comment}\n{nw:12d}\n{nr:12d}\n")
        for start in range(0, nr, 15):
            stream.write("".join(f"{value:5d}" for value in deg[start:start + 15]) + "\n")
        np.savetxt(stream, rows, fmt="%5d%5d%5d%5d%5d%16.8f%16.8f")


def parse_args(argv=None, description=None):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="JSON configuration file")
    config = parser.parse_args(argv).config.resolve()
    settings = json.loads(re.sub(r"(?m)^[ \t]*//[^\n]*", "", config.read_text()))
    args = argparse.Namespace(
        src=config.parent / settings["src"],
        out=config.parent / settings["out"],
        thetas=sorted(map(float, settings["thetas"])),
        mag=settings["mag"],
        lig=settings["lig"],
        efermi=settings["efermi"],
        kmesh=settings["kmesh"],
        smearing=settings["smearing"],
        weight=settings["weight"],
        prefixes=settings["prefixes"],
        bond=settings["bond"],
        elements=settings["elements"],
        rcut=settings["rcut"],
        nz=settings["nz"],
        distance=settings["distance"],
        tolerance=settings["tolerance"],
    )
    if 0.0 not in args.thetas or len(args.thetas) < 3:
        parser.error("thetas must contain 0 and at least three angles")
    
    return args
