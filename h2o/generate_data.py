"""Calculate a constrained 2D RHF/STO-3G potential energy surface for water."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")  # Save the plot without opening a graphical window.
import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, lib, scf


def water_geometry(bond_length, angle_deg):
    """Return Cartesian coordinates in angstrom, with equal O-H distances."""
    angle_rad = np.deg2rad(angle_deg)
    return [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (bond_length, 0.0, 0.0)),
        ("H", (bond_length * np.cos(angle_rad),
               bond_length * np.sin(angle_rad), 0.0)),
    ]


def main():
    # Small molecules benefit from avoiding parallel thread overhead.
    lib.num_threads(1)
    output_dir = Path(__file__).resolve().parent
    bond_lengths = np.linspace(0.8, 1.2, 10)
    angles = np.linspace(85.0, 125.0, 10)
    energies = np.empty((len(bond_lengths), len(angles)))
    rows = []

    # Rows of the energy array correspond to R; columns correspond to angle.
    for i, bond_length in enumerate(bond_lengths):
        for j, angle_deg in enumerate(angles):
            try:
                mol = gto.M(
                    atom=water_geometry(bond_length, angle_deg),
                    unit="Angstrom", basis="sto-3g", charge=0, spin=0,
                    verbose=0,
                )
                mf = scf.RHF(mol)
                mf.conv_tol = 1e-10
                mf.max_cycle = 100
                # RHF.kernel() returns total energy, including nuclear repulsion.
                # Do not add mol.energy_nuc() again.
                energy = mf.kernel()
                if not mf.converged:
                    raise RuntimeError("SCF did not converge within 100 cycles")
                if not np.isfinite(energy):
                    raise RuntimeError("Non-finite total energy returned")
            except Exception as exc:
                # Stop explicitly: never insert a fabricated energy or skip a point.
                raise RuntimeError(
                    f"Failed at R_OH={bond_length:.8f} angstrom, "
                    f"angle_HOH={angle_deg:.8f} degrees: {exc}"
                ) from exc

            energies[i, j] = energy
            rows.append((bond_length, angle_deg, energy))
            print(f"[{len(rows):3d}/100] R={bond_length:.6f} A, "
                  f"angle={angle_deg:.6f} deg, E={energy:.10f} Eh", flush=True)

    # Save only after every geometry has converged successfully.
    np.savetxt(
        output_dir / "water_energies.csv", np.array(rows), delimiter=",",
        header="bond_length_angstrom,angle_deg,energy_hartree",
        comments="", fmt="%.12f",
    )
    min_i, min_j = np.unravel_index(np.argmin(energies), energies.shape)
    min_r = bond_lengths[min_i]
    min_angle = angles[min_j]
    min_energy = energies[min_i, min_j]
    print(f"Grid minimum: R_OH={min_r:.8f} A, angle_HOH={min_angle:.8f} deg, "
          f"E={min_energy:.12f} Eh", flush=True)

    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    # Angle varies along x, bond length along y, matching the array layout.
    surface = ax.contourf(angles, bond_lengths, energies, levels=20, cmap="viridis")
    fig.colorbar(surface, ax=ax, label="RHF total energy (hartree)")
    ax.plot(min_angle, min_r, "r*", markersize=13,
            markeredgecolor="white", label="Minimum on sampled grid")
    ax.annotate(
        f"R = {min_r:.4f} A\nAngle = {min_angle:.4f} deg\nE = {min_energy:.6f} Eh",
        xy=(min_angle, min_r), xytext=(12, 18), textcoords="offset points",
        fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    ax.set(xlabel="H-O-H angle (degrees)", ylabel="Equal O-H bond length (angstrom)",
           title="H2O constrained potential energy surface: RHF / STO-3G")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figures / "potential_energy_surface.png", dpi=200)
    plt.close(fig)
    print(f"Saved CSV and figure in {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
