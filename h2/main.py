"""Compare RHF and FCI total energies for H2 in the STO-3G basis."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a graphical desktop.
import matplotlib.pyplot as plt
import numpy as np
from pyscf import fci, gto, lib, scf


def configuration_weights(mf, ci):
    """Read determinant probabilities for two electrons in two RHF orbitals."""
    if mf.mol.nelec != (1, 1) or mf.mo_coeff.shape != (2, 2):
        raise RuntimeError("This analysis requires H2 / STO-3G: two orbitals, one alpha and one beta")

    # In this two-AO basis, exchanging the H atoms swaps the AO rows.
    # Bonding sigma is symmetric (+1); antibonding sigma* is antisymmetric (-1).
    # Check this instead of identifying orbitals only by their energy order.
    parity = np.diag(mf.mo_coeff.T @ mf.get_ovlp() @ mf.mo_coeff[::-1, :])
    bonding = np.flatnonzero(np.isclose(parity, 1.0, atol=1e-8, rtol=0))
    antibonding = np.flatnonzero(np.isclose(parity, -1.0, atol=1e-8, rtol=0))
    if len(bonding) != 1 or len(antibonding) != 1:
        raise RuntimeError("Could not identify symmetric sigma and antisymmetric sigma* orbitals")

    # CI rows label alpha occupation strings; columns label beta strings.
    # With one electron per spin, bit (1 << orbital) selects that orbital.
    # Ask PySCF for the addresses rather than assuming a determinant order.
    sigma = fci.cistring.str2addr(2, 1, 1 << int(bonding[0]))
    sigma_star = fci.cistring.str2addr(2, 1, 1 << int(antibonding[0]))
    ci = np.asarray(ci)
    if ci.shape != (2, 2) or not np.all(np.isfinite(ci)):
        raise RuntimeError("Invalid FCI coefficient matrix")
    norm_squared = float(np.sum(np.abs(ci) ** 2))
    if not np.isclose(norm_squared, 1.0, atol=1e-8, rtol=0):
        raise RuntimeError(f"FCI coefficients are not normalized: sum |c|^2 = {norm_squared}")

    # Orthonormal determinants have probabilities |c|^2 (also for complex c).
    # Divide by the measured norm only after checking it, to remove roundoff.
    weights = np.abs(ci) ** 2 / norm_squared
    w_sigma = float(weights[sigma, sigma])
    w_sigma_star = float(weights[sigma_star, sigma_star])
    w_mixed_ab = float(weights[sigma, sigma_star])
    w_mixed_ba = float(weights[sigma_star, sigma])
    w_mixed = w_mixed_ab + w_mixed_ba  # Two distinct orthogonal determinants.
    if not np.isclose(w_sigma + w_sigma_star + w_mixed, 1.0, atol=1e-12, rtol=0):
        raise RuntimeError("Configuration weights do not sum to one")
    return norm_squared, w_sigma, w_sigma_star, w_mixed_ab, w_mixed_ba, w_mixed


def main():
    # One thread avoids parallel overhead for this very small molecule.
    lib.num_threads(1)
    output_dir = Path(__file__).resolve().parent
    rows = []

    # Each geometry is an independent, neutral, singlet H2 calculation.
    for distance in np.linspace(0.5, 3.0, 20):
        try:
            mol = gto.M(
                atom=f"H 0 0 0; H 0 0 {distance:.12f}",
                unit="Angstrom", basis="sto-3g", charge=0, spin=0, verbose=0,
            )

            # RHF puts both electrons in the same spatial orbital.
            mf = scf.RHF(mol)
            mf.conv_tol = 1e-10
            mf.max_cycle = 100
            e_hf = mf.kernel()
            if not mf.converged:
                raise RuntimeError("RHF did not converge within 100 cycles")

            # FCI includes every electronic configuration in this finite basis.
            # This interface returns TOTAL energy, including nuclear repulsion:
            # do not add mol.energy_nuc() a second time.
            solver = fci.FCI(mf)
            solver.conv_tol = 1e-10
            e_fci, ci = solver.kernel()
            if not solver.converged:
                raise RuntimeError("FCI did not converge")
            if not np.all(np.isfinite([e_hf, e_fci])):
                raise RuntimeError("Non-finite energy returned")
            if e_fci > e_hf + 1e-8:
                raise RuntimeError("FCI energy exceeds RHF beyond numerical tolerance")
            weights = configuration_weights(mf, ci)
        except Exception as exc:
            raise RuntimeError(f"Calculation failed at R = {distance:.6f} Angstrom: {exc}") from exc

        rows.append((distance, e_hf, e_fci, e_fci - e_hf, *weights))
        print(f"R = {distance:.6f} A | RHF = {e_hf:.10f} Eh | FCI = {e_fci:.10f} Eh", flush=True)

    # Write results only after all 20 geometries succeed; never skip failures.
    data = np.array(rows)
    np.savetxt(
        output_dir / "results.csv", data, delimiter=",", fmt="%.12e",
        header=("distance_angstrom,energy_rhf_hartree,energy_fci_hartree,delta_fci_minus_hf_hartree,"
                "ci_norm_squared,weight_sigma_squared,weight_sigma_star_squared,"
                "weight_sigma_alpha_sigma_star_beta,weight_sigma_star_alpha_sigma_beta,weight_mixed_total"),
        comments="",
    )
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(data[:, 0], data[:, 1], "o-", label="RHF / STO-3G")
    ax.plot(data[:, 0], data[:, 2], "s-", label="FCI / STO-3G")
    ax.set(xlabel="H-H distance (angstrom)", ylabel="Total energy (hartree)",
           title="H2 potential energy curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "total_energy.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(data[:, 0], data[:, 3], "o-", color="tab:purple")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set(xlabel="H-H distance (angstrom)", ylabel="E_FCI - E_HF (hartree)",
           title="Correlation energy in the STO-3G basis")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "correlation_energy.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(data[:, 0], data[:, 5], "o-", label=r"$\sigma^2$")
    ax.plot(data[:, 0], data[:, 6], "s-", label=r"$(\sigma^*)^2$")
    ax.plot(data[:, 0], data[:, 9], "^-", label="Mixed determinants (sum)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set(xlabel="H-H distance (angstrom)", ylabel="Configuration weight",
           title="FCI configuration weights in RHF orbitals", ylim=(-0.03, 1.03))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "configuration_weights.png", dpi=200)
    plt.close(fig)
    print(f"Saved results.csv and three figures in {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
