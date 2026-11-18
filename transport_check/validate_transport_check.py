"""Independent mathematical checks for the no-flow transport calculation.

Run transport_check.py first. This script checks chemistry derivatives, explicit
surface stoichiometry, an exact neutral-diffusion solution, integration accuracy,
and the full axisymmetric component/current PDEs with finite-volume fluxes.
"""

from pathlib import Path
import json

import numpy as np

from transport_check import (
    acid_fractions, conductivity, half_space_source,
    integrate_chemical_path, prepare_reservoir, profile_parameters, species_properties,
    surface_reaction, transport_matrix,
)


def bernoulli(argument):
    result = np.empty_like(argument)
    small = np.abs(argument) < 1e-5
    result[small] = 1 - argument[small] / 2 + argument[small] ** 2 / 12
    result[~small] = argument[~small] / np.expm1(argument[~small])
    return result


def grid_concentrations(states, families):
    concentrations = []
    for group, family in enumerate(families):
        fractions, _ = acid_fractions(states[..., -2], family["pkas"], np.array(family["charges"]))
        concentrations.extend(np.moveaxis(states[..., group, None] * fractions, -1, 0))
    concentrations.extend((10.0 ** (3 - states[..., -2]), 10.0 ** (states[..., -2] - 11)))
    return np.stack(concentrations, axis=-1)


def finite_volume_residual(solution, families, radius, diffusivity_reference, electrons, faraday,
                           count, extent, current_fraction, quadrature_count, maximum_wave_number):
    """Evaluate the original Nernst-Planck PDEs on a separate 2-D mesh.

    Individual protonation states exchange through fast reactions. Conserved
    component fluxes and total electrical current must each be divergence-free.
    All errors are normalized by the total imposed reaction/electron rate.
    """
    faces = np.linspace(0, extent * radius, count + 1)
    centers = (faces[:-1] + faces[1:]) / 2
    widths = np.diff(faces)
    annular_areas = np.diff(faces**2) / 2
    critical_coordinate = float(solution.t_events[0][0])
    current = current_fraction * electrons * faraday * diffusivity_reference * radius * critical_coordinate / profile_parameters(1)[-1]
    source = half_space_source(centers, centers, radius, 1, quadrature_count, maximum_wave_number)
    coordinate = current * source / (electrons * faraday * diffusivity_reference)
    states = solution.sol(coordinate.ravel()).T.reshape(count, count, -1)
    concentrations = grid_concentrations(states, families)
    properties = species_properties(states[0, 0, :-2], states[0, 0, -2], families, diffusivity_reference)
    charges = properties["charges"]
    diffusivities = properties["diffusivities"] * diffusivity_reference
    projection = np.vstack([(properties["membership"] == group).astype(float) for group in range(len(families))]
                           + [charges.astype(float)])
    potential = states[..., -1]
    fluxes = []
    for left, right, voltage_difference, area, distance in (
        (concentrations[:-1, :], concentrations[1:, :], potential[1:, :] - potential[:-1, :],
         faces[1:-1, None] * widths[None, :], np.diff(centers)[:, None]),
        (concentrations[:, :-1], concentrations[:, 1:], potential[:, 1:] - potential[:, :-1],
         annular_areas[:, None] * np.ones((1, count - 1)), np.diff(centers)[None, :]),
    ):
        argument = voltage_difference[..., None] * charges
        species_flux = diffusivities * (area / distance)[..., None] * (bernoulli(argument) * left - bernoulli(-argument) * right)
        fluxes.append(species_flux @ projection.T)
    radial_flux, vertical_flux = fluxes
    divergence_integral = (radial_flux[1:, 1:-1] - radial_flux[:-1, 1:-1]
                           + vertical_flux[1:-1, 1:] - vertical_flux[1:-1, :-1])
    normalizers = np.r_[np.full(len(families), current / (electrons * faraday)), current / faraday]
    residuals = 2 * np.pi * np.sum(np.abs(divergence_integral), axis=(0, 1)) / normalizers
    return dict(zip([family["name"] for family in families] + ["electrical_current"], residuals.tolist()))


if __name__ == "__main__":
    summary_path = Path("transport_check/output/summary.json")  # Existing output from the coupled calculation.
    output_path = Path("transport_check/output/validation.json")  # Independent validation results.
    ph_samples = (7.8, 8.8, 9.8)  # Chemistry states used for finite-difference derivative checks.
    ph_step = 1e-5  # Symmetric finite-difference step, in pH units.
    derivative_tolerance = 2e-8  # Maximum accepted acid-fraction derivative error.
    integration_tolerances = (1e-8, 1e-10, 1e-12)  # Accuracy sequence for the coupled chemical path.
    finite_volume_grids = (24, 48, 96, 192)  # Independent radial and vertical cell counts.
    spatial_extent = 4.0  # Extent of the separate PDE check, in spot radii.
    spatial_current_fraction = 0.9  # Test a physical state close to depletion.
    quadrature_count = 1024  # Hankel nodes for the source function in the PDE check.
    maximum_wave_number = 12.0  # Upper Gaussian-source Hankel wave number.
    balance_tolerance = 1e-9  # Accepted local conservation and electroneutrality errors.
    expected_convergence_ratio = 0.4  # Each mesh doubling must reduce the PDE residual by at least this factor.
    faraday = 96485.33212  # Faraday constant, in C/mol.
    electrons = 2  # Assumed electrons per oxidation event.

    summary = json.loads(summary_path.read_text())
    families = summary["families_unfitted"]
    diffusivity_reference = summary["reference_diffusivity_m2_s"]
    thermal_voltage = summary["thermal_voltage_V"]
    radius = summary["radius_m"]
    totals, reservoir_ph, _ = prepare_reservoir(families, summary["recipe"])

    derivative_errors = []
    for family in families:
        for ph in ph_samples:
            _, analytic_derivative = acid_fractions(ph, family["pkas"], np.array(family["charges"]))
            plus, _ = acid_fractions(ph + ph_step, family["pkas"], np.array(family["charges"]))
            minus, _ = acid_fractions(ph - ph_step, family["pkas"], np.array(family["charges"]))
            derivative_errors.append(float(np.max(np.abs((plus - minus) / (2 * ph_step) - analytic_derivative))))
    maximum_derivative_error = max(derivative_errors)
    assert maximum_derivative_error < derivative_tolerance
    stoichiometry, target = surface_reaction(families, electrons)
    assert stoichiometry["TrisH+"] == 1 and stoichiometry["Tris"] == -1
    np.testing.assert_array_equal(target, np.r_[0, 0, 0, 0, -1, 1, electrons, 0])

    # An independently known exact solution: neutral reactant and product diffusion.
    neutral_families = [
        {"name": "ascorbate", "species": ["neutral_A"], "pkas": [], "charges": [0], "diffusivities": [0.66e-9]},
        {"name": "product", "species": ["neutral_P"], "pkas": [], "charges": [0], "diffusivities": [0.8e-9]},
    ]
    neutral_target = np.array([-1.0, 1.0, 0.0, 0.0])
    neutral_solution, _ = integrate_chemical_path(neutral_families, np.array([0.5, 0.0]), 7.0,
                                                 diffusivity_reference, 0, 1e-11, 1e-13, target=neutral_target)
    neutral_limit = float(neutral_solution.t_events[0][0])
    neutral_surface = neutral_solution.sol(neutral_limit)
    np.testing.assert_allclose(neutral_limit, 0.33, rtol=1e-11)
    np.testing.assert_allclose(neutral_surface, [0.0, 0.4125, 7.0, 0.0], atol=1e-11)

    integrations = []
    for tolerance in integration_tolerances:
        solution, target = integrate_chemical_path(families, totals, reservoir_ph, diffusivity_reference,
                                                  electrons, tolerance, tolerance / 100)
        limit = float(solution.t_events[0][0])
        surface_state = solution.sol(limit)
        surface_sigma = conductivity(surface_state[:-2], surface_state[-2], families, diffusivity_reference,
                                     faraday, thermal_voltage)[0]
        integrations.append({"relative_tolerance": tolerance, "depletion_coordinate": limit,
                             "surface_potential_V": float(surface_state[-1] * thermal_voltage),
                             "surface_pH": float(surface_state[-2]), "surface_conductivity_S_m": surface_sigma})
    for coarse in integrations[:-1]:
        for quantity in ("depletion_coordinate", "surface_potential_V", "surface_pH", "surface_conductivity_S_m"):
            np.testing.assert_allclose(coarse[quantity], integrations[-1][quantity], rtol=2e-7, atol=1e-11)

    local_balance_errors = []
    electroneutrality_errors = []
    for coordinate in np.linspace(0, limit, 151):
        state = solution.sol(coordinate)
        matrix, properties, _ = transport_matrix(state, families, diffusivity_reference)
        derivative = np.linalg.solve(matrix, target)
        local_balance_errors.append(float(np.max(np.abs(matrix @ derivative - target))))
        electroneutrality_errors.append(float(abs(np.dot(properties["charges"], properties["concentrations"]))))
    assert max(local_balance_errors) < balance_tolerance
    assert max(electroneutrality_errors) < balance_tolerance

    finite_volume_results = []
    for count in finite_volume_grids:
        residuals = finite_volume_residual(solution, families, radius, diffusivity_reference, electrons,
                                           faraday, count, spatial_extent, spatial_current_fraction,
                                           quadrature_count, maximum_wave_number)
        finite_volume_results.append({"cells_per_direction": count, "normalized_integrated_absolute_residuals": residuals})
        largest_residual = max(residuals.values())
        print(f"Independent {count}x{count} PDE check: largest integrated residual={largest_residual:.6g}.", flush=True)
    for previous, current in zip(finite_volume_results, finite_volume_results[1:]):
        old = max(previous["normalized_integrated_absolute_residuals"].values())
        new = max(current["normalized_integrated_absolute_residuals"].values())
        assert new / old < expected_convergence_ratio

    result = {"maximum_acid_fraction_derivative_error": maximum_derivative_error,
              "explicit_surface_reaction": stoichiometry, "neutral_diffusion_exact_limit": neutral_limit,
              "integration_convergence": integrations,
              "maximum_local_flux_balance_error": max(local_balance_errors),
              "maximum_electroneutrality_error_mol_m3": max(electroneutrality_errors),
              "independent_axisymmetric_finite_volume_checks": finite_volume_results}
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("All transport_check validation checks passed.", flush=True)
