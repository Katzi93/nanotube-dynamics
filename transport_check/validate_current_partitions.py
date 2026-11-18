"""Check reaction balance, measured-current solutions, and numerical convergence."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from coupled_flow_transport import CoupledTransportGrid
from run_current_partitions import solution_diagnostics
from transport_check import half_space_source, integrate_chemical_path, profile_parameters, surface_reaction


def archive_prefix(record, thermal_voltage):
    return (f"{record['current_label']}_f{record['ascorbate_fraction']:g}_R{record['extent_in_radii']:g}"
            f"_N{record['cells_per_direction']}_zeta{record['zeta_V'] / thermal_voltage:g}VT")


def compare_records(first, second):
    return {
        "surface_ascorbate_difference_mol_m3": abs(first["axis_surface_ascorbate_mol_m3"] - second["axis_surface_ascorbate_mol_m3"]),
        "surface_potential_difference_V": abs(first["axis_surface_potential_V"] - second["axis_surface_potential_V"]),
        "surface_conductivity_change_difference_percentage_points": abs(first["maximum_surface_conductivity_change_percent"] - second["maximum_surface_conductivity_change_percent"]),
        "maximum_slip_difference_m_s": abs(first["maximum_surface_slip_m_s"] - second["maximum_surface_slip_m_s"]),
    }


def save_comparison_figure(records, path, radius, source_label):
    selected = [item for item in records if item["current_label"] == source_label
                and item["extent_in_radii"] == 16 and item["cells_per_direction"] == 64]
    fractions = sorted({item["ascorbate_fraction"] for item in selected})
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)
    quantities = (("axis_surface_ascorbate_mol_m3", 1, "Surface ascorbate (mM)"),
                  ("maximum_surface_conductivity_change_percent", 1, "Maximum conductivity change (%)"),
                  ("axis_surface_potential_V", 1e3, "Surface potential (mV)"))
    for zeta in sorted({item["zeta_V"] for item in selected}, reverse=True):
        series = [next(item for item in selected if item["ascorbate_fraction"] == share and item["zeta_V"] == zeta)
                  for share in fractions]
        label = "No flow" if zeta == 0 else f"Flow, zeta = {zeta * 1e3:.1f} mV"
        style = "o--" if zeta == 0 else "s-"
        for axis, (quantity, scale, title) in zip(axes, quantities):
            axis.plot(np.array(fractions) * 100, [item[quantity] * scale for item in series], style, label=label)
            axis.set_xlabel("Ascorbate share of total current (%)")
            axis.set_ylabel(title)
            axis.grid(alpha=0.2)
    axes[0].axhline(0.5, color="0.6", linewidth=0.8)
    axes[0].text(0.01, 0.97, "Reservoir: 0.5 mM", transform=axes[0].transAxes, va="top", color="0.4", fontsize=9)
    axes[-1].legend(fontsize=8)
    current_nA = selected[0]["prescribed_current_A"] * 1e9
    radius_um = radius * 1e6
    figure.suptitle(f"Prescribed current {current_nA:.3f} nA; spot radius {radius_um:.3f} um\nRemaining current: hypothetical buffered water oxidation")
    figure.savefig(path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    base_path = Path("transport_check/output/summary.json")  # Original chemistry and measured-current inputs.
    output_dir = Path("transport_check/output/current_partitions")  # Existing fixed-current calculations.
    fractions_to_check = (0.0, 0.05, 0.10, 1.0)  # Include the exact ascorbate-only endpoint.
    reference_span_factor = 1.05  # Coverage of the finite chemical reference path.
    source_order = 1  # Gaussian spot profile.
    quadrature_count = 512  # Same accurately integrated source used in the driver.
    maximum_wave_number = 12.0  # Gaussian Hankel cutoff.
    permittivity = 78.3 * 8.8541878128e-12  # Water permittivity in F/m.
    viscosity = 0.89e-3  # Water viscosity in Pa s.
    residual_tolerance = 2.1e-10  # Maximum normalized steady transport residual.
    balance_tolerance = 1e-7  # Maximum global imbalance relative to imposed electron/reaction rate.
    concentration_tolerance = 1e-8  # Electroneutrality and reconstructed surface positivity tolerance in mol/m3.
    mesh_potential_tolerance = 2e-5  # Surface potential change allowed between independent meshes, in V.
    mesh_ascorbate_tolerance = 1e-3  # Surface ascorbate change allowed between meshes, in mol/m3.
    mesh_sigma_tolerance = 0.15  # Conductivity change difference allowed between meshes, in percentage points.
    no_flow_field_tolerance = 3e-3  # Maximum scaled difference from the independently integrated no-flow fields.
    path_relative_tolerance = 1e-11  # Accuracy of the independently integrated chemical reference.
    path_absolute_tolerance = 1e-13  # Absolute chemical reference tolerance.

    base = json.loads(base_path.read_text())
    saved = json.loads((output_dir / "summary.json").read_text())
    records = saved["records"]
    families = saved["families"]
    totals = np.array(list(saved["reservoir_totals_mol_m3"].values()))
    thermal_voltage = base["thermal_voltage_V"]
    diffusivity = base["reference_diffusivity_m2_s"]
    radius = base["radius_m"]
    current_scale = 2 * 96485.33212 * diffusivity * radius
    reference_current = saved["reference_current_A"]
    maximum_coordinate = reference_span_factor * reference_current / current_scale * profile_parameters(source_order)[-1]
    stoichiometry_checks = []
    for fraction in fractions_to_check:
        stoichiometry, target = surface_reaction(families, 2, fraction)
        np.testing.assert_allclose(target, [0, 0, 0, 0, -fraction, fraction, 2, 0], atol=1e-14)
        np.testing.assert_allclose(stoichiometry["TrisH+"], 2 - fraction, atol=1e-14)
        np.testing.assert_allclose(stoichiometry["Tris"], -(2 - fraction), atol=1e-14)
        stoichiometry_checks.append({"ascorbate_fraction": fraction, "target": target.tolist(), "source": stoichiometry})

    solutions = {}
    fresh_checks = []
    no_flow_checks = []
    for record in records:
        if record["current_label"] == "ascorbate_only_audit":
            continue
        if record["status"] != "converged":
            raise RuntimeError("A configured mixed-current case has no converged solution.")
        fraction = record["ascorbate_fraction"]
        if fraction not in solutions:
            _, target = surface_reaction(families, 2, fraction)
            solution, _ = integrate_chemical_path(families, totals, saved["reservoir_pH"], diffusivity, 2,
                                                  path_relative_tolerance, path_absolute_tolerance,
                                                  target=target, maximum_coordinate=maximum_coordinate, require_depletion=False)
            solutions[fraction] = solution, target
        solution, target = solutions[fraction]
        count = record["cells_per_direction"]
        prefix = archive_prefix(record, thermal_voltage)
        with np.load(output_dir / f"{prefix}.npz") as archive:
            states = archive["states"].reshape(count * count, -1)
            faces = archive["radial_faces_m"] / radius
            np.testing.assert_equal(float(archive["current_A"]), record["prescribed_current_A"])
            assert np.all(archive["species_concentrations_mol_m3"] >= 0)
        grid = CoupledTransportGrid(faces, families, solution, target, diffusivity, current_scale,
                                     reference_current, source_order, quadrature_count, maximum_wave_number)
        current_fraction = record["prescribed_current_A"] / reference_current
        mobility = -permittivity * record["zeta_V"] * thermal_voltage / (viscosity * diffusivity)
        flow = grid.flow(states, current_fraction, mobility)
        check, _, _ = solution_diagnostics(grid, states, current_fraction, flow, base)
        assert check["maximum_scaled_coupled_residual"] < residual_tolerance
        assert check["maximum_global_balance_error"] < balance_tolerance
        assert check["maximum_electroneutrality_error_mol_m3"] < concentration_tolerance
        assert check["minimum_surface_ascorbate_mol_m3"] > -concentration_tolerance
        assert check["maximum_wall_normal_liquid_flux"] == 0
        assert check["relative_discrete_fluid_divergence"] < 1e-12
        assert check["maximum_cell_peclet_ascorbate"] < 1
        fresh_checks.append({"case": prefix, "residual": check["maximum_scaled_coupled_residual"],
                             "global_balance": check["maximum_global_balance_error"],
                             "electroneutrality_error_mol_m3": check["maximum_electroneutrality_error_mol_m3"]})
        if record["zeta_V"] == 0:
            coordinate = current_fraction * reference_current / current_scale * half_space_source(
                grid.centers, grid.centers, 1.0, source_order, quadrature_count, maximum_wave_number)
            exact_states = solution.sol(coordinate.ravel()).T
            field_error = float(np.max(np.abs((states - exact_states) / grid.variable_scale)))
            assert field_error < no_flow_field_tolerance
            no_flow_checks.append({"case": prefix, "cells_per_direction": count, "scaled_field_error": field_error})
        print(f"Checked {prefix}: residual={check['maximum_scaled_coupled_residual']:.2e}, balance={check['maximum_global_balance_error']:.2e}.", flush=True)

    mesh_checks = []
    fine_records = [r for r in records if r["cells_per_direction"] == 64 and r["extent_in_radii"] == 16]
    assert len(fine_records) == 12
    for fine in fine_records:
        coarse = next(r for r in records if r["current_label"] == fine["current_label"]
                      and r["ascorbate_fraction"] == fine["ascorbate_fraction"] and r["zeta_V"] == fine["zeta_V"]
                      and r["cells_per_direction"] == 40 and r["extent_in_radii"] == 16)
        difference = compare_records(coarse, fine)
        assert difference["surface_potential_difference_V"] < mesh_potential_tolerance
        assert difference["surface_ascorbate_difference_mol_m3"] < mesh_ascorbate_tolerance
        assert difference["surface_conductivity_change_difference_percentage_points"] < mesh_sigma_tolerance
        mesh_checks.append({"case": archive_prefix(fine, thermal_voltage), **difference})
        if fine["zeta_V"] == 0:
            fine_error = next(r["scaled_field_error"] for r in no_flow_checks if r["case"] == archive_prefix(fine, thermal_voltage))
            coarse_error = next(r["scaled_field_error"] for r in no_flow_checks if r["case"] == archive_prefix(coarse, thermal_voltage))
            assert fine_error < coarse_error
    enlarged_records = [r for r in records if r["extent_in_radii"] == 32]
    assert len(enlarged_records) == 2
    domain_checks = []
    for enlarged in enlarged_records:
        fine = next(r for r in fine_records if r["current_label"] == enlarged["current_label"]
                    and r["ascorbate_fraction"] == enlarged["ascorbate_fraction"] and r["zeta_V"] == enlarged["zeta_V"])
        difference = compare_records(fine, enlarged)
        assert difference["surface_potential_difference_V"] < mesh_potential_tolerance
        assert difference["surface_ascorbate_difference_mol_m3"] < mesh_ascorbate_tolerance
        assert difference["surface_conductivity_change_difference_percentage_points"] < mesh_sigma_tolerance
        domain_checks.append({"case": archive_prefix(enlarged, thermal_voltage), **difference})
    audit = [r for r in records if r["current_label"] == "ascorbate_only_audit"]
    assert len(audit) == 7
    for record in audit:
        if record["status"] == "converged":
            assert record["minimum_surface_ascorbate_mol_m3"] > -concentration_tolerance
            assert record["maximum_scaled_coupled_residual"] < residual_tolerance
        else:
            assert record["status"] == "no_positive_converged_solution_found"
            assert "solver_error" in record
    result = {"reaction_balance": stoichiometry_checks, "fresh_solution_checks": fresh_checks,
              "independent_no_flow_comparison": no_flow_checks, "mesh_comparison": mesh_checks,
              "domain_comparison": domain_checks, "all_checks_passed": True}
    (output_dir / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for current_label in ("light_increment", "illuminated"):
        save_comparison_figure(records, output_dir / f"comparison_{current_label}.png", radius, current_label)
    print("All prescribed-current comparison checks passed.", flush=True)
