"""Manufactured-flow, Jacobian, convergence, and conservation checks for EOF coupling."""

from pathlib import Path
import json

import numpy as np

from coupled_flow_transport import CoupledTransportGrid
from electroosmotic_flow import HalfSpaceSlipFlow
from run_flow_transport import load_chemical_case
from transport_check import half_space_source


def check_manufactured_stokes(counts, extent, mesh_offset, source_depth, mobility):
    records = []
    for count in counts:
        faces = mesh_offset * np.expm1(np.linspace(0, np.log1p(extent / mesh_offset), count + 1))
        radial = faces[:, None]
        shifted_height = faces[None, :] + source_depth
        distance = np.sqrt(radial**2 + shifted_height**2)
        exact_harmonic = 1 / distance
        exact_stream = -mobility * faces[None, :] * radial**2 / distance**3
        solver = HalfSpaceSlipFlow(faces)
        result = solver.solve(exact_harmonic[:, 0], exact_harmonic, mobility)
        exact_radial_flux = -np.diff(exact_stream, axis=1)
        exact_vertical_flux = np.diff(exact_stream, axis=0)
        numerator = np.sum((result["radial_flux"] - exact_radial_flux)**2) + np.sum((result["vertical_flux"] - exact_vertical_flux)**2)
        denominator = np.sum(exact_radial_flux**2) + np.sum(exact_vertical_flux**2)
        flow_error = float(np.sqrt(numerator / denominator))
        harmonic_error = float(np.max(np.abs(result["harmonic_extension"] - exact_harmonic)))
        assert result["relative_discrete_divergence"] < 1e-12
        assert np.max(np.abs(result["vertical_flux"][:, 0])) == 0
        records.append({"cells_per_direction": count, "relative_face_flux_error": flow_error,
                        "maximum_harmonic_extension_error": harmonic_error,
                        "relative_discrete_divergence": result["relative_discrete_divergence"]})
        print(f"Manufactured Stokes {count}x{count}: flow error={flow_error:.6g}, harmonic error={harmonic_error:.6g}.", flush=True)
    for previous, current in zip(records, records[1:]):
        assert current["relative_face_flux_error"] < previous["relative_face_flux_error"] * 0.4
    return records


def check_transport_jacobian(summary, cell_count, extent, difference_step, random_seed):
    case, families, solution, target = load_chemical_case(summary, "unfitted", 1e-11, 1e-13)
    faces = np.linspace(0, extent, cell_count + 1)
    faraday = 96485.33212
    current_scale = 2 * faraday * summary["reference_diffusivity_m2_s"] * summary["radius_m"]
    grid = CoupledTransportGrid(faces, families, solution, target, summary["reference_diffusivity_m2_s"],
                                current_scale, case["current_limits_A"]["gaussian"], 1, 384, 12.0)
    current_fraction = 0.6
    states = grid.initial_state(current_fraction)
    flow = grid.flow(states, current_fraction, 0.5)
    residual, matrix, current_derivative, constraint = grid.assemble(states, current_fraction, flow)
    generator = np.random.default_rng(random_seed)
    direction = generator.normal(size=states.size)
    current_direction = 0.3
    physical_direction = (grid.column_scale * direction).reshape(states.shape)
    plus, _, _, plus_constraint = grid.assemble(states + difference_step * physical_direction,
                                               current_fraction + difference_step * current_direction, flow)
    minus, _, _, minus_constraint = grid.assemble(states - difference_step * physical_direction,
                                                current_fraction - difference_step * current_direction, flow)
    finite_difference = (plus - minus) / (2 * difference_step)
    analytic = matrix @ direction + current_derivative * current_direction
    relative_error = float(np.linalg.norm(finite_difference - analytic) / np.linalg.norm(analytic))
    constraint_derivative = np.dot(grid.constraint_weights, physical_direction[grid.constraint_cells, grid.ascorbate_index])
    constraint_error = float(abs((plus_constraint - minus_constraint) / (2 * difference_step) - constraint_derivative))
    assert relative_error < 2e-7
    assert constraint_error < 1e-8
    print(f"Full transport/current Jacobian relative error={relative_error:.3e}.", flush=True)
    return {"relative_directional_derivative_error": relative_error, "depletion_constraint_derivative_error": constraint_error}


def check_saved_no_flow(summary, flow_summary, output_dir, quadrature_count, maximum_wave_number):
    case, families, solution, _ = load_chemical_case(summary, "unfitted", 1e-11, 1e-13)
    records = []
    for item in flow_summary["records"]:
        if item["chemical_case"] != "unfitted" or item["source_profile"] != "gaussian" or item["zeta_V"] != 0:
            continue
        count = item["cells_per_direction"]
        extent = item["extent_in_radii"]
        archive_path = output_dir / f"unfitted_gaussian_R{extent:g}_N{count}_zeta0VT.npz"
        with np.load(archive_path) as archive:
            faces = archive["radial_faces_m"] / summary["radius_m"]
            centers = (faces[:-1] + faces[1:]) / 2
            current = float(archive["current_A"])
            states = archive["states"]
        source = half_space_source(centers, centers, 1.0, 1, quadrature_count, maximum_wave_number)
        coordinate = current * source / (2 * 96485.33212 * summary["reference_diffusivity_m2_s"] * summary["radius_m"])
        exact = solution.sol(coordinate.ravel()).T.reshape(states.shape)
        totals = solution.sol(0.0)[:-2]
        variable_scale = np.r_[np.maximum(totals, totals[4]), 1.0, 1.0]
        field_error = float(np.max(np.abs((states - exact) / variable_scale)))
        current_error = abs(item["current_over_exact_no_flow_limit"] - 1)
        records.append({"extent_in_radii": extent, "cells_per_direction": count,
                        "maximum_scaled_no_flow_field_error": field_error,
                        "relative_no_flow_current_limit_error": current_error})
        print(f"No-flow reference {count}x{count}, R={extent:g}a: field error={field_error:.4g}, current error={current_error:.4g}.", flush=True)
    fixed_domain = sorted([item for item in records if item["extent_in_radii"] == 16], key=lambda item: item["cells_per_direction"])
    if len(fixed_domain) < 3:
        raise RuntimeError("Run the configured three fixed-domain transport meshes before completing validation.")
    for previous, current in zip(fixed_domain, fixed_domain[1:]):
        assert current["maximum_scaled_no_flow_field_error"] < previous["maximum_scaled_no_flow_field_error"]
        assert current["relative_no_flow_current_limit_error"] < previous["relative_no_flow_current_limit_error"]
    assert fixed_domain[-1]["relative_no_flow_current_limit_error"] < 2e-4
    return records


def check_global_coupled_balance(summary, flow_summary, output_dir, permittivity, viscosity):
    item = max((record for record in flow_summary["records"]
                if record["chemical_case"] == "unfitted" and record["source_profile"] == "gaussian"),
               key=lambda record: (record["extent_in_radii"], -record["zeta_V"]))
    extent = item["extent_in_radii"]
    count = item["cells_per_direction"]
    zeta_multiple = item["zeta_V"] / summary["thermal_voltage_V"]
    archive_path = output_dir / f"unfitted_gaussian_R{extent:g}_N{count}_zeta{zeta_multiple:g}VT.npz"
    with np.load(archive_path) as archive:
        states = archive["states"].reshape(count * count, -1)
        faces = archive["radial_faces_m"] / summary["radius_m"]
    case, families, solution, target = load_chemical_case(summary, "unfitted", 1e-11, 1e-13)
    current_scale = 2 * 96485.33212 * summary["reference_diffusivity_m2_s"] * summary["radius_m"]
    grid = CoupledTransportGrid(faces, families, solution, target, summary["reference_diffusivity_m2_s"],
                                current_scale, case["current_limits_A"]["gaussian"], 1, 512, 12.0)
    mobility = -permittivity * item["zeta_V"] * summary["thermal_voltage_V"] / (viscosity * summary["reference_diffusivity_m2_s"])
    fraction = item["current_over_exact_no_flow_limit"]
    flow = grid.flow(states, fraction, mobility)
    residual, _, _, constraint = grid.assemble(states, fraction, flow)
    physical_residual = residual.reshape(states.shape) / grid.inverse_row_scale
    source_scale = fraction * grid.bottom_source.sum()
    component_balances = physical_residual[:, grid.equation_indices].sum(axis=0) / source_scale
    component_balances[-1] /= 2
    maximum_error = float(np.max(np.abs(component_balances)))
    maximum_residual = float(np.max(np.abs(residual)))
    assert maximum_error < 2e-6
    assert maximum_residual < 3e-10
    assert abs(constraint) < 3e-10
    print(f"Fresh coupled residual={maximum_residual:.3e}; global component/current balance error={maximum_error:.3e}.", flush=True)
    return {"chemical_case": item["chemical_case"], "extent_in_radii": extent, "cells_per_direction": count,
            "zeta_V": item["zeta_V"], "fresh_maximum_coupled_residual": maximum_residual,
            "normalized_global_component_and_charge_balances": component_balances.tolist(),
            "maximum_normalized_global_balance_error": maximum_error}


def check_prescribed_current_mode(summary, currents, permittivity, viscosity):
    case, families, solution, target = load_chemical_case(summary, "unfitted", 1e-11, 1e-13)
    faces = 0.5 * np.expm1(np.linspace(0, np.log1p(16 / 0.5), 25))
    current_scale = 2 * 96485.33212 * summary["reference_diffusivity_m2_s"] * summary["radius_m"]
    reference_current = case["current_limits_A"]["gaussian"]
    grid = CoupledTransportGrid(faces, families, solution, target, summary["reference_diffusivity_m2_s"],
                                current_scale, reference_current, 1, 384, 12.0)
    zeta = -2 * summary["thermal_voltage_V"]
    mobility = -permittivity * zeta * summary["thermal_voltage_V"] / (viscosity * summary["reference_diffusivity_m2_s"])
    checks = []
    for prescribed_current in currents:
        prescribed_fraction = prescribed_current / reference_current
        states, fraction, flow, history = grid.solve_coupled(
            mobility, 2e-10, 2e-8, 12, 10, initial_current_fraction=prescribed_fraction,
            find_depletion_current=False,
        )
        np.testing.assert_equal(fraction, prescribed_fraction)
        residual, _, _, surface_constraint = grid.assemble(states, fraction, flow)
        assert surface_constraint > 0
        assert np.max(np.abs(residual)) < 2e-10
        checks.append({"prescribed_current_A": prescribed_current,
                       "solved_current_A": fraction * reference_current,
                       "reconstructed_axis_surface_ascorbate_mol_m3": surface_constraint * grid.variable_scale[grid.ascorbate_index],
                       "maximum_scaled_coupled_residual": float(np.max(np.abs(residual)))})
        current_nA = prescribed_current * 1e9
        print(f"Prescribed-current mode: {current_nA:.1f} nA remains fixed; surface ascorbate is positive.", flush=True)
    return checks


if __name__ == "__main__":
    summary_path = Path("transport_check/output/summary.json")  # Existing chemical-model inputs.
    flow_output = Path("transport_check/output/flow")  # Existing coupled calculation outputs.
    manufactured_counts = (24, 48, 96)  # Mesh refinement for an analytic half-space Stokes solution.
    manufactured_extent = 8.0  # Outer radius and height of the manufactured flow check.
    manufactured_mesh_offset = 0.5  # Refinement at the electrode and axis.
    source_depth = 1.0  # Depth of the analytic harmonic point source below the wall.
    manufactured_mobility = 0.7  # Mobility coefficient of the manufactured Stokes solution.
    jacobian_cells = 8  # Small independent mesh used to test every transport Jacobian block.
    jacobian_extent = 4.0  # Size of the Jacobian test domain, in spot radii.
    difference_step = 1e-6  # Centered directional finite-difference step in scaled variables.
    random_seed = 9127  # Reproducible mixed chemical, pH, and potential perturbation.
    source_quadrature_count = 1024  # Source integration accuracy for independent no-flow field comparisons.
    source_maximum_wave_number = 12.0  # Gaussian source wave-number cutoff.
    maximum_coupled_residual = 2.1e-10  # Accepted normalized steady transport residual.
    permittivity = 78.3 * 8.8541878128e-12  # Same water permittivity as the coupled driver, in F/m.
    viscosity = 0.89e-3  # Same dynamic viscosity as the coupled driver, in Pa s.
    prescribed_currents = (5e-9, 10e-9, 14e-9)  # Operating-current tests separate from first-depletion diagnostics.

    summary = json.loads(summary_path.read_text())
    flow_summary = json.loads((flow_output / "summary.json").read_text())
    manufactured = check_manufactured_stokes(manufactured_counts, manufactured_extent, manufactured_mesh_offset,
                                            source_depth, manufactured_mobility)
    jacobian = check_transport_jacobian(summary, jacobian_cells, jacobian_extent, difference_step, random_seed)
    no_flow = check_saved_no_flow(summary, flow_summary, flow_output, source_quadrature_count, source_maximum_wave_number)
    for item in flow_summary["records"]:
        assert item["coupling_history"][-1]["coupled_residual"] < maximum_coupled_residual
        assert item["maximum_wall_normal_liquid_flux"] == 0
        assert item["relative_discrete_fluid_divergence"] < 1e-12
        assert item["minimum_cell_ascorbate_mol_m3"] > 0
        assert item["maximum_electroneutrality_error_mol_m3"] < 1e-8
    primary = [item for item in flow_summary["records"] if item["chemical_case"] == "unfitted" and item["source_profile"] == "gaussian"]
    enhancement_comparisons = []
    for zeta in sorted({item["zeta_V"] for item in primary if item["zeta_V"] != 0}):
        fixed = max([item for item in primary if item["zeta_V"] == zeta and item["extent_in_radii"] == 16],
                     key=lambda item: item["cells_per_direction"])
        enlarged = next(item for item in primary if item["zeta_V"] == zeta and item["extent_in_radii"] == 32)
        enhancement_difference = abs(fixed["current_over_same_mesh_no_flow"] - enlarged["current_over_same_mesh_no_flow"])
        assert enhancement_difference < 2e-5
        enhancement_comparisons.append({"zeta_V": zeta, "flow_enhancement_fraction_R16": fixed["current_over_same_mesh_no_flow"] - 1,
                                       "flow_enhancement_fraction_R32": enlarged["current_over_same_mesh_no_flow"] - 1,
                                       "absolute_difference": enhancement_difference})
    result = {"manufactured_stokes": manufactured, "transport_jacobian": jacobian, "no_flow_comparisons": no_flow,
              "larger_domain_comparison": enhancement_comparisons,
              "global_coupled_balance": check_global_coupled_balance(summary, flow_summary, flow_output, permittivity, viscosity),
              "prescribed_current_mode_checks": check_prescribed_current_mode(summary, prescribed_currents, permittivity, viscosity),
              "all_checks_passed": True}
    (flow_output / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("All coupled-flow validation checks passed.", flush=True)
