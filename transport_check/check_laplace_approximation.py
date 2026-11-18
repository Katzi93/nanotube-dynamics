"""Compare saved transport potentials with harmonic and Ohmic approximations.

No ionic transport problem is rerun. One comparison keeps the prescribed
current and a fixed conductivity. A separate harmonic extension retains the
computed surface potential and tests harmonicity of the bulk field itself.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from electroosmotic_flow import HalfSpaceSlipFlow, centered_radial_values_at_faces, extrapolate_to_wall
from transport_check import half_space_source, integrate_chemical_path, surface_reaction


def compare_potential(archive_path, record, saved, base, quadrature_count, maximum_wave_number,
                      inner_extent, reference_span_factor, path_relative_tolerance, path_absolute_tolerance):
    with np.load(archive_path) as archive:
        faces_m = archive["radial_faces_m"]
        states = archive["states"]
    radius = saved["radius_m"]
    faces = faces_m / radius
    centers = (faces[:-1] + faces[1:]) / 2
    current = record["prescribed_current_A"]
    thermal_voltage = saved["thermal_voltage_V"]
    potential = states[:, :, -1] * thermal_voltage
    surface_cells = extrapolate_to_wall(potential, centers)
    surface_nodes = centered_radial_values_at_faces(surface_cells, centers, faces)
    source_nodes = half_space_source(faces, faces, 1.0, saved["source_order"], quadrature_count, maximum_wave_number)
    ohmic_measured = current / base["measured_conductivity_S_m"] / radius * source_nodes[:, 0]
    ohmic_reservoir = current / record["reservoir_conductivity_S_m"] / radius * source_nodes[:, 0]
    center_matched = surface_nodes[0] * source_nodes[:, 0] / source_nodes[0, 0]
    electric_field = -np.gradient(surface_nodes, faces_m, edge_order=2)
    ohmic_field = -np.gradient(ohmic_measured, faces_m, edge_order=2)
    center_matched_field = -np.gradient(center_matched, faces_m, edge_order=2)
    electric_field[0] = ohmic_field[0] = center_matched_field[0] = 0

    # Retain the physical surface potential and match the nonlinear far boundary
    # used by transport. This tests harmonicity without imposing an Ohmic source.
    families = saved["families"]
    totals = np.array(list(saved["reservoir_totals_mol_m3"].values()))
    _, target = surface_reaction(families, 2, record["ascorbate_fraction"])
    current_scale = 2 * 96485.33212 * saved["reference_diffusivity_m2_s"] * radius
    coordinate = current / current_scale * source_nodes
    solution, _ = integrate_chemical_path(
        families, totals, saved["reservoir_pH"], saved["reference_diffusivity_m2_s"], 2,
        path_relative_tolerance, path_absolute_tolerance, target=target,
        maximum_coordinate=reference_span_factor * float(coordinate.max()), require_depletion=False,
    )
    if coordinate.max() > solution.t[-1]:
        raise ValueError("Chemical reference does not cover the required boundary states.")
    outer = solution.sol(coordinate.ravel())[-1].reshape(coordinate.shape)
    harmonic_solver = HalfSpaceSlipFlow(faces)
    harmonic_nodes = harmonic_solver.solve(surface_nodes / thermal_voltage, outer, 0.0)["harmonic_extension"] * thermal_voltage
    harmonic_centers = (harmonic_nodes[:-1, :-1] + harmonic_nodes[1:, :-1]
                        + harmonic_nodes[:-1, 1:] + harmonic_nodes[1:, 1:]) / 4
    inner = centers <= inner_extent
    inner_nodes = faces <= inner_extent
    potential_scale = float(surface_nodes[0])
    electric_field_scale = float(electric_field.max())
    bulk_difference = float(np.max(np.abs((potential - harmonic_centers)[np.ix_(inner, inner)])))
    surface_shape_difference = float(np.max(np.abs((surface_nodes - center_matched)[inner_nodes])))
    field_shape_difference = float(np.max(np.abs((electric_field - center_matched_field)[inner_nodes])))
    peak_index = int(np.argmax(electric_field))
    ohmic_peak_index = int(np.argmax(ohmic_field))
    result = {
        "extent_in_radii": record["extent_in_radii"], "cells_per_direction": record["cells_per_direction"],
        "surface_center_potential_V": potential_scale,
        "ohmic_center_potential_using_measured_conductivity_V": float(ohmic_measured[0]),
        "ohmic_center_potential_using_model_reservoir_conductivity_V": float(ohmic_reservoir[0]),
        "conductivity_needed_to_match_center_potential_S_m": float(current * source_nodes[0, 0] / radius / potential_scale),
        "actual_maximum_surface_electric_field_V_m": electric_field_scale,
        "ohmic_maximum_surface_electric_field_using_measured_conductivity_V_m": float(ohmic_field.max()),
        "actual_peak_field_radius_m": float(faces_m[peak_index]),
        "ohmic_peak_field_radius_m": float(faces_m[ohmic_peak_index]),
        "maximum_center_matched_surface_potential_shape_error_over_center_value": surface_shape_difference / potential_scale,
        "maximum_center_matched_surface_field_error_over_actual_peak": field_shape_difference / electric_field_scale,
        "maximum_bulk_departure_from_surface_matched_harmonic_extension_V": bulk_difference,
        "maximum_bulk_departure_over_center_surface_potential": bulk_difference / potential_scale,
        "comparison_region": f"0 <= r,z <= {inner_extent:g}a; surface comparisons use 0 <= r <= {inner_extent:g}a",
    }
    profiles = {"radial_m": faces_m, "surface_potential_V": surface_nodes,
                "ohmic_potential_measured_sigma_V": ohmic_measured, "center_matched_potential_V": center_matched,
                "surface_electric_field_V_m": electric_field, "ohmic_field_measured_sigma_V_m": ohmic_field,
                "center_matched_field_V_m": center_matched_field}
    return result, profiles


if __name__ == "__main__":
    input_dir = Path("transport_check/output/current_partitions")  # Validated prescribed-current spatial fields.
    base_path = Path("transport_check/output/summary.json")  # Measured bulk conductivity and original inputs.
    output_dir = Path("transport_check/output/laplace_check")  # Diagnostic results, without changing the SI.
    current_label = "illuminated"  # Full localized-current scenario, approximately 172 nA.
    ascorbate_fraction = 0.05  # Same explicit mixed-reaction hypothesis as the enlarged-domain check.
    zeta_multiple = -2.0  # Include the previously solved electroosmotic flow.
    meshes = ((16.0, 40), (16.0, 64), (32.0, 96))  # Check that the comparison survives mesh/domain changes.
    quadrature_count = 512  # Gaussian source integration accuracy.
    maximum_wave_number = 12.0  # Gaussian source Hankel cutoff.
    inner_extent = 4.0  # Compare within four spot radii, where the field drives the main flow.
    reference_span_factor = 1.05  # Chemical reference coverage beyond the maximum requested coordinate.
    path_relative_tolerance = 1e-11  # Remote nonlinear chemical boundary accuracy.
    path_absolute_tolerance = 1e-13  # Absolute chemical reference tolerance.

    saved = json.loads((input_dir / "summary.json").read_text())
    base = json.loads(base_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for extent, count in meshes:
        item = next(r for r in saved["records"] if r["current_label"] == current_label
                    and r["ascorbate_fraction"] == ascorbate_fraction
                    and r["zeta_V"] == zeta_multiple * saved["thermal_voltage_V"]
                    and r["extent_in_radii"] == extent and r["cells_per_direction"] == count)
        prefix = f"{current_label}_f{ascorbate_fraction:g}_R{extent:g}_N{count}_zeta{zeta_multiple:g}VT"
        result, profiles = compare_potential(input_dir / f"{prefix}.npz", item, saved, base, quadrature_count,
                                              maximum_wave_number, inner_extent, reference_span_factor,
                                              path_relative_tolerance, path_absolute_tolerance)
        records.append(result)
        print(json.dumps(result, indent=2), flush=True)
    (output_dir / "summary.json").write_text(json.dumps({
        "reaction_status": saved["alternative_reaction_status"], "current_A": item["prescribed_current_A"],
        "ascorbate_fraction": ascorbate_fraction, "measured_conductivity_S_m": base["measured_conductivity_S_m"],
        "radius_m": saved["radius_m"], "records": records,
        "harmonic_extension_boundary": "Computed surface potential and nonlinear no-flow remote boundary from the transport model",
        "shape_normalization": "Ohmic potential multiplied by one constant to match the computed center surface potential; diagnostic only",
    }, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output_dir / "surface_profiles.npz", **profiles)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    radius_um = profiles["radial_m"] * 1e6
    for axis, quantities, scale, label in (
        (axes[0], ("surface_potential_V", "ohmic_potential_measured_sigma_V", "center_matched_potential_V"), 1e3, "Surface potential (mV)"),
        (axes[1], ("surface_electric_field_V_m", "ohmic_field_measured_sigma_V_m", "center_matched_field_V_m"), 1, "Outward surface electric field (V/m)"),
    ):
        for quantity, style, title in zip(quantities, ("-", "--", ":"),
                                         ("Transport calculation", "Ohmic, measured bulk conductivity", "Ohmic, amplitude matched at center")):
            axis.plot(radius_um, profiles[quantity] * scale, style, label=title)
        axis.set_xlim(0, inner_extent * saved["radius_m"] * 1e6)
        axis.set_xlabel("Radial distance (um)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("172 nA, 5% ascorbate + hypothetical buffered water oxidation\nComparison of potential amplitude and spatial shape")
    figure.savefig(output_dir / "surface_comparison.png", dpi=180)
    plt.close(figure)
