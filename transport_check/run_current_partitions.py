"""Test prescribed measured currents under explicit alternative reaction shares.

The optional second reaction is buffered water oxidation, not an identified
reaction of the experimental ITO electrode. All cases keep the measured total
current fixed. No depleted or failed solution is substituted for that current.
"""

from pathlib import Path
import csv
import json
from time import perf_counter

import numpy as np

from coupled_flow_transport import CoupledTransportGrid, chemistry_fields
from electroosmotic_flow import extrapolate_to_wall
from run_flow_transport import load_chemical_case, save_flow_figure
from transport_check import integrate_chemical_path, profile_parameters, species_properties, surface_reaction


def solution_diagnostics(grid, states, fraction, flow, summary):
    count = grid.count
    diffusivity = summary["reference_diffusivity_m2_s"]
    radius = summary["radius_m"]
    thermal_voltage = summary["thermal_voltage_V"]
    concentrations, _ = chemistry_fields(states, grid.families)
    reservoir, _ = chemistry_fields(grid.solution.sol(0.0)[None, :], grid.families)
    sigma = 96485.33212 / thermal_voltage * diffusivity * (concentrations * grid.charges**2 * grid.diffusivities).sum(axis=1)
    sigma_reference = float(96485.33212 / thermal_voltage * diffusivity * (reservoir * grid.charges**2 * grid.diffusivities).sum())
    surface = np.column_stack([extrapolate_to_wall(states[:, index].reshape(count, count), grid.centers)
                               for index in range(grid.variable_count)])
    axis_weights = grid.constraint_weights * grid.variable_scale[grid.ascorbate_index]
    axis_surface = axis_weights @ states[grid.constraint_cells]
    surface_chemistry, _ = chemistry_fields(surface, grid.families)
    surface_sigma = 96485.33212 / thermal_voltage * diffusivity * (surface_chemistry * grid.charges**2 * grid.diffusivities).sum(axis=1)
    speed = np.hypot(flow["radial_velocity"], flow["vertical_velocity"]) * diffusivity / radius
    residual, _, _, _ = grid.assemble(states, fraction, flow)
    unscaled_residual = residual.reshape(states.shape) / grid.inverse_row_scale
    global_balance = unscaled_residual[:, grid.equation_indices].sum(axis=0) / (fraction * grid.bottom_source.sum())
    global_balance[-1] /= 2
    concentration_field = concentrations.reshape(count, count, -1)
    first_charge_gradient = (concentration_field[0, 1] - concentration_field[0, 0]) / np.diff(grid.centers[:2])[0]
    first_potential_gradient = (states.reshape(count, count, -1)[0, 1, -1]
                                - states.reshape(count, count, -1)[0, 0, -1]) / np.diff(grid.centers[:2])[0]
    first_concentrations = (concentration_field[0, 1] + concentration_field[0, 0]) / 2
    diffusive_current = -float(np.dot(grid.charges * grid.diffusivities, first_charge_gradient))
    migration_current = -float(np.dot(grid.charges**2 * grid.diffusivities, first_concentrations)) * first_potential_gradient
    diffusion_fraction = diffusive_current / (diffusive_current + migration_current)
    cell_peclet = (np.abs(flow["radial_velocity"]) * grid.widths[:, None]
                   + np.abs(flow["vertical_velocity"]) * grid.widths[None, :]) * diffusivity / grid.families[grid.ascorbate_index]["diffusivities"][0]
    record = {
        "reservoir_conductivity_S_m": sigma_reference,
        "minimum_conductivity_change_percent": float(100 * (sigma.min() / sigma_reference - 1)),
        "maximum_conductivity_change_percent": float(100 * (sigma.max() / sigma_reference - 1)),
        "minimum_surface_conductivity_change_percent": float(100 * (surface_sigma.min() / sigma_reference - 1)),
        "maximum_surface_conductivity_change_percent": float(100 * (surface_sigma.max() / sigma_reference - 1)),
        "axis_surface_ascorbate_mol_m3": float(axis_surface[grid.ascorbate_index]),
        "minimum_cell_ascorbate_mol_m3": float(states[:, grid.ascorbate_index].min()),
        "minimum_surface_ascorbate_mol_m3": float(surface[:, grid.ascorbate_index].min()),
        "axis_surface_potential_V": float(axis_surface[-1] * thermal_voltage),
        "axis_surface_pH": float(axis_surface[-2]),
        "axis_surface_component_totals_mol_m3": dict(zip([f["name"] for f in grid.families], axis_surface[:-2].tolist())),
        "maximum_surface_slip_m_s": float(np.max(np.abs(flow["surface_slip"])) * diffusivity / radius),
        "maximum_cell_speed_m_s": float(speed.max()),
        "maximum_speed_peclet_on_spot_radius": float(speed.max() * radius / grid.families[grid.ascorbate_index]["diffusivities"][0]),
        "maximum_cell_peclet_ascorbate": float(cell_peclet.max()),
        "near_axis_first_vertical_face_diffusion_current_fraction": diffusion_fraction,
        "maximum_scaled_coupled_residual": float(np.max(np.abs(residual))),
        "normalized_global_component_and_charge_balances": global_balance.tolist(),
        "maximum_global_balance_error": float(np.max(np.abs(global_balance))),
        "maximum_electroneutrality_error_mol_m3": float(np.max(np.abs(concentrations @ grid.charges))),
        "maximum_wall_normal_liquid_flux": float(np.max(np.abs(flow["vertical_flux"][:, 0]))),
        "relative_discrete_fluid_divergence": flow["relative_discrete_divergence"],
    }
    return record, concentrations, sigma


def save_checkpoint(output_dir, inputs, records):
    (output_dir / "summary.json").write_text(json.dumps({**inputs, "records": records}, indent=2) + "\n", encoding="utf-8")
    fields = ["current_label", "prescribed_current_A", "ascorbate_fraction", "zeta_V", "extent_in_radii",
              "cells_per_direction", "status", "axis_surface_ascorbate_mol_m3", "axis_surface_potential_V",
              "minimum_surface_conductivity_change_percent", "maximum_surface_conductivity_change_percent",
              "maximum_surface_slip_m_s", "maximum_scaled_coupled_residual"]
    with (output_dir / "case_summary.csv").open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    summary_path = Path("transport_check/output/summary.json")  # Existing recipe, chemistry, and measured currents.
    output_dir = Path("transport_check/output/current_partitions")  # Prescribed-current comparison outputs.
    case_name = "unfitted"  # Retain the assumed OPA mobility without fitting conductivity.
    current_labels = ("light_increment", "illuminated")  # Two alternative assignments of localized spot current.
    ascorbate_fractions = (0.0, 0.05, 0.10)  # Explicit hypotheses for the fraction of electron current from ascorbate.
    zeta_multiples = (0.0, -2.0)  # Compare no flow with slip driven by zeta = -2 V_T.
    meshes = ((16.0, 40), (16.0, 64), (32.0, 96))  # The enlarged domain checks the full-current, 5% case only.
    mesh_offset = 0.5  # Geometric refinement at the axis and wall, in spot radii.
    source_order = 1  # Gaussian source with the experimentally estimated half-maximum radius.
    source_quadrature_count = 512  # Hankel quadrature nodes for Gaussian source geometry.
    maximum_wave_number = 12.0  # Gaussian source Hankel cutoff.
    reference_span_factor = 1.05  # Positive no-flow path coverage beyond the largest requested surface coordinate.
    permittivity = 78.3 * 8.8541878128e-12  # Water permittivity near 25 C, in F/m.
    viscosity = 0.89e-3  # Water viscosity near 25 C, in Pa s.
    path_relative_tolerance = 1e-11  # Accuracy of the independent no-flow chemical path.
    path_absolute_tolerance = 1e-13  # Absolute chemical path tolerance.
    newton_tolerance = 2e-10  # Maximum scaled steady transport residual.
    flow_tolerance = 2e-8  # Relative conservative fluid-flux change tolerance.
    maximum_newton_iterations = 16  # Newton steps for each fixed fluid field.
    maximum_flow_iterations = 16  # Alternating Stokes/transport updates.
    audit_currents_nA = (10.0, 14.0, 14.6, 15.0, 20.0, 113.8532, 172.158)  # All-current-to-ascorbate attempts, not a cap.
    audit_count = 40  # Fixed-current positivity audit mesh; depletion was separately mesh-validated.
    save_fields = True  # Save reusable species, potential, conductivity, and liquid fields.
    resume_completed_cases = True  # Reuse completed cases only when their saved model metadata matches.

    summary = json.loads(summary_path.read_text())
    case, families, _, _ = load_chemical_case(summary, case_name, path_relative_tolerance, path_absolute_tolerance)
    totals = np.array(list(case["reservoir_totals_mol_m3"].values()))
    radius = summary["radius_m"]
    diffusivity = summary["reference_diffusivity_m2_s"]
    thermal_voltage = summary["thermal_voltage_V"]
    current_scale = 2 * 96485.33212 * diffusivity * radius
    reference_current = summary["measured_currents_A"]["illuminated"]
    maximum_coordinate = reference_span_factor * reference_current / current_scale * profile_parameters(source_order)[-1]
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "model": "Prescribed total current; locally equilibrated electroneutral Nernst-Planck/Stokes transport",
        "alternative_reaction_status": "Hypothetical buffered water oxidation; not identified experimentally",
        "neutral_oxygen_status": "Stoichiometric source recorded; passive neutral O2 transport and bubbles omitted",
        "current_localization_status": "Each selected current is separately assigned to the same Gaussian spot; these are alternative scenarios",
        "chemical_case": case_name, "radius_m": radius, "reference_current_A": reference_current,
        "measured_currents_A": summary["measured_currents_A"],
        "reference_diffusivity_m2_s": diffusivity, "thermal_voltage_V": thermal_voltage,
        "permittivity_F_m": permittivity, "viscosity_Pa_s": viscosity,
        "reservoir_pH": case["reservoir_pH"], "families": families,
        "reservoir_totals_mol_m3": case["reservoir_totals_mol_m3"],
        "newton_tolerance": newton_tolerance, "flow_tolerance": flow_tolerance,
        "source_order": source_order, "source_quadrature_count": source_quadrature_count,
        "maximum_wave_number": maximum_wave_number, "mesh_offset_in_radii": mesh_offset,
        "reference_span_factor": reference_span_factor,
        "path_relative_tolerance": path_relative_tolerance, "path_absolute_tolerance": path_absolute_tolerance,
    }
    records = []
    if resume_completed_cases and (output_dir / "summary.json").exists():
        previous = json.loads((output_dir / "summary.json").read_text())
        records = previous.pop("records")
        if previous != inputs:
            raise ValueError("Saved current-partition settings differ; use a separate output directory.")
    for extent, count in meshes:
        faces = mesh_offset * np.expm1(np.linspace(0, np.log1p(extent / mesh_offset), count + 1))
        for ascorbate_fraction in ascorbate_fractions:
            if extent > 16 and ascorbate_fraction != 0.05:
                continue
            stoichiometry, target = surface_reaction(families, 2, ascorbate_fraction)
            solution, _ = integrate_chemical_path(
                families, totals, case["reservoir_pH"], diffusivity, 2, path_relative_tolerance, path_absolute_tolerance,
                target=target, maximum_coordinate=maximum_coordinate, require_depletion=False,
            )
            grid = CoupledTransportGrid(faces, families, solution, target, diffusivity, current_scale,
                                         reference_current, source_order, source_quadrature_count, maximum_wave_number)
            for current_label in current_labels:
                if extent > 16 and current_label != "illuminated":
                    continue
                current = summary["measured_currents_A"][current_label]
                current_fraction = current / reference_current
                initial_states = grid.initial_state(current_fraction)
                for zeta_multiple in zeta_multiples:
                    zeta = zeta_multiple * thermal_voltage
                    prefix = f"{current_label}_f{ascorbate_fraction:g}_R{extent:g}_N{count}_zeta{zeta_multiple:g}VT"
                    completed = [r for r in records if r["current_label"] == current_label
                                 and r["ascorbate_fraction"] == ascorbate_fraction and r["zeta_V"] == zeta
                                 and r["extent_in_radii"] == extent and r["cells_per_direction"] == count]
                    if completed:
                        with np.load(output_dir / f"{prefix}.npz") as archive:
                            initial_states = archive["states"].reshape(grid.cell_count, grid.variable_count)
                        continue
                    mobility = -permittivity * zeta * thermal_voltage / (viscosity * diffusivity)
                    current_nA = current * 1e9
                    started = perf_counter()
                    print(f"Prescribed {current_label}: {current_nA:.4f} nA, ascorbate share={ascorbate_fraction:.0%}, zeta/V_T={zeta_multiple:g}, R={extent:g}, N={count}.", flush=True)
                    states, solved_fraction, flow, history = grid.solve_coupled(
                        mobility, newton_tolerance, flow_tolerance, maximum_newton_iterations, maximum_flow_iterations,
                        initial_states=initial_states, initial_current_fraction=current_fraction, find_depletion_current=False,
                    )
                    if solved_fraction != current_fraction:
                        raise RuntimeError("Prescribed current changed during the solve.")
                    record, concentrations, sigma = solution_diagnostics(grid, states, solved_fraction, flow, summary)
                    if record["minimum_surface_ascorbate_mol_m3"] < 0:
                        raise RuntimeError("The reconstructed surface ascorbate is negative; this operating point is unphysical.")
                    record.update({"current_label": current_label, "prescribed_current_A": current,
                                   "ascorbate_fraction": ascorbate_fraction, "ascorbate_current_A": current * ascorbate_fraction,
                                   "zeta_V": zeta, "extent_in_radii": extent, "cells_per_direction": count,
                                   "surface_stoichiometry_per_two_electrons": stoichiometry,
                                   "status": "converged", "coupling_history": history,
                                   "elapsed_seconds": perf_counter() - started})
                    records.append(record)
                    save_checkpoint(output_dir, inputs, records)
                    if save_fields:
                        labels = species_properties(totals, case["reservoir_pH"], families, diffusivity)["labels"]
                        np.savez_compressed(output_dir / f"{prefix}.npz", radial_faces_m=faces * radius,
                                            states=states.reshape(count, count, -1),
                                            state_labels=np.array([f["name"] for f in families] + ["pH", "phi_over_VT"]),
                                            species_concentrations_mol_m3=concentrations.reshape(count, count, -1),
                                            species_labels=np.array(labels), conductivity_S_m=sigma.reshape(count, count),
                                            radial_velocity_m_s=flow["radial_velocity"] * diffusivity / radius,
                                            vertical_velocity_m_s=flow["vertical_velocity"] * diffusivity / radius,
                                            streamfunction_m3_s=flow["streamfunction"] * diffusivity * radius,
                                            current_A=current, ascorbate_fraction=ascorbate_fraction, zeta_V=zeta)
                        save_flow_figure(output_dir / f"{prefix}.png", grid, states, flow, diffusivity, radius,
                                         thermal_voltage, current_nA, zeta * 1e3)
                    initial_states = states.copy()
                    surface_ascorbate = record["axis_surface_ascorbate_mol_m3"]
                    sigma_change = record["maximum_surface_conductivity_change_percent"]
                    phi_mV = record["axis_surface_potential_V"] * 1e3
                    print(f"  RESULT: surface Asc={surface_ascorbate:.5f} mM, phi={phi_mV:.4f} mV, maximum surface conductivity change={sigma_change:.3f}%.", flush=True)

    print("Auditing prescribed current with every electron assigned to ascorbate.", flush=True)
    faces = mesh_offset * np.expm1(np.linspace(0, np.log1p(16 / mesh_offset), audit_count + 1))
    stoichiometry, target = surface_reaction(families, 2)
    solution, _ = integrate_chemical_path(families, totals, case["reservoir_pH"], diffusivity, 2,
                                          path_relative_tolerance, path_absolute_tolerance)
    grid = CoupledTransportGrid(faces, families, solution, target, diffusivity, current_scale,
                                 reference_current, source_order, source_quadrature_count, maximum_wave_number)
    zeta = -2 * thermal_voltage
    mobility = -permittivity * zeta * thermal_voltage / (viscosity * diffusivity)
    initial_states = None
    for current_nA in audit_currents_nA:
        current = current_nA * 1e-9
        if any(r["current_label"] == "ascorbate_only_audit" and r["prescribed_current_A"] == current for r in records):
            continue
        current_fraction = current / reference_current
        record = {"current_label": "ascorbate_only_audit", "prescribed_current_A": current,
                  "ascorbate_fraction": 1.0, "zeta_V": zeta, "extent_in_radii": 16.0, "cells_per_direction": audit_count}
        print(f"  Asc-only fixed-current attempt: {current_nA:.4f} nA.", flush=True)
        try:
            states, solved_fraction, flow, history = grid.solve_coupled(
                mobility, newton_tolerance, flow_tolerance, maximum_newton_iterations, maximum_flow_iterations,
                initial_states=initial_states, initial_current_fraction=current_fraction, find_depletion_current=False,
            )
        except RuntimeError as error:
            # Nonconvergence is the explicit subject of this audit, not a substitute physical solution.
            if not str(error).startswith(("Transport Newton", "Stokes/transport coupling")):
                raise
            record.update({"status": "no_positive_converged_solution_found", "solver_error": str(error)})
            print(f"    No positive converged solution found: {error}", flush=True)
        else:
            diagnostics, _, _ = solution_diagnostics(grid, states, solved_fraction, flow, summary)
            record.update({**diagnostics, "status": "converged", "coupling_history": history})
            initial_states = states.copy()
        records.append(record)
        save_checkpoint(output_dir, inputs, records)
    print("Saved prescribed-current tests and the ascorbate-only audit.", flush=True)
