"""Run the spatial buffer/ion/Stokes calculation, retaining no-flow references."""

from copy import deepcopy
from pathlib import Path
import csv
import json
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from coupled_flow_transport import CoupledTransportGrid, chemistry_fields
from electroosmotic_flow import extrapolate_to_wall
from transport_check import integrate_chemical_path


def load_chemical_case(summary, case_name, relative_tolerance, absolute_tolerance):
    case = next(item for item in summary["cases"] if item["case"] == case_name)
    families = deepcopy(summary["families_unfitted"])
    families[3]["pkas"][1] = case["opa_pka2_assumed"]
    families[3]["diffusivities"] = [case["opa_diffusivity_m2_s"]] * 3
    if case["anionic_product_limit"]:
        families[-1]["species"] = ["product-"]
        families[-1]["charges"] = [-1]
    totals = np.array(list(case["reservoir_totals_mol_m3"].values()))
    solution, target = integrate_chemical_path(families, totals, case["reservoir_pH"],
                                              summary["reference_diffusivity_m2_s"], 2,
                                              relative_tolerance, absolute_tolerance)
    return case, families, solution, target


def save_flow_figure(path, grid, states, flow, diffusivity, radius, thermal_voltage, current_nA, zeta_mV):
    count = grid.count
    radial_um = grid.centers * radius * 1e6
    vertical_um = grid.centers * radius * 1e6
    radial_speed = flow["radial_velocity"] * diffusivity / radius * 1e6
    vertical_speed = flow["vertical_velocity"] * diffusivity / radius * 1e6
    speed = np.hypot(radial_speed, vertical_speed)
    concentrations, _ = chemistry_fields(states, grid.families)
    sigma = (concentrations * grid.charges**2 * grid.diffusivities).sum(axis=-1)
    reservoir_concentrations, _ = chemistry_fields(grid.solution.sol(0.0)[None, :], grid.families)
    sigma_reference = float((reservoir_concentrations * grid.charges**2 * grid.diffusivities).sum())
    sigma_change = 100 * (sigma / sigma_reference - 1)
    fields = (
        (states[:, grid.ascorbate_index].reshape(count, count), "Ascorbate (mM)", "viridis"),
        (sigma_change.reshape(count, count), "Conductivity change (%)", "coolwarm"),
        (states[:, -1].reshape(count, count) * thermal_voltage * 1e3, "Potential (mV)", "viridis"),
        (speed, "Liquid speed (um/s)", "magma"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 8.5), constrained_layout=True)
    plot_extent = 4 * radius * 1e6
    for axis, (values, title, color_map) in zip(axes.flat, fields):
        mesh = axis.pcolormesh(radial_um, vertical_um, values.T, shading="auto", cmap=color_map)
        axis.set_title(title)
        axis.set_xlabel("Radial distance (um)")
        axis.set_ylabel("Height above ITO (um)")
        axis.set_xlim(0, plot_extent)
        axis.set_ylim(0, plot_extent)
        axis.set_aspect("equal")
        figure.colorbar(mesh, ax=axis, shrink=0.85)
    stream = flow["streamfunction"]
    selected = grid.faces <= 4
    local_stream = stream[np.ix_(selected, selected)]
    if np.min(local_stream) < 0:
        levels = np.linspace(float(local_stream.min()), 0.0, 10)[1:-1]
        axes.flat[-1].contour(grid.faces[selected] * radius * 1e6, grid.faces[selected] * radius * 1e6,
                              local_stream.T, levels=levels, colors="white", linewidths=0.65)
    figure.suptitle(f"Coupled no-penetration Stokes flow: I={current_nA:.3f} nA, zeta={zeta_mV:.1f} mV")
    figure.savefig(path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    summary_path = Path("transport_check/output/summary.json")  # Chemistry and no-flow results used as reference data.
    output_dir = Path("transport_check/output/flow")  # Coupled spatial transport/Stokes outputs.
    calculation_cases = (  # Chemistry, source name/order, Hankel cutoff, and explicit domain/mesh sequence.
        ("unfitted", "gaussian", 1, 12.0, ((16.0, 40), (16.0, 64), (16.0, 96), (32.0, 112))),
        ("control_calibrated", "gaussian", 1, 12.0, ((16.0, 64),)),
        ("unfitted", "flat", 16, 100.0, ((16.0, 64),)),
    )
    mesh_offset = 0.5  # Geometric mesh refinement at the electrode and symmetry axis, in spot radii.
    quadrature_density = 0.8  # Hankel nodes per product of domain extent and maximum wave number.
    minimum_quadrature = 384  # Minimum Hankel nodes for accurate source geometry.
    zeta_thermal_multiples = (0.0, -1.0, -2.0)  # Constant zeta potentials divided by thermal voltage.
    permittivity = 78.3 * 8.8541878128e-12  # Water permittivity near 25 C, in F/m.
    viscosity = 0.89e-3  # Water dynamic viscosity near 25 C, in Pa s.
    newton_tolerance = 2e-10  # Scaled spatial residual and first-depletion constraint tolerance.
    flow_tolerance = 2e-8  # Relative convergence tolerance for conservative Stokes face fluxes.
    maximum_newton_iterations = 12  # Maximum Newton steps with the liquid velocity held fixed.
    maximum_flow_iterations = 10  # Maximum updates coupling Stokes flow and electrolyte transport.
    path_relative_tolerance = 1e-11  # Accuracy of the no-flow reference chemical path.
    path_absolute_tolerance = 1e-13  # Absolute reference path tolerance.
    save_fields = True  # Save full spatial fields for subsequent inspection.

    summary = json.loads(summary_path.read_text())
    radius = summary["radius_m"]
    diffusivity = summary["reference_diffusivity_m2_s"]
    thermal_voltage = summary["thermal_voltage_V"]
    faraday = 96485.33212
    current_scale = 2 * faraday * diffusivity * radius
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for case_name, profile_name, profile_order, maximum_wave_number, mesh_settings in calculation_cases:
        case, families, solution, target = load_chemical_case(summary, case_name, path_relative_tolerance, path_absolute_tolerance)
        reference_current = case["current_limits_A"][profile_name]
        for extent, count in mesh_settings:
            started = perf_counter()
            faces = mesh_offset * np.expm1(np.linspace(0, np.log1p(extent / mesh_offset), count + 1))
            quadrature_count = max(minimum_quadrature, int(quadrature_density * extent * maximum_wave_number))
            print(f"Building {case_name}/{profile_name}, extent={extent:g}a, mesh={count}x{count}.", flush=True)
            grid = CoupledTransportGrid(faces, families, solution, target, diffusivity, current_scale,
                                         reference_current, profile_order, quadrature_count, maximum_wave_number)
            no_flow_states = None
            no_flow_fraction = 0.9
            no_flow_current = None
            for zeta_multiple in zeta_thermal_multiples:
                zeta = zeta_multiple * thermal_voltage
                mobility = -permittivity * zeta * thermal_voltage / (viscosity * diffusivity)
                zeta_mV = zeta * 1e3
                print(f"  Solving zeta={zeta_mV:.3f} mV.", flush=True)
                states, fraction, flow, history = grid.solve_coupled(
                    mobility, newton_tolerance, flow_tolerance, maximum_newton_iterations, maximum_flow_iterations,
                    initial_states=no_flow_states, initial_current_fraction=no_flow_fraction,
                )
                current = fraction * reference_current
                if zeta_multiple == 0:
                    no_flow_states, no_flow_fraction, no_flow_current = states.copy(), fraction, current
                concentrations, _ = chemistry_fields(states, families)
                conductivity_values = faraday / thermal_voltage * diffusivity * (concentrations * grid.charges**2 * grid.diffusivities).sum(axis=-1)
                charge_error = float(np.max(np.abs(concentrations @ grid.charges)))
                speeds = np.hypot(flow["radial_velocity"], flow["vertical_velocity"]) * diffusivity / radius
                surface_states = np.column_stack([extrapolate_to_wall(states[:, index].reshape(count, count), grid.centers)
                                                  for index in range(grid.variable_count)])
                record = {
                    "chemical_case": case_name, "source_profile": profile_name, "extent_in_radii": extent,
                    "cells_per_direction": count, "zeta_V": zeta,
                    "limiting_current_A": current, "current_over_exact_no_flow_limit": fraction,
                    "current_over_same_mesh_no_flow": current / no_flow_current,
                    "maximum_cell_speed_m_s": float(speeds.max()),
                    "maximum_surface_slip_m_s": float(np.max(np.abs(flow["surface_slip"])) * diffusivity / radius),
                    "maximum_cell_peclet_ascorbate_on_spot_radius": float(speeds.max() * radius / families[4]["diffusivities"][0]),
                    "maximum_electroneutrality_error_mol_m3": charge_error,
                    "minimum_cell_ascorbate_mol_m3": float(states[:, grid.ascorbate_index].min()),
                    "minimum_reconstructed_surface_ascorbate_mol_m3": float(surface_states[:, grid.ascorbate_index].min()),
                    "minimum_conductivity_S_m": float(conductivity_values.min()),
                    "maximum_conductivity_S_m": float(conductivity_values.max()),
                    "relative_discrete_fluid_divergence": flow["relative_discrete_divergence"],
                    "maximum_wall_normal_liquid_flux": float(np.max(np.abs(flow["vertical_flux"][:, 0]))),
                    "coupling_history": history,
                }
                records.append(record)
                prefix = f"{case_name}_{profile_name}_R{extent:g}_N{count}_zeta{zeta_multiple:g}VT"
                if save_fields:
                    np.savez_compressed(output_dir / f"{prefix}.npz", radial_faces_m=faces * radius,
                                        states=states.reshape(count, count, -1),
                                        species_concentrations_mol_m3=concentrations.reshape(count, count, -1),
                                        conductivity_S_m=conductivity_values.reshape(count, count),
                                        radial_velocity_m_s=flow["radial_velocity"] * diffusivity / radius,
                                        vertical_velocity_m_s=flow["vertical_velocity"] * diffusivity / radius,
                                        streamfunction_m3_s=flow["streamfunction"] * diffusivity * radius,
                                        harmonic_surface_extension=flow["harmonic_extension"],
                                        current_A=current, zeta_V=zeta)
                    save_flow_figure(output_dir / f"{prefix}.png", grid, states, flow, diffusivity, radius,
                                     thermal_voltage, current * 1e9, zeta_mV)
                current_nA = current * 1e9
                enhancement_percent = 100 * (current / no_flow_current - 1)
                print(f"  Limit={current_nA:.6f} nA; same-mesh flow increase={enhancement_percent:.5f}%.", flush=True)
            elapsed = perf_counter() - started
            print(f"Completed mesh in {elapsed:.1f} s.", flush=True)
            (output_dir / "summary.json").write_text(json.dumps({"model": "Coupled equilibrium Nernst-Planck and incompressible Stokes flow with constant-zeta electroosmotic slip", "records": records}, indent=2) + "\n", encoding="utf-8")
    fields = ["chemical_case", "source_profile", "extent_in_radii", "cells_per_direction", "zeta_V",
              "limiting_current_A", "current_over_same_mesh_no_flow", "maximum_surface_slip_m_s",
              "maximum_cell_peclet_ascorbate_on_spot_radius", "maximum_electroneutrality_error_mol_m3"]
    with (output_dir / "case_summary.csv").open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print("Saved transport_check/output/flow/.", flush=True)
