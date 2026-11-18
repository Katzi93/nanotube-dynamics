"""Steady multicomponent electrolyte transport without liquid flow.

The electric potential, pH, and individual species concentrations vary together.
Fast acid-base equilibrium and bulk electroneutrality are enforced locally.
The no-flow half-space problem is reduced exactly to a numerical chemical path
parameterized by a harmonic transport coordinate. See README.md for the proof,
surface reaction, physical assumptions, and input provenance.
"""

from copy import deepcopy
from math import comb
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import j0, roots_legendre


# Physical water-ion properties used throughout the ideal 25 C model.
WATER_ION_DIFFUSIVITIES = {"H+": 9.31e-9, "OH-": 5.27e-9}


def acid_fractions(ph, pkas, charges):
    ph = np.asarray(ph)
    orders = np.arange(len(charges))
    cumulative_pkas = np.r_[0.0, np.cumsum(pkas)]
    logarithmic_weights = np.log(10) * (ph[..., None] * orders - cumulative_pkas)
    logarithmic_weights -= np.max(logarithmic_weights, axis=-1, keepdims=True)
    weights = np.exp(logarithmic_weights)
    fractions = weights / weights.sum(axis=-1, keepdims=True)
    mean_charge = np.sum(fractions * charges, axis=-1, keepdims=True)
    derivatives = np.log(10) * fractions * (mean_charge - charges)
    return fractions, derivatives


def species_properties(totals, ph, families, reference_diffusivity):
    """Return concentrations and their derivatives with respect to totals and pH.

    Totals and concentrations are in mol/m^3. The final two species are H+ and
    OH-. Derivative columns contain all component totals, then pH, then phi/V_T.
    """
    concentrations = []
    charges = []
    diffusivities = []
    labels = []
    membership = []
    derivative_rows = []
    family_count = len(families)
    for group, (total, family) in enumerate(zip(totals, families)):
        fractions, ph_derivatives = acid_fractions(ph, family["pkas"], np.array(family["charges"]))
        for state, (fraction, ph_derivative) in enumerate(zip(fractions, ph_derivatives)):
            derivative = np.zeros(family_count + 2)
            derivative[group] = fraction
            derivative[-2] = total * ph_derivative
            concentrations.append(total * fraction)
            charges.append(family["charges"][state])
            diffusivities.append(family["diffusivities"][state] / reference_diffusivity)
            labels.append(family["species"][state])
            membership.append(group)
            derivative_rows.append(derivative)
    for label, concentration, charge, diffusivity, ph_derivative in (
        ("H+", 10.0 ** (3 - ph), 1, WATER_ION_DIFFUSIVITIES["H+"], -np.log(10) * 10.0 ** (3 - ph)),
        ("OH-", 10.0 ** (ph - 11), -1, WATER_ION_DIFFUSIVITIES["OH-"], np.log(10) * 10.0 ** (ph - 11)),
    ):
        derivative = np.zeros(family_count + 2)
        derivative[-2] = ph_derivative
        concentrations.append(concentration)
        charges.append(charge)
        diffusivities.append(diffusivity / reference_diffusivity)
        labels.append(label)
        membership.append(-1)
        derivative_rows.append(derivative)
    return {
        "concentrations": np.array(concentrations), "charges": np.array(charges),
        "diffusivities": np.array(diffusivities), "labels": labels,
        "membership": np.array(membership), "derivatives": np.array(derivative_rows),
    }


def solution_charge(ph, totals, families):
    charge = 10.0 ** (3 - ph) - 10.0 ** (ph - 11)
    for total, family in zip(totals, families):
        fractions, _ = acid_fractions(ph, family["pkas"], np.array(family["charges"]))
        charge += total * np.dot(fractions, family["charges"])
    return float(charge)


def titrant_tris(total, acid_family, stock_ph, tris_family):
    acid_fractions_at_ph, _ = acid_fractions(stock_ph, acid_family["pkas"], np.array(acid_family["charges"]))
    tris_fractions_at_ph, _ = acid_fractions(stock_ph, tris_family["pkas"], np.array(tris_family["charges"]))
    acid_charge = total * np.dot(acid_fractions_at_ph, acid_family["charges"])
    water_charge = 10.0 ** (3 - stock_ph) - 10.0 ** (stock_ph - 11)
    return float(-(acid_charge + water_charge) / tris_fractions_at_ph[0])


def prepare_reservoir(families, recipe, include_ascorbate=True):
    """Reconstruct stock titrations under the stated ideal-solution assumptions."""
    tris, borate, _, opa, ascorbate, _ = families
    opa_stock_tris = titrant_tris(recipe["opa_stock"], opa, recipe["opa_stock_ph"], tris)
    ascorbate_stock_tris = titrant_tris(recipe["ascorbate_stock"], ascorbate, recipe["ascorbate_stock_ph"], tris)
    tris_fractions, _ = acid_fractions(recipe["tris_stock_ph"], tris["pkas"], np.array(tris["charges"]))
    borate_fractions, _ = acid_fractions(recipe["tris_stock_ph"], borate["pkas"], np.array(borate["charges"]))
    water_charge = 10.0 ** (3 - recipe["tris_stock_ph"]) - 10.0 ** (recipe["tris_stock_ph"] - 11)
    borate_stock = (recipe["tris_stock"] * tris_fractions[0] + water_charge) / borate_fractions[1]
    total_tris = recipe["tris_fraction"] * recipe["tris_stock"] + recipe["opa_fraction"] * opa_stock_tris
    if include_ascorbate:
        total_tris += recipe["ascorbate_fraction"] * ascorbate_stock_tris
    totals = np.array([
        total_tris, recipe["tris_fraction"] * borate_stock, recipe["spermidine_final"],
        recipe["opa_fraction"] * recipe["opa_stock"],
        recipe["ascorbate_fraction"] * recipe["ascorbate_stock"] if include_ascorbate else 0.0,
        0.0,
    ])
    ph = brentq(solution_charge, 2.0, 13.0, args=(totals, families), xtol=1e-12)
    return totals, ph, {"opa_stock_tris_mol_m3": opa_stock_tris,
                       "ascorbate_stock_tris_mol_m3": ascorbate_stock_tris,
                       "borate_stock_mol_m3": float(borate_stock)}


def conductivity(totals, ph, families, reference_diffusivity, faraday, thermal_voltage):
    properties = species_properties(totals, ph, families, reference_diffusivity)
    contributions = faraday / thermal_voltage * reference_diffusivity * properties["diffusivities"] * properties["charges"] ** 2 * properties["concentrations"]
    return float(contributions.sum()), dict(zip(properties["labels"], contributions.tolist()))


def transport_matrix(state, families, reference_diffusivity):
    """Map local state derivatives to component fluxes, charge flux, and EN."""
    properties = species_properties(state[:-2], state[-2], families, reference_diffusivity)
    concentrations = properties["concentrations"]
    charges = properties["charges"]
    diffusion = properties["diffusivities"]
    species_transport = diffusion[:, None] * properties["derivatives"]
    species_transport[:, -1] = diffusion * charges * concentrations
    component_rows = np.array([species_transport[properties["membership"] == group].sum(axis=0) for group in range(len(families))])
    charge_row = (charges[:, None] * species_transport).sum(axis=0)
    electroneutrality_row = (charges[:, None] * properties["derivatives"]).sum(axis=0)
    matrix = np.vstack((component_rows, charge_row, electroneutrality_row))
    return matrix, properties, species_transport


def surface_reaction(families, electrons, ascorbate_fraction=1.0):
    """Project the explicit surface reaction onto conserved chemical quantities.

    Fast acid-base reactions redistribute protons within an unresolved reaction
    layer. Component fluxes and charge flux are invariant under that exchange.
    Positive stoichiometry denotes release into the electrolyte. Optionally,
    assign the remaining electron current to buffered water oxidation. Neutral
    water and oxygen are included in the reaction bookkeeping but not solved
    as transport components; they do not enter the ideal electrical chemistry.
    """
    if not 0 <= ascorbate_fraction <= 1:
        raise ValueError("The ascorbate current fraction must lie between zero and one.")
    if electrons != 2:
        raise ValueError("This reaction bookkeeping uses two electron equivalents.")
    anionic_product = families[-1]["charges"][0] == -1
    ascorbate_protons = 2 if anionic_product else 1
    proton_release = ascorbate_fraction * ascorbate_protons + 2 * (1 - ascorbate_fraction)
    product_label = families[-1]["species"][0]
    stoichiometry = {"AscH-": -ascorbate_fraction, "Tris": -float(proton_release),
                     "TrisH+": float(proton_release), product_label: ascorbate_fraction}
    if ascorbate_fraction < 1:
        stoichiometry.update({"H2O": -(1 - ascorbate_fraction), "O2": (1 - ascorbate_fraction) / 2})
    target = np.zeros(len(families) + 2)
    charge_release = 0.0
    for group, family in enumerate(families):
        for label, charge in zip(family["species"], family["charges"]):
            coefficient = stoichiometry.get(label, 0.0)
            target[group] += coefficient
            charge_release += charge * coefficient
    if not np.isclose(charge_release, electrons, rtol=0.0, atol=1e-14):
        raise ValueError("Surface-reaction stoichiometry does not match the imposed electron count.")
    target[-2] = charge_release
    return stoichiometry, target


def integrate_chemical_path(families, totals, ph, reference_diffusivity, electrons, relative_tolerance, absolute_tolerance,
                            target=None, maximum_coordinate=None, require_depletion=True):
    """Integrate the no-flow chemical path, stopping at ascorbate depletion.

    The default requires depletion. A specified finite coordinate interval can
    instead supply a positive far-field reference for prescribed-current runs,
    including mixtures that do not consume any ascorbate.
    """
    ascorbate_index = next(index for index, family in enumerate(families) if family["name"] == "ascorbate")
    if target is None:
        _, target = surface_reaction(families, electrons)

    def right_hand_side(coordinate, state):
        matrix, _, _ = transport_matrix(state, families, reference_diffusivity)
        return np.linalg.solve(matrix, target)

    def ascorbate_depletion(coordinate, state):
        return state[ascorbate_index]

    ascorbate_depletion.terminal = True
    ascorbate_depletion.direction = -1
    initial_state = np.r_[totals, ph, 0.0]
    reference_length_in_concentration_units = (10 * totals[ascorbate_index]
                                             if maximum_coordinate is None else maximum_coordinate)
    solution = solve_ivp(right_hand_side, (0.0, reference_length_in_concentration_units), initial_state,
                         method="DOP853", rtol=relative_tolerance, atol=absolute_tolerance,
                         events=ascorbate_depletion, dense_output=True)
    if not solution.success:
        raise RuntimeError(solution.message)
    if require_depletion and len(solution.t_events[0]) != 1:
        raise RuntimeError("Ascorbate depletion was not reached within the requested transport coordinate interval.")
    return solution, target


def profile_parameters(order):
    components = np.arange(1, order + 1)
    coefficients = np.array([(-1) ** (index + 1) * comb(order, index) for index in components])
    width = 1 / np.sqrt(-np.log(1 - 0.5 ** (1 / order)))
    weights = coefficients / components / np.sum(1 / components)
    center_coefficient = np.sqrt(np.pi) / width * np.dot(weights, np.sqrt(components)) / (2 * np.pi)
    return components, weights, width, float(center_coefficient)


def half_space_source(radial_positions, vertical_positions, radius, order, quadrature_count, maximum_wave_number):
    """Harmonic source H with -dH/dz=p(r) and integral(p dA)=1."""
    components, weights, width, _ = profile_parameters(order)
    nodes, quadrature_weights = roots_legendre(quadrature_count)
    wave_numbers = (nodes + 1) * maximum_wave_number / 2
    spectrum = np.sum(weights[:, None] * np.exp(-(width * wave_numbers[None, :]) ** 2 / (4 * components[:, None])), axis=0)
    spectral_weights = quadrature_weights * maximum_wave_number / 2 * spectrum
    radial_kernel = j0(np.outer(radial_positions / radius, wave_numbers))
    vertical_kernel = np.exp(-np.outer(wave_numbers, vertical_positions / radius))
    return (radial_kernel * spectral_weights) @ vertical_kernel / (2 * np.pi * radius)


def load_current_measurements(data_dir, voltage, tolerance, conversion):
    currents = {}
    for label, filename in (("illuminated", "on900s.csv"), ("dark", "off900s.csv")):
        with (data_dir / filename).open(newline="") as input_file:
            rows = list(csv.DictReader(input_file))
        voltages = np.array([float(row["Mean"]) for row in rows if row["Channel"] == "CH1"])
        current_values = np.array([float(row["Mean"]) for row in rows if row["Channel"] == "CH2"]) * conversion
        selected = np.abs(voltages - voltage) < tolerance
        currents[label] = float(current_values[selected].mean())
    currents["light_increment"] = currents["illuminated"] - currents["dark"]
    return currents


def path_diagnostics(solution, target, families, reference_diffusivity, sample_count):
    coordinates = np.linspace(0, float(solution.t_events[0][0]), sample_count)
    states = solution.sol(coordinates).T
    charges = []
    errors = []
    diffusion_current_fractions = []
    for state in states:
        matrix, properties, _ = transport_matrix(state, families, reference_diffusivity)
        derivative = np.linalg.solve(matrix, target)
        errors.append(float(np.max(np.abs(matrix @ derivative - target))))
        charges.append(float(np.dot(properties["charges"], properties["concentrations"])))
        conductivity_coefficient = float(np.dot(properties["charges"] ** 2 * properties["diffusivities"], properties["concentrations"]))
        diffusion_current_fractions.append(float(1 - conductivity_coefficient * derivative[-1] / target[-2]))
    return {"maximum_local_transport_balance_error": max(errors),
            "maximum_electroneutrality_error_mol_m3": float(np.max(np.abs(charges))),
            "minimum_diffusion_current_fraction": min(diffusion_current_fractions),
            "maximum_diffusion_current_fraction": max(diffusion_current_fractions)}


def save_spatial_solution(output_dir, solution, families, reference_diffusivity, thermal_voltage, faraday,
                          radius, electrons, case_name, profile_order, current_fraction, radial_count,
                          vertical_count, extent_in_radii, quadrature_count, maximum_wave_number):
    """Map the coupled solution to an axisymmetric half-space region."""
    radial_positions = np.linspace(0, extent_in_radii * radius, radial_count)
    vertical_positions = np.linspace(0, extent_in_radii * radius, vertical_count)
    center_coefficient = profile_parameters(profile_order)[-1]
    coordinate_limit = float(solution.t_events[0][0])
    critical_current = electrons * faraday * reference_diffusivity * radius * coordinate_limit / center_coefficient
    current = current_fraction * critical_current
    source = half_space_source(radial_positions, vertical_positions, radius, profile_order,
                               quadrature_count, maximum_wave_number)
    coordinate = current * source / (electrons * faraday * reference_diffusivity)
    states = solution.sol(coordinate.ravel()).T.reshape(radial_count, vertical_count, -1)
    family_names = [family["name"] for family in families]
    ascorbate_index = family_names.index("ascorbate")
    tris_index = family_names.index("tris")
    tris_fraction, _ = acid_fractions(states[..., -2], families[tris_index]["pkas"], np.array(families[tris_index]["charges"]))
    tris_h = states[..., tris_index] * tris_fraction[..., 0]
    sigma = np.zeros(coordinate.shape)
    for group, family in enumerate(families):
        fractions, _ = acid_fractions(states[..., -2], family["pkas"], np.array(family["charges"]))
        weighted_diffusion = np.array(family["diffusivities"]) * np.array(family["charges"]) ** 2
        sigma += faraday / thermal_voltage * states[..., group] * np.sum(fractions * weighted_diffusion, axis=-1)
    sigma += faraday / thermal_voltage * (WATER_ION_DIFFUSIVITIES["H+"] * 10.0 ** (3 - states[..., -2])
                                           + WATER_ION_DIFFUSIVITIES["OH-"] * 10.0 ** (states[..., -2] - 11))
    reservoir_state = solution.sol(0.0)
    reservoir_sigma = conductivity(reservoir_state[:-2], reservoir_state[-2], families,
                                   reference_diffusivity, faraday, thermal_voltage)[0]
    conductivity_change = 100 * (sigma / reservoir_sigma - 1)
    all_species_concentrations = []
    all_species_labels = []
    for group, family in enumerate(families):
        fractions, _ = acid_fractions(states[..., -2], family["pkas"], np.array(family["charges"]))
        all_species_concentrations.extend(np.moveaxis(states[..., group, None] * fractions, -1, 0))
        all_species_labels.extend(family["species"])
    all_species_concentrations.extend((10.0 ** (3 - states[..., -2]), 10.0 ** (states[..., -2] - 11)))
    all_species_labels.extend(("H+", "OH-"))
    np.savez_compressed(output_dir / f"{case_name}_fields.npz", radial_m=radial_positions, vertical_m=vertical_positions,
                        transport_coordinate_mol_m3=coordinate, states=states, total_ascorbate_mol_m3=states[..., ascorbate_index],
                        tris_h_mol_m3=tris_h, conductivity_S_m=sigma, potential_V=states[..., -1] * thermal_voltage,
                        current_A=current, current_fraction_of_depletion=current_fraction,
                        species_labels=np.array(all_species_labels),
                        species_concentrations_mol_m3=np.stack(all_species_concentrations, axis=-1),
                        state_labels=np.array(family_names + ["pH", "phi_over_thermal_voltage"]))
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 7.5), constrained_layout=True)
    panels = (
        (states[..., ascorbate_index], "Total ascorbate (mM)", "viridis"),
        (tris_h, "TrisH+ (mM)", "viridis"),
        (conductivity_change, "Conductivity change (%)", "coolwarm"),
        (states[..., -1] * thermal_voltage * 1e3, "Electrolyte potential (mV)", "viridis"),
        (states[..., -2], "pH", "viridis"),
        (states[..., family_names.index("opa")], "Total dissolved OPA (mM)", "viridis"),
    )
    radial_um = radial_positions * 1e6
    vertical_um = vertical_positions * 1e6
    for axis, (values, title, color_map) in zip(axes.flat, panels):
        mesh = axis.pcolormesh(radial_um, vertical_um, values.T, shading="auto", cmap=color_map)
        axis.set_title(title)
        axis.set_xlabel("Radial distance (um)")
        axis.set_ylabel("Height above ITO (um)")
        axis.set_aspect("equal")
        figure.colorbar(mesh, ax=axis, shrink=0.85)
    current_nA = current * 1e9
    figure.suptitle(f"No flow: coupled buffer transport, {case_name}, I = {current_nA:.2f} nA")
    figure.savefig(output_dir / f"{case_name}_fields.png", dpi=180)
    plt.close(figure)


def save_path_figure(output_dir, summaries, solutions, reference_diffusivity, faraday, thermal_voltage, sample_count):
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    fractions = np.linspace(0, 1, sample_count)
    for case_name in ("unfitted", "control_calibrated", "product_anion_limit"):
        solution, _, local_families = solutions[case_name]
        coordinate_limit = float(solution.t_events[0][0])
        states = solution.sol(fractions * coordinate_limit).T
        summary = next(item for item in summaries if item["case"] == case_name)
        sigma = np.array([conductivity(state[:-2], state[-2], local_families, reference_diffusivity,
                                      faraday, thermal_voltage)[0] for state in states])
        sigma_change = 100 * (sigma / summary["reservoir_conductivity_S_m"] - 1)
        ph_change = states[:, -2] - summary["reservoir_pH"]
        axes[0].plot(fractions, sigma_change, label=case_name)
        axes[1].plot(fractions, ph_change)
        axes[2].plot(fractions, states[:, -1] * thermal_voltage * 1e3)
    for axis, ylabel in zip(axes, ("Conductivity change (%)", "pH change", "Electrolyte potential (mV)")):
        axis.set_xlabel("Local transport coordinate / depletion value")
        axis.set_ylabel(ylabel)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(output_dir / "chemical_response.png", dpi=180)
    plt.close(figure)


def save_conductivity_breakdown(output_dir, summary, families):
    names = []
    changes = []
    for family in families[:-1]:
        names.append(family["name"])
        labels = family["species"]
        bulk = sum(summary["reservoir_conductivity_contributions_S_m"][label] for label in labels)
        surface = sum(summary["surface_at_depletion_conductivity_contributions_S_m"][label] for label in labels)
        changes.append((surface - bulk) * 1e4)
    names.append("total, including water ions")
    changes.append((summary["surface_at_depletion_conductivity_S_m"] - summary["reservoir_conductivity_S_m"]) * 1e4)
    colors = ["#2166ac" if change >= 0 else "#b2182b" for change in changes]
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.barh(names, changes, color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Conductivity change at first ascorbate depletion (uS/cm)")
    axis.set_title("Species redistribution: unfitted baseline, no flow")
    axis.spines[["top", "right"]].set_visible(False)
    axis.invert_yaxis()
    figure.savefig(output_dir / "conductivity_breakdown.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    # Physical inputs and explicit modelling assumptions.
    output_dir = Path("transport_check/output")  # Generated numerical results and scientific figures.
    measurement_dir = Path("ito_light_response/data/ito_characterization_1/voltage_sweep")  # Read-only current records.
    applied_voltage = 2.0  # Select the current measurements near this imposed electrode voltage, in volts.
    voltage_tolerance = 0.05  # Keep points with |V - applied_voltage| below this value, in volts.
    current_conversion = 1e-6  # CH2 V to A using the existing 1 kOhm shunt and 1000x amplifier.
    radius = 52.541e-6  # Half-maximum illumination radius, in metres.
    reference_diffusivity = 1e-9  # Reference diffusivity, in m^2/s.
    thermal_voltage = 0.02569258  # RT/F near 25 C, in volts.
    faraday = 96485.33212  # Faraday constant, in C/mol.
    electrons = 2  # Electrons released per assumed ascorbate oxidation.
    measured_conductivity = 0.0435  # Conductometer reading with ascorbate stock, in S/m.
    measured_control_conductivity = 0.0390  # Conductometer reading without ascorbate stock, in S/m.
    relative_tolerance = 1e-10  # Relative tolerance for the coupled chemical-path integration.
    absolute_tolerance = 1e-12  # Absolute tolerance for totals, pH, and phi/V_T.
    diagnostic_samples = 101  # Number of states used to check transport and electroneutrality balances.
    source_profiles = (("gaussian", 1), ("flat", 16))  # Smooth prescribed current profiles with the same half-maximum radius.
    sample_fractions = (0.0, 0.25, 0.5, 0.75, 0.95, 1.0)  # Fractions of the transport coordinate at first ascorbate depletion.
    spatial_current_fraction = 0.95  # Map a physical state below the first ascorbate-depletion threshold.
    spatial_grid = (161, 161)  # Radial and vertical samples for exported fields.
    spatial_extent = 4.0  # Plot radius and height, in measured spot radii.
    spatial_quadrature_count = 1024  # Hankel integration nodes for the Gaussian source maps.
    spatial_maximum_wave_number = 12.0  # Upper dimensionless Hankel wave number for the Gaussian source.
    path_plot_samples = 151  # Samples along the coupled chemical path for summary figures.
    recipe = {
        "opa_stock": 5.0, "opa_stock_ph": 8.0, "opa_fraction": 0.5,  # OPA stock in mol/m^3 and its final volume fraction.
        "ascorbate_stock": 100.0, "ascorbate_stock_ph": 9.0, "ascorbate_fraction": 0.005,  # Nominal ascorbate stock and titration pH.
        "tris_stock": 1000.0, "tris_stock_ph": 9.3, "tris_fraction": 0.015,  # Tris stock titrated with boric acid.
        "spermidine_final": 0.75,  # Free-base spermidine in the final mixture, in mol/m^3.
    }
    families = [
        {"name": "tris", "species": ["TrisH+", "Tris"], "pkas": [8.06], "charges": [1, 0], "diffusivities": [0.757931084e-9] * 2},
        {"name": "borate", "species": ["B(OH)3", "B(OH)4-"], "pkas": [9.237], "charges": [0, -1], "diffusivities": [1e-9] * 2},
        {"name": "spermidine", "species": ["SpdH3+3", "SpdH2+2", "SpdH+", "Spd"], "pkas": [8.25, 9.71, 10.90], "charges": [3, 2, 1, 0], "diffusivities": [0.49e-9] * 4},
        {"name": "opa", "species": ["OPA", "OPA-", "OPA-2"], "pkas": [2.6, 8.0], "charges": [0, -1, -2], "diffusivities": [0.5e-9] * 3},
        {"name": "ascorbate", "species": ["H2Asc", "AscH-", "Asc-2"], "pkas": [4.17, 11.57], "charges": [0, -1, -2], "diffusivities": [0.66e-9] * 3},
        {"name": "product", "species": ["DHA"], "pkas": [], "charges": [0], "diffusivities": [0.66e-9]},
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    current_measurements = load_current_measurements(measurement_dir, applied_voltage, voltage_tolerance, current_conversion)
    control_totals, control_ph, _ = prepare_reservoir(families, recipe, include_ascorbate=False)
    control_sigma, control_contributions = conductivity(control_totals, control_ph, families, reference_diffusivity, faraday, thermal_voltage)
    fitted_opa_diffusivity = families[3]["diffusivities"][0] * (measured_control_conductivity - control_sigma + control_contributions["OPA-"] + control_contributions["OPA-2"]) / (control_contributions["OPA-"] + control_contributions["OPA-2"])
    case_parameters = (
        ("unfitted", 8.0, 0.5e-9, False),
        ("control_calibrated", 8.0, fitted_opa_diffusivity, False),
        ("opa_pka_7p5", 7.5, 0.5e-9, False),
        ("opa_pka_8p5", 8.5, 0.5e-9, False),
        ("opa_diffusion_low", 8.0, 0.35e-9, False),
        ("opa_diffusion_high", 8.0, 0.8e-9, False),
        ("product_anion_limit", 8.0, 0.5e-9, True),
    )
    summaries = []
    solutions = {}
    for case_name, opa_pka, opa_diffusivity, anionic_product in case_parameters:
        local_families = deepcopy(families)
        local_families[3]["pkas"][1] = opa_pka
        local_families[3]["diffusivities"] = [opa_diffusivity] * 3
        if anionic_product:
            local_families[-1]["species"] = ["product-"]
            local_families[-1]["charges"] = [-1]
        totals, reservoir_ph, stock_details = prepare_reservoir(local_families, recipe)
        reservoir_sigma, reservoir_contributions = conductivity(totals, reservoir_ph, local_families, reference_diffusivity, faraday, thermal_voltage)
        print(f"Solving {case_name}: reservoir pH={reservoir_ph:.5f}.", flush=True)
        solution, target = integrate_chemical_path(local_families, totals, reservoir_ph, reference_diffusivity,
                                                 electrons, relative_tolerance, absolute_tolerance)
        stoichiometry, _ = surface_reaction(local_families, electrons)
        coordinate_limit = float(solution.t_events[0][0])
        surface_state = solution.sol(coordinate_limit)
        surface_sigma, surface_contributions = conductivity(surface_state[:-2], surface_state[-2], local_families,
                                                           reference_diffusivity, faraday, thermal_voltage)
        diagnostics = path_diagnostics(solution, target, local_families, reference_diffusivity, diagnostic_samples)
        current_limits = {name: electrons * faraday * reference_diffusivity * radius * coordinate_limit / profile_parameters(order)[-1]
                          for name, order in source_profiles}
        current_limits["absorbing_disk"] = 4 * electrons * faraday * reference_diffusivity * radius * coordinate_limit
        current_limits["uniform_flux_disk"] = np.pi * electrons * faraday * reference_diffusivity * radius * coordinate_limit
        sampled_coordinates = coordinate_limit * np.array(sample_fractions)
        sampled_states = solution.sol(sampled_coordinates).T
        sigma_values = [conductivity(state[:-2], state[-2], local_families, reference_diffusivity, faraday, thermal_voltage)[0] for state in sampled_states]
        bulk_properties = species_properties(totals, reservoir_ph, local_families, reference_diffusivity)
        surface_properties = species_properties(surface_state[:-2], surface_state[-2], local_families, reference_diffusivity)
        summary = {
            "case": case_name, "opa_pka2_assumed": opa_pka, "opa_diffusivity_m2_s": opa_diffusivity,
            "anionic_product_limit": anionic_product, "stock_reconstruction": stock_details,
            "surface_reaction_stoichiometry": stoichiometry,
            "reservoir_totals_mol_m3": dict(zip([family["name"] for family in local_families], totals.tolist())),
            "reservoir_pH": reservoir_ph, "reservoir_conductivity_S_m": reservoir_sigma,
            "reservoir_species_mol_m3": dict(zip(bulk_properties["labels"], bulk_properties["concentrations"].tolist())),
            "reservoir_conductivity_contributions_S_m": reservoir_contributions,
            "depletion_coordinate_mol_m3": coordinate_limit, "current_limits_A": current_limits,
            "surface_at_depletion_pH": float(surface_state[-2]), "surface_at_depletion_potential_V": float(surface_state[-1] * thermal_voltage),
            "surface_at_depletion_conductivity_S_m": surface_sigma,
            "surface_at_depletion_species_mol_m3": dict(zip(surface_properties["labels"], surface_properties["concentrations"].tolist())),
            "surface_at_depletion_conductivity_contributions_S_m": surface_contributions,
            "sample_coordinate_fractions": list(sample_fractions), "sample_states": sampled_states.tolist(),
            "sample_conductivities_S_m": sigma_values, "diagnostics": diagnostics,
            "measured_current_feasibility": {
                profile: {measurement: {"current_over_first_depletion_limit": measured / limit_current,
                                         "nonnegative_steady_solution": bool(measured <= limit_current)}
                          for measurement, measured in current_measurements.items() if measurement != "dark"}
                for profile, limit_current in current_limits.items()
            },
        }
        summaries.append(summary)
        solutions[case_name] = (solution, target, local_families)
        disk_limit_nA = current_limits["absorbing_disk"] * 1e9
        conductivity_change_percent = (surface_sigma / reservoir_sigma - 1) * 100
        surface_potential_mV = surface_state[-1] * thermal_voltage * 1e3
        print(f"  Disk limit={disk_limit_nA:.4f} nA, surface phi={surface_potential_mV:.4f} mV, conductivity change={conductivity_change_percent:+.4f}%.", flush=True)
    output = {
        "model": "Steady electroneutral Nernst-Planck transport with local acid-base equilibrium; no liquid flow. All species and the electric potential vary.",
        "radius_m": radius, "reference_diffusivity_m2_s": reference_diffusivity, "thermal_voltage_V": thermal_voltage,
        "measured_currents_A": current_measurements, "measured_conductivity_S_m": measured_conductivity,
        "measured_control_conductivity_S_m": measured_control_conductivity,
        "control_calibration_explanation": "Only the case named control_calibrated adjusts the effective dissolved-OPA diffusivity to the no-ascorbate conductivity. It is not a measured OPA mobility. Other cases do not fit conductivity.",
        "recipe": recipe, "families_unfitted": families, "cases": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for case_name in ("unfitted", "control_calibrated"):
        solution, _, local_families = solutions[case_name]
        save_spatial_solution(output_dir, solution, local_families, reference_diffusivity, thermal_voltage,
                              faraday, radius, electrons, case_name, source_profiles[0][1], spatial_current_fraction,
                              *spatial_grid, spatial_extent, spatial_quadrature_count, spatial_maximum_wave_number)
    save_path_figure(output_dir, summaries, solutions, reference_diffusivity, faraday, thermal_voltage, path_plot_samples)
    save_conductivity_breakdown(output_dir, summaries[0], families)
    with (output_dir / "case_summary.csv").open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["case", "bulk_pH", "bulk_conductivity_uS_cm", "depletion_conductivity_change_percent",
                         "depletion_pH", "depletion_phi_mV", "gaussian_limit_nA", "flat_limit_nA",
                         "absorbing_disk_limit_nA", "uniform_flux_disk_limit_nA", "diffusion_current_fraction_min", "diffusion_current_fraction_max"])
        for item in summaries:
            change_percent = 100 * (item["surface_at_depletion_conductivity_S_m"] / item["reservoir_conductivity_S_m"] - 1)
            writer.writerow([item["case"], item["reservoir_pH"], item["reservoir_conductivity_S_m"] * 1e4, change_percent,
                             item["surface_at_depletion_pH"], item["surface_at_depletion_potential_V"] * 1e3,
                             item["current_limits_A"]["gaussian"] * 1e9, item["current_limits_A"]["flat"] * 1e9,
                             item["current_limits_A"]["absorbing_disk"] * 1e9, item["current_limits_A"]["uniform_flux_disk"] * 1e9,
                             item["diagnostics"]["minimum_diffusion_current_fraction"], item["diagnostics"]["maximum_diffusion_current_fraction"]])
    print("Saved transport_check/output/summary.json.", flush=True)
