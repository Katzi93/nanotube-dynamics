"""Spatial finite-volume equilibrium Nernst-Planck transport coupled to Stokes flow.

The unknowns in each axisymmetric cell are six component totals, pH, and phi/V_T.
The current can be prescribed. An optional diagnostic instead solves for the
current at first ascorbate depletion. Stokes flow is updated from the computed
surface potential until the coupled system converges.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from electroosmotic_flow import HalfSpaceSlipFlow, centered_radial_values_at_faces, extrapolate_to_wall
from transport_check import (
    WATER_ION_DIFFUSIVITIES, acid_fractions, half_space_source, profile_parameters,
    species_properties, transport_matrix,
)


def bernoulli_and_derivative(argument):
    values = np.empty_like(argument)
    derivatives = np.empty_like(argument)
    small = np.abs(argument) < 1e-4
    small_argument = argument[small]
    values[small] = 1 - small_argument / 2 + small_argument**2 / 12 - small_argument**4 / 720
    derivatives[small] = -0.5 + small_argument / 6 - small_argument**3 / 180
    exponential_difference = np.expm1(argument[~small])
    values[~small] = argument[~small] / exponential_difference
    derivatives[~small] = (exponential_difference - argument[~small] * np.exp(argument[~small])) / exponential_difference**2
    return values, derivatives


def chemistry_fields(states, families):
    cell_count, variable_count = states.shape
    species_count = sum(len(family["charges"]) for family in families) + 2
    concentrations = np.empty((cell_count, species_count))
    derivatives = np.zeros((cell_count, species_count, variable_count))
    start = 0
    for group, family in enumerate(families):
        count = len(family["charges"])
        fractions, ph_derivatives = acid_fractions(states[:, -2], family["pkas"], np.array(family["charges"]))
        concentrations[:, start:start + count] = states[:, group, None] * fractions
        derivatives[:, start:start + count, group] = fractions
        derivatives[:, start:start + count, -2] = states[:, group, None] * ph_derivatives
        start += count
    concentrations[:, -2] = 10.0 ** (3 - states[:, -2])
    concentrations[:, -1] = 10.0 ** (states[:, -2] - 11)
    derivatives[:, -2, -2] = -np.log(10) * concentrations[:, -2]
    derivatives[:, -1, -2] = np.log(10) * concentrations[:, -1]
    return concentrations, derivatives


def transport_face_flux(left, right, left_chemistry, right_chemistry, conductance, volume_flux,
                        charges, diffusivities, projection, component_count):
    left_c, left_derivative = left_chemistry
    right_c, right_derivative = right_chemistry
    argument = (right[:, -1] - left[:, -1])[:, None] * charges
    positive_b, positive_derivative = bernoulli_and_derivative(argument)
    negative_b, negative_derivative = bernoulli_and_derivative(-argument)
    coefficient = conductance[:, None] * diffusivities
    species_flux = coefficient * (positive_b * left_c - negative_b * right_c)
    left_species_derivative = (coefficient * positive_b)[:, :, None] * left_derivative
    right_species_derivative = -(coefficient * negative_b)[:, :, None] * right_derivative
    potential_derivative = coefficient * charges * (positive_derivative * left_c + negative_derivative * right_c)
    left_species_derivative[:, :, -1] -= potential_derivative
    right_species_derivative[:, :, -1] += potential_derivative
    flux = species_flux @ projection.T
    left_jacobian = np.einsum("is,fsu->fiu", projection, left_species_derivative)
    right_jacobian = np.einsum("is,fsu->fiu", projection, right_species_derivative)
    flux[:, :component_count] += volume_flux[:, None] * (left[:, :component_count] + right[:, :component_count]) / 2
    components = np.arange(component_count)
    left_jacobian[:, components, components] += volume_flux[:, None] / 2
    right_jacobian[:, components, components] += volume_flux[:, None] / 2
    return flux, left_jacobian, right_jacobian


class CoupledTransportGrid:
    """Fixed mesh, source profile, chemistry, and sparse Newton geometry."""

    def __init__(self, faces, families, solution, target, reference_diffusivity, current_scale,
                 reference_current, source_order, quadrature_count, maximum_wave_number):
        self.faces = faces
        self.centers = (faces[:-1] + faces[1:]) / 2
        self.widths = np.diff(faces)
        self.annular_areas = np.diff(faces**2) / 2
        self.count = self.centers.size
        self.cell_count = self.count**2
        self.families = families
        self.component_count = len(families)
        self.variable_count = self.component_count + 2
        self.solution = solution
        self.target = np.r_[target[:self.component_count], target[-2]]
        self.reference_diffusivity = reference_diffusivity
        self.reference_current = reference_current
        self.coordinate_coefficient = reference_current / current_scale
        self.equation_indices = np.r_[np.arange(self.component_count), self.variable_count - 1]
        reservoir = solution.sol(0.0)
        properties = species_properties(reservoir[:-2], reservoir[-2], families, reference_diffusivity)
        self.charges = properties["charges"]
        self.diffusivities = properties["diffusivities"]
        self.projection = np.vstack([(properties["membership"] == group).astype(float) for group in range(self.component_count)]
                                    + [self.charges.astype(float)])
        ascorbate = next(index for index, family in enumerate(families) if family["name"] == "ascorbate")
        self.variable_scale = np.r_[np.maximum(reservoir[:-2], reservoir[ascorbate]), 1.0, 1.0]
        conductivity_coefficient = np.dot(self.charges**2 * self.diffusivities, properties["concentrations"])
        charge_coefficient = np.dot(np.abs(self.charges), properties["concentrations"])
        self.indices = np.arange(self.cell_count).reshape(self.count, self.count)
        self.source_centers = half_space_source(self.centers, self.centers, 1.0, source_order, quadrature_count, maximum_wave_number)
        self.source_vertices = half_space_source(faces, faces, 1.0, source_order, quadrature_count, maximum_wave_number)
        self.boundary_source = np.r_[
            half_space_source(faces[-1:], self.centers, 1.0, source_order, quadrature_count, maximum_wave_number).ravel(),
            half_space_source(self.centers, faces[-1:], 1.0, source_order, quadrature_count, maximum_wave_number).ravel(),
        ]
        components, weights, width, _ = profile_parameters(source_order)
        tail = np.sum(weights[:, None] * np.exp(-components[:, None] * (faces[None, :] / width)**2), axis=0)
        self.bottom_source = self.coordinate_coefficient * np.diff(-tail) / (2 * np.pi)
        self.bottom_cells = self.indices[:, 0]
        self.interior = []
        conductance_sum = np.zeros(self.cell_count)
        for left, right, area, distance, orientation in (
            (self.indices[:-1, :], self.indices[1:, :], faces[1:-1, None] * self.widths[None, :],
             np.diff(self.centers)[:, None], "radial"),
            (self.indices[:, :-1], self.indices[:, 1:], self.annular_areas[:, None] * np.ones((1, self.count - 1)),
             np.diff(self.centers)[None, :], "vertical"),
        ):
            conductance = np.broadcast_to(area / distance, left.shape).ravel()
            self.interior.append((left.ravel(), right.ravel(), conductance, orientation))
            np.add.at(conductance_sum, left.ravel(), conductance)
            np.add.at(conductance_sum, right.ravel(), conductance)
        self.boundary_cells = np.r_[self.indices[-1, :], self.indices[:, -1]]
        self.boundary_conductance = np.r_[faces[-1] * self.widths / (faces[-1] - self.centers[-1]),
                                         self.annular_areas / (faces[-1] - self.centers[-1])]
        np.add.at(conductance_sum, self.boundary_cells, self.boundary_conductance)
        row_scale = np.empty((self.cell_count, self.variable_count))
        row_scale[:, :self.component_count] = conductance_sum[:, None] * self.variable_scale[:self.component_count]
        row_scale[:, -2] = charge_coefficient
        row_scale[:, -1] = conductance_sum * conductivity_coefficient
        self.inverse_row_scale = 1 / row_scale
        self.column_scale = np.tile(self.variable_scale, self.cell_count)
        vertical_nodes = self.centers[:3]
        z_weights = np.array([np.prod(-np.delete(vertical_nodes, index) / (vertical_nodes[index] - np.delete(vertical_nodes, index)))
                              for index in range(3)])
        radial_squared = self.centers[:2]**2
        r_weights = np.array([radial_squared[1], -radial_squared[0]]) / np.diff(radial_squared)[0]
        self.constraint_cells = self.indices[:2, :3].ravel()
        self.constraint_columns = self.constraint_cells * self.variable_count + ascorbate
        self.constraint_weights = (r_weights[:, None] * z_weights[None, :]).ravel() / self.variable_scale[ascorbate]
        self.ascorbate_index = ascorbate
        self.flow_solver = HalfSpaceSlipFlow(faces)
        reservoir_matrix, _, _ = transport_matrix(reservoir, families, reference_diffusivity)
        reservoir_target = np.r_[self.target[:self.component_count], self.target[-1], 0.0]
        self.reservoir_derivative = np.linalg.solve(reservoir_matrix, reservoir_target)

    def initial_state(self, current_fraction):
        coordinate = current_fraction * self.coordinate_coefficient * self.source_centers
        return self.solution.sol(coordinate.ravel()).T

    def far_states(self, current_fraction):
        coordinates = current_fraction * self.coordinate_coefficient * self.boundary_source
        if np.max(coordinates) > self.solution.t[-1] * (1 + 1e-12):
            raise ValueError("The remote boundary exceeds the integrated chemical path; extrapolation is not allowed.")
        states = self.solution.sol(coordinates).T
        derivatives = []
        full_target = np.r_[self.target[:self.component_count], self.target[-1], 0.0]
        for state, source in zip(states, self.boundary_source):
            matrix, _, _ = transport_matrix(state, self.families, self.reference_diffusivity)
            derivatives.append(np.linalg.solve(matrix, full_target) * self.coordinate_coefficient * source)
        return states, np.array(derivatives)

    def flow(self, states, current_fraction, mobility_coefficient):
        surface_cells = extrapolate_to_wall(states[:, -1].reshape(self.count, self.count), self.centers)
        surface_faces = centered_radial_values_at_faces(surface_cells, self.centers, self.faces)
        outer = self.reservoir_derivative[-1] * current_fraction * self.coordinate_coefficient * self.source_vertices
        return self.flow_solver.solve(surface_faces, outer, mobility_coefficient)

    def assemble(self, states, current_fraction, flow):
        chemistry = chemistry_fields(states, self.families)
        residual = np.zeros_like(states)
        current_derivative = np.zeros_like(states)
        block_rows = []
        block_columns = []
        block_values = []

        def add_blocks(row_cells, column_cells, blocks):
            rows = row_cells[:, None, None] * self.variable_count + self.equation_indices[None, :, None]
            columns = column_cells[:, None, None] * self.variable_count + np.arange(self.variable_count)[None, None, :]
            rows = np.broadcast_to(rows, blocks.shape)
            columns = np.broadcast_to(columns, blocks.shape)
            selected = blocks != 0
            block_rows.append(rows[selected])
            block_columns.append(columns[selected])
            block_values.append(blocks[selected])

        for left, right, conductance, orientation in self.interior:
            volume_flux = flow["radial_flux"][1:-1, :].ravel() if orientation == "radial" else flow["vertical_flux"][:, 1:-1].ravel()
            left_chemistry = chemistry[0][left], chemistry[1][left]
            right_chemistry = chemistry[0][right], chemistry[1][right]
            flux, left_jacobian, right_jacobian = transport_face_flux(
                states[left], states[right], left_chemistry, right_chemistry, conductance, volume_flux,
                self.charges, self.diffusivities, self.projection, self.component_count,
            )
            for equation, position in enumerate(self.equation_indices):
                np.add.at(residual[:, position], left, flux[:, equation])
                np.add.at(residual[:, position], right, -flux[:, equation])
            add_blocks(left, left, left_jacobian)
            add_blocks(left, right, right_jacobian)
            add_blocks(right, left, -left_jacobian)
            add_blocks(right, right, -right_jacobian)

        boundary_states, boundary_derivative = self.far_states(current_fraction)
        boundary_chemistry = chemistry_fields(boundary_states, self.families)
        boundary_volume_flux = np.r_[flow["radial_flux"][-1, :], flow["vertical_flux"][:, -1]]
        flux, inside_jacobian, outside_jacobian = transport_face_flux(
            states[self.boundary_cells], boundary_states,
            (chemistry[0][self.boundary_cells], chemistry[1][self.boundary_cells]), boundary_chemistry,
            self.boundary_conductance, boundary_volume_flux, self.charges, self.diffusivities,
            self.projection, self.component_count,
        )
        boundary_current_derivative = np.einsum("fij,fj->fi", outside_jacobian, boundary_derivative)
        for equation, position in enumerate(self.equation_indices):
            np.add.at(residual[:, position], self.boundary_cells, flux[:, equation])
            np.add.at(current_derivative[:, position], self.boundary_cells, boundary_current_derivative[:, equation])
            residual[self.bottom_cells, position] -= current_fraction * self.bottom_source * self.target[equation]
            current_derivative[self.bottom_cells, position] -= self.bottom_source * self.target[equation]
        add_blocks(self.boundary_cells, self.boundary_cells, inside_jacobian)

        residual[:, -2] = chemistry[0] @ self.charges
        charge_derivative = np.einsum("s,fsj->fj", self.charges, chemistry[1])
        charge_rows = np.broadcast_to(np.arange(self.cell_count)[:, None] * self.variable_count + self.variable_count - 2,
                                      charge_derivative.shape)
        charge_columns = np.arange(self.cell_count)[:, None] * self.variable_count + np.arange(self.variable_count)
        selected = charge_derivative != 0
        block_rows.append(charge_rows[selected])
        block_columns.append(charge_columns[selected])
        block_values.append(charge_derivative[selected])
        rows = np.concatenate(block_rows)
        columns = np.concatenate(block_columns)
        values = np.concatenate(block_values)
        inverse_row_scale = self.inverse_row_scale.ravel()
        values *= inverse_row_scale[rows] * self.column_scale[columns]
        matrix = coo_matrix((values, (rows, columns)), shape=(states.size,) * 2).tocsc()
        scaled_residual = (residual * self.inverse_row_scale).ravel()
        scaled_current_derivative = (current_derivative * self.inverse_row_scale).ravel()
        constraint = float(np.dot(self.constraint_weights, states[self.constraint_cells, self.ascorbate_index]))
        return scaled_residual, matrix, scaled_current_derivative, constraint

    def solve_transport(self, initial_states, initial_current_fraction, flow, tolerance, maximum_iterations,
                        find_depletion_current=True):
        states = initial_states.copy()
        current_fraction = initial_current_fraction
        history = []
        for iteration in range(maximum_iterations):
            residual, matrix, current_derivative, constraint = self.assemble(states, current_fraction, flow)
            norm = max(float(np.max(np.abs(residual))), abs(constraint)) if find_depletion_current else float(np.max(np.abs(residual)))
            history.append(norm)
            if norm < tolerance:
                return states, current_fraction, history
            factorization = splu(matrix, permc_spec="MMD_AT_PLUS_A")
            right_hand_sides = np.column_stack((-residual, -current_derivative)) if find_depletion_current else -residual[:, None]
            corrections = factorization.solve(right_hand_sides)
            del factorization
            if find_depletion_current:
                base_step, current_step = corrections[:, 0], corrections[:, 1]
                constraint_scaled_weights = self.constraint_weights * self.column_scale[self.constraint_columns]
                numerator = -constraint - np.dot(constraint_scaled_weights, base_step[self.constraint_columns])
                denominator = np.dot(constraint_scaled_weights, current_step[self.constraint_columns])
                current_correction = numerator / denominator
                scaled_state_correction = base_step + current_step * current_correction
            else:
                current_correction = 0.0
                scaled_state_correction = corrections[:, 0]
            state_correction = (self.column_scale * scaled_state_correction).reshape(states.shape)
            step = 1.0
            for line_search in range(20):
                candidate = states + step * state_correction
                candidate_current = current_fraction + step * current_correction
                if candidate_current > 0 and np.all(candidate[:, :self.component_count] >= 0):
                    candidate_residual, _, _, candidate_constraint = self.assemble(candidate, candidate_current, flow)
                    candidate_norm = (max(float(np.max(np.abs(candidate_residual))), abs(candidate_constraint))
                                      if find_depletion_current else float(np.max(np.abs(candidate_residual))))
                    if candidate_norm < norm * (1 - 1e-4 * step):
                        states, current_fraction = candidate, candidate_current
                        break
                step /= 2
            else:
                raise RuntimeError(f"Transport Newton line search failed at residual {norm:.3e}.")
        raise RuntimeError(f"Transport Newton iteration did not converge: {history}.")

    def solve_coupled(self, mobility_coefficient, newton_tolerance, flow_tolerance, maximum_newton_iterations,
                      maximum_flow_iterations, initial_states=None, initial_current_fraction=0.9,
                      find_depletion_current=True):
        if initial_states is not None:
            states = initial_states.copy()
        elif find_depletion_current:
            states = self.initial_state(initial_current_fraction)
        else:
            states = np.tile(self.solution.sol(0.0), (self.cell_count, 1))
        current_fraction = initial_current_fraction
        flow = self.flow(states, current_fraction, mobility_coefficient)
        history = []
        for iteration in range(maximum_flow_iterations):
            states, current_fraction, newton_history = self.solve_transport(
                states, current_fraction, flow, newton_tolerance, maximum_newton_iterations,
                find_depletion_current=find_depletion_current,
            )
            updated_flow = self.flow(states, current_fraction, mobility_coefficient)
            flux_scale = max(float(np.max(np.abs(updated_flow["radial_flux"]))), float(np.max(np.abs(updated_flow["vertical_flux"]))))
            flux_difference = max(float(np.max(np.abs(updated_flow["radial_flux"] - flow["radial_flux"]))),
                                  float(np.max(np.abs(updated_flow["vertical_flux"] - flow["vertical_flux"]))))
            relative_change = flux_difference / flux_scale if flux_scale else 0.0
            residual, _, _, constraint = self.assemble(states, current_fraction, updated_flow)
            coupled_residual = (max(float(np.max(np.abs(residual))), abs(constraint))
                                if find_depletion_current else float(np.max(np.abs(residual))))
            history.append({"iteration": iteration + 1, "current_fraction": current_fraction,
                            "newton_residual_history": newton_history, "flow_relative_change": relative_change,
                            "coupled_residual": coupled_residual})
            print(f"    Coupling {iteration + 1}: I/I_reference={current_fraction:.9f}, residual={coupled_residual:.2e}, flow change={relative_change:.2e}.", flush=True)
            flow = updated_flow
            if relative_change < flow_tolerance and coupled_residual < newton_tolerance:
                return states, current_fraction, flow, history
        raise RuntimeError(f"Stokes/transport coupling did not converge: {history}.")
