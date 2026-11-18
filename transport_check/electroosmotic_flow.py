"""Incompressible half-space Stokes flow driven by potential-gradient slip.

The harmonic function used here extends the SURFACE potential. It is an auxiliary
Stokes representation and is not a replacement for the computed bulk potential.
Coordinates are scaled by the spot radius, speeds by D_ref/a, and streamfunctions
by D_ref*a. Axisymmetric face volume fluxes omit a common factor 2*pi.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def differentiation_weights(nodes, location, derivative):
    offsets = np.asarray(nodes) - location
    powers = np.arange(offsets.size)
    matrix = offsets[None, :] ** powers[:, None]
    target = np.zeros(offsets.size)
    target[derivative] = 1 if derivative == 1 else 2
    return np.linalg.solve(matrix, target)


def extrapolate_to_wall(values, vertical_centers):
    nodes = vertical_centers[:3]
    weights = np.array([np.prod(-np.delete(nodes, index) / (nodes[index] - np.delete(nodes, index)))
                        for index in range(3)])
    return values[:, :3] @ weights


def centered_radial_values_at_faces(values, radial_centers, radial_faces):
    result = np.empty(radial_faces.size)
    for index, location in enumerate(radial_faces):
        if index == 0:
            nodes = radial_centers[:2] ** 2
            result[index] = (nodes[1] * values[0] - nodes[0] * values[1]) / (nodes[1] - nodes[0])
        else:
            start = min(max(np.searchsorted(radial_centers, location) - 1, 0), radial_centers.size - 3)
            nodes = radial_centers[start:start + 3]
            weights = np.array([np.prod((location - np.delete(nodes, offset)) / (nodes[offset] - np.delete(nodes, offset)))
                                for offset in range(3)])
            result[index] = weights @ values[start:start + 3]
    return result


class HalfSpaceSlipFlow:
    """Reuse a harmonic-extension factorization on one fixed axisymmetric mesh."""

    def __init__(self, faces):
        self.faces = faces
        self.centers = (faces[:-1] + faces[1:]) / 2
        self.node_count = faces.size
        indices = np.arange(faces.size**2).reshape(faces.size, faces.size)
        rows = []
        columns = []
        data = []
        boundary = np.zeros(indices.shape, dtype=bool)
        boundary[:, 0] = True
        boundary[-1, :] = True
        boundary[:, -1] = True
        for radial in range(faces.size - 1):
            for vertical in range(1, faces.size - 1):
                index = indices[radial, vertical]
                if radial == 0:
                    radial_columns = indices[:2, vertical]
                    radial_values = np.array([-4.0, 4.0]) / faces[1] ** 2
                else:
                    nodes = faces[radial - 1:radial + 2]
                    radial_columns = indices[radial - 1:radial + 2, vertical]
                    radial_values = differentiation_weights(nodes, faces[radial], 2)
                    radial_values += differentiation_weights(nodes, faces[radial], 1) / faces[radial]
                vertical_values = differentiation_weights(faces[vertical - 1:vertical + 2], faces[vertical], 2)
                rows.extend([index] * 6 if radial else [index] * 5)
                columns.extend(radial_columns.tolist() + indices[radial, vertical - 1:vertical + 2].tolist())
                data.extend((-radial_values).tolist() + (-vertical_values).tolist())
        boundary_indices = indices[boundary]
        rows.extend(boundary_indices.tolist())
        columns.extend(boundary_indices.tolist())
        data.extend(np.ones(boundary_indices.size).tolist())
        matrix = coo_matrix((data, (rows, columns)), shape=(faces.size**2,) * 2).tocsc()
        self.factorization = splu(matrix)
        self.boundary = boundary
        self.indices = indices

    def solve(self, surface_potential, outer_potential, mobility_coefficient):
        boundary_values = outer_potential.copy()
        boundary_values[:, 0] = surface_potential
        right_hand_side = np.zeros_like(boundary_values)
        right_hand_side[self.boundary] = boundary_values[self.boundary]
        harmonic = self.factorization.solve(right_hand_side.ravel()).reshape(boundary_values.shape)
        radial_derivative = np.gradient(harmonic, self.faces, axis=0, edge_order=2)
        radial_derivative[0, :] = 0
        streamfunction = mobility_coefficient * self.faces[:, None] * self.faces[None, :] * radial_derivative
        radial_flux = -np.diff(streamfunction, axis=1)
        vertical_flux = np.diff(streamfunction, axis=0)
        divergence = (radial_flux[1:, :] - radial_flux[:-1, :]
                      + vertical_flux[:, 1:] - vertical_flux[:, :-1])
        flux_scale = max(float(np.max(np.abs(radial_flux))), float(np.max(np.abs(vertical_flux))))
        relative_divergence = float(np.max(np.abs(divergence)) / flux_scale) if flux_scale else 0.0
        widths = np.diff(self.faces)
        annular_areas = np.diff(self.faces**2) / 2
        radial_velocity_faces = np.zeros_like(radial_flux)
        radial_velocity_faces[1:, :] = radial_flux[1:, :] / (self.faces[1:, None] * widths[None, :])
        vertical_velocity_faces = vertical_flux / annular_areas[:, None]
        radial_velocity = (radial_velocity_faces[:-1, :] + radial_velocity_faces[1:, :]) / 2
        vertical_velocity = (vertical_velocity_faces[:, :-1] + vertical_velocity_faces[:, 1:]) / 2
        return {
            "radial_flux": radial_flux, "vertical_flux": vertical_flux, "streamfunction": streamfunction,
            "radial_velocity": radial_velocity, "vertical_velocity": vertical_velocity,
            "harmonic_extension": harmonic, "relative_discrete_divergence": relative_divergence,
            "surface_slip": -mobility_coefficient * radial_derivative[:, 0],
        }
