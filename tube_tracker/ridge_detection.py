"""Find likely local tube ridges using oriented convolution kernels."""

import numpy as np
from scipy.signal import fftconvolve

from tube_tracker import configuration


def make_convolution_kernels():
    """Return an ordered tuple of angle and oriented-line-kernel pairs."""
    perpendicular_fwhm = configuration.convolution_perpendicular_fwhm
    parallel_fwhm = configuration.convolution_parallel_fwhm
    angle_step = configuration.convolution_angle_step
    parallel_profile_power = configuration.convolution_parallel_power

    assert perpendicular_fwhm > 0.0
    assert parallel_fwhm > 0.0
    assert 0.0 < angle_step < np.pi
    assert parallel_profile_power > 0.0

    perpendicular_sigma = perpendicular_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    parallel_scale = parallel_fwhm / (2.0 * (2.0 * np.log(2.0)) ** (1.0 / parallel_profile_power))

    # At these radii both profiles have fallen to exp(-8) of their peaks.
    perpendicular_radius = 4.0 * perpendicular_sigma
    parallel_radius = parallel_scale * 16.0 ** (1.0 / parallel_profile_power)
    kernel_radius = int(np.ceil(np.hypot(perpendicular_radius, parallel_radius)))
    coordinates = np.arange(-kernel_radius, kernel_radius + 1)
    horizontal, vertical = np.meshgrid(coordinates, coordinates)

    kernels = []
    for angle in np.arange(0.0, np.pi, angle_step):
        parallel_coordinate = horizontal * np.cos(angle) + vertical * np.sin(angle)
        perpendicular_coordinate = -horizontal * np.sin(angle) + vertical * np.cos(angle)
        parallel_weight = np.exp(-0.5 * (np.abs(parallel_coordinate) / parallel_scale) ** parallel_profile_power)
        perpendicular_weight = np.exp(-0.5 * (perpendicular_coordinate / perpendicular_sigma) ** 2)
        kernel = parallel_weight * perpendicular_weight
        kernel -= kernel.mean()
        kernel /= np.sqrt(np.sum(kernel**2))
        kernels.append((float(angle), kernel.astype(np.float32)))

    return tuple(kernels)


def convolve_locally(image, center, kernel, search_radius):
    """Convolve one kernel around one position and return the local response."""
    # Find the image region needed to calculate every kernel response inside the search radius.
    center_column = int(round(center[0]))
    center_row = int(round(center[1]))
    kernel_radius = kernel.shape[0] // 2
    crop_margin = kernel_radius + search_radius
    row_minimum = center_row - crop_margin
    row_maximum = center_row + crop_margin + 1
    column_minimum = center_column - crop_margin
    column_maximum = center_column + crop_margin + 1

    local_image = image[row_minimum:row_maximum, column_minimum:column_maximum]
    return fftconvolve(local_image, kernel, mode="valid")


def find_likely_ridge(image, clicked_center, convolution_kernels, search_radius):
    """Return the strongest nearby ridge center, angle, and response score."""
    clicked_column = int(round(clicked_center[0]))
    clicked_row = int(round(clicked_center[1]))

    best_score = -np.inf
    best_center = None
    best_angle = None
    for angle, kernel in convolution_kernels:
        response = convolve_locally(image, clicked_center, kernel, search_radius)
        maximum_index = np.unravel_index(np.argmax(response), response.shape)
        maximum_score = response[maximum_index]
        if maximum_score > best_score:
            best_score = maximum_score
            best_center = np.array(
                [clicked_column + maximum_index[1] - search_radius, clicked_row + maximum_index[0] - search_radius],
                dtype=np.float64,
            )
            best_angle = angle

    return best_center, best_angle, best_score
