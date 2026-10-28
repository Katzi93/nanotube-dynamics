"""Fit local ridges and connect them into complete tube traces."""

import numpy as np
from scipy.optimize import least_squares

from tube_tracker import configuration


def finite_gaussian_ridge(
    parallel_positions,
    perpendicular_positions,
    parallel_fwhm,
    parallel_profile_power,
    perpendicular_fwhm,
    amplitude,
    background,
):
    """Return the intensity of a finite ridge at specified parallel and perpendicular positions."""
    # The super-Gaussian is nearly constant along the ridge and falls sharply at its two ends.
    # The ordinary Gaussian gives the ridge a bright center and a smooth cross-tube intensity profile.
    # Both profiles equal one half at plus or minus one half of their respective FWHM values.
    parallel_profile = np.exp(-np.log(2.0) * np.abs(2.0 * parallel_positions / parallel_fwhm) ** parallel_profile_power)
    perpendicular_profile = np.exp(-4.0 * np.log(2.0) * (perpendicular_positions / perpendicular_fwhm) ** 2)
    # Multiplying the two profiles produces one finite bright line segment on a constant background.
    return background + amplitude * parallel_profile * perpendicular_profile


def fit_finite_ridge(
    image,
    initial_center,
    initial_angle,
):
    """Fit one finite ridge without allowing displacement parallel to the ridge."""
    fitting_radius = configuration.fitting_radius
    parallel_fwhm = configuration.ridge_parallel_fwhm
    parallel_profile_power = configuration.ridge_parallel_power
    initial_perpendicular_fwhm = configuration.initial_perpendicular_fwhm
    initial_center = np.asarray(initial_center, dtype=np.float64)

    # Select one fixed set of pixels satisfying (x - center_x)^2 + (y - center_y)^2 <= fitting_radius^2.
    x_minimum = int(np.floor(initial_center[0] - fitting_radius))
    x_maximum = int(np.ceil(initial_center[0] + fitting_radius)) + 1
    y_minimum = int(np.floor(initial_center[1] - fitting_radius))
    y_maximum = int(np.ceil(initial_center[1] + fitting_radius)) + 1
    if y_minimum < 0 or x_minimum < 0 or y_maximum > image.shape[0] or x_maximum > image.shape[1]:
        raise ValueError("The complete ridge-fitting region must lie inside the image.")

    # Create the integer (x, y) coordinate of every pixel in the square enclosing the fitting circle.
    y_coordinates, x_coordinates = np.mgrid[y_minimum:y_maximum, x_minimum:x_maximum]
    inside_fitting_circle = (x_coordinates - initial_center[0]) ** 2 + (y_coordinates - initial_center[1]) ** 2 <= fitting_radius**2
    # Store the selected coordinates as an N x 2 array of [x, y] pairs and read their matching intensities.
    selected_coordinates = np.stack((x_coordinates[inside_fitting_circle], y_coordinates[inside_fitting_circle]), axis=1).astype(np.float64)
    measured_intensities = image[y_coordinates[inside_fitting_circle], x_coordinates[inside_fitting_circle]].astype(np.float64)
    # Use local intensity statistics only as starting guesses; amplitude and background are fitted below.
    initial_background = np.median(measured_intensities)
    initial_amplitude = measured_intensities.max() - initial_background

    def residuals(parameters):
        perpendicular_shift, angle_change, perpendicular_fwhm, amplitude, background = parameters
        angle = initial_angle + angle_change
        # These unit vectors point respectively along and across the candidate ridge.
        parallel_direction = np.array([np.cos(angle), np.sin(angle)])
        perpendicular_direction = np.array([-np.sin(angle), np.cos(angle)])
        # The center may move only across the ridge; there is no fitted displacement along the ridge.
        center = initial_center + perpendicular_shift * perpendicular_direction
        relative_positions = selected_coordinates - center
        # The @ operator takes the dot product of every [x, y] position with a direction vector.
        # The results are the signed distances of every selected pixel along and across the ridge.
        parallel_positions = relative_positions @ parallel_direction
        perpendicular_positions = relative_positions @ perpendicular_direction
        predicted_intensities = finite_gaussian_ridge(
            parallel_positions,
            perpendicular_positions,
            parallel_fwhm,
            parallel_profile_power,
            perpendicular_fwhm,
            amplitude,
            background,
        )
        # least_squares minimizes the sum of these squared pixel-intensity differences.
        return predicted_intensities - measured_intensities

    # Parameter order: perpendicular shift, angle change, perpendicular FWHM, amplitude, and background.
    initial_parameters = np.array([0.0, 0.0, initial_perpendicular_fwhm, initial_amplitude, initial_background])
    # Width and amplitude must be nonnegative; shift, angle, and background remain unrestricted during fitting.
    lower_bounds = np.array([-np.inf, -np.inf, 0.0, 0.0, -np.inf])
    upper_bounds = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
    fit = least_squares(residuals, initial_parameters, bounds=(lower_bounds, upper_bounds), x_scale="jac")

    perpendicular_shift, angle_change, perpendicular_fwhm, amplitude, background = fit.x
    optimizer_angle = initial_angle + angle_change
    perpendicular_direction = np.array([-np.sin(optimizer_angle), np.cos(optimizer_angle)])
    center = initial_center + perpendicular_shift * perpendicular_direction
    # A ridge is an undirected line, so angles separated by pi are identical. Store its angle in [0, pi).
    angle = optimizer_angle % np.pi
    # fit.fun contains the final intensity difference for every selected pixel.
    root_mean_squared_error = np.sqrt(np.mean(fit.fun**2))

    return {
        "converged": fit.success,
        "center": center,
        "angle": angle,
        "perpendicular_shift": perpendicular_shift,
        "perpendicular_fwhm": perpendicular_fwhm,
        "amplitude": amplitude,
        "background": background,
        "root_mean_squared_error": root_mean_squared_error,
        "function_evaluations": fit.nfev,
    }


def validate_ridge(
    ridge,
    predicted_angle,
):
    """Return the reasons a fitted ridge fails the agreed acceptance criteria."""
    minimum_accepted_fwhm = configuration.minimum_accepted_fwhm
    maximum_accepted_fwhm = configuration.maximum_accepted_fwhm
    minimum_amplitude_to_error = configuration.minimum_amplitude_to_error
    maximum_accepted_angle_change = configuration.maximum_accepted_angle_change
    maximum_accepted_perpendicular_shift = configuration.maximum_accepted_perpendicular_shift

    # abs(cos(...)) treats angles separated by pi as the same line; arccos returns a difference in [0, pi/2].
    angle_change = np.arccos(np.abs(np.cos(ridge["angle"] - predicted_angle)))
    rejection_reasons = []
    if not ridge["converged"]:
        rejection_reasons.append("fit did not converge")
    if not minimum_accepted_fwhm <= ridge["perpendicular_fwhm"] <= maximum_accepted_fwhm:
        rejection_reasons.append(f"FWHM {ridge['perpendicular_fwhm']:.2f} is outside the accepted range")
    if ridge["amplitude"] < minimum_amplitude_to_error * ridge["root_mean_squared_error"]:
        amplitude_to_error = ridge["amplitude"] / ridge["root_mean_squared_error"]
        rejection_reasons.append(f"amplitude/RMSE {amplitude_to_error:.2f} is below {minimum_amplitude_to_error:.2f}")
    if angle_change > maximum_accepted_angle_change:
        rejection_reasons.append(f"angle change {np.rad2deg(angle_change):.2f} degrees is too large")
    if abs(ridge["perpendicular_shift"]) > maximum_accepted_perpendicular_shift:
        rejection_reasons.append(f"perpendicular shift {ridge['perpendicular_shift']:.2f} pixels is too large")
    return rejection_reasons


def predict_next_ridge(current_ridge, previous_parallel_direction, step_length):
    """Predict the next center and retain the current branch's direction along the ridge."""
    current_parallel_direction = np.array([np.cos(current_ridge["angle"]), np.sin(current_ridge["angle"])])
    # The stored line angle has two possible directions. Select the one agreeing with the previous step.
    if np.dot(current_parallel_direction, previous_parallel_direction) < 0.0:
        current_parallel_direction = -current_parallel_direction
    predicted_center = current_ridge["center"] + step_length * current_parallel_direction
    return predicted_center, current_ridge["angle"], current_parallel_direction


def extend_tube(
    image,
    initial_ridge,
    direction,
):
    """Repeatedly predict and fit ridges in one direction until a candidate is rejected."""
    accepted_ridges = []
    current_ridge = initial_ridge
    # The two calls to extend_tube use opposite signs and therefore begin toward opposite tube ends.
    previous_parallel_direction = direction * np.array([np.cos(initial_ridge["angle"]), np.sin(initial_ridge["angle"])])
    step_lengths = (
        configuration.extension_step_length,
        configuration.extension_step_length * configuration.long_retry_step_multiplier,
        configuration.extension_step_length * configuration.short_retry_step_multiplier,
    )
    for point_number in range(1, configuration.maximum_points_per_direction + 1):
        # Try the normal step first, then jump farther past a difficult region, then try one shorter final step.
        for attempt_number, step_length in enumerate(step_lengths, start=1):
            predicted_center, predicted_angle, current_parallel_direction = predict_next_ridge(
                current_ridge,
                previous_parallel_direction,
                step_length,
            )
            candidate_ridge = fit_finite_ridge(image, predicted_center, predicted_angle)
            rejection_reasons = validate_ridge(candidate_ridge, predicted_angle)
            if not rejection_reasons:
                if attempt_number > 1 and configuration.fitting_verbose:
                    print(f"Direction {direction:+d}, point {point_number}: accepted retry with a {step_length:.2f}-pixel step.")
                break
            if configuration.fitting_verbose:
                print(
                    f"Direction {direction:+d}, point {point_number}: rejected {step_length:.2f}-pixel step: "
                    + "; ".join(rejection_reasons)
                )

        if rejection_reasons:
            if configuration.fitting_verbose:
                print(f"Direction {direction:+d} stopped after {len(accepted_ridges)} extensions: every step length was rejected.")
            break

        accepted_ridges.append(candidate_ridge)
        current_ridge = candidate_ridge
        previous_parallel_direction = current_parallel_direction
        if configuration.fitting_verbose:
            amplitude_to_error = candidate_ridge["amplitude"] / candidate_ridge["root_mean_squared_error"]
            print(
                f"Direction {direction:+d}, point {point_number}: center=({candidate_ridge['center'][0]:.2f}, "
                f"{candidate_ridge['center'][1]:.2f}), angle={np.rad2deg(candidate_ridge['angle']):.2f} degrees, "
                f"FWHM={candidate_ridge['perpendicular_fwhm']:.2f}, amplitude/RMSE={amplitude_to_error:.2f}"
            )

    return accepted_ridges


def construct_tube(
    image,
    initial_center,
    initial_angle,
):
    """Fit an initial ridge and construct the complete tube in both directions."""
    initial_ridge = fit_finite_ridge(image, initial_center, initial_angle)
    rejection_reasons = validate_ridge(initial_ridge, initial_angle)
    if rejection_reasons:
        if configuration.fitting_verbose:
            print("Initial ridge rejected: " + "; ".join(rejection_reasons))
        return []

    if configuration.fitting_verbose:
        initial_amplitude_to_error = initial_ridge["amplitude"] / initial_ridge["root_mean_squared_error"]
        print(
            f"Initial ridge: center=({initial_ridge['center'][0]:.2f}, {initial_ridge['center'][1]:.2f}), "
            f"angle={np.rad2deg(initial_ridge['angle']):.2f} degrees, FWHM={initial_ridge['perpendicular_fwhm']:.2f}, "
            f"amplitude/RMSE={initial_amplitude_to_error:.2f}"
        )

    backward_ridges = extend_tube(
        image,
        initial_ridge,
        -1,
    )
    forward_ridges = extend_tube(
        image,
        initial_ridge,
        +1,
    )
    return list(reversed(backward_ridges)) + [initial_ridge] + forward_ridges
