"""Store the algorithm parameters selected for one tracking run."""

import numpy as np


convolution_search_radius: int = 12  # Half-width in pixels of the square searched around a click or tracking seed.
convolution_perpendicular_fwhm: float = 6.0  # Width in pixels across each convolution kernel.
convolution_parallel_fwhm: float = 30.0  # Length in pixels along each convolution kernel.
convolution_angle_step: float = np.pi / 90.0  # Angular spacing in radians between convolution kernels.
convolution_parallel_power: float = 4.0  # Power controlling the convolution kernel's longitudinal profile.

fitting_radius: float = 21.0  # Radius in pixels of the fixed circular fitting region.
ridge_parallel_fwhm: float = 30.0  # Fixed longitudinal FWHM in pixels of the fitted ridge.
ridge_parallel_power: float = 12.0  # Power controlling the sharpness of the fitted ridge ends.
initial_perpendicular_fwhm: float = 6.2  # Starting estimate of the fitted tube width in pixels.

extension_step_length: float = 6.0  # Distance in pixels between consecutive predicted ridge centers.
long_retry_step_multiplier: float = 2.0  # Multiplier used to jump past a rejected normal prediction.
short_retry_step_multiplier: float = 0.5  # Multiplier used for the final shorter prediction attempt.
maximum_points_per_direction: int = 100  # Maximum number of ridges constructed toward each tube end.

minimum_accepted_fwhm: float = 2.0  # Smallest fitted tube width accepted as a ridge.
maximum_accepted_fwhm: float = 12.0  # Largest fitted tube width accepted as a ridge.
minimum_amplitude_to_error: float = 1.2  # Smallest accepted ratio between ridge amplitude and fitting RMSE.
maximum_accepted_angle_change: float = np.pi / 9.0  # Largest accepted angle change from the predicted ridge.
maximum_accepted_perpendicular_shift: float = 8.0  # Largest accepted sideways displacement in pixels.
fitting_verbose: bool = True  # Print detailed ridge fitting, retry, and rejection messages.
tracking_verbose: bool = False  # Print every tracking trial, rejection reason, and internal timing measurement.

maximum_relative_length_difference: float = 0.15  # Largest accepted fractional difference from the running mean length.
maximum_center_position_difference: float = 25.0  # Largest accepted tube-center displacement in pixels between frames.
maximum_overall_angle_difference: float = np.pi / 6.0  # Largest accepted overall tube-angle change between frames.
maximum_rescue_frames: int = 3  # Number of additional future frames searched after a tube is first lost.


def set_parameters(**values):
    """Change algorithm parameters before starting a tracking run."""
    for name, value in values.items():
        # __annotations__ contains exactly the parameter names declared above.
        if name not in __annotations__:
            raise ValueError(f"Unknown tracking parameter: {name}")
        globals()[name] = value
