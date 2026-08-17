"""Threshold the maximum oriented convolution response."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


if __name__ == "__main__":
    # Read-only stack containing one convolution-response image per angle.
    response_stack_path = Path("test_outputs/angle_responses.tif")
    # Single binary threshold image, overwritten on every run.
    output_path = Path("test_outputs/thresholded_response.png")
    # Minimum maximum-angle response classified as tube-like.
    response_threshold = 50.0
    # Size of the binary threshold figure in inches.
    figure_size = (10, 9)
    # Resolution of the saved binary threshold figure.
    figure_dpi = 150

    assert response_stack_path.is_file(), (
        f"Missing angle-response stack: {response_stack_path}"
    )

    print(f"Reading angle responses from {response_stack_path}...")
    response_stack = tifffile.memmap(response_stack_path, mode="r")
    assert response_stack.ndim == 3

    print("Calculating the maximum response across all angles...")
    maximum_response = np.max(response_stack, axis=0)
    thresholded_response = maximum_response >= response_threshold

    figure, axis = plt.subplots(figsize=figure_size)
    axis.imshow(
        thresholded_response,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axis.set_title(f"Maximum response >= {response_threshold:g}")
    axis.set_axis_off()
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(figure)

    selected_pixel_count = np.count_nonzero(thresholded_response)
    selected_fraction = selected_pixel_count / thresholded_response.size
    print(
        f"Selected {selected_pixel_count} pixels "
        f"({selected_fraction:.3%} of the image)."
    )
    print(f"Saved threshold image: {output_path}")
