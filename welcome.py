"""Open a test TIFF stack and browse it frame by frame."""

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.widgets import Slider
import numpy as np
import tifffile


def show_frame(
    frame_index: float,
    *,
    frames: np.ndarray,
    displayed_frame: AxesImage,
    axis: Axes,
    stack_name: str,
) -> None:
    """Display the selected frame in the existing image artist."""
    index = int(frame_index)
    displayed_frame.set_data(frames[index])
    axis.set_title(f"{stack_name} - frame {index + 1}/{len(frames)}")
    axis.figure.canvas.draw_idle()


if __name__ == "__main__":
    # Analysis parameters
    input_path = Path(__file__).parent / "test_data" / "test_1.tif"  # TIFF stack to display.
    contrast_percentiles = (1, 99.8)  # Percentiles mapped to black and white.
    color_map = "gray"  # Color map used to display the image intensity.
    figure_size = (8, 7)  # Width and height of the viewer in inches.

    print(f"Loading TIFF stack: {input_path}")
    frames = tifffile.imread(input_path)
    assert frames.ndim == 3, f"Expected a 3-D TIFF stack, found shape {frames.shape}"

    print("Calculating display contrast...")
    contrast_low, contrast_high = np.percentile(frames, contrast_percentiles)

    print(f"Loaded {len(frames)} frames with shape {frames.shape[1:]} and dtype {frames.dtype}")
    print(f"Display range: {contrast_low:g} to {contrast_high:g}")

    figure, axis = plt.subplots(figsize=figure_size)
    figure.subplots_adjust(bottom=0.14)
    displayed_frame = axis.imshow(
        frames[0], cmap=color_map, vmin=contrast_low, vmax=contrast_high
    )
    axis.set_title(f"{input_path.name} - frame 1/{len(frames)}")
    axis.set_axis_off()

    slider_axis = figure.add_axes((0.18, 0.05, 0.64, 0.03))
    frame_slider = Slider(
        slider_axis,
        "Frame",
        valmin=0,
        valmax=len(frames) - 1,
        valinit=0,
        valstep=1,
    )
    frame_slider.on_changed(
        partial(
            show_frame,
            frames=frames,
            displayed_frame=displayed_frame,
            axis=axis,
            stack_name=input_path.name,
        )
    )

    print("Opening interactive viewer...")
    plt.show()
