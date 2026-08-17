"""Interactively segment one nanotube using a box on the first frame."""

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
import numpy as np
from PIL import Image, ImageFilter
from sam3.model_builder import build_sam3_predictor
import tifffile
import torch


def scale_grayscale_to_rgb(frame, low_percentile, high_percentile):
    """Scale a grayscale frame to an RGB uint8 image using percentile limits."""
    low_value, high_value = np.percentile(
        frame, [low_percentile, high_percentile]
    )
    assert high_value > low_value, "Contrast limits must span a nonzero range"

    scaled_frame = np.clip(
        (frame.astype(np.float32) - low_value) / (high_value - low_value),
        0.0,
        1.0,
    )
    grayscale_uint8 = np.rint(scaled_frame * 255).astype(np.uint8)
    rgb_image = np.repeat(grayscale_uint8[:, :, np.newaxis], 3, axis=2)
    return rgb_image, low_value, high_value


def handle_box_selection(click_event, release_event, bounding_box, box_artist):
    """Store and display a box selected by dragging the left mouse button."""
    x_min, x_max = sorted([click_event.xdata, release_event.xdata])
    y_min, y_max = sorted([click_event.ydata, release_event.ydata])
    assert x_max > x_min and y_max > y_min, "The selected box has zero area"

    bounding_box[:] = [x_min, y_min, x_max, y_max]
    box_artist.set_xy((x_min, y_min))
    box_artist.set_width(x_max - x_min)
    box_artist.set_height(y_max - y_min)
    box_artist.set_visible(True)
    box_artist.figure.canvas.draw_idle()
    print(
        f"Selected box: x={x_min:.1f} to {x_max:.1f}, "
        f"y={y_min:.1f} to {y_max:.1f}"
    )


def segment_from_box(
    tracking_model,
    inference_state,
    bounding_box,
    frame_index,
    output_probability_threshold,
):
    """Run SAM with one pixel-coordinate box and return its highest-scoring mask."""
    assert len(bounding_box) == 4, "Draw a box before running SAM"
    x_min, y_min, x_max, y_max = bounding_box
    image_width = inference_state["orig_width"]
    image_height = inference_state["orig_height"]
    relative_box_xywh = [
        x_min / image_width,
        y_min / image_height,
        (x_max - x_min) / image_width,
        (y_max - y_min) / image_height,
    ]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        returned_frame_index, outputs = tracking_model.add_prompt(
            inference_state=inference_state,
            frame_idx=frame_index,
            points=None,
            point_labels=None,
            boxes_xywh=torch.tensor([relative_box_xywh], dtype=torch.float32),
            box_labels=torch.tensor([1], dtype=torch.int32),
            clear_old_boxes=True,
            output_prob_thresh=output_probability_threshold,
        )

    assert returned_frame_index == frame_index
    masks = np.asarray(outputs["out_binary_masks"])
    output_scores = np.asarray(outputs["out_probs"])
    assert masks.shape[0] > 0, "SAM returned no masks for the selected box"

    selected_index = int(np.argmax(output_scores))
    return masks[selected_index].astype(bool), float(output_scores[selected_index])


def save_result(mask, figure, output_directory, frame_index):
    """Save the binary mask and current annotated figure."""
    output_directory.mkdir(parents=True, exist_ok=True)
    mask_path = output_directory / f"frame_{frame_index:03d}_mask.tif"
    overlay_path = output_directory / f"frame_{frame_index:03d}_overlay.png"

    tifffile.imwrite(mask_path, mask.astype(np.uint8), photometric="minisblack")
    figure.savefig(overlay_path, dpi=150, bbox_inches="tight")
    print(f"Saved mask: {mask_path}")
    print(f"Saved overlay: {overlay_path}")


def handle_key_press(
    event,
    axes,
    tracking_model,
    inference_state,
    bounding_box,
    box_artist,
    mask_artist,
    output_directory,
    frame_index,
    output_probability_threshold,
    mask_color,
    mask_alpha,
):
    """Clear the selection or run segmentation when a key is pressed."""
    if event.key == "c":
        bounding_box.clear()
        box_artist.set_visible(False)
        mask_artist.set_data(np.zeros((*mask_artist.get_array().shape[:2], 4)))
        axes.set_title("Selection cleared")
        axes.figure.canvas.draw_idle()
        print("Cleared the selected box and predicted mask")
        return

    if event.key != "enter":
        return

    axes.set_title("Running SAM 3.1 box segmentation...")
    axes.figure.canvas.draw()
    axes.figure.canvas.flush_events()
    print("Running SAM 3.1 box-prompt segmentation...")

    mask, output_score = segment_from_box(
        tracking_model,
        inference_state,
        bounding_box,
        frame_index,
        output_probability_threshold,
    )

    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask, :3] = mask_color
    overlay[mask, 3] = mask_alpha
    mask_artist.set_data(overlay)
    axes.set_title(f"SAM 3.1 box result, score: {output_score:.3f}")
    axes.figure.canvas.draw_idle()

    mask_fraction = 100.0 * mask.mean()
    print(f"SAM output score: {output_score:.3f}")
    print(f"Masked pixels: {mask.sum()} ({mask_fraction:.4f}% of the frame)")
    save_result(mask, axes.figure, output_directory, frame_index)


if __name__ == "__main__":
    # Read-only microscopy TIFF stack used for this experiment.
    input_path = Path("test_data/test_1.tif")
    # Local SAM 3.1 checkpoint downloaded from Hugging Face.
    checkpoint_path = Path("checkpoints/sam3.1_multiplex.pt")
    # Directory for generated masks and annotated images.
    output_directory = Path("test_outputs/sam3_first_frame_smoothed")
    # Directory containing the generated one-frame JPEG input for SAM's video API.
    frame_directory = output_directory / "input_frames"
    # Zero-based TIFF frame to segment.
    frame_index = 0
    # Lower percentile mapped to black during display scaling.
    contrast_low_percentile = 1.0
    # Upper percentile mapped to white during display scaling.
    contrast_high_percentile = 99.8
    # Gaussian blur radius applied after contrast scaling to suppress pixel noise.
    gaussian_blur_radius = 1.5
    # Minimum object probability retained in the SAM output.
    output_probability_threshold = 0.0
    # Maximum number of objects exposed by this single-object test.
    max_num_objects = 1
    # Native object bucket size encoded by the SAM 3.1 checkpoint.
    multiplex_count = 16
    # Avoid the optional Flash Attention 3 dependency on the laptop GPU.
    use_flash_attention = False
    # Use the checkpoint's native complex-valued rotary position encoding.
    use_real_rope = False
    # Allow faster Ampere matrix operations during inference.
    allow_tf32 = True
    # Size of the interactive Matplotlib window in inches.
    figure_size = (11, 11)
    # RGB color used for the predicted mask overlay.
    mask_color = (0.0, 0.8, 1.0)
    # Opacity used for the predicted mask overlay.
    mask_alpha = 0.45
    # Color used for the manually selected bounding box.
    box_color = "lime"
    # Width of the manually selected bounding-box line.
    box_line_width = 2
    # Minimum box width and height accepted by the selector in display pixels.
    minimum_box_span = 5

    assert input_path.is_file(), f"Missing input TIFF: {input_path}"
    assert checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}"
    assert torch.cuda.is_available(), "PyTorch cannot access a CUDA GPU"

    print(f"Reading frame {frame_index} from {input_path}...")
    frame = tifffile.imread(input_path, key=frame_index)
    assert frame.ndim == 2, f"Expected a grayscale frame, got shape {frame.shape}"

    rgb_array, contrast_low_value, contrast_high_value = scale_grayscale_to_rgb(
        frame, contrast_low_percentile, contrast_high_percentile
    )
    rgb_image = Image.fromarray(rgb_array).filter(
        ImageFilter.GaussianBlur(radius=gaussian_blur_radius)
    )
    rgb_array = np.asarray(rgb_image)
    print(
        f"Display contrast: {contrast_low_value:.1f} to "
        f"{contrast_high_value:.1f} intensity units"
    )
    print(f"Gaussian blur radius: {gaussian_blur_radius:.1f} pixels")

    frame_directory.mkdir(parents=True, exist_ok=True)
    generated_frame_path = frame_directory / "00000.jpg"
    rgb_image.save(generated_frame_path, quality=100, subsampling=0)
    print(f"Generated SAM input frame: {generated_frame_path}")

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    print("Loading the SAM 3.1 video predictor...")
    predictor = build_sam3_predictor(
        checkpoint_path=str(checkpoint_path),
        version="sam3.1",
        compile=False,
        warm_up=False,
        max_num_objects=max_num_objects,
        multiplex_count=multiplex_count,
        use_fa3=use_flash_attention,
        use_rope_real=use_real_rope,
        async_loading_frames=False,
    )

    print("Starting a one-frame SAM session...")
    tracking_model = predictor.model
    inference_state = tracking_model.init_state(
        resource_path=str(frame_directory),
        offload_video_to_cpu=False,
        async_loading_frames=False,
    )
    print("SAM session is ready.")
    print("Drag with the left mouse button to draw a tight box around one nanotube")
    print("C: clear the box and mask")
    print("Enter: run SAM and save the result")

    bounding_box = []

    figure, axes = plt.subplots(figsize=figure_size)
    axes.imshow(rgb_array, cmap="gray")
    mask_artist = axes.imshow(np.zeros((*frame.shape, 4), dtype=np.float32))
    box_artist = Rectangle(
        (0, 0),
        0,
        0,
        fill=False,
        edgecolor=box_color,
        linewidth=box_line_width,
        visible=False,
    )
    axes.add_patch(box_artist)
    axes.set_title("Drag a tight box around one nanotube, then press Enter")
    axes.set_xlim(0, frame.shape[1])
    axes.set_ylim(frame.shape[0], 0)

    rectangle_selector = RectangleSelector(
        axes,
        partial(
            handle_box_selection,
            bounding_box=bounding_box,
            box_artist=box_artist,
        ),
        useblit=True,
        button=[1],
        minspanx=minimum_box_span,
        minspany=minimum_box_span,
        spancoords="pixels",
        interactive=True,
    )
    figure.canvas.mpl_connect(
        "key_press_event",
        partial(
            handle_key_press,
            axes=axes,
            tracking_model=tracking_model,
            inference_state=inference_state,
            bounding_box=bounding_box,
            box_artist=box_artist,
            mask_artist=mask_artist,
            output_directory=output_directory,
            frame_index=frame_index,
            output_probability_threshold=output_probability_threshold,
            mask_color=mask_color,
            mask_alpha=mask_alpha,
        ),
    )

    plt.show()
