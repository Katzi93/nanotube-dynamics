"""Compare text-only SAM 3.1 prompts on the first microscopy frame."""

from pathlib import Path

import matplotlib.pyplot as plt
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
    return np.repeat(grayscale_uint8[:, :, np.newaxis], 3, axis=2)


def run_text_prompt(
    tracking_model,
    inference_state,
    text_prompt,
    frame_index,
    output_probability_threshold,
):
    """Run one text prompt and return all masks, scores, and boxes."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        returned_frame_index, outputs = tracking_model.add_prompt(
            inference_state=inference_state,
            frame_idx=frame_index,
            text_str=text_prompt,
            points=None,
            point_labels=None,
            boxes_xywh=None,
            box_labels=None,
            output_prob_thresh=output_probability_threshold,
        )

    assert returned_frame_index == frame_index
    return (
        np.asarray(outputs["out_binary_masks"]),
        np.asarray(outputs["out_probs"]),
        np.asarray(outputs["out_boxes_xywh"]),
    )


def save_prompt_result(
    rgb_image,
    masks,
    scores,
    text_prompt,
    prompt_index,
    output_directory,
    mask_alpha,
):
    """Save masks and a colored overlay for one text prompt."""
    prompt_directory = output_directory / f"{prompt_index:02d}_{text_prompt.replace(' ', '_')}"
    prompt_directory.mkdir(parents=True, exist_ok=True)

    color_map = plt.get_cmap("tab10")
    overlay = np.zeros((*rgb_image.shape[:2], 4), dtype=np.float32)
    for mask_index, mask in enumerate(masks):
        color = color_map(mask_index % 10)[:3]
        overlay[mask, :3] = color
        overlay[mask, 3] = mask_alpha
        tifffile.imwrite(
            prompt_directory / f"mask_{mask_index:02d}.tif",
            mask.astype(np.uint8),
            photometric="minisblack",
        )

    figure, axes = plt.subplots(figsize=(10, 10))
    axes.imshow(rgb_image)
    axes.imshow(overlay)
    axes.set_title(
        f"Text: {text_prompt!r} | masks: {len(masks)} | "
        f"scores: {np.round(scores, 3).tolist()}"
    )
    axes.axis("off")
    figure.savefig(prompt_directory / "overlay.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    # Read-only microscopy TIFF stack used for this experiment.
    input_path = Path("test_data/test_1.tif")
    # Local SAM 3.1 checkpoint downloaded from Hugging Face.
    checkpoint_path = Path("checkpoints/sam3.1_multiplex.pt")
    # Directory for generated frames and text-prompt results.
    output_directory = Path("test_outputs/sam3_text_prompts")
    # Directory exposed to SAM as a one-frame video.
    frame_directory = output_directory / "input_frames"
    # Zero-based TIFF frame tested by every text prompt.
    frame_index = 0
    # Short concepts compared without any spatial prompt.
    text_prompts = (
        "filament",
        "fiber",
        "thin rod",
        "bright line",
        "fluorescent filament",
    )
    # Lower percentile mapped to black during display scaling.
    contrast_low_percentile = 1.0
    # Upper percentile mapped to white during display scaling.
    contrast_high_percentile = 99.8
    # Gaussian blur radius applied after contrast scaling.
    gaussian_blur_radius = 1.5
    # Minimum text-conditioned object probability retained in the output.
    output_probability_threshold = 0.2
    # Maximum number of text-detected objects retained by the predictor.
    max_num_objects = 16
    # Native object bucket size encoded by the SAM 3.1 checkpoint.
    multiplex_count = 16
    # Avoid the optional Flash Attention 3 dependency on the laptop GPU.
    use_flash_attention = False
    # Use the checkpoint's native complex-valued rotary position encoding.
    use_real_rope = False
    # Allow faster Ampere matrix operations during inference.
    allow_tf32 = True
    # Opacity used when displaying multiple predicted masks.
    mask_alpha = 0.45

    assert input_path.is_file(), f"Missing input TIFF: {input_path}"
    assert checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}"
    assert torch.cuda.is_available(), "PyTorch cannot access a CUDA GPU"

    print(f"Reading frame {frame_index} from {input_path}...")
    frame = tifffile.imread(input_path, key=frame_index)
    assert frame.ndim == 2, f"Expected a grayscale frame, got shape {frame.shape}"

    rgb_array = scale_grayscale_to_rgb(
        frame, contrast_low_percentile, contrast_high_percentile
    )
    rgb_image = Image.fromarray(rgb_array).filter(
        ImageFilter.GaussianBlur(radius=gaussian_blur_radius)
    )
    rgb_array = np.asarray(rgb_image)

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
    tracking_model = predictor.model

    for prompt_index, text_prompt in enumerate(text_prompts):
        print(f"Testing text prompt {text_prompt!r}...")
        inference_state = tracking_model.init_state(
            resource_path=str(frame_directory),
            offload_video_to_cpu=False,
            async_loading_frames=False,
        )
        masks, scores, boxes = run_text_prompt(
            tracking_model,
            inference_state,
            text_prompt,
            frame_index,
            output_probability_threshold,
        )
        assert masks.ndim == 3, f"Unexpected mask shape: {masks.shape}"
        assert len(masks) == len(scores) == len(boxes)

        print(f"Prompt {text_prompt!r}: {len(masks)} masks")
        print(f"Scores: {np.round(scores, 3).tolist()}")
        save_prompt_result(
            rgb_array,
            masks,
            scores,
            text_prompt,
            prompt_index,
            output_directory,
            mask_alpha,
        )
    print(f"Saved text-prompt comparisons under {output_directory}")
