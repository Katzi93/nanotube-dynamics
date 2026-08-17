"""Confirm that the local SAM 3.1 checkpoint loads on the CUDA GPU."""

from pathlib import Path

import torch
from sam3.model_builder import build_sam3_predictor


if __name__ == "__main__":
    # Path to the downloaded SAM 3.1 model checkpoint.
    checkpoint_path = Path("checkpoints/sam3.1_multiplex.pt")
    # Model release to construct for the downloaded checkpoint.
    model_version = "sam3.1"
    # Number of objects exposed by the smoke-test predictor.
    max_num_objects = 1
    # Native object bucket size encoded by the SAM 3.1 checkpoint.
    multiplex_count = 16
    # Keep compilation disabled during the first model-load test.
    compile_model = False
    # Avoid the optional Flash Attention 3 dependency on the laptop GPU.
    use_flash_attention = False
    # Use the checkpoint's native complex-valued rotary position encoding.
    use_real_rope = False

    assert checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}"
    assert torch.cuda.is_available(), "PyTorch cannot access a CUDA GPU"

    print(f"Loading {checkpoint_path} on {torch.cuda.get_device_name(0)}...")
    torch.cuda.reset_peak_memory_stats()

    predictor = build_sam3_predictor(
        checkpoint_path=str(checkpoint_path),
        version=model_version,
        compile=compile_model,
        warm_up=False,
        max_num_objects=max_num_objects,
        multiplex_count=multiplex_count,
        use_fa3=use_flash_attention,
        use_rope_real=use_real_rope,
        async_loading_frames=False,
    )

    torch.cuda.synchronize()
    allocated_gib = torch.cuda.memory_allocated() / 1024**3
    peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3

    print(f"Loaded predictor: {type(predictor).__name__}")
    print(f"Current CUDA allocation: {allocated_gib:.2f} GiB")
    print(f"Peak CUDA allocation: {peak_allocated_gib:.2f} GiB")
