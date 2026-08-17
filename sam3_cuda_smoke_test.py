"""Confirm that SAM 3 imports and PyTorch can execute on the NVIDIA GPU."""

import sam3
import torch


if __name__ == "__main__":
    assert torch.cuda.is_available(), "PyTorch cannot access a CUDA GPU"

    device = torch.device("cuda")
    values = torch.tensor([1.0, 2.0, 3.0], device=device)
    doubled_values = values * 2

    print(f"SAM 3: {sam3.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(device)}")
    print(f"CUDA tensor result: {doubled_values.tolist()}")
