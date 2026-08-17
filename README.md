# Tube tracking

Microscopy data-analysis project for testing Meta SAM 3.1 video segmentation on DNA nanotubes.

The documented environment was verified on Windows with:

- Python 3.12.13
- NVIDIA GeForce RTX 3060 Laptop GPU with 6 GB VRAM
- NVIDIA driver 581.57
- PyTorch 2.10.0 with the CUDA 12.8 runtime
- SAM 3.1 commit `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`

The PyTorch wheel includes the required CUDA runtime. A separate CUDA Toolkit installation is not required for this setup. A compatible NVIDIA driver is still required.

Install Anaconda, Git, and a current NVIDIA display driver before creating the environment. Git is required because the tested SAM version is installed from an exact source commit.

## Create the environment

These commands use Anaconda's Command Prompt or a VS Code terminal configured as Command Prompt. Replace the Anaconda path if it is installed elsewhere.

```cmd
C:\Users\Flori\anaconda3\Scripts\activate
conda create --name tube-tracking python=3.12.13 pip -y
conda activate tube-tracking
```

The next command installs several gigabytes, including CUDA-enabled PyTorch. Run it manually and allow it to finish:

```cmd
python -m pip install -r requirements-windows-cu128.txt
```

Verify dependency consistency and CUDA access:

```cmd
python -m pip check
python sam3_cuda_smoke_test.py
```

The CUDA smoke test must report the NVIDIA GPU and complete the CUDA tensor calculation.

## Download the gated SAM 3.1 checkpoint

1. Sign in to Hugging Face and request access to <https://huggingface.co/facebook/sam3.1>.
2. After Meta approves the request, authenticate the command-line client:

```cmd
hf auth login
hf auth whoami
```

3. Create the checkpoint directory and manually download the approximately 3.5 GB checkpoint:

```cmd
mkdir checkpoints
hf download facebook/sam3.1 sam3.1_multiplex.pt config.json --local-dir checkpoints
```

The expected files are:

```text
checkpoints\sam3.1_multiplex.pt
checkpoints\config.json
```

The entire `checkpoints/` directory is excluded by `.gitignore`. Do not commit or redistribute Meta's model weights.

## Test the complete model load

```cmd
python sam3_model_smoke_test.py
```

The verified RTX 3060 result was:

```text
Loaded predictor: Sam3MultiplexVideoPredictor
Current CUDA allocation: 3.65 GiB
Peak CUDA allocation: 3.65 GiB
```

The current upstream builder prints a large intermediate `Missing keys` diagnostic while it assembles the internal tracker from the merged checkpoint. The final predictor load is successful when the script reaches the three lines above and exits with code 0. Warnings about `pkg_resources` and `timm.models.layers` are also emitted by the current upstream SAM dependencies.

Model loading alone does not guarantee that a complete video inference workload will fit in 6 GB VRAM. Frame resolution, stored video state, and the number of objects will add memory usage.

## TIFF viewer

Run the existing interactive test-data viewer with:

```cmd
python welcome.py
```

Files under `raw_data/` and `test_data/` are read-only inputs.

## Interactive first-frame SAM test

Run the box-prompt experiment with:

```cmd
python tests\segment_first_frame.py
```

The script reads frame 0 from `test_data\test_1.tif`, writes a contrast-scaled one-frame SAM input under `test_outputs\`, and loads the SAM 3.1 video predictor. In the displayed frame:

- Hold the left mouse button and drag a tight box around one complete nanotube.
- Press `C` to clear the box and mask.
- Press Enter to run SAM and save the mask and overlay under `test_outputs\sam3_first_frame_smoothed\`.

The displayed and model input image uses a 1.5-pixel Gaussian blur after contrast scaling. Change the explicit `gaussian_blur_radius` parameter to compare other values. The source TIFF is never modified. An empty SAM result raises an error instead of silently saving an empty mask.

Run the synthetic circle control with:

```cmd
python tests\segment_synthetic_circle.py
```

This generates a known gray circle, prompts it with a tight box, and reports intersection over union against the exact reference mask. The verified result was an IoU of 0.997. Setting `prompt_type = "point"` in the script reproduces the currently empty SAM 3.1 point-prompt result.

## Text and short tracking evaluations

Run the text-only comparison with:

```cmd
python tests\test_text_prompts.py
```

This tests `filament`, `fiber`, `thin rod`, `bright line`, and `fluorescent filament` on the same smoothed first frame. With an output threshold of 0.2, the current SAM 3.1 checkpoint returned zero masks for every phrase.

Run the ten-frame exact-object propagation test with:

```cmd
python tests\test_ten_frame_propagation.py
```

The script uses one frame-0 box to obtain the initial SAM 3 mask, converts its centroid into an exact-object refinement prompt, and propagates that object through ten frames. In the verified run, frame 0 produced a plausible but overly thick mask and the tracker returned no mask for frames 1 through 9. The result is therefore a failed tracking test, not a working pipeline.

PyTorch for Windows does not provide the Flash Attention kernel that SAM 3.1 exclusively requests during this propagation path. The test applies a process-local PyTorch math-attention fallback and does not modify the installed SAM package. This matches the upstream diagnosis in [facebookresearch/sam3 issue 523](https://github.com/facebookresearch/sam3/issues/523). On the tested RTX 3060 laptop GPU, the fallback completed ten-frame exact-object propagation in about 85 seconds after model initialization.

## Troubleshooting

### Pip repeatedly contacts `pypi.ngc.nvidia.com`

Inspect every active pip configuration source:

```cmd
python -m pip config debug
```

If the user configuration contains the obsolete NVIDIA extra index, remove only those entries:

```cmd
python -m pip config --user unset global.extra-index-url
python -m pip config --user unset global.trusted-host
```

Confirm that they are gone:

```cmd
python -m pip config list
```

### PowerShell blocks Conda activation scripts

Use Command Prompt as the VS Code terminal profile and activate Conda with:

```cmd
C:\Users\Flori\anaconda3\Scripts\activate
conda activate tube-tracking
```

This avoids changing the Windows PowerShell execution policy.

### SAM import fails after installation

Do not install unpinned replacements one at a time. Reapply the tested dependency file:

```cmd
python -m pip install -r requirements-windows-cu128.txt
python -m pip check
```
