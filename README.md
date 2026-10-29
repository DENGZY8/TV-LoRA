# TV-LoRA

Sparse-view CT reconstruction with a frozen two-dimensional NCSN++ score prior, three-axis anisotropic total variation, and slice-wise singular-value shrinkage interleaved through ADMM.

Repository: <https://github.com/DENGZY8/TV-LoRA>

This repository contains the TV-LoRA reconstruction and evaluation code for LDCT and CTHD. The supplied core sampler is retained; the command-line interface makes dataset paths, checkpoint selection, and reconstruction parameters explicit. See [中文运行说明](README_CN.md).

## Installation

The supplied server snapshot reports Python 3.9.0, PyTorch 2.8.0, and two RTX 4090 GPUs. An existing working environment can be reused:

```bash
conda activate mypy
python -m pip install -r requirements.txt
```

For a separate environment, run from this repository directory:

```bash
conda env create -f environment.yml
conda activate tv-lora
```

Choose the PyTorch 2.8.0 CUDA or CPU wheel appropriate to the system using the [official installation commands](https://pytorch.org/get-started/previous-versions/#v280). The `nvidia-smi` CUDA field alone does not identify the CUDA runtime packaged with PyTorch. Inspect it with `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`.

The default `native` resampling backend uses ordinary PyTorch operations on CPU or CUDA and does not compile an extension. The optional `cuda` backend uses the supplied legacy CUDA kernel and requires a matching CUDA toolkit, compiler, and `ninja` package. The backend is recorded for each run.

## Data

Dataset pages:

- [AAPM 2016 Low-Dose CT Grand Challenge](https://www.aapm.org/GrandChallenge/LowDoseCT/).
- [CT Heart Dataset](https://www.kaggle.com/datasets/nikhilroxtomar/ct-heart-segmentation).

Obtain data under the original providers' access and usage conditions. This release expects the **already preprocessed, numerically ordered 256 x 256 slices** used by the reconstruction pipeline, stored as real-valued `.npy` arrays on the [0,1] scale. It also accepts one stacked `[S,H,W]` or `[S,1,H,W]` `.npy` volume. No DICOM-to-HU conversion, resizing, clipping, or input normalization is silently applied. Keep the original slice order and preprocessing record. Raw dataset downloads alone are not a substitute for the preprocessing used in an experiment.

The example profiles expect 198 slices for LDCT and 112 slices for CTHD. They check the complete volume and do not truncate it. Use `--data-dir` to supply an actual local path.

## Pretrained score checkpoint

Both dataset profiles use the same frozen AAPM NCSN++ score prior, identified in the supplied entry point as checkpoint 185. Reconstruction does not train or fine-tune this network. The model architecture is included under `models/`; pretrained weights are separate.

Place the original checkpoint at `checkpoints/checkpoint_185.pth`, or pass its exact filename with `--checkpoint`. A filename without `.pth` is accepted. The loader checks model keys, tensor shapes, EMA parameter count, and finite values. Only trainable parameters are paired with the saved EMA list; the frozen Fourier-embedding parameter is restored from the model state.

```bash
python tools/check_checkpoint.py --checkpoint /path/to/checkpoint_185 --output checkpoint_info.json
```

The report records the file's SHA256 and size. Copy the SHA256 into `checkpoint_sha256` in the experiment profile to pin a specific file. See [weight distribution](docs/model_weights.md). A verified public download link and hash for the particular experiment checkpoint have not yet been supplied with this code package.

## Reconstruction

Run the setup check before a long reconstruction:

```bash
python run.py --config configs/experiments/ldct.json \
  --data-dir /path/to/LDCT/slices --checkpoint /path/to/checkpoint_185 \
  --device cuda:0 --doctor
```

Then reconstruct the three view settings:

```bash
python run.py --config configs/experiments/ldct.json \
  --data-dir /path/to/LDCT/slices --checkpoint /path/to/checkpoint_185 \
  --device cuda:0 --views 8 4 2

python run.py --config configs/experiments/cthd.json \
  --data-dir /path/to/CTHD/slices --checkpoint /path/to/checkpoint_185 \
  --device cuda:0 --views 8 4 2
```

The primary profiles use:

| Views | TV penalty | Spectral penalty | TV weight | Spectral weight |
| --- | --- | --- | --- | --- |
| 8 | 15 | 1 | 0.04 | 0.5 |
| 4 | 15 | 5 | 0.04 | 0.5 |
| 2 | 15 | 1 | 0.04 | 0.5 |

Each run uses 2000 reverse steps, one reverse-diffusion predictor, one Langevin corrector at SNR 0.16, and one ADMM correction with one CG recurrence. Nominal slice batch size is 12. The original slice partition function is retained. `seed` controls new runs; saved historical stochastic reconstructions need not be bitwise identical to newly seeded runs.

`configs/experiments/cthd_legacy_visual.json` reproduces the **parameter choices** of the saved CTHD qualitative pipeline: TV penalty 10 and spectral penalty 1, 3, 3 at 8, 4, 2 views. Its outputs have their own directory. The primary profile uses the table above. There is no automatic search for or selection of a result based on its measured score.

Relative paths are resolved against the repository root. Override paths with command-line arguments instead of editing source files. `python run.py --config configs/experiments/ldct.json --check-config` prints the resolved profile without loading the checkpoint. The retained filename `generator_AAPM_3d_sparseview.py` accepts the same arguments as `run.py`.

### tmux

From the activated Python environment:

```bash
bash scripts/tmux_run.sh configs/experiments/ldct.json 0 \
  --data-dir /path/to/LDCT/slices --checkpoint /path/to/checkpoint_185

bash scripts/tmux_run.sh configs/experiments/cthd.json 1 \
  --data-dir /path/to/CTHD/slices --checkpoint /path/to/checkpoint_185
```

The second argument selects the physical GPU through `CUDA_VISIBLE_DEVICES`. The launcher captures the activated Python executable and prints the session name and log path. Attach with `tmux attach -t SESSION`; detach with **Ctrl+b**, then **d**. A completed session closes and its log remains. `--skip-completed` skips completed, hash-matched view outputs; it does not resume an interrupted diffusion trajectory. Keep the same nominal batch size when comparing numerical runs.

## Outputs and metrics

Each `outputs/DATASET/viewN/` directory contains:

- `reconstruction.raw.npy`: sampler output before export rescaling.
- `reconstruction.npy`, `reference.npy`: independently min-max normalized **whole volumes**, matching the supplied legacy volume-export processing.
- `measurement.npy`: the simulated projection array.
- `metrics.json`: metrics on the exported floating-point volumes.
- `metrics.raw.json`: metrics on the raw output and original input scale.
- `run.json`, `config.resolved.json`: parameters, slice ordering, hashes, checkpoint identity, timing boundaries, runtime, and completion status.

`results.csv` contains orientation-specific metrics for the requested views. PSNR and SSIM use data range 1.0; SSIM uses a uniform 7 x 7 window. Each orientation reports the mean of its section scores. Dice is intensity-thresholded overlap at values greater than 0.5, with empty/empty Dice equal to zero. It is not evaluation against anatomical segmentation annotations. Evaluation does not apply additional clipping or maximum-based rescaling.

```bash
python tools/evaluate.py --reference outputs/LDCT/view8/reference.npy \
  --prediction outputs/LDCT/view8/reconstruction.npy --output metrics.json
```

Reconstruction time covers the sampler call with CUDA synchronization at both boundaries. `view_wall_seconds` also includes projection, export, and evaluation. New release timings should be measured with their recorded backend and timing boundary.

## Figures

Draw matched axial, coronal, and sagittal sections with 8/4/2-view columns and one ground-truth column:

```bash
python tools/plot_reconstruction.py --config configs/visualization/ldct.json --output figures/LDCT
python tools/plot_reconstruction.py --config configs/visualization/cthd.json --output figures/CTHD
```

The panels show view count and section PSNR/SSIM, without repeated method names. Yellow boxes and insets use shared, configurable ROIs. A single font family and display window are used. Default indices select the central section in each orientation; set `indices` and `roi` explicitly for a chosen anatomical example. Figure metadata records the source hashes and section selection.

```bash
python tools/plot_error_maps.py --config configs/visualization/error_ldct.json --output figures/LDCT_errors
python tools/plot_error_maps.py --config configs/visualization/error_cthd.json --output figures/CTHD_errors
python tools/plot_curves.py --csv outputs/LDCT/results.csv outputs/CTHD/results.csv --output figures/LDCT_CTHD_curves
```

For numeric baseline error maps, edit `configs/visualization/error_baselines.example.json` to point to actual matched reconstruction arrays. The template includes Chung and Lahiri. These tools do not replace missing baseline outputs with example values. See [the original Fig. 5 rendering path](docs/paper_fig5.md) for the saved ADMM-TV/FBPConvNet image panels retained in the manuscript.

The curve tool accepts measured CSV rows with columns `dataset,method,views,plane,psnr,ssim,dice`. Add baseline rows or CSV files to include their curves. Every point requires three orientation means; duplicate records and incomplete curves are rejected.

## Validation and implementation

```bash
python -m pip install pytest
python -m pytest -q tests
```

The package includes tests of finite-difference adjoints, the slice SVT map, EMA restoration, input ordering, metrics, a small NCSN++ forward pass, and a short synthetic CT trajectory. Real-checkpoint and full-volume GPU validation require the original weight file and local input data; run `--doctor` on the target server.

For tensors `[S,1,H,W]`, the original SVT code decomposes one-row matrices of size `1 x HW`. Its proximal map is therefore slice-wise group shrinkage. The CT backprojector and the one-step CG recurrence are retained. See [implementation details](docs/implementation.md).

## Acknowledgments and references

The score-model, SDE, sampling, and CT implementation build on [DiffusionMBIR](https://github.com/hyungjin-chung/DiffusionMBIR) and [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch). Original copyright notices and the supplied Apache-2.0 license are retained. See [NOTICE](NOTICE).

- H. Chung, D. Ryu, M. T. McCann, M. L. Klasky, J. C. Ye. *Solving 3D Inverse Problems using Pre-trained 2D Diffusion Models*. CVPR, 2023.
- Y. Song et al. *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR, 2021.
