# TV-LoRA

[English](README.md) | [简体中文](README_CN.md)

Implementation of **Interleaved Score-Based Diffusion and Structure-Constrained Optimization for Sparse-View CT Reconstruction**.

TV-LoRA interleaves a pretrained two-dimensional score-based diffusion model with ADMM updates for projection fidelity, three-dimensional anisotropic total variation, and slice-wise singular-value shrinkage. This repository provides reconstruction, evaluation, and visualization tools for LDCT and CTHD at 2, 4, and 8 projection views.

## 1. Installation

Clone the repository and create the environment:

```bash
git clone https://github.com/DENGZY8/TV-LoRA.git
cd TV-LoRA
conda env create -f environment.yml
conda activate tv-lora
```

The environment specifies Python 3.9 and PyTorch 2.8.0. For GPU reconstruction, use a PyTorch CUDA build compatible with your NVIDIA driver; installation options are listed in the [PyTorch documentation](https://pytorch.org/get-started/previous-versions/#v280). Check the active environment with:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

The default `native` backend runs with PyTorch operations on CPU or CUDA and requires no extension compilation. The optional `cuda` backend requires a compatible CUDA toolkit, C++ compiler, and `ninja`.

## 2. Download the pretrained model

LDCT and CTHD use the same frozen NCSN++ score model. No training or fine-tuning is required for reconstruction.

Download [checkpoint_185.pth](https://github.com/DENGZY8/TV-LoRA/releases/download/v1.0.0/checkpoint_185.pth) into `checkpoints/`:

```bash
mkdir -p checkpoints
curl --fail --location --retry 3 \
  --output checkpoints/checkpoint_185.pth \
  https://github.com/DENGZY8/TV-LoRA/releases/download/v1.0.0/checkpoint_185.pth
```

File size: **983,643,489 bytes**. SHA256:

```text
88f6a5b48dd8c0a21ebe948e15b693b6fa4e33e21b04d09ad12079efb77486de
```

Check the local file and its model and exponential-moving-average (EMA) tensors on CPU:

```bash
python tools/check_checkpoint.py \
  --checkpoint checkpoints/checkpoint_185.pth \
  --output checkpoint_info.json
```

The experiment profiles verify this SHA256 during loading. To use weights stored elsewhere, pass their exact path through `--checkpoint`.

## 3. Prepare the data

Dataset access:

- **LDCT:** [AAPM Low-Dose CT Grand Challenge](https://www.aapm.org/GrandChallenge/LowDoseCT/).
- **CTHD:** [CT Heart Dataset](https://www.kaggle.com/datasets/nikhilroxtomar/ct-heart-segmentation).

Obtain the datasets under their providers' access and usage conditions. Prepare preprocessed floating-point CT arrays on the **[0, 1] scale**, with **256 × 256 pixels per axial slice**. Store each volume as either:

- A directory of `.npy` slices with filenames encoding their numerical order, such as `000.npy`, `001.npy`, and so on.
- A single `.npy` volume with shape `[S, 256, 256]` or `[S, 1, 256, 256]`, where `S` is the slice count.

Preserve the anatomical slice order and use the same preprocessing for all compared methods. The loader expects preprocessed arrays; it does not convert DICOM files or resize images.

The provided profiles use the following complete volumes:

| Dataset | Slices | Example data directory | Experiment profile |
| --- | ---: | --- | --- |
| LDCT | 198 | `data/LDCT/` | `configs/experiments/ldct.json` |
| CTHD | 112 | `data/CTHD/` | `configs/experiments/cthd.json` |

The commands below explicitly pass these example data directories. Replace them with the paths to your own prepared volumes. Run commands from the repository root.

## 4. Reconstruct a volume

Before reconstruction, check the input, checkpoint, score-model forward pass, and projection operator:

```bash
python run.py --config configs/experiments/ldct.json \
  --data-dir data/LDCT --device cuda:0 --doctor

python run.py --config configs/experiments/cthd.json \
  --data-dir data/CTHD --device cuda:0 --doctor
```

Reconstruct the 8-, 4-, and 2-view settings sequentially:

```bash
python run.py --config configs/experiments/ldct.json \
  --data-dir data/LDCT --device cuda:0 --views 8 4 2

python run.py --config configs/experiments/cthd.json \
  --data-dir data/CTHD --device cuda:0 --views 8 4 2
```

Each profile uses 2,000 reverse steps, a reverse-diffusion predictor, one Langevin corrector step with SNR 0.16, and one ADMM correction with one CG recurrence per reverse step. The nominal slice batch size is 12; the TV and spectral weights are 0.04 and 0.5, respectively.

| Projection views | TV penalty | Spectral penalty |
| ---: | ---: | ---: |
| 8 | 15 | 1 |
| 4 | 15 | 5 |
| 2 | 15 | 1 |

Use `--views 8` to run one view setting. Other options include `--checkpoint`, `--output-dir`, `--batch-size`, `--seed`, and `--op-backend`. For all options, run `python run.py --help`.

The CTHD qualitative configuration is provided separately in `configs/experiments/cthd_qualitative.json`. It uses TV penalty 10 and spectral penalties 1, 3, and 3 for 8, 4, and 2 views, respectively, and saves results under `outputs/CTHD_qualitative/`.

To reconstruct and visualize this configuration:

```bash
python run.py --config configs/experiments/cthd_qualitative.json \
  --data-dir data/CTHD --device cuda:0 --views 8 4 2
python tools/plot_reconstruction.py --config configs/visualization/cthd_qualitative.json --output figures/CTHD_qualitative
```

### Run with tmux

Install `tmux` if needed, activate the environment, and launch:

```bash
bash scripts/tmux_run.sh configs/experiments/ldct.json 0 --data-dir data/LDCT
```

After LDCT finishes, launch CTHD on the same GPU:

```bash
bash scripts/tmux_run.sh configs/experiments/cthd.json 0 --data-dir data/CTHD
```

The second argument selects the physical GPU. To run both jobs concurrently on two GPUs, use `1` for CTHD. Each launcher prints its session name and log path.

```bash
tmux ls
tmux attach -t SESSION_NAME
```

Detach without stopping the job with **Ctrl+b**, then **d**. After completion, the log remains in `logs/`.

Add `--skip-completed` when relaunching to skip completed view settings with matching configurations and output hashes. An interrupted view restarts from its first diffusion step. Use `--overwrite` only when replacing existing results.

## 5. Outputs and evaluation

Results are saved under `outputs/LDCT/viewN/` and `outputs/CTHD/viewN/`, where `N` is 2, 4, or 8.

| File | Contents |
| --- | --- |
| `reconstruction.raw.npy` | Reconstruction before export rescaling |
| `reconstruction.npy` | Reconstruction after whole-volume min-max scaling |
| `reference.npy` | Reference after independent whole-volume min-max scaling |
| `measurement.npy` | Simulated sparse-view projections |
| `metrics.json` | Metrics on the exported reconstruction and reference |
| `metrics.raw.json` | Metrics before export rescaling |
| `run.json` | Runtime, timing, checkpoint identity, and completion status |
| `config.resolved.json` | Parameters used for the view setting |

Each dataset output directory also contains `results.csv`. PSNR, SSIM, and Dice are computed section-wise and averaged separately in the axial, coronal, and sagittal orientations. PSNR and SSIM use data range 1.0; SSIM uses a uniform 7 × 7 window. Dice uses intensity masks thresholded at values greater than 0.5, with an empty/empty score of zero.

Evaluate saved arrays independently:

```bash
python tools/evaluate.py \
  --reference outputs/LDCT/view8/reference.npy \
  --prediction outputs/LDCT/view8/reconstruction.npy \
  --output outputs/LDCT/view8/metrics_recomputed.json
```

## 6. Visualization

Generate reconstruction panels with axial, coronal, and sagittal rows and columns for 8 views, 4 views, 2 views, and ground truth:

```bash
python tools/plot_reconstruction.py --config configs/visualization/ldct.json --output figures/LDCT
python tools/plot_reconstruction.py --config configs/visualization/cthd.json --output figures/CTHD
```

The panels display view counts, section PSNR/SSIM, and magnified regions of interest. Set `indices` and `roi` in the visualization JSON files to select sections and regions. If using a different reconstruction output directory, update `reference` and `reconstructions` in the corresponding visualization configuration.

Generate absolute-error maps and the combined LDCT/CTHD PSNR and SSIM curves:

```bash
python tools/plot_error_maps.py --config configs/visualization/error_ldct.json --output figures/LDCT_errors
python tools/plot_error_maps.py --config configs/visualization/error_cthd.json --output figures/CTHD_errors
python tools/plot_curves.py --csv outputs/LDCT/results.csv outputs/CTHD/results.csv --output figures/LDCT_CTHD_curves
```

To include baseline error maps, edit `configs/visualization/error_baselines.example.json` with paths to matched reconstruction arrays and pass that file to `plot_error_maps.py`. All methods must use the same reference, slice order, and intensity scale.

To include baseline curves, supply CSV files with columns `dataset,method,views,plane,psnr,ssim,dice`. Each method requires results for 2, 4, and 8 views and all three orientations within each dataset it appears in. Curve points are equal-weight averages of the three orientation means.

Figures are exported as PNG, PDF, and SVG, together with a JSON record of their source arrays and plotting settings.

## 7. Tests

Run the unit and synthetic integration tests:

```bash
python -m pip install pytest
python -m pytest -q tests
```

## 8. License and acknowledgments

See [LICENSE](LICENSE) and [NOTICE](NOTICE). The score-model, SDE, sampling, and CT components build on [DiffusionMBIR](https://github.com/hyungjin-chung/DiffusionMBIR) and [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch).
