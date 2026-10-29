# Model weights

The model architecture (`models/`) and model configuration are included in Git. The pretrained checkpoint is a separate binary asset. Both LDCT and CTHD profiles use the same checkpoint-185 prior.

## Download

Download [checkpoint_185.pth](https://github.com/DENGZY8/TV-LoRA/releases/download/v1.0.0/checkpoint_185.pth) from [release v1.0.0](https://github.com/DENGZY8/TV-LoRA/releases/tag/v1.0.0). Run from the repository root:

```bash
mkdir -p checkpoints
curl --fail --location --retry 3 --output checkpoints/checkpoint_185.pth https://github.com/DENGZY8/TV-LoRA/releases/download/v1.0.0/checkpoint_185.pth
```

The following identity fields are reported by the [GitHub release API](https://api.github.com/repos/DENGZY8/TV-LoRA/releases/tags/v1.0.0), inspected on 2026-10-29:

| Field | Value |
| --- | --- |
| Checkpoint | `checkpoint_185.pth` |
| Release tag | `v1.0.0` |
| File size | 983,643,489 bytes |
| SHA256 | `88f6a5b48dd8c0a21ebe948e15b693b6fa4e33e21b04d09ad12079efb77486de` |
| Local path | `checkpoints/checkpoint_185.pth` |
| Model configuration | `configs/ve/AAPM_256_ncsnpp_continuous.py` |
| EMA | Applied by the loader after model-state restoration |

All supplied experiment profiles set `checkpoint_sha256` to this digest. If a different checkpoint is deliberately used, verify its contents and update the profile digest explicitly.

## Validate the local file

The checkpoint should contain `model`, `ema`, and optionally `optimizer` and `step`. EMA shadow parameters correspond to trainable parameters only. The Fourier-embedding parameter with `requires_grad=False` comes from the model state. The `sigmas` buffer is regenerated from the model configuration and requested number of reverse steps.

```bash
python tools/check_checkpoint.py --checkpoint checkpoints/checkpoint_185.pth --output checkpoint_info.json
```

This command records the local file's SHA256 and size and checks the model and EMA tensors on CPU. A filename can include or omit `.pth`; the bytes and architecture checks determine validity. A directory containing multiple plausible checkpoint files must be disambiguated explicitly.

The release metadata was retrieved without downloading the full binary. Its reported digest does not establish that the checkpoint architecture, training history, or clinical results have been independently validated. Run the local check and `run.py --doctor` on the target server before reconstruction.

## Distribution and upstream provenance

The checkpoint is distributed through GitHub Releases. GitHub blocks ordinary Git files larger than 100 MiB; browser uploads are limited to 25 MiB per file. See [GitHub's file-size documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

The upstream [DiffusionMBIR repository](https://github.com/hyungjin-chung/DiffusionMBIR) links pretrained CT weights. Its downloadable file has not been byte-compared to the author's checkpoint in this review. Do not describe the two as identical without checking their hashes and contents.
