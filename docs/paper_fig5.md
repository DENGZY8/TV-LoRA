# Saved Fig. 5 rendering

`tools/paper_fig5/error_map_inputs.py` and `tools/paper_fig5/plot_error_maps.py` are the previously delivered manuscript rendering scripts. They are retained without changes. They use the saved numerical sections for TV_z, Chung, Lahiri, and TV-LoRA, and retain the original colored ADMM-TV/FBPConvNet panels from an earlier figure.

The exact rendering requires the corresponding original input bundle with:

- `error_sections/fig5_sections.npz` and `fig5_sources.json`;
- `source_figures/ldct_error_maps_view8_original.png`.

These image/section assets are not included in this source-only package. With the original bundle available:

```bash
python tools/paper_fig5/plot_error_maps.py \
  --base /path/to/Fig5_Update_20261028 --output figures/paper_fig5
```

The script checks the original source image hash before cropping. It preserves the two numeric calibrations used for the legacy colored panels and the four matched numerical methods. The RGB panels are not converted back into scalar reconstruction errors.

For a new figure with one common numerical error scale, use `tools/plot_error_maps.py` and supply the actual saved `.npy` reconstruction arrays for every included method. Edit `configs/visualization/error_baselines.example.json` with their true paths. An unavailable baseline volume must not be replaced with another method's reconstruction or inferred from a colored screenshot.

The reconstruction-panel tools have been generalized from the earlier visualization workflow to require explicit LDCT/CTHD inputs. They do not retain the old LIDC-specific hashes, automatic result ranking, or server-specific search paths.
