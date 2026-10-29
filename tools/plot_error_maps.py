#!/usr/bin/env python3
"""One error-map figure using matched numeric reconstructions and one scale."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--indices', nargs=3, type=int)
    p.add_argument('--error-max', type=float, default=0.35)
    p.add_argument('--dpi', type=int, default=400)
    args = p.parse_args()
    if args.error_max <= 0:
        raise ValueError('error-max must be positive')
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from tvlora.figures import style, read_figure_config, section, save_figure, PLANES
    cfg, reference, arrays, sources = read_figure_config(args.config, args.indices)
    style()
    names = list(arrays)
    ratios = [1, reference.shape[0] / reference.shape[1], reference.shape[0] / reference.shape[2]]
    fig, axes = plt.subplots(3, len(names) + 1, squeeze=False,
        figsize=(7.2, max(2.6, 7.2 / (len(names) + 1) * sum(ratios) + 0.55)),
        gridspec_kw={'height_ratios': ratios, 'wspace': 0.025, 'hspace': 0.035})
    fig.subplots_adjust(left=0.045, right=0.995, top=0.91, bottom=0.16)
    records = []
    for axis, plane in enumerate(PLANES):
        gt = section(reference, axis, cfg['indices'][axis])
        axes[axis, 0].imshow(gt, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[axis, 0].set_ylabel(plane.capitalize(), fontsize=8)
        for column, name in enumerate(names, 1):
            error = np.abs(section(arrays[name], axis, cfg['indices'][axis]).astype(np.float64) - gt)
            axes[axis, column].imshow(error, cmap='magma', vmin=0, vmax=args.error_max, interpolation='nearest')
            records.append({'plane': plane, 'method': name, 'section_mae': float(error.mean()),
                            'fraction_above_display_max': float(np.mean(error > args.error_max))})
        for ax in axes[axis]:
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#d8dde2'); spine.set_linewidth(0.3)
    for ax, label in zip(axes[0], ['Reference'] + names):
        ax.set_title(label, fontsize=8.0, pad=5, fontweight='bold' if label == 'TV-LoRA' else 'normal')
    cax = fig.add_axes([0.24, 0.075, 0.54, 0.025])
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(0, args.error_max), cmap='magma'),
                           cax=cax, orientation='horizontal')
    colorbar.set_label('Absolute error', fontsize=8, labelpad=2)
    colorbar.ax.tick_params(labelsize=7, length=2, pad=2)
    save_figure(fig, args.output, {'dataset': cfg['dataset'], 'views': cfg.get('views'),
        'indices': cfg['indices'], 'sources': sources, 'reference_window': [0, 1],
        'common_error_window': [0, args.error_max], 'section_statistics': records,
        'errors_computed_from_numeric_arrays': True}, args.dpi)


if __name__ == '__main__':
    main()
