#!/usr/bin/env python3
"""Render a matched 3x4 grid: 8, 4, 2 views, and reference."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True, help='Output stem without extension')
    p.add_argument('--indices', nargs=3, type=int, metavar=('AXIAL', 'CORONAL', 'SAGITTAL'))
    p.add_argument('--dpi', type=int, default=400)
    args = p.parse_args()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.patheffects as pe
    from tvlora.figures import style, read_figure_config, section, save_figure, PLANES
    from tvlora.evaluation import section_metrics
    cfg, reference, arrays, sources = read_figure_config(args.config, args.indices)
    if set(arrays) != {'8', '4', '2'}:
        raise ValueError('Reconstruction figure requires exactly the 8, 4, and 2 view volumes')
    style()
    row_ratios = [1, reference.shape[0] / reference.shape[1], reference.shape[0] / reference.shape[2]]
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 1.8 * sum(row_ratios)),
                             gridspec_kw={'height_ratios': row_ratios, 'wspace': 0.012, 'hspace': 0.012})
    records = []
    for axis, plane in enumerate(PLANES):
        index = cfg['indices'][axis]
        gt = section(reference, axis, index)
        roi = cfg.get('roi', {}).get(plane, [0.32, 0.32, 0.16, 0.16])
        h, w = gt.shape
        x, y, rw, rh = [int(round(v * n)) for v, n in zip(roi, (w, h, w, h))]
        if min(rw, rh) < 2 or x < 0 or y < 0 or x + rw > w or y + rh > h:
            raise ValueError('ROI lies outside the displayed ' + plane + ' section')
        for column, view in enumerate(('8', '4', '2', None)):
            a = section(arrays[view], axis, index) if view else gt
            ax = axes[axis, column]
            ax.imshow(a, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if view:
                metrics = section_metrics(gt, a)
                text = '%s views\n%.2f / %.3f' % (view, metrics['psnr'], metrics['ssim'])
                records.append(dict(plane=plane, view=int(view), index=index, **metrics))
            else:
                text = 'Ground truth'
            ax.text(0.025, 0.975, text, transform=ax.transAxes, ha='left', va='top',
                    fontsize=7.8, color='white', linespacing=1.15,
                    path_effects=[pe.withStroke(linewidth=1.1, foreground='black')])
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), rw, rh, fill=False, edgecolor='#ffff35', linewidth=0.7))
            inset = ax.inset_axes([0.70, 0.0, 0.30, 0.30])
            inset.imshow(a[y:y+rh, x:x+rw], cmap='gray', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
            inset.set_xticks([]); inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_color('#ffff35'); spine.set_linewidth(0.75)
    save_figure(fig, args.output, {'dataset': cfg['dataset'], 'indices': cfg['indices'],
        'sources': sources, 'section_metrics': records, 'display_window': [0, 1],
        'method_names_inside_panels': False, 'roi': cfg.get('roi'), 'font': 'DejaVu Sans'}, args.dpi)


if __name__ == '__main__':
    main()
