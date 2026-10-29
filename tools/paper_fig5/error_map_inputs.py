#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw Fig. 5 using only exported methods with verified matching references."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

METHODS = ('TV_z', 'Chung', 'Lahiri', 'TV-LoRA')
AVAILABLE_NAMES = ('ADMM-TV', 'FBPConvNet', 'TV_z', 'Chung', 'Lahiri', 'TV-LoRA')
PLANES = ('axial', 'coronal', 'sagittal')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font_setup():
    for folder in (Path(__file__).resolve().parent / 'fonts',
                   Path(__file__).resolve().parent.parent / 'fonts'):
        for path in sorted(folder.glob('*.ttf')):
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8,
        'mathtext.fontset': 'dejavusans', 'mathtext.default': 'regular',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.unicode_minus': False, 'savefig.facecolor': 'white',
    })


def compact_layout(fig, headers, ratios, title=None):
    """Physical dimensions in inches; every panel has the same native x scale."""
    width, height = fig.get_size_inches()
    left, right, top, bottom = 0.25, 0.15, 0.38, 0.52
    gap, ref_gap, row_gap = 0.025, 0.105, 0.055
    if title:
        top += 0.16
    ncols = len(headers)
    cell = (width - left - right - ref_gap - (ncols - 2) * gap) / ncols
    row_heights = [cell * r for r in ratios]
    needed = top + bottom + sum(row_heights) + 2 * row_gap
    fig.set_size_inches(width, needed, forward=True)
    height = needed
    x_positions = [left]
    for i in range(1, ncols):
        x_positions.append(x_positions[-1] + cell + (ref_gap if i == 1 else gap))
    y = height - top
    axes = []
    for r, row_height in enumerate(row_heights):
        y -= row_height
        row = [fig.add_axes([x / width, y / height, cell / width, row_height / height])
               for x in x_positions]
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#c9cdd1')
                spine.set_linewidth(0.35)
        axes.append(row)
        fig.text(0.10 / width, (y + row_height / 2) / height,
                 PLANES[r].capitalize(), rotation=90, ha='center', va='center',
                 fontsize=8.5, color='#252c33')
        y -= row_gap
    for c, name in enumerate(headers):
        label = r'$\mathrm{TV}_z$' if name == 'TV_z' else name
        fig.text((x_positions[c] + cell/2) / width,
                 (height - top + 0.085) / height, label,
                 ha='center', va='bottom', fontsize=8.3,
                 fontweight='bold' if name == 'TV-LoRA' else 'normal', color='#252c33')
    if title:
        fig.text(left / width, (height - 0.055) / height, title,
                 ha='left', va='top', fontsize=8, color='#626b73')
    error_start = x_positions[1]
    error_end = x_positions[-1] + cell
    cax = fig.add_axes([error_start / width, 0.32 / height,
                        (error_end - error_start) / width, 0.055 / height])
    return axes, cax


def validate_bundle(folder, methods=METHODS):
    source = json.loads((folder / 'fig5_sources.json').read_text(encoding='utf-8'))
    if not methods or len(set(methods)) != len(methods):
        raise ValueError('Select distinct verified methods')
    for method in methods:
        if method not in AVAILABLE_NAMES:
            raise ValueError('Unknown method: ' + method)
        record = source.get('methods', {}).get(method, {})
        if record.get('status') != 'verified' or record.get('reference_max_abs_error', 1) > 1e-6:
            raise ValueError('Method has no verified matching reference: ' + method)
    if any(e.get('source') in ('reference',) + tuple(methods)
           for e in source.get('errors', [])):
        raise ValueError('Selected input has an unresolved data error')
    if (source.get('dataset'), source.get('case'), source.get('views')) != ('LDCT', 'L067', 8):
        raise ValueError('The export is not the selected LDCT L067 eight-view comparison')
    if source.get('indices_zero_based') != dict(axial=99, coronal=128, sagittal=128):
        raise ValueError('Unexpected section indices')
    path = folder / 'fig5_sections.npz'
    if file_hash(path) != source['section_archive_sha256']:
        raise ValueError('Section archive changed after export')
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    for plane in PLANES:
        expected = (256, 256) if plane == 'axial' else (198, 256)
        for method in ('reference',) + tuple(methods):
            a = arrays[method + '__' + plane]
            if a.shape != expected or not np.isfinite(a).all():
                raise ValueError('Section shape or finite check failed: ' + method + ' ' + plane)
    return source, arrays


def render(folder, output, error_max=0.35, dpi=600, formats=('png', 'pdf', 'svg'), methods=METHODS):
    source, arrays = validate_bundle(folder, methods)
    output.mkdir(parents=True, exist_ok=True)
    font_setup()
    fig = plt.figure(figsize=(7.16, 3.3), facecolor='white')
    axes, cax = compact_layout(fig, ('Reference',) + tuple(methods),
                                ratios=(1, 198/256, 198/256))
    statistics = []
    any_saturated = False
    for r, plane in enumerate(PLANES):
        ref = arrays['reference__' + plane]
        # Same vertical flip as the original Fig. 5, applied to every method.
        axes[r][0].imshow(np.flipud(ref), cmap='gray', vmin=0, vmax=1,
                           interpolation='nearest', aspect='equal')
        for c, method in enumerate(methods, 1):
            pred = arrays[method + '__' + plane]
            error = np.abs(pred.astype(np.float64) - ref.astype(np.float64))
            axes[r][c].imshow(np.flipud(error), cmap='magma', vmin=0, vmax=error_max,
                              interpolation='nearest', aspect='equal')
            fraction = float(np.mean(error > error_max))
            any_saturated |= fraction > 0
            statistics.append({'method': method, 'plane': plane,
                               'section_mae': float(error.mean()),
                               'section_rmse': float(np.sqrt(np.mean(error**2))),
                               'maximum_error': float(error.max()),
                               'fraction_above_display_max': fraction})
    bar = fig.colorbar(ScalarMappable(norm=Normalize(0, error_max), cmap='magma'),
                       cax=cax, orientation='horizontal',
                       extend='max' if any_saturated else 'neither')
    bar.set_ticks(np.linspace(0, error_max, 8))
    bar.ax.tick_params(labelsize=7.2, length=2, width=0.4, pad=2)
    bar.outline.set_linewidth(0.4)
    bar.set_label('Absolute error (normalized intensity)', fontsize=8, labelpad=2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(match=plt.Text):
        if item.get_visible() and item.get_text():
            bounds = item.get_window_extent(renderer)
            if (bounds.x0 < -1 or bounds.y0 < -1 or
                bounds.x1 > fig.bbox.width+1 or bounds.y1 > fig.bbox.height+1):
                raise ValueError('Text outside figure boundary: ' + item.get_text())
    stem = 'Fig5_LDCT_8views'
    paths = []
    for extension in formats:
        path = output / (stem + '.' + extension)
        fig.savefig(path, dpi=dpi, facecolor='white', pad_inches=0,
                     metadata={'Creator': 'Matplotlib'} if extension == 'pdf' else None)
        paths.append(path)
    size = list(fig.get_size_inches())
    plt.close(fig)
    report = {
        'figure': 'Fig. 5', 'dataset': 'LDCT', 'case': 'L067', 'views': 8,
        'methods': list(methods),
        'layout': '3 orientations x (reference + %s error maps)' % len(methods),
        'excluded_methods': {name: record for name, record in source['methods'].items()
                             if name not in methods},
        'indices_zero_based': source['indices_zero_based'], 'font': 'DejaVu Sans',
        'size_inches': size, 'png_dpi': dpi,
        'reference_window': [0, 1], 'error_window': [0, error_max],
        'colormap': 'magma', 'pixel_processing': 'abs(saved_prediction - reference)',
        'display_orientation': 'flipud for every section, as in original Fig. 5',
        'same_scale_for_all_methods_and_planes': True,
        'cropped_to_body': False, 'intensity_rescaling': False,
        'interpolation': 'nearest', 'saturation_indicator': 'upper-end colorbar triangle',
        'section_statistics': statistics,
        'source_archive_sha256': source['section_archive_sha256'],
        'outputs': {p.name: file_hash(p) for p in paths},
    }
    (output / 'fig5_render_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for path in paths:
        print('FIGURE=' + str(path.resolve()))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent.parent
    parser.add_argument('--input', type=Path, default=base / 'error_sections')
    parser.add_argument('--output', type=Path, default=base / 'figures')
    parser.add_argument('--error-max', type=float, default=0.35)
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--methods', nargs='+', default=METHODS, choices=AVAILABLE_NAMES)
    args = parser.parse_args()
    if args.error_max <= 0 or args.dpi < 100:
        parser.error('error-max must be positive and dpi must be at least 100')
    try:
        render(args.input, args.output, args.error_max, args.dpi, methods=tuple(args.methods))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, 'ERROR=' + str(exc) + '\n')


if __name__ == '__main__':
    main()
