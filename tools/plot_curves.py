#!/usr/bin/env python3
"""Plot LDCT/CTHD PSNR and SSIM curves from measured orientation rows."""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', nargs='+', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    import numpy as np
    import matplotlib.pyplot as plt
    from tvlora.figures import style, save_figure
    from tvlora.io import file_sha256
    groups = defaultdict(dict)
    for path in args.csv:
        with Path(path).open(newline='', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                key = (r['dataset'], r['method'], int(r['views']))
                if r['plane'] in groups[key]:
                    raise ValueError('Duplicate dataset/method/view/plane row: ' + str(key))
                groups[key][r['plane']] = r
    if {k[0] for k in groups} != {'LDCT', 'CTHD'}:
        raise ValueError('Supply records for both LDCT and CTHD')
    means = {}
    for key, planes in groups.items():
        if set(planes) != {'axial', 'coronal', 'sagittal'}:
            raise ValueError('Each point requires all three orientation means: ' + str(key))
        means[key] = {m: float(np.mean([float(r[m]) for r in planes.values()])) for m in ('psnr', 'ssim')}
    methods = sorted({k[1] for k in groups}, key=lambda n: (n == 'TV-LoRA', n.lower()))
    style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7), sharex=True, sharey='row')
    palette = plt.get_cmap('tab20')
    colors = {name: palette(i % 20) for i, name in enumerate(methods)}
    colors['TV-LoRA'] = '#b33235'
    records = []
    for column, dataset in enumerate(('LDCT', 'CTHD')):
        axes[0, column].set_title(dataset, fontsize=10, fontweight='bold')
        for row, metric in enumerate(('psnr', 'ssim')):
            ax = axes[row, column]
            for name in methods:
                if not any((dataset, name, v) in means for v in (2, 4, 8)):
                    continue
                if not all((dataset, name, v) in means for v in (2, 4, 8)):
                    raise ValueError('Incomplete 2/4/8 view curve: ' + dataset + '/' + name)
                values = [means[(dataset, name, v)][metric] for v in (2, 4, 8)]
                ax.plot((2, 4, 8), values, color=colors[name], marker='D' if name == 'TV-LoRA' else 'o',
                        linewidth=1.8 if name == 'TV-LoRA' else 1.0, markersize=3.8,
                        label=name, zorder=5 if name == 'TV-LoRA' else 2)
                records.append(dict(dataset=dataset, method=name, metric=metric, views=[2, 4, 8], values=values))
            ax.grid(True, color='#e7ebef', linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.set_xticks((2, 4, 8))
            ax.set_ylabel('PSNR (dB)' if metric == 'psnr' else 'SSIM')
            if row == 1:
                ax.set_xlabel('Number of projection views')
    handles = {}
    for ax in axes.flat:
        hs, ls = ax.get_legend_handles_labels()
        handles.update(zip(ls, hs))
    fig.legend([handles[n] for n in methods], methods, loc='lower center',
               ncol=min(4, len(methods)), frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.12 if len(methods) > 4 else 0.07, 1, 1), h_pad=0.9, w_pad=1.4)
    save_figure(fig, args.output, {'sources': {str(p): file_sha256(p) for p in args.csv},
                'aggregation': 'equal mean of three orientation means', 'curves': records})


if __name__ == '__main__':
    main()
