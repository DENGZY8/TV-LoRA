"""Shared styling and matched-section loading for scientific figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .io import project_path, load_volume, file_sha256, write_json

PLANES = ('axial', 'coronal', 'sagittal')


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.unicode_minus': False, 'savefig.facecolor': 'white'})


def section(a, axis, index):
    return np.flipud(np.take(a, index, axis=axis))


def read_figure_config(path, indices=None):
    cfg = json.loads(project_path(path).read_text())
    if cfg['dataset'] not in ('LDCT', 'CTHD'):
        raise ValueError('Expected an LDCT or CTHD visualization profile')
    reference_path = project_path(cfg['reference'])
    reference = load_volume(reference_path)
    if reference.shape[0] != cfg['expected_slices']:
        raise ValueError('Reference slice count does not match the dataset profile')
    methods = cfg.get('reconstructions', cfg.get('methods'))
    if not methods:
        raise ValueError('Specify reconstruction or method paths')
    arrays = {}
    sources = {'reference': {'file': str(reference_path), 'sha256': file_sha256(reference_path)}}
    for name, value in methods.items():
        p = project_path(value)
        a = load_volume(p)
        if a.shape != reference.shape:
            raise ValueError('Reconstruction shape differs from reference: ' + name)
        arrays[name] = a
        sources[name] = {'file': str(p), 'sha256': file_sha256(p)}
        # New run outputs include their reference, which must match exactly.
        sibling_reference = p.parent / 'reference.npy'
        if sibling_reference.is_file() and sibling_reference.resolve() != reference_path:
            if not np.array_equal(load_volume(sibling_reference), reference):
                raise ValueError('Reconstruction was evaluated against a different reference: ' + name)
    selected = indices or cfg.get('indices') or [n // 2 for n in reference.shape]
    if len(selected) != 3 or any(not 0 <= i < n for i, n in zip(selected, reference.shape)):
        raise ValueError('Section indices are outside the reference volume')
    cfg['indices'] = list(selected)
    return cfg, reference, arrays, sources


def save_figure(fig, output, report, dpi=400):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for extension in ('png', 'pdf', 'svg'):
        path = output.with_suffix('.' + extension)
        fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight', pad_inches=0.025)
        files.append(path)
    plt.close(fig)
    report['outputs'] = {p.name: file_sha256(p) for p in files}
    write_json(output.with_suffix('.json'), report)
    print('FIGURE=' + str(output.with_suffix('.pdf').resolve()))
