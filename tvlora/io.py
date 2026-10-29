"""Explicit paths, complete-volume loading, and provenance records."""
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def project_path(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    a = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(a.dtype).encode())
    digest.update(str(a.shape).encode())
    digest.update(memoryview(a).cast('B'))
    return digest.hexdigest()


def _json_value(obj):
    if isinstance(obj, dict):
        return {str(k): _json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_value(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        if math.isnan(float(obj)):
            raise ValueError('NaN cannot be written to a result record')
        return str(float(obj)) if math.isinf(float(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + '.tmp')
    temporary.write_text(json.dumps(_json_value(obj), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(p)


def load_volume(path):
    a = np.load(path, allow_pickle=False)
    if a.ndim == 4 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim != 3 or not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a):
        raise ValueError('Expected a real numeric [S,H,W] or [S,1,H,W] array: ' + str(path))
    if not np.isfinite(a).all():
        raise ValueError('Non-finite volume: ' + str(path))
    return a


def slice_key(path):
    # Match the supplied float-based numeric ordering where that key is valid.
    text = ''.join(c for c in path.name if c.isdigit() or c == '.')
    try:
        return (0, float(text))
    except ValueError:
        return (1, tuple((0, int(t)) if t.isdigit() else (1, t.lower())
                         for t in re.split(r'(\d+)', path.name)))


def load_input(path, expected_slices=None, image_size=256):
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob('*.npy'), key=slice_key)
        if not files:
            raise ValueError('No .npy slices in ' + str(path))
        keys = [slice_key(p) for p in files]
        if len(set(keys)) != len(keys):
            raise ValueError('Ambiguous slice ordering; provide a stacked volume instead')
        slices = []
        for p in files:
            a = np.load(p, allow_pickle=False)
            if a.shape != (image_size, image_size):
                raise ValueError('Expected one square 2D slice in ' + str(p))
            slices.append(a)
        a = np.stack(slices)
    elif path.is_file():
        a = load_volume(path)
        files = [path]
    else:
        raise FileNotFoundError(path)
    if expected_slices is not None and a.shape[0] != expected_slices:
        raise ValueError('Expected %d slices; found %d. No truncation is applied.' % (expected_slices, a.shape[0]))
    if a.shape[1:] != (image_size, image_size):
        raise ValueError('Expected %dx%d slices' % (image_size, image_size))
    if np.iscomplexobj(a) or not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
        raise ValueError('Input must contain finite real numeric values')
    if float(a.min()) < -1e-6 or float(a.max()) > 1 + 1e-6:
        raise ValueError('Provide the preprocessed [0,1] input used in the experiments; no automatic HU conversion is applied')
    a = np.asarray(a, dtype=np.float32)
    record = {'shape': list(a.shape), 'dtype': str(a.dtype), 'array_sha256': array_sha256(a),
              'files': [{'name': p.name, 'sha256': file_sha256(p)} for p in files]}
    return a, record


def minmax_volume(a):
    a = np.array(a, dtype=np.float32, copy=True)
    a -= a.min()
    span = float(a.max())
    if span > 0:
        a /= span
    return a
