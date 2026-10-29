"""Section means on fixed-range floating-point arrays; no image rescaling."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PLANES = ('axial', 'coronal', 'sagittal')


def section_metrics(reference, prediction, threshold=0.5):
    p = float(peak_signal_noise_ratio(reference, prediction, data_range=1.0))
    s = float(structural_similarity(reference, prediction, data_range=1.0, win_size=7))
    g, x = reference > threshold, prediction > threshold
    denominator = int(g.sum()) + int(x.sum())
    dice = 2.0 * int(np.logical_and(g, x).sum()) / denominator if denominator else 0.0
    return {'psnr': p, 'ssim': s, 'dice': dice}


def evaluate_volume(reference, prediction, threshold=0.5):
    if reference.shape != prediction.shape or reference.ndim != 3:
        raise ValueError('Reference and reconstruction must have the same [S,H,W] shape')
    if min(reference.shape) < 7:
        raise ValueError('Each section must support the fixed 7x7 SSIM window')
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError('Metrics require finite arrays')
    result = {}
    for axis, name in enumerate(PLANES):
        values = [section_metrics(a, b, threshold) for a, b in zip(
            np.moveaxis(reference, axis, 0), np.moveaxis(prediction, axis, 0))]
        result[name] = {key: float(np.mean([v[key] for v in values]))
                        for key in ('psnr', 'ssim', 'dice')}
        result[name]['sections'] = reference.shape[axis]
    result['orientation_average'] = {
        key: float(np.mean([result[p][key] for p in PLANES])) for key in ('psnr', 'ssim', 'dice')}
    result['protocol'] = {'data_range': 1.0, 'ssim_win_size': 7, 'dice_threshold': threshold,
                          'empty_empty_dice': 0.0, 'aggregation': 'mean of section metrics in each orientation',
                          'prediction_rescaling_during_evaluation': False}
    return result
