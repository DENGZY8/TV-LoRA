import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ['TVLORA_OP_BACKEND'] = 'native'

from controllable_generation_TV_lora import _Dx, _DxT, _Dy, _DyT, _Dz, _DzT, SVT, get_pc_radon_ADMM_TV_lora_vol
from tvlora.io import load_input
from tvlora.prior import load_pretrained
from tvlora.evaluation import evaluate_volume


@pytest.mark.parametrize('forward,transpose', [(_Dx, _DxT), (_Dy, _DyT), (_Dz, _DzT)])
def test_circular_differences_are_adjoint(forward, transpose):
    generator = torch.Generator().manual_seed(31)
    x = torch.randn((9, 1, 8, 8), dtype=torch.float64, generator=generator)
    y = torch.randn(x.shape, dtype=x.dtype, generator=generator)
    assert torch.allclose((forward(x) * y).sum(), (x * transpose(y)).sum(), atol=1e-10, rtol=1e-10)


def test_slice_svt_matches_group_shrinkage():
    x = torch.randn((5, 1, 7, 8), dtype=torch.float64)
    tau = 0.73
    norms = torch.linalg.vector_norm(x.flatten(1), dim=1)[:, None, None, None]
    expected = x * (1 - tau / norms).clamp(min=0)
    assert torch.allclose(SVT(x, tau), expected, atol=1e-12, rtol=1e-12)


def test_input_order_and_slice_count(tmp_path):
    for n in (10, 2, 1):
        np.save(tmp_path / ('%d.npy' % n), np.full((8, 8), n / 10, np.float32))
    (tmp_path / 'README.txt').write_text('Not a slice')
    x, info = load_input(tmp_path, 3, image_size=8)
    assert info['files'][0]['name'] == '1.npy'
    assert info['files'][1]['name'] == '2.npy'
    assert np.allclose(x[:, 0, 0], [0.1, 0.2, 1.0])
    with pytest.raises(ValueError, match='No truncation'):
        load_input(tmp_path, 2, image_size=8)


class TinyPrior(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fourier = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 3.0]))
        self.register_buffer('sigmas', torch.tensor([1.0, 0.1]))


def saved_prior(path, model, shadows):
    torch.save({'model': {'module.' + k: v for k, v in model.state_dict().items()},
                'ema': {'shadow_params': shadows}, 'step': 185}, path)


def test_checkpoint_filters_nontrainable_fourier_parameter(tmp_path):
    model = TinyPrior()
    path = tmp_path / 'checkpoint_185'
    saved_prior(path, model, [torch.tensor([5.0, 7.0])])
    info = load_pretrained(model, path)
    assert torch.equal(model.weight, torch.tensor([5.0, 7.0]))
    assert torch.equal(model.fourier, torch.tensor([2.0]))
    assert info['ema_parameter_count'] == 1
    assert info['frozen_parameter_count'] == 1
    with pytest.raises(ValueError, match='SHA256'):
        load_pretrained(model, path, '0' * 64)


def test_checkpoint_rejects_ema_count_mismatch(tmp_path):
    model = TinyPrior()
    path = tmp_path / 'wrong.pth'
    saved_prior(path, model, [torch.ones(2), torch.ones(1)])
    with pytest.raises(ValueError, match='EMA count mismatch'):
        load_pretrained(model, path)


def test_missing_checkpoint_cannot_use_random_weights(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pretrained(TinyPrior(), tmp_path / 'missing.pth')


def test_metrics_empty_dice_convention_and_orientations():
    a = np.zeros((9, 10, 11), np.float32)
    result = evaluate_volume(a, a)
    for name, count in [('axial', 9), ('coronal', 10), ('sagittal', 11)]:
        assert result[name]['sections'] == count
        assert result[name]['ssim'] == pytest.approx(1.0)
        assert result[name]['dice'] == 0.0
        assert np.isinf(result[name]['psnr'])


def test_ncsnpp_native_forward_smoke():
    from configs.ve.AAPM_256_ncsnpp_continuous import get_config
    from models.ncsnpp import NCSNpp
    cfg = get_config()
    cfg.data.image_size = 32
    cfg.model.nf = 8
    cfg.model.ch_mult = (1, 2)
    cfg.model.num_res_blocks = 1
    cfg.model.num_scales = 4
    model = NCSNpp(cfg).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, 32, 32), torch.ones(1))
    assert output.shape == (1, 1, 32, 32)
    assert torch.isfinite(output).all()


def test_original_sampler_short_synthetic_ct_run():
    from physics.ct import CT
    from sampling import ReverseDiffusionPredictor, LangevinCorrector
    from sde_lib import VESDE
    torch.manual_seed(11)

    class GaussianScore(torch.nn.Module):
        def forward(self, x, std):
            return -x / (1 + std[:, None, None, None] ** 2)

    data = torch.rand(9, 1, 16, 16)
    geometry = CT(16, 4, uniform=True, circle=False, device='cpu')
    sde = VESDE(sigma_min=0.1, sigma_max=1.0, N=3)
    sampler = get_pc_radon_ADMM_TV_lora_vol(
        sde, ReverseDiffusionPredictor, LangevinCorrector, lambda x: x,
        snr=0.16, n_steps=1, continuous=True, radon=geometry,
        save_progress=False, final_consistency=True, img_shape=data.shape,
        lamb_1=0.04, rho_1=15, rho_2=1, lamb_lowrank=0.5)
    result = sampler(GaussianScore(), data, measurement=geometry.A(data), batch_size=4)
    assert result.shape == data.shape
    assert torch.isfinite(result).all()


def test_reconstruction_figure_rejects_mismatched_reference(tmp_path):
    from tvlora.figures import read_figure_config
    shared = tmp_path / 'reference.npy'
    result_dir = tmp_path / 'view8'
    result_dir.mkdir()
    a = np.ones((9, 8, 8), np.float32)
    np.save(shared, a)
    np.save(result_dir / 'reference.npy', a * 0.5)
    np.save(result_dir / 'reconstruction.npy', a)
    config = tmp_path / 'figure.json'
    config.write_text(json.dumps({'dataset': 'LDCT', 'expected_slices': 9,
        'reference': str(shared), 'methods': {'TV-LoRA': str(result_dir / 'reconstruction.npy')}}))
    with pytest.raises(ValueError, match='different reference'):
        read_figure_config(config)


def test_cli_writes_outputs_and_skips_completed_subset(tmp_path, monkeypatch):
    import csv
    from types import SimpleNamespace
    import run
    import tvlora.prior
    from sde_lib import VESDE

    class GaussianScore(torch.nn.Module):
        def forward(self, x, std):
            return -x / (1 + std[:, None, None, None] ** 2)

    def synthetic_prior(checkpoint, steps, device, digest=None):
        return (GaussianScore(), VESDE(sigma_min=0.1, sigma_max=1, N=steps),
                SimpleNamespace(data=SimpleNamespace(centered=False)),
                {'checkpoint_sha256': '1' * 64, 'checkpoint_file': 'synthetic-test-only'})

    monkeypatch.setattr(tvlora.prior, 'create_prior', synthetic_prior)
    reference = tmp_path / 'input.npy'
    np.save(reference, np.random.default_rng(3).random((7, 256, 256), dtype=np.float32))
    cfg = {'dataset': 'LDCT', 'data_dir': str(reference), 'checkpoint': str(tmp_path / 'synthetic'),
           'output_dir': str(tmp_path / 'outputs'), 'expected_slices': 7, 'steps': 2, 'batch_size': 4,
           'lambda_tv': 0.04, 'lambda_lr': 0.5, 'snr': 0.16, 'seed': 0, 'op_backend': 'native',
           'view_parameters': {str(v): {'rho_tv': 15, 'rho_lr': 5 if v == 4 else 1} for v in (2, 4, 8)}}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg))
    run.main(['--config', str(path), '--device', 'cpu', '--views', '8', '4'])
    output = Path(cfg['output_dir'])
    first = (output / 'view8/run.json').read_bytes()
    run.main(['--config', str(path), '--device', 'cpu', '--views', '8', '--skip-completed'])
    assert (output / 'view8/run.json').read_bytes() == first
    with (output / 'results.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert {r['views'] for r in rows} == {'8', '4'}
    assert len(rows) == 6
    assert json.loads(first)['status'] == 'completed'
    assert (output / 'view8/reconstruction.raw.npy').is_file()
