#!/usr/bin/env python3
"""TV-LoRA reconstruction of complete, preprocessed CT volumes."""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, help='Dataset profile JSON')
    p.add_argument('--data-dir', help='Directory of ordered .npy slices, or one stacked .npy volume')
    p.add_argument('--checkpoint', help='Exact pretrained score checkpoint file')
    p.add_argument('--output-dir')
    p.add_argument('--views', nargs='+', type=int, choices=(2, 4, 8), default=[8, 4, 2])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--steps', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--op-backend', choices=('native', 'cuda'))
    p.add_argument('--doctor', action='store_true', help='Validate input, checkpoint, score forward pass, and projection shapes; do not reconstruct')
    p.add_argument('--check-config', action='store_true', help='Print resolved configuration without importing PyTorch')
    p.add_argument('--skip-completed', action='store_true', help='Skip completed views whose configuration and output hashes still match')
    p.add_argument('--overwrite', action='store_true', help='Replace existing outputs for requested views')
    return p


def seed_all(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv=None):
    args = parser().parse_args(argv)
    from tvlora.config import load_config
    from tvlora.io import project_path, load_input, write_json, array_sha256, minmax_volume, file_sha256
    cfg = load_config(args.config)
    overrides = {'data_dir': args.data_dir, 'checkpoint': args.checkpoint, 'output_dir': args.output_dir,
                 'batch_size': args.batch_size, 'steps': args.steps, 'seed': args.seed,
                 'op_backend': args.op_backend}
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    if cfg['batch_size'] <= 0 or cfg['steps'] < 2:
        raise ValueError('batch-size must be positive and steps must be at least two')
    for key in ('data_dir', 'checkpoint', 'output_dir'):
        cfg[key] = str(project_path(cfg[key]))
    cfg['views'] = list(dict.fromkeys(args.views))
    cfg['device'] = args.device
    if args.check_config:
        print(json.dumps(cfg, indent=2))
        return
    os.environ['TVLORA_OP_BACKEND'] = cfg.get('op_backend', 'native')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import numpy as np
    import torch
    from controllable_generation_TV_lora import get_pc_radon_ADMM_TV_lora_vol
    from models.utils import get_score_fn
    from physics.ct import CT
    from sampling import ReverseDiffusionPredictor, LangevinCorrector
    from tvlora.evaluation import evaluate_volume
    from tvlora.prior import create_prior
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    reference, input_info = load_input(cfg['data_dir'], cfg['expected_slices'])
    model, sde, model_cfg, checkpoint_info = create_prior(
        cfg['checkpoint'], cfg['steps'], device, cfg.get('checkpoint_sha256'))
    x_reference = torch.from_numpy(reference[:, None]).to(device)
    runtime = {'python': sys.version.split()[0], 'pytorch': torch.__version__,
               'pytorch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
               'device': str(device), 'op_backend': cfg.get('op_backend', 'native'),
               'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
    if args.doctor:
        with torch.no_grad():
            score = get_score_fn(sde, model, train=False, continuous=True)(
                torch.zeros((1, 1, 256, 256), device=device), torch.full((1,), 0.5, device=device))
            if not torch.isfinite(score).all():
                raise ValueError('Score smoke check produced non-finite values')
            geometry = CT(256, cfg['views'][0], uniform=True, circle=False, device=device)
            projection = geometry.A(x_reference[:1])
            backprojection = geometry.AT(projection)
            if backprojection.shape != x_reference[:1].shape or not torch.isfinite(backprojection).all():
                raise ValueError('Projection smoke check failed')
        report = {'status': 'doctor_passed', 'configuration': cfg, 'input': input_info,
                  'checkpoint': checkpoint_info, 'runtime': runtime,
                  'score_shape': list(score.shape), 'projection_shape': list(projection.shape)}
        write_json(Path(cfg['output_dir']) / 'doctor.json', report)
        print(json.dumps(report, indent=2))
        return
    root = Path(cfg['output_dir'])
    rows = []
    for view in cfg['views']:
        params = cfg['view_parameters'][str(view)]
        destination = root / ('view%d' % view)
        config_view = dict(cfg, current_view=view, rho_tv=params['rho_tv'], rho_lr=params['rho_lr'])
        identity_config = {k: v for k, v in config_view.items() if k != 'views'}
        identity = hashlib.sha256(json.dumps({'configuration': identity_config,
                                             'reference': input_info['array_sha256'],
                                             'checkpoint': checkpoint_info['checkpoint_sha256']}, sort_keys=True).encode()).hexdigest()
        run_path = destination / 'run.json'
        if run_path.exists() and not args.overwrite:
            old = json.loads(run_path.read_text())
            if old.get('status') == 'completed':
                if args.skip_completed and old.get('identity') == identity and all(
                    (destination / name).is_file() and file_sha256(destination / name) == digest
                    for name, digest in old.get('output_sha256', {}).items()
                ) and len(old.get('output_sha256', {})) >= 4:
                    print('SKIPPED_COMPLETED_VIEW=%d' % view, flush=True)
                    for plane, metrics in json.loads((destination / 'metrics.json').read_text()).items():
                        if plane in ('axial', 'coronal', 'sagittal'):
                            rows.append(dict(dataset=cfg['dataset'], method='TV-LoRA', views=view, plane=plane,
                                             **{k: metrics[k] for k in ('psnr', 'ssim', 'dice')}))
                    continue
                raise FileExistsError('Completed outputs already exist in %s; use --skip-completed or --overwrite' % destination)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'recon/progress').mkdir(parents=True, exist_ok=True)
        write_json(destination / 'config.resolved.json', config_view)
        report = {'status': 'running', 'identity': identity, 'input': input_info,
                  'checkpoint': checkpoint_info, 'runtime': runtime, 'parameters': config_view}
        write_json(run_path, report)
        wall_start = time.perf_counter()
        try:
            seed_all(cfg.get('seed', 0), torch, np)
            geometry = CT(256, view, uniform=True, circle=False, device=device)
            with torch.no_grad():
                measurement = geometry.A(x_reference)
            np.save(destination / 'measurement.npy', measurement.detach().cpu().numpy())
            inverse_scaler = (lambda x: (x + 1.0) / 2.0) if model_cfg.data.centered else (lambda x: x)
            data = 2 * x_reference - 1 if model_cfg.data.centered else x_reference
            sampler = get_pc_radon_ADMM_TV_lora_vol(
                sde, ReverseDiffusionPredictor, LangevinCorrector, inverse_scaler,
                snr=cfg['snr'], n_steps=1, probability_flow=False, continuous=True,
                denoise=True, radon=geometry, save_progress=cfg.get('save_progress', False),
                save_root=destination, final_consistency=cfg.get('final_consistency', True),
                img_shape=data.shape, lamb_1=cfg['lambda_tv'], rho_1=params['rho_tv'],
                rho_2=params['rho_lr'], lamb_lowrank=cfg['lambda_lr'])
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            reconstruction_start = time.perf_counter()
            prediction_tensor = sampler(model, data, measurement=measurement, batch_size=cfg['batch_size'])
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            reconstruction_seconds = time.perf_counter() - reconstruction_start
            raw = prediction_tensor.detach().cpu().numpy()[:, 0]
            if not np.isfinite(raw).all():
                raise ValueError('Non-finite reconstruction; outputs are not marked complete')
            prediction = minmax_volume(raw)
            exported_reference = minmax_volume(reference)
            np.save(destination / 'reconstruction.raw.npy', raw)
            np.save(destination / 'reconstruction.npy', prediction)
            np.save(destination / 'reference.npy', exported_reference)
            metrics = evaluate_volume(exported_reference, prediction)
            write_json(destination / 'metrics.json', metrics)
            write_json(destination / 'metrics.raw.json', evaluate_volume(reference, raw))
            report.update({'status': 'completed', 'reconstruction_seconds': reconstruction_seconds,
                           'view_wall_seconds': time.perf_counter() - wall_start,
                           'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                           'measurement_array_sha256': array_sha256(measurement.detach().cpu().numpy()),
                           'export_processing': 'independent whole-volume min-max scaling of reconstruction and reference',
                           'output_sha256': {name: file_sha256(destination / name) for name in (
                               'reference.npy', 'reconstruction.npy', 'reconstruction.raw.npy', 'measurement.npy', 'metrics.json', 'metrics.raw.json')}})
            write_json(run_path, report)
            for plane in ('axial', 'coronal', 'sagittal'):
                rows.append(dict(dataset=cfg['dataset'], method='TV-LoRA', views=view, plane=plane,
                                 **{k: metrics[plane][k] for k in ('psnr', 'ssim', 'dice')}))
            print('COMPLETED_VIEW=%d OUTPUT=%s' % (view, destination), flush=True)
        except Exception as error:
            report.update(status='failed', error=repr(error), view_wall_seconds=time.perf_counter() - wall_start)
            write_json(run_path, report)
            raise
        finally:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    # Preserve other completed views when a subset is requested later.
    rows = []
    for view in (8, 4, 2):
        directory = root / ('view%d' % view)
        if not (directory / 'run.json').is_file():
            continue
        previous = json.loads((directory / 'run.json').read_text())
        if previous.get('status') != 'completed':
            continue
        previous_cfg = previous['parameters']
        keys = ('dataset', 'profile', 'steps', 'batch_size', 'seed', 'lambda_tv', 'lambda_lr', 'snr', 'op_backend', 'final_consistency')
        if any(previous_cfg.get(k) != cfg.get(k) for k in keys):
            continue
        if previous['input']['array_sha256'] != input_info['array_sha256'] or previous['checkpoint']['checkpoint_sha256'] != checkpoint_info['checkpoint_sha256']:
            continue
        if any(previous_cfg[k] != cfg['view_parameters'][str(view)][k] for k in ('rho_tv', 'rho_lr')):
            continue
        if not previous.get('output_sha256') or any(
            not (directory / name).is_file() or file_sha256(directory / name) != digest
            for name, digest in previous['output_sha256'].items()
        ):
            raise ValueError('Completed output hashes differ in ' + str(directory))
        metrics = json.loads((directory / 'metrics.json').read_text())
        for plane in ('axial', 'coronal', 'sagittal'):
            rows.append(dict(dataset=cfg['dataset'], method='TV-LoRA', views=view, plane=plane,
                             **{k: metrics[plane][k] for k in ('psnr', 'ssim', 'dice')}))
    with (root / 'results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'method', 'views', 'plane', 'psnr', 'ssim', 'dice'])
        writer.writeheader()
        writer.writerows(rows)
    print('RESULTS_CSV=' + str(root / 'results.csv'))


if __name__ == '__main__':
    main()
