"""Strict loading of the frozen NCSN++ score prior and its EMA parameters."""
from pathlib import Path
import torch

from .io import file_sha256


def resolve_checkpoint(path):
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        matches = [p / n for n in ('checkpoint_185.pth', 'checkpoint_185', 'checkpoint.pth') if (p / n).is_file()]
        if len(matches) == 1:
            return matches[0]
        raise ValueError('Checkpoint directory is empty or ambiguous; pass the exact file path: ' + str(p))
    raise FileNotFoundError('Checkpoint does not exist: ' + str(p))


def load_pretrained(model, path, expected_sha256=None):
    path = resolve_checkpoint(path)
    digest = file_sha256(path)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError('Checkpoint SHA256 differs from the configured value')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(saved, dict) or 'model' not in saved or 'ema' not in saved:
        raise ValueError('Expected the original checkpoint dictionary with model and ema fields')
    state = {}
    for key, value in saved['model'].items():
        key = key[7:] if key.startswith('module.') else key
        if key in state:
            raise ValueError('Duplicate model key after removing module prefix: ' + key)
        if key != 'sigmas':
            state[key] = value
    incompatible = model.load_state_dict(state, strict=False)
    missing = [k for k in incompatible.missing_keys if k != 'sigmas']
    if missing or incompatible.unexpected_keys:
        raise ValueError('Model architecture mismatch: missing=%s, unexpected=%s' % (missing, incompatible.unexpected_keys))
    named_parameters = list(model.named_parameters())
    trainable = [(n, p) for n, p in named_parameters if p.requires_grad]
    shadows = saved['ema'].get('shadow_params')
    if not isinstance(shadows, (list, tuple)) or len(shadows) != len(trainable):
        raise ValueError('EMA count mismatch: checkpoint=%s, model trainable=%d, model all=%d' %
                         (len(shadows) if isinstance(shadows, (list, tuple)) else 'invalid', len(trainable), len(named_parameters)))
    with torch.no_grad():
        for (name, parameter), shadow in zip(trainable, shadows):
            if tuple(parameter.shape) != tuple(shadow.shape):
                raise ValueError('EMA shape mismatch for ' + name)
            if not torch.isfinite(shadow).all():
                raise ValueError('Non-finite EMA parameter: ' + name)
            parameter.copy_(shadow.to(device=parameter.device, dtype=parameter.dtype))
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError('Non-finite model tensor: ' + name)
    return {'checkpoint_file': path.name, 'checkpoint_sha256': digest,
            'checkpoint_bytes': path.stat().st_size, 'checkpoint_step': saved.get('step'),
            'ema_parameter_count': len(shadows), 'frozen_parameter_count': len(named_parameters) - len(trainable),
            'ema_applied': True, 'sigma_buffer_from_configuration': True}


def create_prior(checkpoint, steps, device, expected_sha256=None):
    from configs.ve.AAPM_256_ncsnpp_continuous import get_config
    from models.ncsnpp import NCSNpp
    from sde_lib import VESDE
    cfg = get_config()
    cfg.model.num_scales = steps
    cfg.device = torch.device(device)
    model = NCSNpp(cfg)
    info = load_pretrained(model, checkpoint, expected_sha256)
    model.to(cfg.device).eval()
    model.requires_grad_(False)
    sde = VESDE(sigma_min=cfg.model.sigma_min, sigma_max=cfg.model.sigma_max, N=steps)
    info.update({'model': cfg.model.name, 'image_size': cfg.data.image_size,
                 'sigma_min': cfg.model.sigma_min, 'sigma_max': cfg.model.sigma_max,
                 'steps': steps, 'centered': cfg.data.centered})
    return model, sde, cfg, info
