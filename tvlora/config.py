"""Dataset profiles; all relative paths are anchored at the repository root."""
import json
from .io import project_path


def load_config(path):
    cfg = json.loads(project_path(path).read_text(encoding='utf-8'))
    required = ['dataset', 'data_dir', 'checkpoint', 'output_dir', 'expected_slices',
                'steps', 'batch_size', 'lambda_tv', 'lambda_lr', 'snr', 'view_parameters']
    for key in required:
        if key not in cfg:
            raise ValueError('Missing configuration field: ' + key)
    if cfg['dataset'] not in ('LDCT', 'CTHD'):
        raise ValueError('This release provides LDCT and CTHD profiles')
    for key in ('expected_slices', 'steps', 'batch_size'):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    if cfg['steps'] < 2:
        raise ValueError('At least two reverse steps are required')
    for key in ('lambda_tv', 'lambda_lr', 'snr'):
        if not float(cfg[key]) > 0:
            raise ValueError(key + ' must be positive')
    if set(cfg['view_parameters']) != {'2', '4', '8'}:
        raise ValueError('Specify parameters for 2, 4, and 8 views')
    for params in cfg['view_parameters'].values():
        if not float(params['rho_tv']) > 0 or not float(params['rho_lr']) > 0:
            raise ValueError('ADMM penalties must be positive')
    if cfg.get('op_backend', 'native') not in ('native', 'cuda'):
        raise ValueError('op_backend must be native or cuda')
    if not isinstance(cfg.get('seed', 0), int):
        raise ValueError('seed must be an integer')
    return cfg
