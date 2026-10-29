#!/usr/bin/env python3
"""Check checkpoint architecture/EMA and record its exact hash on CPU."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', default='checkpoint_info.json')
    p.add_argument('--sha256-only', action='store_true', help='Hash and size only; does not deserialize weights')
    args = p.parse_args()
    from tvlora.io import file_sha256, write_json
    path = Path(args.checkpoint).expanduser().resolve()
    if args.sha256_only:
        if not path.is_file():
            raise FileNotFoundError(path)
        info = {'checkpoint_file': path.name, 'checkpoint_bytes': path.stat().st_size,
                'checkpoint_sha256': file_sha256(path), 'architecture_and_ema_checked': False}
    else:
        os.environ['TVLORA_OP_BACKEND'] = 'native'
        from tvlora.prior import create_prior
        model, sde, cfg, info = create_prior(path, 2000, 'cpu')
        info['architecture_and_ema_checked'] = True
    write_json(args.output, info)
    print(json.dumps(info, indent=2))
    print('REPORT=' + str(Path(args.output).resolve()))


if __name__ == '__main__':
    main()
