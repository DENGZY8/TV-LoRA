#!/usr/bin/env python3
"""Evaluate a saved reconstruction without intensity rescaling."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True)
    p.add_argument('--prediction', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    from tvlora.io import load_volume, file_sha256, write_json
    from tvlora.evaluation import evaluate_volume
    result = evaluate_volume(load_volume(args.reference), load_volume(args.prediction))
    result['source_files'] = {'reference_sha256': file_sha256(args.reference),
                              'prediction_sha256': file_sha256(args.prediction)}
    write_json(args.output, result)
    print('METRICS=' + str(Path(args.output).resolve()))


if __name__ == '__main__':
    main()
