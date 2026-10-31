"""Read recorded economics and concentration without collecting or settling."""

import argparse
import json
from pathlib import Path

from research.forward_diagnostics import diagnose_trial
from .paths import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.out and args.out.resolve().is_relative_to(args.trial.resolve()):
        parser.error('--out must be outside the recorded trial directory')
    body = diagnose_trial(args.trial)
    if args.out:
        write_json(args.out, body)
    print(json.dumps(body, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
