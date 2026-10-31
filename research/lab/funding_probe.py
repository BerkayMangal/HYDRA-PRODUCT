"""Archive native public funding before applying the fixed feasibility screen."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

from collectors.hyperliquid_research import PublicHyperliquid
from research.forward_ledger import digest
from research.funding_feasibility import HOUR_MS, SPECIFICATION, funding_screen
from .paths import write_json


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    end = int(time.time() * 1000)
    start = end - SPECIFICATION['lookback_hours'] * HOUR_MS
    repo = Path(__file__).resolve().parents[2]
    declaration = {'declared_at_ms': end, 'window_start_ms': start, 'window_end_ms': end,
                   'specification': SPECIFICATION,
                   'implementation_sha256': {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                                              for name in ['research/funding_feasibility.py', 'research/lab/funding_probe.py']},
                   'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()}
    with (out / 'probe-spec.json').open('x') as stream:
        stream.write(json.dumps(declaration, indent=2) + '\n')
    client = PublicHyperliquid()
    results, archives = [], {}
    try:
        for coin in SPECIFICATION['assets']:
            sample = {'coin': coin, 'started_ms': int(time.time() * 1000), 'body': None}
            try:
                sample['body'] = client.query('fundingHistory', coin=coin, startTime=start, endTime=end)
                sample['observed_ms'] = int(time.time() * 1000)
                result = funding_screen(sample['body'], coin, start, end, sample['observed_ms'])
            except Exception as exc:
                sample.update(observed_ms=int(time.time() * 1000), error=type(exc).__name__ + ': ' + str(exc)[:160])
                result = {'coin': coin, 'status': 'unavailable', 'error': sample['error'],
                          'orders_enabled': False, 'alpha_validated': False}
            raw = json.dumps(sample, sort_keys=True, separators=(',', ':'), allow_nan=True).encode()
            packed = gzip.compress(raw, mtime=0)
            filename = coin + '.json.gz'
            (out / filename).write_bytes(packed)
            archives[filename] = {'sha256': hashlib.sha256(packed).hexdigest(), 'decoded_sha256': hashlib.sha256(raw).hexdigest()}
            results.append(result)
    finally:
        client.close()
    report = {'schema_version': 1, 'status': 'funding_feasibility_only',
              'orders_enabled': False, 'alpha_validated': False,
              'completed_at': datetime.now(timezone.utc).isoformat(),
              'declaration_sha256': digest(declaration), 'window_start_ms': start, 'window_end_ms': end,
              'assumptions': SPECIFICATION, 'results': results, 'archives': archives}
    write_json(out / 'summary.json', report)
    return report


def audit(directory):
    directory = Path(directory)
    declared = json.loads((directory / 'probe-spec.json').read_text())
    report = json.loads((directory / 'summary.json').read_text())
    if declared['specification'] != SPECIFICATION or report['declaration_sha256'] != digest(declared):
        raise ValueError('Funding probe definition differs')
    if ([row['coin'] for row in report['results']] != SPECIFICATION['assets']
            or set(report['archives']) != {coin + '.json.gz' for coin in SPECIFICATION['assets']}
            or report['orders_enabled'] is not False or report['alpha_validated'] is not False):
        raise ValueError('Funding probe scope differs')
    for result in report['results']:
        coin = result['coin']
        filename = coin + '.json.gz'
        if coin not in SPECIFICATION['assets']:
            raise ValueError('Unspecified funding asset')
        packed = (directory / filename).read_bytes()
        raw = gzip.decompress(packed)
        if (hashlib.sha256(packed).hexdigest() != report['archives'][filename]['sha256']
                or hashlib.sha256(raw).hexdigest() != report['archives'][filename]['decoded_sha256']):
            raise ValueError('Funding archive digest differs')
        sample = json.loads(raw)
        if not declared['declared_at_ms'] <= sample['started_ms'] <= sample['observed_ms']:
            raise ValueError('Invalid receipt order')
        if sample.get('error'):
            if result['status'] != 'unavailable' or result.get('error') != sample['error']:
                raise ValueError('Source error cannot become a successful result')
        elif funding_screen(sample['body'], coin, declared['window_start_ms'], declared['window_end_ms'], sample['observed_ms']) != result:
            raise ValueError('Funding screen does not reproduce')
    return {'archives_verified': len(report['archives']), 'screening_reproduced': True,
            'orders_enabled': False, 'alpha_validated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    report = audit(args.out) if args.audit else run(args.out)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
