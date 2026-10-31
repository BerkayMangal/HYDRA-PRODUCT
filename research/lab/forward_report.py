"""Reproduce a recorded trial report offline, without creating or settling trades."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from types import MappingProxyType

from research.forward_ledger import SignalLedger, digest
from .paths import write_json


class RecordedLedger(SignalLedger):
    def __init__(self, directory):
        self.directory = Path(directory)
        self.protocol = json.loads((self.directory / 'protocol.json').read_text())
        self.protocol_hash = digest(self.protocol)
        repo = Path(__file__).resolve().parents[2]
        for relative, expected in self.protocol['implementation_sha256'].items():
            if hashlib.sha256((repo / relative).read_bytes()).hexdigest() != expected:
                raise ValueError('Recorded implementation differs; use its archived source revision')
        self.spec = MappingProxyType(self.protocol['specification'])
        self.started_ms = self.protocol['created_ms']
        self.wallets = sorted({wallet.lower() for group in self.protocol['cohorts'].values() for wallet in group['wallets']})
        self.db = sqlite3.connect((self.directory / 'ledger.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA query_only=ON')
        self.db.execute('BEGIN')
        stored = self.db.execute('SELECT protocol_hash FROM metadata WHERE id=1').fetchone()[0]
        if stored != self.protocol_hash:
            self.db.close()
            raise ValueError('Recorded definition differs from the journal')
        latest = [self.started_ms]
        for table in ('quotes', 'batches', 'decisions', 'entries', 'outcomes', 'run_events'):
            row = self.db.execute(f'SELECT MAX(observed_ms) FROM {table}').fetchone()[0]
            if row is not None:
                latest.append(row)
        self.as_of_ms = max(latest)
        self.clock = lambda: self.as_of_ms / 1000


def recorded_report(directory):
    ledger = RecordedLedger(directory)
    try:
        return ledger.report(now=ledger.as_of_ms, runtime_state='recorded_snapshot')
    finally:
        ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--trial', type=Path, required=True)
    args = parser.parse_args()
    if args.out and args.out.resolve().is_relative_to(args.trial.resolve()):
        parser.error('--out must be outside the recorded trial directory')
    report = recorded_report(args.trial)
    if args.out:
        write_json(args.out, report)
    print(json.dumps({key: report[key] for key in ['protocol_sha256', 'started_at', 'generated_at',
                                                'decisions', 'pending_events', 'unknown_active_outcomes', 'results']}, indent=2))


if __name__ == '__main__':
    main()
