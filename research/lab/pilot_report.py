"""Reproduce a pilot checkpoint offline at its last recorded ingress time."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from research.liquidation_pilot import PilotJournal, SPEC, digest
from .paths import write_json


def recorded_report(directory):
    directory = Path(directory)
    protocol = json.loads((directory / 'protocol.json').read_text())
    if protocol['specification'] != SPEC:
        raise ValueError('Use the archived pilot source revision for this specification')
    repo = Path(__file__).resolve().parents[2]
    for name, expected in protocol['implementation_sha256'].items():
        if hashlib.sha256((repo / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Use the archived pilot source revision')
    ledger = object.__new__(PilotJournal)
    ledger.directory = directory
    ledger.protocol = protocol; ledger.protocol_hash = digest(protocol)
    ledger.started_ms = protocol['created_ms']; ledger.wallets = sorted(protocol['cohort']['wallets'])
    ledger.db = sqlite3.connect((directory / 'pilot.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
    ledger.db.row_factory = sqlite3.Row
    try:
        ledger.db.execute('PRAGMA query_only=ON'); ledger.db.execute('BEGIN')
        if ledger.db.execute('SELECT hash FROM metadata').fetchone()[0] != ledger.protocol_hash:
            raise ValueError('Pilot definition hash differs')
        latest = [ledger.started_ms]
        for table in ('frames','feed_events','run_events'):
            value = ledger.db.execute('SELECT MAX(observed_ms) FROM '+table).fetchone()[0]
            if value is not None: latest.append(value)
        ledger.clock = lambda: max(latest) / 1000
        until = None
        for row in ledger.db.execute('SELECT payload FROM run_events ORDER BY id DESC'):
            event = json.loads(row[0])
            if event.get('state') == 'started': until = event.get('until'); break
        return ledger.report('recorded_snapshot',until)
    finally:
        ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,required=True); parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    if args.out and args.out.resolve().is_relative_to(args.pilot.resolve()):
        parser.error('--out must be outside the recorded pilot directory')
    body = recorded_report(args.pilot)
    if args.out: write_json(args.out,body)
    print(json.dumps({k:body[k] for k in ['generated_at','protocol_sha256','frames','live_ws_unique_btc_fills','live_latency_seconds']},indent=2))


if __name__=='__main__': main()
