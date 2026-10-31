import hashlib
import sqlite3

import pytest

from research.lab.data_audit import audit_journal


def test_missing_journal_cannot_be_created_or_pass_integrity(tmp_path):
    path = tmp_path / 'absent.sqlite3'
    with pytest.raises(sqlite3.OperationalError):
        audit_journal(path, 'okx')
    assert not path.exists()


def test_audit_detects_tampering_future_time_and_unit_arithmetic_without_repair(tmp_path):
    path = tmp_path / 'journal.sqlite3'
    db = sqlite3.connect(path)
    db.executescript('''CREATE TABLE snapshots(payload TEXT, sha256 TEXT);
      CREATE TABLE trades(instrument TEXT,event_ms INTEGER,observed_ms INTEGER,price REAL,btc_size REAL,quote_size REAL,raw_size REAL);
      CREATE TABLE health(status TEXT);''')
    db.execute('INSERT INTO snapshots VALUES(?,?)', ('{}', hashlib.sha256(b'{}').hexdigest()))
    db.execute("INSERT INTO trades VALUES('BTC-USDT',100,101,10,2,20,2)")
    db.commit()
    assert audit_journal(path, 'okx')['integrity_checks_passed']
    db.execute("UPDATE snapshots SET payload='changed'")
    db.execute('UPDATE trades SET event_ms=102,quote_size=2000')
    db.commit()
    before = path.read_bytes()
    result = audit_journal(path, 'okx')
    assert result['raw_digest_mismatches'] == result['invalid_trade_times'] == result['invalid_trade_arithmetic'] == 1
    assert result['integrity_checks_passed'] is False
    assert result['coverage_certified_complete'] is False
    assert path.read_bytes() == before
    db.close()
