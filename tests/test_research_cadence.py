"""Network work must not turn four-minute polls into missing 15m windows."""

from unittest.mock import patch

import pytest

from research.lab.runtime import next_poll_deadline
from research.lab import hyperliquid_shadow as worker


def test_network_duration_does_not_accumulate_into_poll_spacing():
    started = 0.0
    observations = []
    for duration in [31.8, 30.4, 33.0, 32.1, 31.0, 30.5]:
        finished = started + duration
        observations.append(finished)
        started = next_poll_deadline(started, finished, 240)
    assert all(238 <= b - a <= 243 for a, b in zip(observations, observations[1:]))
    # Four observations apart now fit the unchanged 14–18 minute contract.
    assert all(14 * 60 <= observations[i] - observations[i - 4] <= 18 * 60
               for i in range(4, len(observations)))
    legacy_spacing = 240 + 31.8
    assert 3 * legacy_spacing < 14 * 60
    assert 4 * legacy_spacing > 18 * 60


def test_overrun_skips_elapsed_slots_without_a_catch_up_burst():
    assert next_poll_deadline(100, 371, 240) == 580
    assert next_poll_deadline(100, 580, 240) == 580
    assert next_poll_deadline(100, 581, 240) == 820


@pytest.mark.parametrize('args', [(1, 0, 240), (0, 1, 0), (0, float('nan'), 240)])
def test_invalid_schedule_is_rejected(args):
    with pytest.raises(ValueError):
        next_poll_deadline(*args)


METADATA = [{'universe': [{'name': 'BTC'}]}, [{'markPx': '80000',
    'openInterest': '40000', 'funding': '0.00001', 'dayNtlVlm': '1000000'}]]
WALLET = '0x' + '1' * 40


class Client:
    def query(self, kind, **kwargs):
        if kind == 'metaAndAssetCtxs':
            return METADATA
        if kind == 'clearinghouseState':
            return {'time': int(worker.time.time() * 1000), 'assetPositions': []}
        return []


def test_missing_reference_remains_unknown_and_is_visible(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    db = worker.connect()
    cohort = {'wallets': [WALLET], 'scope': 'test', 'frozen_at_ms': 0}
    with patch.object(worker.time, 'sleep'):
        with patch.object(worker.time, 'time', return_value=1000):
            first = worker.cycle(db, Client(), cohort, 360, 240)
        with patch.object(worker.time, 'time', return_value=2200):
            missing = worker.cycle(db, Client(), cohort, 360, 240)
    assert first['inventory_window_status'] == 'warmup'
    assert missing['inventory_window_status'] == 'missing_reference'
    assert missing['inventory_change_actual_window_minutes'] is None
    assert missing['collection_protocol']['scheduler'] == 'fixed_start_v1'
    db.close()


def test_fixed_start_snapshots_make_valid_window_without_rewriting_old_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    db = worker.connect()
    cohort = {'wallets': [WALLET], 'scope': 'test', 'frozen_at_ms': 0}
    first_payload = None
    with patch.object(worker.time, 'sleep'):
        for tick in [1000, 1240, 1480, 1720, 1960]:
            with patch.object(worker.time, 'time', return_value=tick):
                result = worker.cycle(db, Client(), cohort, 360, 240)
            if first_payload is None:
                first_payload = db.execute('SELECT payload FROM features ORDER BY observed_ms LIMIT 1').fetchone()[0]
    assert result['inventory_window_status'] == 'observed'
    assert result['inventory_change_actual_window_minutes'] == 16
    assert db.execute('SELECT payload FROM features ORDER BY observed_ms LIMIT 1').fetchone()[0] == first_payload
    from research.lab.report import hyperliquid_summary
    summary, _ = hyperliquid_summary(db)
    quality = summary['inventory_window_quality']
    assert quality['snapshots'] == 5
    assert quality['comparable_14_18m_snapshots'] == 1
    assert quality['unknown_change_snapshots'] == 4
    assert quality['collection_protocol_counts'] == {'fixed_start_v1': 5}
    db.close()
