from datetime import datetime, timezone
import json

from fastapi.testclient import TestClient

import dashboard
from collectors.hyperliquid_research import PublicHyperliquid
from research.pilot_summary import load_pilot_summary


def body():
    return {'schema_version':1,'status':'liquidation_data_pilot','protocol_sha256':'a'*64,
            'orders_enabled':False,'strategy_enabled':False,'alpha_validated':False,'source_review_ready':False,
            'runtime_state':'collecting','generated_at':datetime.fromtimestamp(1000,timezone.utc).isoformat()}


def test_pilot_staleness_preserves_time_and_rejects_strategy_enablement(tmp_path,monkeypatch):
    path=tmp_path/'pilot.json';b=body();path.write_text(json.dumps(b))
    monkeypatch.setenv('HYDRA_PILOT_REPORT',str(path))
    fresh=load_pilot_summary(clock=lambda:1001)
    assert fresh['is_stale'] is False and fresh['generated_at']==b['generated_at']
    assert load_pilot_summary(clock=lambda:1200)['collection_state']=='stale_report'
    b['strategy_enabled']=True;path.write_text(json.dumps(b))
    assert load_pilot_summary()['status']=='unavailable'
    path.unlink()
    assert load_pilot_summary()['status']=='unavailable'


def test_pilot_endpoint_does_not_poll_upstream_or_create_worker(tmp_path,monkeypatch):
    path=tmp_path/'pilot.json';path.write_text(json.dumps(body()))
    monkeypatch.setenv('HYDRA_PILOT_REPORT',str(path))
    def forbidden(*args,**kwargs):raise AssertionError('Page must only read status')
    monkeypatch.setattr(PublicHyperliquid,'query',forbidden)
    with TestClient(dashboard.app) as client:r=client.get('/api/research/pilot')
    assert r.status_code==200 and r.headers['cache-control']=='no-store'
    assert r.json()['is_stale'] is True and r.json()['orders_enabled'] is False
