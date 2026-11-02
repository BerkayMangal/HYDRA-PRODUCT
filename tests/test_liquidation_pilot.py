from copy import deepcopy
import json
import sqlite3

import pytest

from research.liquidation_pilot import PilotJournal
from research.lab.liquidation_pilot import audit
from research.lab.pilot_cohort import select

WALLET = '0x' + '1' * 40
VICTIM = '0x' + '2' * 40


def fill(tid=1, when=1001000, liquidation=False):
    row = {'coin': 'BTC', 'tid': tid, 'time': when, 'px': '80000', 'sz': '.1', 'side': 'B'}
    if liquidation: row['liquidation'] = {'liquidatedUser': VICTIM, 'method': 'market'}
    return row


def journal(tmp_path):
    return PilotJournal(tmp_path / 'pilot', {'wallets': [WALLET]}, clock=lambda: 1000)


def test_bootstrap_and_reconnect_never_refresh_first_receipt(tmp_path):
    p = journal(tmp_path)
    p.ingest('hl_ws', [fill(when=999000)], 1002000, wallet=WALLET, bootstrap=True)
    p.ingest('hl_ws', [fill(when=999000)], 1003000, wallet=WALLET)
    row = p.db.execute('SELECT observed_ms,phase FROM fills').fetchone()
    assert tuple(row) == (1002000, 'bootstrap')
    p.ingest('hl_ws', [fill(2,1004000)],1005000,wallet=WALLET)
    p.ingest('hl_ws', [fill(2,1004000),fill(3,1005500)],1006000,wallet=WALLET,bootstrap=True)
    rows = {r['tid']:r['phase'] for r in p.db.execute('SELECT tid,phase FROM fills')}
    assert rows == {'1':'bootstrap','2':'live','3':'bootstrap'}
    p.close()


def test_later_rest_metadata_has_its_own_knowledge_time(tmp_path):
    p = journal(tmp_path)
    p.ingest('hl_ws',[fill()],1002000,wallet=WALLET)
    p.ingest('hl_rest',[fill(liquidation=True)],1010000,wallet=WALLET)
    a = p.db.execute('SELECT observed_ms,payload FROM annotations').fetchone()
    assert a['observed_ms'] == 1010000
    assert json.loads(a['payload'])['liquidated_position_side'] == 'long'
    report = p.report()
    assert report['rest_explicit_with_live_ws_fill'] == 1
    assert report['matched_rest_liquidations_without_ws_flag'] == 1
    assert report['source_review_ready'] is False
    assert 'ws_liquidation_metadata_missing_in_matched_cases' in report['readiness_blockers']
    assert audit(p.directory)['metadata_before_fill_receipt'] == 0
    p.close()


def test_prior_rest_snapshot_cannot_become_a_new_ws_signal(tmp_path):
    p = journal(tmp_path)
    p.ingest('hl_rest',[fill()],1002000,wallet=WALLET,bootstrap=True)
    p.ingest('hl_ws',[fill(liquidation=True)],1003000,wallet=WALLET)
    assert p.db.execute("SELECT phase FROM fills WHERE source='hl_ws'").fetchone()[0] == 'bootstrap'
    assert p.db.execute('SELECT phase FROM annotations').fetchone()[0] == 'bootstrap'
    p.close()


def test_future_conflicting_or_nonfinite_frame_is_preserved_but_not_normalized(tmp_path):
    p = journal(tmp_path)
    assert not p.ingest('hl_ws',[fill(when=1003000)],1002000,wallet=WALLET)['valid']
    assert p.db.execute('SELECT COUNT(*) FROM fills').fetchone()[0] == 0
    assert p.ingest('hl_ws',[fill()],1002000,wallet=WALLET)['valid']
    changed=fill();changed['px']='81000'
    assert not p.ingest('hl_ws',[changed],1004000,wallet=WALLET)['valid']
    invalid=fill(2);invalid['sz']='nan'
    assert not p.ingest('hl_ws',[invalid],1005000,wallet=WALLET)['valid']
    checks=audit(p.directory)
    assert checks['raw_frames_checked']==4 and checks['raw_digest_mismatches']==0
    assert checks['future_fills']==0
    p.close()


def test_liquidation_conflict_and_victim_direction_are_checked(tmp_path):
    p=journal(tmp_path)
    p.ingest('hl_ws',[fill(liquidation=True)],1002000,wallet=WALLET)
    changed=fill(liquidation=True);changed['liquidation']['liquidatedUser']=WALLET
    assert not p.ingest('hl_rest',[changed],1004000,wallet=WALLET)['valid']
    p.close()


def spot(when,tid):
    return {'code':'0','data':[{'tradeId':str(tid),'ts':str(when),'px':'100','sz':'1','side':'buy'}]}


def test_spot_flow_requires_continuous_overlap_and_never_backfills_old_decision(tmp_path):
    p=journal(tmp_path)
    p.ingest('spot',spot(1000000,1),1000000,bootstrap=True)
    assert p.flow(1000000,5)['imbalance'] is None
    previous=1
    for index,when in enumerate(range(1005000,1305001,5000),2):
        body=spot(when,index)
        body['data'].append(spot(when-5000,previous)['data'][0])
        p.ingest('spot',body,when);previous=index
    assert p.flow(1305000,5)['usable'] is True
    assert p.flow(1305000,5)['imbalance']==1
    old=p.flow(1305000,5)
    p.ingest('spot',spot(1310000,900),1400000)
    assert p.flow(1400000,5)['usable'] is False
    assert p.flow(1305000,5)==old
    p.close()


def test_venues_with_equal_trade_ids_do_not_share_bootstrap_state(tmp_path):
    p=journal(tmp_path)
    p.ingest('hl_market',[fill()],1002000,bootstrap=True)
    p.ingest('spot',spot(1002000,1),1003000)
    assert p.db.execute("SELECT phase FROM fills WHERE source='spot'").fetchone()[0]=='live'
    p.close()


def test_journal_rejects_mutation_and_changed_protocol(tmp_path):
    p=journal(tmp_path)
    p.ingest('hl_ws',[fill()],1002000,wallet=WALLET)
    for sql in ["DELETE FROM fills","UPDATE fills SET observed_ms=1",'INSERT OR REPLACE INTO fills SELECT * FROM fills']:
        with pytest.raises(sqlite3.IntegrityError,match='append-only'):p.db.execute(sql)
        p.db.rollback()
    p.close()
    with pytest.raises(ValueError,match='differs'):
        PilotJournal(tmp_path/'pilot',{'wallets':[VICTIM]},clock=lambda:2000)


def test_activity_selection_ignores_pnl_and_caps_scope():
    rows=[{'wallet':str(n),'btc_fill_count':n,'pnl':-n} for n in range(20)]
    assert len(select(rows))==8
    assert select(rows)[0]=='19'
    rows[19]['error']='bad'
    assert select(rows)[0]=='18'


def test_missing_audit_never_creates_database(tmp_path):
    with pytest.raises(sqlite3.OperationalError):audit(tmp_path/'missing')
    assert not (tmp_path/'missing').exists()


def test_conflicting_metadata_inside_one_frame_rejects_entire_frame(tmp_path):
    p=journal(tmp_path)
    changed=fill(liquidation=True);changed['liquidation']['liquidatedUser']=WALLET
    assert not p.ingest('hl_ws',[fill(liquidation=True),changed],1002000,wallet=WALLET)['valid']
    assert p.db.execute('SELECT COUNT(*) FROM annotations').fetchone()[0]==0
    assert p.db.execute('SELECT COUNT(*) FROM fills').fetchone()[0]==0
    p.close()


def test_receiver_handles_missing_snapshot_flag_conservatively(tmp_path,monkeypatch):
    import asyncio
    from datetime import datetime,timedelta,timezone
    import time
    from research.lab import liquidation_pilot as runner
    p=PilotJournal(tmp_path/'receiver',{'wallets':[WALLET]})
    event=int(time.time()*1000)
    def item(tid):return fill(tid,event)
    frames=[{'channel':'userFills','data':{'user':WALLET,'fills':[item(1)]}},
            {'channel':'userFills','data':{'user':WALLET,'fills':[item(2)]}},
            {'channel':'userFills','data':{'user':WALLET,'isSnapshot':True,'fills':[item(2),item(3)]}}]
    sent=[]
    async def run():
        stop=asyncio.Event()
        class Socket:
            async def send(self,value):sent.append(json.loads(value))
            async def recv(self):
                row=frames.pop(0)
                if not frames:stop.set()
                return json.dumps(row)
        class Connection:
            async def __aenter__(self):return Socket()
            async def __aexit__(self,*args):pass
        monkeypatch.setattr(runner,'connect',lambda *args,**kwargs:Connection())
        await runner.collect_hyperliquid(p,stop,datetime.now(timezone.utc)+timedelta(minutes=1))
    asyncio.run(run())
    phases=dict(p.db.execute('SELECT tid,phase FROM fills'))
    assert phases=={'1':'bootstrap','2':'live','3':'bootstrap'}
    assert all(row['method']=='subscribe' for row in sent)
    p.close()
