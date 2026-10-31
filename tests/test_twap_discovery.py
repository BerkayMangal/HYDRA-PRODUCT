import pytest

from research.lab.twap_discovery import audit_discovery, run, screen_account, select_accounts, trade_participants

WALLET = '0x' + '1'*40


def sample(event=999000, size='2', body=None):
    return {'wallet':WALLET,'started_ms':1000000,'observed_ms':1001000,
            'body':body if body is not None else [{'twapId':9,'fill':{'coin':'BTC','tid':1,
                'time':event,'side':'B','px':'80000','sz':size}}]}


def test_screen_requires_large_fresh_observed_flow_and_never_claims_forward():
    row=screen_account(sample(),1000000)
    assert row['economic_activity_eligible'] is True
    assert row['signed_notional_usd']==160000
    assert row['selected_for_future_study'] is False
    assert row['observation_phase']=='history_at_first_observation'
    assert screen_account(sample(size='.001'),1000000)['economic_activity_eligible'] is False
    assert screen_account(sample(event=800000),1000000)['economic_activity_eligible'] is False
    # Events observed after the common window cutoff cannot boost eligibility.
    assert screen_account(sample(event=1000500),1000000)['economic_activity_eligible'] is False
    assert screen_account(sample(event=1002000),1000000)['status']=='unknown'


def test_capped_or_changed_history_cannot_select_a_wallet():
    body=[{'twapId':9,'fill':{'coin':'BTC','tid':index,'time':999000,'side':'B','px':'80000','sz':'2'}} for index in range(2000)]
    assert screen_account(sample(body=body),1000000)['status']=='unknown'
    body[1]['fill']['tid']=0;body[1]['fill']['sz']='3'
    assert screen_account(sample(body=body),1000000)['status']=='unknown'


def test_candidates_and_selection_ignore_performance_and_reject_future_sources():
    body=[{'coin':'BTC','time':999000,'users':[WALLET,WALLET.upper().replace('0X','0x'),'0x'+'0'*40]}]
    assert trade_participants(body,1000000)=={WALLET}
    with pytest.raises(ValueError,match='future'):
        trade_participants(body,998000)
    rows=[{'wallet':'b','economic_activity_eligible':True,'pnl':100000},
          {'wallet':'a','economic_activity_eligible':True,'pnl':-100000},
          {'wallet':'c','economic_activity_eligible':False,'pnl':999999}]
    assert select_accounts(rows)==['a','b']


def test_discovery_declares_before_reads_archives_raw_and_reproduces(tmp_path):
    out=tmp_path/'discovery'
    calls=[]
    class Client:
        def query(self,kind,**params):
            assert (out/'discovery-spec.json').is_file()
            calls.append(kind)
            if kind=='recentTrades':
                return [{'coin':'BTC','time':999000,'users':[WALLET]}]
            assert kind=='userTwapSliceFills'
            return sample()['body']
        def close(self): pass
    result=run(out,client_factory=Client,clock=lambda:1000,wait=lambda _:None)
    assert set(calls)=={'recentTrades','userTwapSliceFills'}
    assert result['selected_accounts']==[WALLET]
    assert result['orders_enabled'] is result['alpha_validated'] is False
    assert audit_discovery(out)['screening_reproduced'] is True
    with pytest.raises(FileExistsError):
        run(out,client_factory=Client,clock=lambda:1000,wait=lambda _:None)
    filename=next(iter(result['archives']))
    (out/filename).write_bytes(b'tampered')
    with pytest.raises((ValueError,OSError)):
        audit_discovery(out)


def test_invalid_upstream_snapshot_is_archived_and_cannot_supply_candidates(tmp_path):
    class Client:
        def query(self,kind,**params):
            assert kind=='recentTrades'
            return [{'coin':'BTC','time':1001000,'users':[WALLET]}]
        def close(self): pass
    out=tmp_path/'failed-source'
    result=run(out,client_factory=Client,clock=lambda:1000,wait=lambda _:None)
    assert len(result['errors'])==30
    assert result['accounts_probed']==0 and result['selected_accounts']==[]
    assert not (out/'candidate-cohort.json').exists()
    assert audit_discovery(out)['archives_verified']==30
