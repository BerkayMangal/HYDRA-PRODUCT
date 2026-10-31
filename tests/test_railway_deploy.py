"""Wrong target, stale commit and weak health must never pass a live release."""

from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import tarfile
from unittest.mock import Mock

import pytest

import dashboard
from scripts import railway_deploy as deploy


def edge_nodes(items):
    return {'edges': [{'node': row} for row in items]}


class API:
    project_token = True

    def __init__(self, domains):
        self.domains = domains

    def query(self, query, variables=None):
        if 'projectToken' in query:
            return {'projectToken': {'project': {'id': 'project'}, 'environment': {'id': 'environment'}}}
        return {'environment': {'serviceInstances': {
            **edge_nodes([{'serviceId': 'service-' + str(i), 'domains': {'serviceDomains': [{'domain': domain}], 'customDomains': []}}
                          for i, domain in enumerate(self.domains)]),
            'pageInfo': {'hasNextPage': False, 'endCursor': None},
        }}}


def test_missing_auth_fails_before_api_or_cli_is_used():
    with pytest.raises(deploy.DeployError, match='Railway access missing'):
        deploy.Railway({})
    headers, project = deploy.credentials({'RAILWAY_TOKEN': 'project-secret', 'RAILWAY_API_TOKEN': 'account-secret'})
    assert project is True
    assert headers == {'Project-Access-Token': 'project-secret'}
    assert deploy.credentials({'RAILWAY_API_TOKEN': 'account-secret'}) == ({'Authorization': 'Bearer account-secret'}, False)


def test_domain_match_cannot_choose_another_service_or_create_one():
    host = 'web-production-57fc0.up.railway.app'
    assert deploy.find_target(API(['other.up.railway.app', host]), {}) == deploy.Target('project', 'environment', 'service-1')
    for domains in [[], ['other.up.railway.app'], [host, host]]:
        with pytest.raises(deploy.DeployError):
            deploy.find_target(API(domains), {})
    with pytest.raises(deploy.DeployError):
        deploy.find_target(API([host]), {'RAILWAY_PROJECT_ID': 'wrong-project'})


def test_upload_has_explicit_target_and_cannot_enter_project_creation(tmp_path, monkeypatch):
    call = Mock(return_value=subprocess.CompletedProcess([], 0, '{"deploymentId":"deployment"}\n', ''))
    monkeypatch.setattr(deploy.subprocess, 'run', call)
    target = deploy.Target('project', 'environment', 'service')
    assert deploy.upload(Path('/tmp/railway'), tmp_path, target, 'a' * 40, {}) == 'deployment'
    args = call.call_args.args[0]
    assert args[args.index('--project') + 1] == 'project'
    assert args[args.index('--environment') + 1] == 'environment'
    assert args[args.index('--service') + 1] == 'service'
    assert not {'--new', '--yes', '-y'}.intersection(args)


def test_cli_errors_never_expose_the_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy.subprocess, 'run', Mock(return_value=subprocess.CompletedProcess([], 1, '', 'failed: secret-value')))
    with pytest.raises(deploy.DeployError) as error:
        deploy.upload(Path('/tmp/railway'), tmp_path, deploy.Target('p', 'e', 's'), 'a' * 40, {'RAILWAY_TOKEN': 'secret-value'})
    assert 'secret-value' not in str(error.value)
    assert '[redacted]' in str(error.value)


def live_payloads():
    return {
        '/api/version': {'commit': 'a' * 40, 'release_id': 'release'},
        '/api/health': {'cycles': 1},
        '/api/wallets': {'status': 'observed', 'requested_accounts': 22, 'successful_accounts': 22,
                         'is_stale': False, 'journal_written': True, 'orders_enabled': False, 'errors': {}},
        '/api/status': {'market': {'last_price': 80000, 'last_price_event_ms': 999000}},
        '/api/research/costs': {'status': 'descriptive_only', 'orders_enabled': False, 'alpha_validated': False},
        '/api/ml': {'status': 'disabled'},
    }


def test_exact_commit_and_real_data_readiness_are_required(monkeypatch):
    rows = live_payloads()
    monkeypatch.setattr(deploy, 'public_json', lambda path: rows[path])
    assert deploy.verify_live('a' * 40, 'release', clock=lambda: 1000)
    for path, key, value in [('/api/version', 'commit', 'b' * 40),
                             ('/api/health', 'cycles', 0),
                             ('/api/wallets', 'successful_accounts', 21),
                             ('/api/wallets', 'journal_written', False),
                             ('/api/wallets', 'is_stale', True),
                             ('/api/ml', 'status', 'enabled')]:
        changed = deepcopy(rows)
        changed[path][key] = value
        monkeypatch.setattr(deploy, 'public_json', lambda path: changed[path])
        assert not deploy.verify_live('a' * 40, 'release', clock=lambda: 1000)
    for price, timestamp in [(80000, 1), (80000, 1001000), (float('inf'), 999000), (True, 999000)]:
        changed = deepcopy(rows)
        changed['/api/status']['market'] = {'last_price': price, 'last_price_event_ms': timestamp}
        monkeypatch.setattr(deploy, 'public_json', lambda path: changed[path])
        assert not deploy.verify_live('a' * 40, 'release', clock=lambda: 1000)


def test_staging_contains_only_committed_source_and_a_build_stamp(tmp_path, monkeypatch):
    packed = io.BytesIO()
    with tarfile.open(fileobj=packed, mode='w') as tar:
        for name, content in [('release.json', b'{"release_id":"release"}'), ('main.py', b'# app')]:
            item = tarfile.TarInfo(name)
            item.size = len(content)
            tar.addfile(item, io.BytesIO(content))
    def git(args, **kwargs):
        if args[1] == 'rev-parse':
            return 'a' * 40 + '\n'
        if args[1] == 'ls-remote':
            return 'a' * 40 + '\trefs/heads/main\n'
        return packed.getvalue()
    monkeypatch.setattr(deploy.subprocess, 'check_output', git)
    version = deploy.stage_release(tmp_path, tmp_path / 'stage', 'a' * 40)
    assert version['build_commit'] == 'a' * 40
    assert not (tmp_path / 'stage/.git').exists()
    assert not (tmp_path / 'stage/.env').exists()


def test_uploaded_build_stamp_wins_over_old_railway_git_metadata(monkeypatch):
    original = Path.read_text
    def read(path, *args, **kwargs):
        if path.name == 'release.json':
            return json.dumps({'release_id': 'release', 'build_commit': 'b' * 40})
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setenv('RAILWAY_GIT_COMMIT_SHA', 'a' * 40)
    assert json.loads(dashboard.api_version().body)['commit'] == 'b' * 40
