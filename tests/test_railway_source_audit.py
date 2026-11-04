"""Source diagnosis cannot deploy, read application secrets, or expose tokens."""
from unittest.mock import Mock

import pytest

from scripts import railway_source_audit as source
from scripts import railway_deploy as deploy


def test_missing_credentials_performs_no_remote_operation(monkeypatch):
    client = Mock(side_effect=AssertionError('Must not call Railway without access'))
    monkeypatch.setattr(source, 'ReadOnlyRailway', client)
    assert source.audit({})['status'] == 'missing_railway_access'
    client.assert_not_called()


def test_mutations_rejected_before_network(monkeypatch):
    send = Mock(side_effect=AssertionError('Network must not be reached'))
    monkeypatch.setattr(deploy.Railway, 'query', send)
    api = source.ReadOnlyRailway({'RAILWAY_TOKEN': 'test-key'})
    for query in ['mutation { serviceDelete(id:"x") }', 'query Q { mutation }', '{ projects { edges { node { id } } } }']:
        with pytest.raises(deploy.DeployError, match='queries only'):
            api.query(query)
    send.assert_not_called()


def test_source_diagnosis_keeps_only_safe_fields_and_matching_service(monkeypatch):
    target = deploy.Target('project', 'production', 'service')
    monkeypatch.setattr(source, 'find_target', lambda api, env: target)
    deployment = {'id': 'd1', 'status': 'SUCCESS', 'createdAt': '2026-11-04',
                  'meta': {'repo': 'https://user:secret@github.com/BerkayMangal/HYDRA-PRODUCT?token=secret',
                           'branch': 'main', 'commitHash': 'a'*40, 'variables': {'PRIVATE_KEY': 'must-not-leak'}}}
    payload = {'serviceInstance': {'serviceName': 'web', 'source': {'repo': 'BerkayMangal/HYDRA-PRODUCT', 'image': None},
               'rootDirectory': '/app', 'activeDeployments': [deployment], 'latestDeployment': deployment,
               'service': {'repoTriggers': {'edges': [
                   {'node': {'repository': 'BerkayMangal/HYDRA-PRODUCT', 'branch': 'main', 'environmentId': 'production', 'serviceId': 'service'}},
                   {'node': {'repository': 'other/project', 'branch': 'dev', 'environmentId': 'other', 'serviceId': 'service'}}],
                   'pageInfo': {'hasNextPage': False}}}}}
    monkeypatch.setattr(deploy.Railway, 'query', Mock(return_value=payload))
    result = source.audit({'RAILWAY_TOKEN': 'secret'})
    assert result['status'] == 'source_metadata_read'
    assert result['deployments_started'] == 0
    assert result['matching_repository_triggers'] == [{'repository': 'BerkayMangal/HYDRA-PRODUCT', 'branch': 'main'}]
    assert result['active_deployments'][0]['repository'] == 'https://github.com/BerkayMangal/HYDRA-PRODUCT'
    assert 'must-not-leak' not in str(result)
    assert 'secret' not in str(result).replace('repository and production environment secrets', '')


def test_vendor_failure_does_not_echo_raw_error_or_credentials(monkeypatch):
    monkeypatch.setattr(source, 'find_target', Mock(side_effect=deploy.DeployError('raw test-key error')))
    result = source.audit({'RAILWAY_TOKEN': 'test-key'})
    assert result['status'] == 'source_metadata_unavailable'
    assert 'test-key' not in str(result)


def test_deployment_metadata_does_not_promote_invalid_commit_or_command():
    result = source.safe_deployment({'meta': {'commitHash': 'not-a-commit', 'branch': 'main\napi-key', 'startCommand': 'secret'}})
    assert result['commit'] is None
    assert result['branch'] is None
    assert 'startCommand' not in str(result)
