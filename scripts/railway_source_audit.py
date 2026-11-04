"""Read the existing Railway service's source metadata without deploying.

Uses the same exact-domain target discovery as the deployer. Reads no application
variables, logs, commands or source files. Only explicitly selected metadata is
written. The workflow has no upload, restart or exchange operation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

from scripts.railway_deploy import DeployError, Railway, TARGET, find_target

SOURCE_QUERY = '''query($environmentId:String!,$serviceId:String!){
  serviceInstance(environmentId:$environmentId,serviceId:$serviceId){
    serviceId serviceName rootDirectory source{repo image}
    activeDeployments{id status createdAt meta}
    latestDeployment{id status createdAt meta}
    service{repoTriggers(first:100){
      edges{node{repository branch environmentId serviceId}}
      pageInfo{hasNextPage}
    }}
  }
}'''


class ReadOnlyRailway(Railway):
    def query(self, query, variables=None):
        if not query.lstrip().startswith('query') or re.search(r'\bmutation\b', query):
            raise DeployError('Source audit permits GraphQL queries only.')
        return super().query(query, variables)


def clean_repo(value):
    if not isinstance(value, str):
        return None
    if re.fullmatch(r'[\w.-]+/[\w.-]+', value):
        return value
    url = urlsplit(value)
    if url.scheme in ('https', 'http') and url.hostname:
        return urlunsplit((url.scheme, url.hostname, url.path, '', ''))
    return None


def safe_name(value):
    return value if isinstance(value, str) and re.fullmatch(r'[\w./-]{1,250}', value) else None


def safe_deployment(row):
    if not row:
        return None
    meta = row.get('meta') or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except ValueError:
            meta = {}
    if not isinstance(meta, dict):
        meta = {}
    sha = meta.get('commitHash') or meta.get('commitSha')
    return {
        'id': row.get('id'), 'status': row.get('status'),
        'created_at': row.get('createdAt'),
        'repository': clean_repo(meta.get('repo') or meta.get('repository')),
        'branch': safe_name(meta.get('branch')),
        'commit': sha if isinstance(sha, str) and re.fullmatch('[a-fA-F0-9]{7,40}', sha) else None,
    }


def audit(env):
    result = {'target_url': TARGET, 'mode': 'read_only_source_audit',
              'deployments_started': 0, 'service_changes': 0,
              'credential_scope': 'repository and production environment secrets'}
    if not any(env.get(key, '').strip() for key in ('RAILWAY_TOKEN', 'RAILWAY_API_TOKEN')):
        return {**result, 'status': 'missing_railway_access',
                'next_step': 'Identify the live service in the Railway management dashboard.'}
    try:
        api = ReadOnlyRailway(env)
        target = find_target(api, env)
        data = api.query(SOURCE_QUERY, {'environmentId': target.environment, 'serviceId': target.service})['serviceInstance']
        source = data.get('source') or {}
        connection = (data.get('service') or {}).get('repoTriggers') or {}
        triggers = []
        for edge in connection.get('edges', []):
            row = edge['node']
            if row.get('environmentId') == target.environment and row.get('serviceId') in (None, target.service):
                triggers.append({'repository': clean_repo(row.get('repository')), 'branch': safe_name(row.get('branch'))})
        result.update({
            'status': 'source_metadata_read', 'target_ids': asdict(target),
            'management_url': f'https://railway.com/project/{target.project}/service/{target.service}?environmentId={target.environment}',
            'service_name': safe_name(data.get('serviceName')),
            'root_directory': safe_name(data.get('rootDirectory')),
            'source_repository': clean_repo(source.get('repo')),
            'container_image_configured': bool(source.get('image')),
            'matching_repository_triggers': triggers,
            'triggers_truncated': bool(connection.get('pageInfo', {}).get('hasNextPage')),
            'active_deployments': [safe_deployment(row) for row in data.get('activeDeployments', [])],
            'latest_deployment': safe_deployment(data.get('latestDeployment')),
        })
    except (DeployError, KeyError, TypeError, ValueError):
        result.update(status='source_metadata_unavailable', next_step='Management access or source metadata could not be verified; no deployment was attempted.')
    # An inline credential in vendor metadata must never enter an artifact.
    text = json.dumps(result, ensure_ascii=False)
    for key in ('RAILWAY_TOKEN', 'RAILWAY_API_TOKEN'):
        token = env.get(key, '')
        if token:
            text = text.replace(token, '[redacted]')
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(os.environ)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('Railway read-only source audit:', result['status'])
    print('Deployments started: 0; service changes: 0')
    if result['status'] == 'source_metadata_read':
        public_source = {key: result[key] for key in ('source_repository', 'matching_repository_triggers', 'management_url')}
        public_source['active_commits'] = [row['commit'] for row in result['active_deployments']]
        print('::notice::Railway source metadata: ' + json.dumps(public_source))
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        Path(summary).write_text('# Railway source audit\n\nStatus: '+result['status']+'\n\nNo deployment or service change was performed.\n')
    if result['status'] != 'source_metadata_read':
        print('::warning::Railway source could not be read. See the audit artifact; no deployment was attempted.')


if __name__ == '__main__':
    main()
