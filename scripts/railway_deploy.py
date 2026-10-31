"""Deploy only HYDRA's existing Railway domain, then verify the exact release."""

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

API = 'https://backboard.railway.com/graphql/v2'
TARGET = 'https://web-production-57fc0.up.railway.app'
REPOSITORY = 'BerkayMangal/HYDRA-PRODUCT'
CLI_VERSION = '5.71.0'
CLI_URL = f'https://github.com/railwayapp/cli/releases/download/v{CLI_VERSION}/railway-v{CLI_VERSION}-x86_64-unknown-linux-musl.tar.gz'
CLI_SHA256 = '8b1394f787d85cefdaa480027b5718212ad74c46b57a6ee21cd4436667dfc8e1'


class DeployError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def credentials(env):
    if env.get('RAILWAY_TOKEN', '').strip():
        return {'Project-Access-Token': env['RAILWAY_TOKEN']}, True
    if env.get('RAILWAY_API_TOKEN', '').strip():
        return {'Authorization': 'Bearer ' + env['RAILWAY_API_TOKEN']}, False
    raise DeployError('Railway access missing: configure RAILWAY_TOKEN or RAILWAY_API_TOKEN in GitHub Actions secrets. No deployment was attempted.')


class Railway:
    def __init__(self, env):
        self.headers, self.project_token = credentials(env)

    def query(self, query, variables=None):
        request = Request(API, data=json.dumps({'query': query, 'variables': variables or {}}).encode(),
                          headers={'Content-Type': 'application/json', **self.headers})
        try:
            with build_opener(NoRedirect()).open(request, timeout=30) as response:
                body = json.load(response)
        except HTTPError as exc:
            raise DeployError(f'Railway API HTTP {exc.code}; token scope or service access could not be verified.') from None
        except (URLError, TimeoutError, ValueError):
            raise DeployError('Railway API could not be read; deployment stopped before choosing a target.') from None
        if body.get('errors') or not isinstance(body.get('data'), dict):
            raise DeployError('Railway API rejected the query; token scope or service access could not be verified.')
        return body['data']


@dataclass(frozen=True)
class Target:
    project: str
    environment: str
    service: str


def nodes(connection):
    return [edge['node'] for edge in connection.get('edges', [])]


def find_target(api, env):
    """IDs are accepted only after an exact existing-domain match."""
    project_hint = env.get('RAILWAY_PROJECT_ID')
    environment_hint = env.get('RAILWAY_ENVIRONMENT_ID')
    service_hint = env.get('RAILWAY_SERVICE_ID')
    if api.project_token:
        token = api.query('query { projectToken { project { id } environment { id } } }')['projectToken']
        contexts = [(token['project']['id'], token['environment']['id'])]
    elif project_hint:
        project = api.query('query($id:String!){ project(id:$id){ id environments { edges { node { id deletedAt } } } } }', {'id': project_hint})['project']
        contexts = [(project['id'], item['id']) for item in nodes(project['environments']) if not item.get('deletedAt')]
    else:
        projects = api.query('query { projects { edges { node { id environments { edges { node { id deletedAt } } } } } } }')['projects']
        contexts = [(project['id'], item['id']) for project in nodes(projects)
                    for item in nodes(project['environments']) if not item.get('deletedAt')]
    contexts = [(p, e) for p, e in contexts if (not project_hint or p == project_hint)
                and (not environment_hint or e == environment_hint)]
    found = set()
    for project, environment in contexts:
        cursor = None
        for _ in range(100):
            body = api.query('''query($project:String!,$environment:String!,$after:String){
                environment(id:$environment,projectId:$project){
                  serviceInstances(first:100,after:$after){
                    edges{node{serviceId domains{serviceDomains{domain} customDomains{domain}}}}
                    pageInfo{hasNextPage endCursor}
                  }
                }
            }''', {'project': project, 'environment': environment, 'after': cursor})
            connection = body['environment']['serviceInstances']
            for item in nodes(connection):
                domains = item.get('domains') or {}
                hostnames = [row['domain'].lower().rstrip('.') for category in ('serviceDomains', 'customDomains')
                             for row in domains.get(category, [])]
                if urlsplit(TARGET).hostname in hostnames and (not service_hint or item['serviceId'] == service_hint):
                    found.add(Target(project, environment, item['serviceId']))
            page = connection['pageInfo']
            if not page['hasNextPage']:
                break
            if not page.get('endCursor') or page['endCursor'] == cursor:
                raise DeployError('Railway service pagination is incomplete; target selection stopped.')
            cursor = page['endCursor']
        else:
            raise DeployError('Railway service discovery exceeded its bound; target selection stopped.')
    if len(found) != 1:
        raise DeployError(f'Expected exactly one existing service for {urlsplit(TARGET).hostname}; found {len(found)}. No service was created or changed.')
    return next(iter(found))


def install_cli(destination):
    # No credentials are attached to the public vendor binary download.
    from urllib.request import urlopen
    with urlopen(CLI_URL, timeout=60) as response:
        packed = response.read(50_000_001)
    if len(packed) > 50_000_000:
        raise DeployError('Railway CLI download exceeds the expected size.')
    with tarfile.open(fileobj=io.BytesIO(packed), mode='r:gz') as archive:
        members = [item for item in archive.getmembers() if item.isfile() and Path(item.name).name == 'railway']
        if len(members) != 1:
            raise DeployError('Unexpected Railway CLI archive layout.')
        binary = archive.extractfile(members[0]).read()
    if hashlib.sha256(binary).hexdigest() != CLI_SHA256:
        raise DeployError('Railway CLI checksum does not match the pinned release.')
    destination.write_bytes(binary)
    destination.chmod(0o755)


def stage_release(repo, destination, sha):
    if not re.fullmatch('[0-9a-f]{40}', sha):
        raise DeployError('A full immutable Git commit SHA is required.')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if head != sha:
        raise DeployError('Checkout does not match the validated release commit.')
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=repo, text=True).split()
    if not remote or remote[0] != sha:
        raise DeployError('Release was superseded on main; refusing to deploy an older commit.')
    archive = subprocess.check_output(['git', 'archive', '--format=tar', sha], cwd=repo)
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as tar:
        tar.extractall(destination, filter='data')
    shutil.rmtree(destination / '.github', ignore_errors=True)
    version = json.loads((destination / 'release.json').read_text())
    version['build_commit'] = sha
    (destination / 'release.json').write_text(json.dumps(version, indent=2) + '\n')
    return version


def upload(cli, directory, target, sha, env):
    command = [str(cli), 'up', str(directory), '--path-as-root', '--project', target.project,
               '--environment', target.environment, '--service', target.service,
               '--detach', '--json', '--message', 'HYDRA ' + sha]
    result = subprocess.run(command, cwd=directory, env=env, text=True,
                            capture_output=True, timeout=180)
    if result.returncode:
        message = (result.stderr or result.stdout)[-3000:]
        for name in ('RAILWAY_TOKEN', 'RAILWAY_API_TOKEN'):
            if env.get(name):
                message = message.replace(env[name], '[redacted]')
        raise DeployError('Railway upload failed: ' + message)
    for line in reversed(result.stdout.splitlines()):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('deploymentId'):
            return row['deploymentId']
    raise DeployError('Railway did not return an uploaded deployment ID.')


def public_json(path):
    from urllib.request import urlopen
    request = Request(TARGET + path + '?verify=' + str(time.time_ns()), headers={'Cache-Control': 'no-cache'})
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def verify_live(sha, release_id, clock=time.time):
    version = public_json('/api/version')
    if version.get('commit') != sha or version.get('release_id') != release_id:
        return False
    health = public_json('/api/health')
    wallets = public_json('/api/wallets')
    status = public_json('/api/status')
    costs = public_json('/api/research/costs')
    price = status.get('market', {}).get('last_price')
    price_time = status.get('market', {}).get('last_price_event_ms')
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in (price, price_time)):
        return False
    age = clock() - price_time / 1000
    return (health.get('cycles', 0) >= 1 and wallets.get('status') == 'observed'
            and wallets.get('requested_accounts') == wallets.get('successful_accounts') == 22
            and wallets.get('is_stale') is False and wallets.get('journal_written') is True
            and wallets.get('orders_enabled') is False and not wallets.get('errors')
            and price > 0 and 0 <= age <= 600
            and costs.get('status') == 'descriptive_only' and costs.get('orders_enabled') is False
            and costs.get('alpha_validated') is False and public_json('/api/ml').get('status') == 'disabled')


def wait_for_release(api, deployment, sha, release_id, timeout=1200):
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        status = api.query('query($id:String!){deployment(id:$id){status}}', {'id': deployment})['deployment']['status']
        if status != previous:
            print('Railway deployment status:', status, flush=True)
            previous = status
        if status in ('FAILED', 'CRASHED', 'REMOVED', 'CANCELED', 'CANCELLED', 'SKIPPED'):
            raise DeployError(f'Railway deployment ended with {status}; deployment ID {deployment}.')
        if status == 'SUCCESS':
            try:
                if verify_live(sha, release_id):
                    return
            except (HTTPError, URLError, TimeoutError, ValueError, TypeError):
                pass
        time.sleep(15)
    raise DeployError('New release did not pass exact-commit and public-data readiness checks before timeout.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-access', action='store_true')
    parser.add_argument('--install-cli', action='store_true')
    parser.add_argument('--deploy', action='store_true')
    args = parser.parse_args()
    result = {'target': TARGET, 'verified': False}
    result_path = Path(os.environ.get('RUNNER_TEMP', tempfile.gettempdir())) / 'hydra-release-verification.json'
    try:
        if args.install_cli:
            install_cli(Path(os.environ['RAILWAY_CLI']))
            print('Pinned Railway CLI installed and checksum verified.')
            return
        api = Railway(os.environ)
        target = find_target(api, os.environ)
        print('Authenticated existing Railway target matched:', urlsplit(TARGET).hostname, flush=True)
        if args.check_access:
            return
        if not args.deploy:
            raise DeployError('Choose --check-access, --install-cli or --deploy.')
        if os.environ.get('GITHUB_REPOSITORY') != REPOSITORY:
            raise DeployError('Deployment is restricted to the authorized repository.')
        sha = os.environ['HYDRA_DEPLOY_SHA']
        result['commit'] = sha
        with tempfile.TemporaryDirectory(prefix='hydra-release-') as temporary:
            directory = Path(temporary) / 'app'
            version = stage_release(Path.cwd(), directory, sha)
            deployment = upload(Path(os.environ['RAILWAY_CLI']), directory, target, sha, dict(os.environ))
            result.update(deployment=deployment, release_id=version['release_id'])
            print('Existing-service upload accepted; waiting for readiness.', flush=True)
            wait_for_release(api, deployment, sha, version['release_id'])
        result['verified'] = True
        print('Public release, exact commit, source freshness and 22-account readiness verified.', flush=True)
    except (DeployError, KeyError, TypeError, AttributeError, OSError, subprocess.SubprocessError, ValueError) as exc:
        message = str(exc)
        for name in ('RAILWAY_TOKEN', 'RAILWAY_API_TOKEN'):
            if os.environ.get(name):
                message = message.replace(os.environ[name], '[redacted]')
        result['error'] = message
        result_path.write_text(json.dumps(result, indent=2) + '\n')
        print('::error title=Railway deployment blocked::' + message.replace('\n', ' ')[:3000], flush=True)
        raise SystemExit(1)
    finally:
        if args.deploy and result.get('verified'):
            result_path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
