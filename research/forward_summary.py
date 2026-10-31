"""Serve a recorded forward report without triggering collection or simulation."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time


def reject_constant(_value):
    raise ValueError('Nonfinite report value')


def load_forward_summary(clock=time.time):
    from .lab.paths import ROOT
    explicit = os.environ.get('HYDRA_FORWARD_REPORT')
    path = Path(explicit) if explicit else ROOT / 'twap-forward/latest-summary.json'
    unavailable = {'schema_version': 1, 'status': 'unavailable' if explicit else 'not_started',
                   'orders_enabled': False, 'alpha_validated': False,
                   'source_current_healthy': False, 'is_stale': True}
    try:
        if path.stat().st_size > 1_000_000:
            raise ValueError('Oversized forward report')
        body = json.loads(path.read_text(), parse_constant=reject_constant)
        if (body.get('schema_version') != 1 or body.get('status') != 'prospective_research_only'
                or body.get('orders_enabled') is not False or body.get('alpha_validated') is not False
                or not re.fullmatch('[0-9a-f]{64}', body.get('protocol_sha256', ''))):
            raise ValueError('Invalid read-only report')
        for variant in ('strategy', 'control'):
            stats = body['results'][variant]
            count, mean = stats['completed_trades'], stats['mean_modeled_net_return']
            if type(count) is not int or count < 0 or (count == 0 and mean is not None):
                raise ValueError('Observation counts and results disagree')
            if mean is not None and (isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean)):
                raise ValueError('Finite modeled movement or unknown required')
        generated = datetime.fromisoformat(body['generated_at'])
        if generated.tzinfo is None:
            raise ValueError('Timezone-aware report required')
        now = clock()
        age = now - generated.astimezone(timezone.utc).timestamp()
        batch = body.get('last_batch')
        quote_ms = body.get('last_valid_quote_ms')
        quote_age = now - quote_ms / 1000 if quote_ms is not None else None
        batch_age = now - batch['observed_ms'] / 1000 if batch else None
        stale = age < -5 or age > 90
        fresh = (not stale and body['runtime_state'] == 'collecting' and quote_age is not None
                 and 0 <= quote_age <= 60 and batch_age is not None and 0 <= batch_age <= 420
                 and batch['requested_accounts'] == batch['successful_accounts'] == body['account_count']
                 and not batch['errors'])
        body.update(report_age_seconds=age, quote_age_seconds=quote_age, batch_age_seconds=batch_age,
                    source_current_healthy=fresh, is_stale=stale,
                    collection_state='stale_report' if stale else body['runtime_state'])
        return body
    except FileNotFoundError:
        return unavailable
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        return {**unavailable, 'status': 'unavailable'}
