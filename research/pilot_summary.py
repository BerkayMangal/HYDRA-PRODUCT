"""Read the pilot's atomic status file; page polling never starts collection."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time

from .forward_summary import reject_constant


def load_pilot_summary(clock=time.time):
    from .lab.paths import ROOT
    explicit = os.environ.get('HYDRA_PILOT_REPORT')
    path = Path(explicit) if explicit else ROOT / 'liquidation-pilot/latest-summary.json'
    absent = {'schema_version': 1, 'status': 'unavailable' if explicit else 'not_started',
              'orders_enabled': False, 'strategy_enabled': False, 'alpha_validated': False,
              'is_stale': True, 'source_review_ready': False}
    try:
        if path.stat().st_size > 1000000: raise ValueError('Oversized pilot report')
        body = json.loads(path.read_text(), parse_constant=reject_constant)
        if (body.get('status') != 'liquidation_data_pilot' or body.get('schema_version') != 1
                or any(body.get(key) is not False for key in ('orders_enabled', 'strategy_enabled', 'alpha_validated', 'source_review_ready'))
                or not re.fullmatch('[0-9a-f]{64}', body.get('protocol_sha256', ''))):
            raise ValueError('Invalid observational pilot report')
        generated = datetime.fromisoformat(body['generated_at'])
        if generated.tzinfo is None: raise ValueError('Timezone-aware status required')
        age = clock() - generated.astimezone(timezone.utc).timestamp()
        stale = not -5 <= age <= 90
        body.update(report_age_seconds=age, is_stale=stale,
                    collection_state='stale_report' if stale else body['runtime_state'])
        return body
    except FileNotFoundError:
        return absent
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {**absent, 'status': 'unavailable'}
