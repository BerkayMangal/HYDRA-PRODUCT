"""Serve a bounded, recorded research summary independently of live signals."""

import json
import os
from pathlib import Path


def load_summary():
    repo = Path(__file__).resolve().parents[1]
    explicit = os.environ.get("HYDRA_RESEARCH_REPORT")
    if explicit:
        path = Path(explicit)
    else:
        from .lab.paths import ROOT

        path = ROOT / "reports/summary.json"
        if not path.is_file():
            path = repo / "docs/research/evidence/deployment-summary.json"
            if not path.is_file():
                path = repo / "docs/research/evidence/latest-summary.json"
    unavailable = {
        "status": "unavailable",
        "orders_enabled": False,
        "collection_state": "unknown",
    }
    try:
        if path.stat().st_size > 2_000_000:
            return unavailable
        report = json.loads(path.read_text())
        if (
            report.get("schema_version") != 1
            or report.get("status") != "research_only_alpha_unproven"
            or report.get("orders_enabled") is not False
        ):
            return unavailable
        # Detailed row-level observations remain in the report, not in the poll API.
        result = {
            key: report[key]
            for key in (
                "schema_version",
                "generated_at",
                "status",
                "orders_enabled",
                "collection_state",
                "missing_inputs",
            )
        }
        for key in (
            "historical_search",
            "okx",
            "hyperliquid",
            "additional_wallet_samples",
            "additional_wallet_errors",
            "wallet_sample_coverage",
        ):
            if key in report:
                result[key] = report[key]
        replay = report.get("receipt_replay", {})
        result["price_movement_diagnostics"] = {
            "inventory": replay.get("inventory", {}).get("all_inventory_directions", {}),
            "liquidation_reversal": replay.get("liquidations", {}).get("all_reversals", {}),
            "spot_absorption": replay.get("liquidations", {}).get("spot_absorption_reversals", {}),
        }
        result["native_twap_probe"] = load_twap_probe(repo)
        return result
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return unavailable


def load_twap_probe(repo):
    """A dated native API trial, separate from current wallet/strategy results."""
    try:
        path = repo / 'docs/research/evidence/twap-20261030/manifest.json'
        if path.stat().st_size > 100_000:
            raise ValueError('Oversized trial manifest')
        probe = json.loads(path.read_text())
        if probe['orders_enabled'] is not False or probe['alpha_validated'] is not False:
            raise ValueError('Read-only unvalidated trial required')
        return {'status': 'recorded_native_probe', **{key: probe[key] for key in (
            'started_at', 'completed_at', 'query', 'requested_accounts',
            'successful_accounts', 'accounts_with_btc_slices',
            'possibly_truncated_accounts', 'orders_enabled', 'alpha_validated', 'limits')}}
    except (OSError, ValueError, KeyError, TypeError):
        return {'status': 'unavailable'}
