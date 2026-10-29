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
        return result
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return unavailable
