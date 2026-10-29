"""Exploratory account-breadth controls; no performance-selected wallet list."""

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from contextlib import closing

from research.wallet_breadth import btc_wallet_change
from .event_replay import PriceTape, outcomes, summarize
from .report import read_only
from .paths import ROOT, write_json

SPECIFICATION = {
    "version": "btc_wallet_breadth_diagnostic_v1",
    "design": "Exploratory replay on previously collected snapshots; not prospective strategy validation",
    "sampling": "Earliest usable 14-18 minute frozen-cohort change in each UTC quarter-hour",
    "directions": [
        "sign of aggregate BTC base change",
        "sign of equal-account buying-minus-selling breadth",
        "aggregate direction only if every single-account removal preserves it",
    ],
    "zero_breadth": "Abstain; do not count flat predictions as zero-return successes",
    "epsilon_btc": 1e-9,
    "forward_horizons_minutes": [15, 60, 240],
    "orders_enabled": False,
}
SPECIFICATION_SHA256 = hashlib.sha256(
    json.dumps(SPECIFICATION, sort_keys=True).encode()
).hexdigest()


def analyze(features, tape):
    by_time = dict(features)
    selected = {}
    for observed, current in features:
        duration = current.get("inventory_change_actual_window_minutes")
        if (
            duration is None
            or not 14 <= duration <= 18
            or current.get("sampled_inventory_change") is None
        ):
            continue
        expected = round(observed - duration * 60000)
        previous = by_time.get(expected)
        if previous is None:
            continue
        breadth = btc_wallet_change(previous, current)
        if breadth is not None:
            selected.setdefault(observed // 900000, (observed, breadth))
    variants = {"aggregate": [], "equal_account_breadth": [], "robust_aggregate": []}
    evidence = []
    for observed, breadth in selected.values():
        evidence.append({"observed_ms": observed, "breadth": breadth})
        total = breadth["sampled_net_base_change"]
        equal = breadth["equal_account_direction_balance"]
        for name, direction in (
            ("aggregate", total),
            ("equal_account_breadth", equal),
            (
                "robust_aggregate",
                (
                    total
                    if breadth[
                        "aggregate_direction_survives_every_single_account_removal"
                    ]
                    else 0.0
                ),
            ),
        ):
            if abs(direction) > SPECIFICATION["epsilon_btc"]:
                variants[name].append(
                    {
                        "decision_ms": observed,
                        **outcomes(tape, observed, 1 if direction > 0 else -1),
                    }
                )
    return {
        "usable_quarter_hours": len(selected),
        "aggregate_direction_fragile_quarter_hours": sum(
            abs(row[1]["sampled_net_base_change"]) > SPECIFICATION["epsilon_btc"]
            and not row[1]["aggregate_direction_survives_every_single_account_removal"]
            for row in selected.values()
        ),
        "variants": {
            name: {
                "directional_observations": len(rows),
                "price_movements": summarize(rows),
            }
            for name, rows in variants.items()
        },
        "account_changes": evidence,
    }


def generate(hl_path, okx_path, out):
    with closing(read_only(hl_path)) as hl:
        features = [
            (observed, json.loads(payload))
            for observed, payload in hl.execute(
                "SELECT observed_ms,payload FROM features ORDER BY observed_ms"
            )
        ]
    with closing(read_only(okx_path)) as okx:
        tape = PriceTape.from_db(okx)
    now = datetime.now(timezone.utc)
    report = {
        "generated_at": now.isoformat(),
        "generated_at_istanbul": now.astimezone(
            ZoneInfo("Europe/Istanbul")
        ).isoformat(),
        "specification": SPECIFICATION,
        "specification_sha256": SPECIFICATION_SHA256,
        "orders_enabled": False,
        "results": analyze(features, tape),
        "limits": "Common owners, correlated accounts and external hedges are unknown. Descriptive price references, not filled trades or a portfolio. Overlapping intervals and exploratory comparisons do not establish alpha.",
    }
    out = Path(out)
    write_json(out / "breadth-summary.json", report)
    r = report["results"]
    lines = [
        "# Cüzdan yönü: tek hesap etkisi kontrolü",
        "",
        report["generated_at_istanbul"],
        "",
        f"Kullanılabilir çeyrek saat: {r['usable_quarter_hours']}; bir hesabı çıkarınca aggregate yönünü koruyamayan: {r['aggregate_direction_fragile_quarter_hours']}.",
        "",
        "| Tanım | Yön gözlemi | 15m ortalama brüt | 60m ortalama brüt | 4h ortalama brüt |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, variant in r["variants"].items():
        moves = [
            variant["price_movements"][str(h)]["mean_signed_gross_price_move"]
            for h in (15, 60, 240)
        ]
        text = [
            f"{move:+.3%}" if move is not None else "henüz ölçülmedi" for move in moves
        ]
        lines.append(
            f"| {name} | {variant['directional_observations']} | {' | '.join(text)} |"
        )
    lines += [
        "",
        "Bu yeni kontrol geçmişte toplanmış veriye sonradan uygulanmıştır. Aynı sonuçları ileriye dönük başarı diye sunmaz. Yönü sıfır olan hesap dengesi abstain sayılır. Fiyat hareketleri fill, funding/slippage sonrası getiri veya bağımsız olay kanıtı değildir. Cüzdan/parametre seçimi PnL ile yapılmaz; gerçek emir yoktur.",
        "",
    ]
    (out / "BREADTH_TR.md").write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hyperliquid-db", type=Path, default=ROOT / "hyperliquid/observations.sqlite3"
    )
    parser.add_argument("--okx-db", type=Path, default=ROOT / "data/shadow.sqlite3")
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    result = generate(args.hyperliquid_db, args.okx_db, args.out)
    print(
        json.dumps(
            {
                key: value
                for key, value in result["results"].items()
                if key != "account_changes"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
