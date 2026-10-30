"""Build one auditable report from existing journals, without network calls."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo
from contextlib import closing

from research.hyperliquid_features import SPECIFICATION_SHA256 as HL_SPEC_HASH
from research.spot_confirmation import (
    SPECIFICATION_SHA256 as SPOT_SPEC_HASH,
    replay_paired_portfolios,
)
from .event_replay import (
    PriceTape,
    liquidation_replay,
    inventory_replay,
    SPECIFICATION,
    SPECIFICATION_SHA256,
)
from .paths import ROOT, write_json


def read_only(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def iso(timestamp):
    return (
        datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat()
        if timestamp is not None
        else None
    )


def count(db, table):
    # Table names only come from constants in this module, never from HTTP input.
    return db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def okx_summary(db):
    health = db.execute(
        "SELECT observed_ms,status,payload FROM health ORDER BY observed_ms"
    ).fetchall()
    errors = Counter()
    for _, _, payload in health:
        errors.update(json.loads(payload).get("errors", {}).keys())
    decisions = [
        json.loads(row[0])
        for row in db.execute(
            "SELECT payload FROM comparison_decisions ORDER BY observed_ms"
        )
    ]
    current = [
        row for row in decisions if row.get("specification_sha256") == SPOT_SPEC_HASH
    ]
    return {
        "first_observed_at": iso(health[0][0]) if health else None,
        "last_observed_at": iso(health[-1][0]) if health else None,
        "cycles": len(health),
        "degraded_cycles": sum(status != "collecting" for _, status, _ in health),
        "error_cycle_counts": dict(errors),
        "trade_rows": dict(
            db.execute("SELECT instrument,COUNT(*) FROM trades GROUP BY instrument")
        ),
        "forecast_rows": count(db, "forecasts"),
        "all_comparison_rows": len(decisions),
        "comparison_specification_sha256": SPOT_SPEC_HASH,
        "comparison": replay_paired_portfolios(current),
        "comparison_cost_stress": replay_paired_portfolios(
            current, cost_per_side=0.0022
        ),
        "limits": "Observed sampled trades; coverage failures abstain. Flat portfolios are not evidence of predictive skill.",
    }


def hyperliquid_summary(db):
    health = [
        json.loads(row[0])
        for row in db.execute("SELECT payload FROM health ORDER BY observed_ms")
    ]
    features = [
        (timestamp, json.loads(payload))
        for timestamp, payload in db.execute(
            "SELECT observed_ms,payload FROM features WHERE spec_hash=? ORDER BY observed_ms",
            (HL_SPEC_HASH,),
        )
    ]
    errors = Counter()
    for row in health:
        errors.update(row.get("errors", {}).keys())
        if row.get("cycle_error"):
            errors.update(["cycle:" + row["cycle_error"]])
    latest = features[-1][1] if features else {}
    window_rows = [row for _, row in features]
    valid_windows = sum(
        isinstance(row.get("inventory_change_actual_window_minutes"), (int, float))
        and 14 <= row["inventory_change_actual_window_minutes"] <= 18
        and row.get("sampled_inventory_change") is not None
        for row in window_rows
    )
    protocol_counts = Counter(
        row.get("collection_protocol", {}).get("scheduler", "legacy_unclassified")
        for row in window_rows
    )
    return {
        "first_observed_at": iso(features[0][0]) if features else None,
        "last_observed_at": iso(features[-1][0]) if features else None,
        "cycles": len(health),
        "feature_snapshots": len(features),
        "all_wallets_successful_cycles": sum(
            row.get("wallets_requested", 0) > 0
            and row.get("wallets_successful") == row.get("wallets_requested")
            for row in health
        ),
        "error_cycle_counts": dict(errors),
        "fill_truncation_cycles": sum(
            bool(row.get("fills_may_be_truncated")) for row in health
        ),
        "raw_snapshots": count(db, "raw_snapshots"),
        "unique_fills": count(db, "fills"),
        "explicit_liquidation_fills": count(db, "liquidations"),
        "liquidation_fills_by_asset": dict(
            db.execute("SELECT coin,COUNT(*) FROM liquidations GROUP BY coin")
        ),
        "requested_wallets": latest.get("requested_wallets"),
        "latest_successful_wallets": latest.get("successful_wallets"),
        "latest_position_rows": len(latest.get("positions", [])),
        "latest_btc_sample": latest.get("sampled_assets", {}).get("BTC"),
        "latest_inventory_change": latest.get("sampled_inventory_change"),
        "inventory_window_quality": {
            "snapshots": len(window_rows),
            "comparable_14_18m_snapshots": valid_windows,
            "unknown_change_snapshots": len(window_rows) - valid_windows,
            "latest_status": latest.get("inventory_window_status", "legacy_unclassified"),
            "latest_window_minutes": latest.get("inventory_change_actual_window_minutes"),
            "collection_protocol_counts": dict(protocol_counts),
        },
        "specification_sha256": HL_SPEC_HASH,
        "scope": latest.get("scope"),
        "cohort_sha256": latest.get("cohort_sha256"),
        "limits": "Size-selected frozen default-DEX cohort; incomplete market coverage, concentrated accounts, unknown external hedges.",
    }, features


def markdown(report):
    lines = [
        "# HYDRA — tamamlanan çalışma ve güncel kanıt",
        "",
        f"Rapor: {report['generated_at_istanbul']}",
        "",
        "**Karar: doğrulanmış, maliyetlere dayanıklı alfa henüz bulunmadı.** "
        "Yeni araştırma modülleri üretim sinyallerini değiştirmiyor. Emir gönderimi kapalı.",
        "",
    ]
    search = report.get("historical_search")
    if search:
        lines += [
            f"Tarihsel tarama: {search['declared_variants']:,} varyasyon; "
            f"{search['distinct_discovery_paths']:,} farklı eğitim pozisyon yolu; "
            f"test öncesi dondurulan {search['selected_before_test']} aday; "
            f"kabul ölçütlerini geçen {search['promotion_eligible_count']}.",
            "",
        ]
    okx = report.get("okx")
    if okx:
        lines += [
            f"OKX: {sum(okx['trade_rows'].values()):,} tekil işlem; "
            f"{okx['cycles']:,} döngünün {okx['degraded_cycles']} tanesinde kalite/coverage sorunu.",
            f"Gözlem aralığı: {okx['first_observed_at']} — {okx['last_observed_at']}.",
            f"Aynı donmuş EMA/spot sözleşmesiyle {okx['comparison']['observations']} karar; "
            f"spot doğrulaması {okx['comparison']['confirmation_available']} kararda kullanılabilir. "
            f"Baseline giriş sayısı {okx['comparison']['portfolios']['baseline']['entries']}. "
            "İşlemsizlik getirisi alfa sayılmaz.",
            "",
        ]
    hl = report.get("hyperliquid")
    if hl:
        lines += [
            f"Hyperliquid: {hl['requested_wallets']} donmuş hesap, {hl['feature_snapshots']} snapshot; "
            f"{hl['unique_fills']:,} tekil fill; {hl['explicit_liquidation_fills']} açık liquidation fill'i "
            f"({hl['liquidation_fills_by_asset'].get('BTC', 0)} BTC).",
            f"Gözlem aralığı: {hl['first_observed_at']} — {hl['last_observed_at']}.",
            f"Fill API sayfa sınırına ulaşılan {hl['fill_truncation_cycles']} döngü var; "
            "bu pencerelerin bütün fill'leri içerdiği varsayılmaz.",
        ]
        btc = hl.get("latest_btc_sample")
        if btc:
            lines += [
                f"Son BTC örnekleminde tek hesabın payı %{100 * btc['largest_account_gross_share']:.1f}; "
                f"gross ağırlıklarına göre etkin hesap sayısı {btc['effective_account_count_by_gross']:.2f}. "
                "Bu, bütün piyasanın veya bağımsız akıllı paranın görüşü değildir."
            ]
        lines += [""]
        quality = hl.get("inventory_window_quality")
        if quality:
            lines += [
                f"14–18 dakikalık karşılaştırılabilir snapshot: {quality['comparable_14_18m_snapshots']}/{quality['snapshots']}; "
                f"bilinmeyen değişim: {quality['unknown_change_snapshots']}. Son pencere durumu: {quality['latest_status']}.",
                "",
            ]
    for sample in report.get("additional_wallet_samples", []):
        lines += [
            f"Ek, ayrı dondurulmuş örneklem: {sample['requested_wallets']} hesap; "
            f"son çevrimde {sample['latest_successful_wallets']} başarılı okuma. "
            f"Son gözlem: {sample['last_observed_at']}.",
            "",
        ]
        btc = sample.get("latest_btc_sample")
        if btc:
            lines += [
                f"Bu ek örneklemde en büyük hesabın BTC gross payı %{100 * btc['largest_account_gross_share']:.1f}. "
                "Hesap sayısı artışı, notional yoğunlaşmasının azaldığını göstermez.",
                "",
            ]
        quality = sample.get("inventory_window_quality")
        if quality:
            lines += [
                f"Ek örneklemde 14–18 dakikalık karşılaştırılabilir snapshot: {quality['comparable_14_18m_snapshots']}/{quality['snapshots']}; "
                f"bilinmeyen değişim: {quality['unknown_change_snapshots']}. Son pencere durumu: {quality['latest_status']}.",
                "",
            ]
    if report.get("wallet_sample_coverage"):
        coverage = report["wallet_sample_coverage"]
        lines += [
            f"Örneklemler arasında tekilleştirilmiş public hesap sayısı: {coverage['distinct_public_accounts']}. "
            "Ortak sahiplik bilinmiyor; bütün piyasanın kapsamı değildir. Likidasyon hacimleri farklı örneklemlerden toplanmaz.",
            "",
        ]
    replay = report.get("receipt_replay")
    if replay:
        liq = replay["liquidations"]
        lines += [
            f"BTC fill'leri erişim dakikası/yönüne göre {liq['receipt_minute_side_groups']} gruba indi. "
            f"İki spot penceresinde ters yön desteği {liq['spot_absorption_groups']} grupta var; "
            f"coverage bilinmeyen {liq['unknown_spot_coverage_groups']} grup ayrı tutuldu.",
            "",
            "| Ufuk | Gözlenen grup | Örtüşmeyen aralık | Tüm ters yönler: ortalama brüt hareket | Spot desteği olanlar: ortalama brüt hareket |",
            "|---|---:|---:|---:|---:|",
        ]
        for horizon in (15, 60, 240):
            all_rows = liq["all_reversals"][str(horizon)]
            filtered = liq["spot_absorption_reversals"][str(horizon)]
            fmt = lambda value: f"{value:+.3%}" if value is not None else "ölçülmedi"
            lines.append(
                f"| {horizon}m | {all_rows['observed_groups']} | {all_rows['nonoverlapping_intervals']} | "
                f"{fmt(all_rows['mean_signed_gross_price_move'])} | {fmt(filtered['mean_signed_gross_price_move'])} |"
            )
        lines += [
            "",
            "Bunlar erişim zamanı korunarak yapılmış geriye dönük açıklayıcı ölçümlerdir; "
            "kilitlenmiş ileriye dönük strateji testi değildir. Örtüşmeyen aralıklar da istatistiksel "
            "bağımsızlık garantilemez. Brüt fiyat hareketi; gerçekleşmiş kâr, short/spot portföy getirisi "
            "veya funding/slippage içeren backtest değildir. Örneklem bütün likidasyonları kapsamaz. "
            "Gidiş-geliş 24 bp maliyet eşiği JSON'da ayrıca gösterilir; Sharpe/p-değeri üretilmez.",
            "",
        ]
        inventory = replay["inventory"]
        lines += [
            f"Cüzdan baz envanter yönü: {inventory['quarter_hour_observations']} kullanılabilir çeyrek saat "
            "gözlemi; en erken erişilebilir snapshot seçildi, fiyat etkisi baz miktardan ayrıldı.",
            "",
            "| Ufuk | Gözlenen yön | Örtüşmeyen aralık | Ortalama yönlü brüt fiyat hareketi |",
            "|---|---:|---:|---:|",
        ]
        for horizon in (15, 60, 240):
            row = inventory["all_inventory_directions"][str(horizon)]
            move = row["mean_signed_gross_price_move"]
            formatted = f"{move:+.3%}" if move is not None else "ölçülmedi"
            lines.append(
                f"| {horizon}m | {row['observed_groups']} | {row['nonoverlapping_intervals']} | {formatted} |"
            )
        available = [
            row["mean_signed_gross_price_move"]
            for row in inventory["all_inventory_directions"].values()
            if row["mean_signed_gross_price_move"] is not None
        ]
        if available and all(
            move <= SPECIFICATION["illustrative_round_trip_cost"] for move in available
        ):
            lines += [
                "",
                "Bu örneklemde envanter yönünün ortalama hareketi de 24 bp maliyet eşiğini geçmedi.",
            ]
        lines += [
            "",
            "Kısa gözlem süresi ve yoğunlaşmış cohort ile performans genellenemez.",
            "",
        ]
    lines += [
        "Araştırma önceliği: gerçekleşen likidasyon + spot absorpsiyonu; ardından aynı donmuş "
        "hesapların baz envanter değişimi. Sonuçlara bakarak eşik veya cüzdan seçimi değiştirilmedi.",
        "",
        "Rakip ve kaynak analizi: `docs/research/COMPETITORS_TR.md`. Pain ve bazı ticari "
        "arayüzler ortamın ağ kısıtı nedeniyle incelenemedi; fiyat/özellik doğrulaması tamamlanmış sayılmaz.",
        "",
        "Yerel kod ve raporlar hazırdır. Railway yayını ve sürüm eşleşmesi doğrulanmadı. "
        "Eski ML simülatörü doğrulanmış değil; ML kapalı kalır.",
        "",
    ]
    if report["missing_inputs"]:
        lines += ["Eksik girdiler: " + ", ".join(report["missing_inputs"]), ""]
    return "\n".join(lines)


def generate(
    okx_path=None, hl_path=None, results=None, out=None, additional_hl_paths=None
):
    if additional_hl_paths is None:
        wide = ROOT.parent / "research-wide/hyperliquid/observations.sqlite3"
        additional_hl_paths = [wide] if hl_path is None and wide.is_file() else []
    okx_path = Path(okx_path or ROOT / "data/shadow.sqlite3")
    hl_path = Path(hl_path or ROOT / "hyperliquid/observations.sqlite3")
    results = Path(results or ROOT / "results")
    out = Path(out or ROOT / "reports")
    now = datetime.now(timezone.utc)
    report = {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "generated_at_istanbul": now.astimezone(
            ZoneInfo("Europe/Istanbul")
        ).isoformat(),
        "status": "research_only_alpha_unproven",
        "orders_enabled": False,
        "collection_state": "recorded_snapshot",
        "missing_inputs": [],
    }
    if (results / "summary.json").is_file():
        search = json.loads((results / "summary.json").read_text())
        report["historical_search"] = {
            key: search[key]
            for key in (
                "declared_variants",
                "distinct_discovery_paths",
                "shortlisted_before_validation",
                "selected_before_test",
                "promotion_eligible_count",
                "completed_at",
                "manifest_sha256",
            )
        }
    else:
        report["missing_inputs"].append("historical_search")
    okx = read_only(okx_path) if okx_path.is_file() else None
    hl = read_only(hl_path) if hl_path.is_file() else None
    try:
        if okx:
            report["okx"] = okx_summary(okx)
        else:
            report["missing_inputs"].append("okx_journal")
        if hl:
            report["hyperliquid"], features = hyperliquid_summary(hl)
        else:
            report["missing_inputs"].append("hyperliquid_journal")
            features = []
        known_wallets = (
            (
                set(features[-1][1].get("wallets", []))
                - set(features[-1][1].get("unavailable_wallets", []))
            )
            if features
            else set()
        )
        samples, sample_errors = [], []
        seen_paths = {hl_path.resolve()}
        for path in additional_hl_paths:
            path = Path(path).resolve()
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                with closing(read_only(path)) as extra_db:
                    sample, extra_features = hyperliquid_summary(extra_db)
                if not extra_features:
                    sample_errors.append(
                        {"source": path.parent.name, "error_type": "NoRecordedFeatures"}
                    )
                    continue
                samples.append(sample)
                if extra_features:
                    known_wallets.update(
                        set(extra_features[-1][1].get("wallets", []))
                        - set(extra_features[-1][1].get("unavailable_wallets", []))
                    )
            except (sqlite3.Error, ValueError, KeyError, TypeError) as exc:
                sample_errors.append(
                    {"source": path.parent.name, "error_type": type(exc).__name__}
                )
        if samples or sample_errors:
            report["additional_wallet_samples"] = samples
            report["additional_wallet_errors"] = sample_errors
            report["wallet_sample_coverage"] = {
                "distinct_public_accounts": len(known_wallets),
                "cohort_count": int(bool(features)) + len(samples),
                "market_wide": False,
                "liquidation_counts_are_per_sample": True,
            }
        if okx and hl and features:
            tape = PriceTape.from_db(okx)
            report["receipt_replay"] = {
                "specification": SPECIFICATION,
                "specification_sha256": SPECIFICATION_SHA256,
                "liquidations": liquidation_replay(
                    hl, okx, tape, features[0][1]["cohort_frozen_at_ms"]
                ),
                "inventory": inventory_replay(features, tape),
            }
    finally:
        if okx:
            okx.close()
        if hl:
            hl.close()
    write_json(out / "summary.json", report)
    (out / "SUMMARY_TR.md").write_text(markdown(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--okx-db", type=Path)
    parser.add_argument("--hyperliquid-db", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--additional-hyperliquid-db", action="append", type=Path)
    args = parser.parse_args()
    report = generate(
        args.okx_db,
        args.hyperliquid_db,
        args.results,
        args.out,
        args.additional_hyperliquid_db,
    )
    print(
        json.dumps(
            {key: report[key] for key in ("generated_at", "status", "missing_inputs")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
