"""Historical funding-only cost screen; no executable carry or alpha claim."""

import math

HOUR_MS = 3600000
SPECIFICATION = {
    'version': 'public_funding_cost_screen_v1',
    'assets': ['BTC', 'ETH', 'SOL', 'HYPE'],
    'lookback_hours': 168,
    'spot_fee_per_side_bps': 8,
    'perp_fee_per_side_bps': 4.5,
    'slippage_per_leg_bps': 4,
    'trade_legs': 4,
    'design': 'Predeclared descriptive funding-only feasibility screen, not a selected trading rule or backtest.',
    'position_assumption': 'Long spot and short an equal constant perp notional; positive funding is received by the short.',
    'source': 'Native public Hyperliquid fundingHistory; hourly records must cover every expected funding bucket.',
    'unknown_costs': ['Executable spot/perp spread and basis entry/exit', 'Spot venue and asset availability',
                      'Financing and collateral opportunity cost', 'Stablecoin conversion and transfers',
                      'Margin, liquidation and venue risk'],
    'orders_enabled': False, 'alpha_validated': False,
}


def funding_screen(body, coin, start_ms, end_ms, observed_ms, spec=SPECIFICATION):
    if not 0 < start_ms < end_ms <= observed_ms:
        raise ValueError('Valid requested and observed funding interval required')
    if not isinstance(body, list) or len(body) > 500:
        raise ValueError('Bounded native funding list required')
    expected = set(range(start_ms // HOUR_MS + 1, end_ms // HOUR_MS + 1))
    rates = {}
    for row in body:
        event, value = row['time'], row['fundingRate']
        if isinstance(event, bool) or not isinstance(event, int) or not 0 < event <= observed_ms or row['coin'] != coin:
            raise ValueError('Invalid instrument or future funding record')
        if isinstance(value, bool):
            raise ValueError('Finite funding rate required')
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Finite funding rate required')
        if not start_ms < event <= end_ms:
            continue
        if event % HOUR_MS > 1000:
            raise ValueError('Funding timestamp does not identify an hourly payment')
        bucket = event // HOUR_MS
        if bucket in rates and rates[bucket] != value:
            raise ValueError('Conflicting funding payment in the same hourly bucket')
        rates[bucket] = value
    complete = set(rates) == expected
    cost = (2 * (spec['spot_fee_per_side_bps'] + spec['perp_fee_per_side_bps'])
            + spec['trade_legs'] * spec['slippage_per_leg_bps'])
    gross = sum(rates.values()) * 10000
    annual = sum(rates.values()) / len(rates) * 24 * 365 if rates else None
    if not math.isfinite(gross) or (annual is not None and not math.isfinite(annual)):
        raise ValueError('Nonfinite funding arithmetic')
    day_buckets = set(range(end_ms // HOUR_MS - 23, end_ms // HOUR_MS + 1))
    day_complete = day_buckets <= set(rates)
    week = gross if complete else None
    return {'coin': coin, 'status': 'complete_funding_history' if complete else 'incomplete_funding_history',
            'expected_hourly_payments': len(expected), 'observed_hourly_payments': len(rates),
            'missing_hourly_buckets': sorted(expected - set(rates)),
            'observed_funding_sum_bps': gross,
            'full_window_short_received_funding_bps': week,
            'last_24h_short_received_funding_bps': sum(rates[bucket] for bucket in day_buckets) * 10000 if day_complete else None,
            'positive_payment_hours': sum(rate > 0 for rate in rates.values()),
            'negative_payment_hours': sum(rate < 0 for rate in rates.values()),
            'simple_annualized_observed_funding_on_perp_notional': annual,
            'assumed_entry_exit_cost_bps_before_unknown_costs': cost,
            'full_window_funding_less_assumed_cost_bps': week - cost if complete else None,
            'screen': 'unknown_history' if not complete else 'funding_did_not_cover_assumed_costs' if week <= cost else 'requires_executable_basis_and_cost_verification',
            'orders_enabled': False, 'alpha_validated': False,
            'limits': 'Past funding on constant perp notional, not realized PnL or return on total capital. Basis changes, spreads and financing are unknown. Future funding may reverse; passing this lower-bound cost screen is not alpha.'}
