"""Dated Farside BTC ETF totals. Missing cells are never interpreted as zero."""
import math
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser

import requests
from loguru import logger

from collectors.base import BaseCollector


class _Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.table = None
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def _number(text):
    text = text.strip().replace(",", "").replace("−", "-").replace("\xa0", "")
    if text in ("", "-", "–", "—"):
        return None
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    try:
        value = float(text)
        return value if math.isfinite(value) else None
    except ValueError:
        return None


def parse_flows(html, today=None):
    """Use Date/Total columns; 7d means seven calendar days ending on the flow date."""
    today = today or datetime.now(timezone.utc).date()
    parser = _Tables()
    parser.feed(html)
    published = {}
    pending = []
    for table in parser.tables:
        header = next((row for row in table if "total" in [c.lower() for c in row]
                       and ("date" in [c.lower() for c in row]
                            or (len(row) > 2 and not row[0] and row[-1].lower() == "total"))), None)
        if header is None:
            continue
        names = [c.lower() for c in header]
        date_col = names.index("date") if "date" in names else 0
        total_col = names.index("total")
        for row in table:
            if len(row) != len(header):
                continue
            try:
                date = datetime.strptime(row[date_col], "%d %b %Y").date()
            except ValueError:
                continue
            if date > today:
                continue
            funds = [_number(cell) for i, cell in enumerate(row) if i not in (date_col, total_col)]
            total = _number(row[total_col])
            if not funds or any(value is None for value in funds) or total is None:
                pending.append(date)
                continue
            if not math.isclose(sum(funds), total, abs_tol=.1 * (len(funds) + 1)):
                raise ValueError("ETF fund columns do not reconcile with Total")
            if date in published and published[date] != total:
                raise ValueError("Conflicting dated ETF rows")
            published[date] = total
    if not published:
        return {}
    latest = max(published)
    if (today - latest).days > 4:
        return {}
    window = {date: value for date, value in published.items() if date >= latest - timedelta(days=6)}
    return {
        "etf_net_flow_daily": published[latest] * 1_000_000,
        "etf_net_flow_7d": sum(window.values()) * 1_000_000,
        "etf_flow_date": latest.isoformat(),
        "etf_flow_days_in_7d": len(window),
        "etf_flow_window": "7_calendar_days_ending_on_flow_date",
        "etf_pending_date": max(pending).isoformat() if pending else None,
    }


class ETFFlowCollector(BaseCollector):
    def __init__(self, config):
        super().__init__("ETFFlow", config)
        self._identity = None
        self._observed_at = None

    def fetch(self):
        data = self._scrape_farside()
        if data:
            identity = (data["etf_flow_date"], data["etf_net_flow_daily"], data["etf_net_flow_7d"])
            if identity != self._identity:
                self._identity, self._observed_at = identity, datetime.now(timezone.utc).timestamp()
        return data

    def observation_times(self, data, fetched_at):
        return {key: self._observed_at if key.startswith("etf_net_flow_") else fetched_at for key in data}

    def _scrape_farside(self):
        try:
            response = requests.get("https://farside.co.uk/btc/", headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            response.raise_for_status()
            return parse_flows(response.text)
        except (requests.RequestException, ValueError) as exc:
            logger.warning("[ETFFlow] Source unavailable or invalid: {}", exc)
            return {}

    def health_check(self):
        return bool(self._scrape_farside())
