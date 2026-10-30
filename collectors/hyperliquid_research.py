"""Only public /info reads. No signing, exchange/order endpoint or credentials."""
import re

import requests


class PublicHyperliquid:
    ALLOWED_TYPES = frozenset({"metaAndAssetCtxs", "recentTrades", "clearinghouseState",
                               "userFillsByTime", "userNonFundingLedgerUpdates", "fundingHistory", "candleSnapshot",
                               "userTwapSliceFills"})

    def __init__(self, session=None):
        self.session = session or requests.Session()

    def close(self):
        self.session.close()

    def query(self, kind, **params):
        if kind not in self.ALLOWED_TYPES:
            raise ValueError("Only explicitly allowed public information queries are supported")
        if "type" in params:
            raise ValueError("Query type cannot be overridden")
        if "user" in params and not re.fullmatch(r"0x[0-9a-fA-F]{40}", params["user"]):
            raise ValueError("Valid public account address required")
        response = self.session.post("https://api.hyperliquid.xyz/info", json={"type": kind, **params}, timeout=20)
        response.raise_for_status()
        return response.json()
