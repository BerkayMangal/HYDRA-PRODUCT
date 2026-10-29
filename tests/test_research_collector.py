import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.lab.okx_shadow import (
    connect,
    insert_trades,
    get,
    save_snapshot,
    cycle,
    recover_window,
)


class CollectorIntegrity(unittest.TestCase):
    def setUp(self):
        self.db = connect(":memory:")
        self.trade = {
            "tradeId": "123",
            "ts": "1793127600000",
            "side": "buy",
            "px": "80000",
            "sz": "5",
        }

    def tearDown(self):
        self.db.close()

    def test_contract_to_base_and_quote_units(self):
        insert_trades(self.db, "BTC-USDT-SWAP", [self.trade], 100, 0.01)
        row = self.db.execute("SELECT btc_size,quote_size FROM trades").fetchone()
        self.assertAlmostEqual(row[0], 0.05)
        self.assertAlmostEqual(row[1], 4000.0)

    def test_duplicate_is_not_new_observation(self):
        self.assertEqual(insert_trades(self.db, "BTC-USDT", [self.trade], 100, 1.0), 1)
        self.assertEqual(insert_trades(self.db, "BTC-USDT", [self.trade], 200, 1.0), 0)
        self.assertEqual(
            self.db.execute("SELECT observed_ms FROM trades").fetchone()[0], 100
        )

    def test_trade_side_validation(self):
        with self.assertRaises(ValueError):
            insert_trades(
                self.db, "BTC-USDT", [dict(self.trade, side="unknown")], 100, 1.0
            )

    def test_nonfinite_trade_values_are_rejected(self):
        for bad in ["nan", "inf", "-inf"]:
            with self.assertRaises(ValueError):
                insert_trades(self.db, "BTC-USDT", [dict(self.trade, sz=bad)], 100, 1.0)

    def test_private_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            get("trade/order", {})

    def test_snapshot_preserves_payload(self):
        data = {"original": 303432780707.017}
        save_snapshot(self.db, 100, "sample", "BTC-USDT", data)
        self.assertEqual(data["original"], 303432780707.017)
        self.assertEqual(
            self.db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0], 1
        )

    def test_gap_recovery_requires_known_trade_id(self):
        insert_trades(self.db, "BTC-USDT", [self.trade], 100, 1.0)
        rows = [dict(self.trade, tradeId="130", ts="1793127600001")]
        with patch(
            "research.lab.okx_shadow.get",
            return_value={"code": "0", "data": [self.trade]},
        ) as request:
            backfills, complete = recover_window(self.db, "BTC-USDT", rows)
        self.assertTrue(complete)
        self.assertEqual(len(backfills), 1)
        self.assertEqual(request.call_args.args[0], "market/history-trades")

    def test_same_timestamp_unknown_id_is_not_overlap(self):
        insert_trades(self.db, "BTC-USDT", [self.trade], 100, 1.0)
        with patch(
            "research.lab.okx_shadow.get", return_value={"code": "0", "data": []}
        ):
            _, complete = recover_window(
                self.db, "BTC-USDT", [dict(self.trade, tradeId="124")]
            )
        self.assertFalse(complete)


if __name__ == "__main__":
    unittest.main()
