# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for tushare_skill (HTTP mocked; no token / no credits).

Run:
    python -m unittest discover -s tests -v
    TUSHARE_LIVE=1 python -m unittest discover -s tests -v   # + real-API smoke
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

SKILL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_DIR))

import run as runner  # noqa: E402
import schema  # noqa: E402


# ---------------------------------------------------------------------------
# schema.py
# ---------------------------------------------------------------------------


class TestSchema(unittest.TestCase):
    def test_normalize_date(self):
        self.assertEqual(schema.normalize_date("20240701"), "20240701")
        self.assertEqual(schema.normalize_date("2024-07-01"), "20240701")
        for bad in ("202407", "2024/07/01", "20241301", "abc"):
            with self.assertRaises(ValueError):
                schema.normalize_date(bad)

    def test_validate_ts_code(self):
        self.assertEqual(schema.validate_ts_code("600519.sh"), "600519.SH")
        self.assertEqual(schema.validate_ts_code("000001.SZ"), "000001.SZ")
        for bad in ("600519", "600519.XX", "60051.SH", "AAPL"):
            with self.assertRaises(ValueError):
                schema.validate_ts_code(bad)

    def test_model_columns(self):
        cols = schema.model_columns("daily")
        self.assertEqual(cols[:2], ["ts_code", "trade_date"])
        self.assertIn("close_qfq", cols)
        self.assertEqual(schema.model_columns("trade-cal"), ["cal_date", "is_open"])

    def test_envelope(self):
        items = [{"a": 1}, {"a": 2}]
        body = schema.envelope("daily", {"x": 1, "y": None}, items, as_of="20240701")
        self.assertEqual(body["count"], 2)
        self.assertEqual(body["items"], items)
        self.assertNotIn("y", body["params"])

        body_out = schema.envelope("daily", {}, items, out="/tmp/f.csv")
        self.assertNotIn("items", body_out)
        self.assertEqual(body_out["out"], "/tmp/f.csv")


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------


class TestToken(unittest.TestCase):
    def test_parse_token_text(self):
        self.assertEqual(runner._parse_token_text("abc123\n"), "abc123")
        self.assertEqual(runner._parse_token_text('TUSHARE_TOKEN="abc123"'), "abc123")
        self.assertEqual(runner._parse_token_text("TOKEN = 'x'"), "x")

    def test_env_var_wins(self):
        cfg = {"token": {"env_var": "TUSHARE_TOKEN", "file": "", "token_walk_filename": "nope"}}
        with mock.patch.dict(os.environ, {"TUSHARE_TOKEN": "from-env"}):
            self.assertEqual(runner.resolve_token(cfg), "from-env")

    def test_explicit_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("TUSHARE_TOKEN=from-file")
            path = f.name
        try:
            cfg = {"token": {"env_var": "TUSHARE_TOKEN", "file": path, "token_walk_filename": "nope"}}
            with mock.patch.dict(os.environ):
                os.environ.pop("TUSHARE_TOKEN", None)
                self.assertEqual(runner.resolve_token(cfg), "from-file")
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Point-in-time helpers + qfq math (HTTP mocked)
# ---------------------------------------------------------------------------


def _payload(fields: list[str], items: list[list]) -> dict:
    return {"code": 0, "msg": None, "data": {"fields": fields, "items": items}}


_DAILY_FIELDS = [
    "ts_code", "trade_date", "open", "high", "low", "close",
    "pre_close", "pct_chg", "vol", "amount",
]
_DAILY_ITEMS = [
    # descending order, as the real API returns
    ["600519.SH", "20240702", 30, 30, 30, 30.0, 20, 50.0, 1, 1],
    ["600519.SH", "20240701", 20, 20, 20, 20.0, 10, 100.0, 1, 1],
    ["600519.SH", "20240628", 10, 10, 10, 10.0, 10, 0.0, 1, 1],
]
_ADJ_ITEMS = [
    ["20240702", 2.5],
    ["20240701", 2.0],
    ["20240628", 1.0],
]
_FIN_FIELDS = [
    "ts_code", "ann_date", "end_date", "eps", "roe",
    "grossprofit_margin", "netprofit_margin", "or_yoy", "netprofit_yoy",
]
_FIN_ITEMS = [
    ["600519.SH", "20240430", "20240331", 1.0, 5.0, 90.0, 50.0, 10.0, 12.0],
    ["600519.SH", "20240830", "20240630", 2.0, 9.0, 90.0, 50.0, 11.0, 13.0],  # future vs as_of
    ["600519.SH", None, "20231231", 3.0, 15.0, 90.0, 50.0, 12.0, 14.0],       # missing ann_date
]


def _fake_post(url, json=None, timeout=None):  # noqa: A002 - mirror requests signature
    api = json["api_name"]
    resp = mock.Mock()
    resp.raise_for_status = mock.Mock()
    if api == "daily":
        resp.json = mock.Mock(return_value=_payload(_DAILY_FIELDS, _DAILY_ITEMS))
    elif api == "adj_factor":
        resp.json = mock.Mock(return_value=_payload(["trade_date", "adj_factor"], _ADJ_ITEMS))
    elif api == "fina_indicator":
        resp.json = mock.Mock(return_value=_payload(_FIN_FIELDS, _FIN_ITEMS))
    else:
        raise AssertionError(f"unexpected api {api}")
    return resp


class TestPointInTime(unittest.TestCase):
    def setUp(self):
        self.cfg = runner.load_config()

    def test_effective_end(self):
        self.assertEqual(runner.effective_end("20241231", "20240701"), "20240701")
        self.assertEqual(runner.effective_end(None, "20240701"), "20240701")
        self.assertEqual(runner.effective_end("20240630", None), "20240630")
        self.assertIsNone(runner.effective_end(None, None))

    @mock.patch.object(runner.requests, "post", side_effect=_fake_post)
    def test_daily_qfq_and_asof_cut(self, _post):
        args = Namespace(
            ts_code="600519.SH", start="20240601", end=None, as_of="20240701",
            adjust="qfq", fields=None,
        )
        df, params = runner.cmd_daily(args, self.cfg, token="t")

        # 20240702 row cut by as_of
        self.assertEqual(list(df["trade_date"]), ["20240628", "20240701"])
        self.assertEqual(params["end_date"], "20240701")

        # qfq basis = factor at as_of (2.0): 10*1/2=5, 20*2/2=20
        self.assertAlmostEqual(df["close_qfq"].iloc[0], 5.0)
        self.assertAlmostEqual(df["close_qfq"].iloc[1], 20.0)

    @mock.patch.object(runner.requests, "post", side_effect=_fake_post)
    def test_financials_ann_date_cut(self, _post):
        args = Namespace(
            ts_code="600519.SH", start=None, end=None, as_of="20240701", fields=None,
        )
        df, _ = runner.cmd_financials(args, self.cfg, token="t")
        # future announcement and null ann_date both dropped
        self.assertEqual(list(df["ann_date"]), ["20240430"])

    @mock.patch.object(runner.requests, "post", side_effect=_fake_post)
    def test_emit_envelope_stdout(self, _post):
        args = Namespace(
            ts_code="600519.SH", start=None, end=None, as_of="20240701",
            adjust="raw", fields=None, format="json", out=None,
        )
        df, params = runner.cmd_daily(args, self.cfg, token="t")
        buf = io.StringIO()
        with redirect_stdout(buf):
            runner.emit(df, "daily", params, args)
        body = json.loads(buf.getvalue())
        self.assertEqual(body["api"], "daily")
        self.assertEqual(body["count"], 2)
        self.assertEqual(len(body["items"]), 2)


class TestRetry(unittest.TestCase):
    def test_rate_limit_then_success(self):
        cfg = runner.load_config()
        cfg["request"] = {"timeout_seconds": 1, "max_retries": 2, "backoff_seconds": 0}

        limited = mock.Mock()
        limited.raise_for_status = mock.Mock()
        limited.json = mock.Mock(return_value={"code": -1, "msg": "抱歉，您每分钟最多访问该接口1次"})
        ok = mock.Mock()
        ok.raise_for_status = mock.Mock()
        ok.json = mock.Mock(return_value=_payload(["cal_date", "is_open"], [["20240701", 1]]))

        with mock.patch.object(runner.requests, "post", side_effect=[limited, ok]):
            df = runner.tushare_query(cfg, "t", "trade_cal", {}, "cal_date,is_open")
        self.assertEqual(len(df), 1)

    def test_non_retryable_error_raises(self):
        cfg = runner.load_config()
        cfg["request"] = {"timeout_seconds": 1, "max_retries": 2, "backoff_seconds": 0}
        bad = mock.Mock()
        bad.raise_for_status = mock.Mock()
        bad.json = mock.Mock(return_value={"code": -1, "msg": "token无效"})
        with mock.patch.object(runner.requests, "post", return_value=bad):
            with self.assertRaises(RuntimeError):
                runner.tushare_query(cfg, "t", "daily", {}, "f")


# ---------------------------------------------------------------------------
# Optional live smoke (needs token + network + credits)
# ---------------------------------------------------------------------------


@unittest.skipUnless(os.getenv("TUSHARE_LIVE") == "1", "set TUSHARE_LIVE=1 for live smoke")
class TestLiveSmoke(unittest.TestCase):
    def test_daily_live(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            runner.main([
                "daily", "--ts-code", "600519.SH",
                "--start", "20240601", "--as-of", "20240701",
            ])
        body = json.loads(buf.getvalue())
        self.assertGreater(body["count"], 0)
        self.assertTrue(all(r["trade_date"] <= "20240701" for r in body["items"]))
        self.assertIn("close_qfq", body["items"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
