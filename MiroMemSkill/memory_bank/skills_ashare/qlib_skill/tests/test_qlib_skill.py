# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for qlib_skill.

Offline parts (schema + converter) run in any env with numpy/pandas.
The end-to-end train/signal/backtest test auto-skips when pyqlib is not
importable, and runs against the real local cache in the Qlib env:

    /home/msj_team/.conda/envs/Qlib/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

SKILL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_DIR))

import qlib_dump  # noqa: E402
import schema  # noqa: E402

try:
    import qlib  # noqa: F401

    HAS_QLIB = True
except Exception:
    HAS_QLIB = False


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


class TestSchema(unittest.TestCase):
    def test_normalize_date(self):
        self.assertEqual(schema.normalize_date("20250101"), "2025-01-01")
        self.assertEqual(schema.normalize_date("2025-01-01"), "2025-01-01")
        with self.assertRaises(ValueError):
            schema.normalize_date("2025/01/01")

    def test_validate_run_name(self):
        self.assertEqual(schema.validate_run_name("demo_1.a-b"), "demo_1.a-b")
        for bad in ("a/b", "../x", "a b", ""):
            with self.assertRaises(ValueError):
                schema.validate_run_name(bad)

    def test_label_expression(self):
        self.assertEqual(schema.label_expression(20), "Ref($close,-21)/Ref($close,-1)-1")
        with self.assertRaises(ValueError):
            schema.label_expression(0)

    def test_envelope(self):
        body = schema.envelope("train", {"a": 1, "b": None}, {"count": 3}, out="/tmp/x")
        self.assertEqual(body["count"], 3)
        self.assertEqual(body["out"], "/tmp/x")
        self.assertNotIn("b", body["params"])


# ---------------------------------------------------------------------------
# converter (synthetic cache in a tmp dir)
# ---------------------------------------------------------------------------


def _make_cache(root: Path) -> None:
    # 6 calendar days, one closed; stock suspended on 0106 (missing row).
    cal = pd.DataFrame(
        {
            "cal_date": ["20250102", "20250103", "20250104", "20250106", "20250107", "20250108"],
            "is_open": [1, 1, 0, 1, 1, 1],
        }
    )
    cal.to_csv(root / "trade_cal.csv", index=False)

    daily = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 4,
            "trade_date": ["20250102", "20250103", "20250107", "20250108"],
            "open": [10.0, 10.5, 11.0, 11.5],
            "high": [10.6, 10.9, 11.4, 11.9],
            "low": [9.9, 10.2, 10.8, 11.2],
            "close": [10.5, 10.8, 11.2, 11.6],
            "pre_close": [10.0, 10.5, 10.8, 11.2],
            "pct_chg": [5.0, 2.86, 3.7, 3.57],
            "vol": [1000.0, 1200.0, 0.0, 1500.0],
            "amount": [1050.0, 1300.0, 0.0, 1750.0],
            "adj_factor": [2.0, 2.0, 2.0, 2.0],
            "open_qfq": [10.0, 10.5, 11.0, 11.5],
            "high_qfq": [10.6, 10.9, 11.4, 11.9],
            "low_qfq": [9.9, 10.2, 10.8, 11.2],
            "close_qfq": [10.5, 10.8, 11.2, 11.6],
        }
    )
    daily.to_csv(root / "daily_000001.SZ.csv", index=False)

    index = pd.DataFrame(
        {
            "ts_code": ["000300.SH"] * 5,
            "trade_date": ["20250102", "20250103", "20250106", "20250107", "20250108"],
            "open": [4000.0, 4010, 4020, 4030, 4040],
            "high": [4050.0, 4060, 4070, 4080, 4090],
            "low": [3990.0, 4000, 4010, 4020, 4030],
            "close": [4010.0, 4020, 4030, 4040, 4050],
            "vol": [1e6] * 5,
            "amount": [1e7] * 5,
        }
    )
    index.to_csv(root / "index_000300.SH.csv", index=False)


class TestConverter(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="qlib_skill_test_"))
        self.src = self.tmp / "cache"
        self.dst = self.tmp / "provider"
        self.src.mkdir()
        _make_cache(self.src)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_convert_layout_and_alignment(self):
        summary = qlib_dump.convert(self.src, self.dst)
        self.assertEqual(summary["instruments"], 1)
        self.assertEqual(summary["benchmark"], "SH000300")

        # Calendar: only open days.
        cal = (self.dst / "calendars" / "day.txt").read_text().split()
        self.assertEqual(
            cal, ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
        )

        # Universe: stock only, index excluded.
        instruments = list(qlib_dump.iter_instruments(self.dst))
        self.assertEqual(instruments, [("SZ000001", "2025-01-02", "2025-01-08")])

        # close bin: starts at calendar idx 0, suspension day (0106) -> NaN.
        start, values = qlib_dump.read_bin(self.dst, "SZ000001", "close")
        self.assertEqual(start, 0)
        self.assertEqual(len(values), 5)
        np.testing.assert_allclose(values[[0, 1, 3, 4]], [10.5, 10.8, 11.2, 11.6], rtol=1e-6)
        self.assertTrue(np.isnan(values[2]))

        # vwap: amount*10/vol * qfq ratio (=1 here); zero-volume day -> NaN.
        _, vwap = qlib_dump.read_bin(self.dst, "SZ000001", "vwap")
        np.testing.assert_allclose(vwap[0], 1050.0 * 10 / 1000.0, rtol=1e-6)
        self.assertTrue(np.isnan(vwap[2]))

        # Benchmark bins exist.
        _, bench_close = qlib_dump.read_bin(self.dst, "SH000300", "close")
        np.testing.assert_allclose(bench_close, [4010, 4020, 4030, 4040, 4050], rtol=1e-6)

    def test_code_mapping(self):
        self.assertEqual(qlib_dump.to_qlib_code("600519.SH"), "SH600519")
        self.assertEqual(qlib_dump.to_qlib_code("300012.sz"), "SZ300012")


# ---------------------------------------------------------------------------
# end-to-end on the real cache (Qlib env only)
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_QLIB, "pyqlib not importable in this interpreter")
class TestEndToEnd(unittest.TestCase):
    RUN = "_test_e2e"

    def test_full_pipeline(self):
        import run as runner

        cfg = runner.load_config()
        if not Path(cfg["data"]["csv_cache_dir"], "trade_cal.csv").exists():
            self.skipTest("local A-share CSV cache not present")

        # Fast training profile for tests.
        cfg["experiment"]["lgbm"].update(num_boost_round=20, early_stopping_rounds=10)

        buf = io.StringIO()
        with redirect_stdout(buf):
            runner.cmd_convert(Namespace(), cfg)
            runner.cmd_train(Namespace(run_name=self.RUN), cfg)
            runner.cmd_signal(Namespace(run_name=self.RUN), cfg)
            runner.cmd_backtest(Namespace(run_name=self.RUN, start=None, end=None), cfg)
            runner.cmd_report(Namespace(run_name=self.RUN), cfg)

        out = SKILL_DIR / cfg["output_dir"] / self.RUN
        signal = json.loads((out / "signal.json").read_text())
        bt = json.loads((out / "backtest.json").read_text())
        self.assertGreater(signal["n_days"], 50)
        self.assertTrue(-1.0 < signal["rank_ic_mean"] < 1.0)
        self.assertIn("excess_annualized_return", bt)
        self.assertTrue((out / "report.md").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
