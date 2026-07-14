"""Offline tests for the point-in-time relative-momentum skill."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

SKILL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SKILL_DIR.parents[2]
sys.path.insert(0, str(SKILL_DIR))
sys.path.insert(0, str(REPO_ROOT))

import run as runner  # noqa: E402
from src.utils.ashare_momentum import (  # noqa: E402
    build_relative_momentum_baseline,
)


def _write_fixture(data_dir: Path) -> tuple[dict[str, dict[str, str]], str]:
    dates = pd.bdate_range("2024-01-02", periods=27).strftime("%Y%m%d").tolist()
    as_of = dates[-2]
    codes = [f"{index:06d}.SZ" for index in range(1, 6)]
    pool = {
        code: {"name": f"stock-{index}", "industry": "test"}
        for index, code in enumerate(codes, start=1)
    }
    (data_dir / "meta.json").write_text(
        json.dumps(
            {
                "index_code": "000300.SH",
                "stock_pool": pool,
                "start_date": dates[0],
                "end_date": dates[-1],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "trade_date": dates,
            "close": [100.0 + 0.05 * day for day in range(len(dates))],
        }
    ).to_csv(data_dir / "index_000300.SH.csv", index=False)

    rates = [0.15, 0.12, 0.09, 0.06, 0.03]
    for code, rate in zip(codes, rates):
        closes = [
            100.0 * (1.0 + rate * day / (len(dates) - 2))
            for day in range(len(dates) - 1)
        ]
        closes.append(1_000_000.0)
        pd.DataFrame(
            {
                "trade_date": dates,
                "close_qfq": closes,
                "vol": [100.0] * len(dates),
                "amount": [1000.0] * len(dates),
            }
        ).to_csv(data_dir / f"daily_{code}.csv", index=False)
    return pool, as_of


class TestMomentumBaseline(unittest.TestCase):
    def test_top4_ranking_and_future_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pool, as_of = _write_fixture(data_dir)
            first = build_relative_momentum_baseline(
                as_of,
                pool,
                data_dir=data_dir,
            )
            expected = list(pool)[:4]
            self.assertEqual(list(first["weights"]), expected)
            self.assertTrue(all(weight == 0.25 for weight in first["weights"].values()))
            self.assertEqual(first["cash"], 0.0)

            for code in pool:
                path = data_dir / f"daily_{code}.csv"
                frame = pd.read_csv(path, dtype={"trade_date": str})
                frame.loc[frame["trade_date"] > as_of, "close_qfq"] = -999_999.0
                frame.to_csv(path, index=False)
            second = build_relative_momentum_baseline(
                as_of,
                pool,
                data_dir=data_dir,
            )
            self.assertEqual(first, second)

    def test_rejects_invalid_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pool, as_of = _write_fixture(data_dir)
            with self.assertRaises(ValueError):
                build_relative_momentum_baseline(
                    as_of,
                    pool,
                    data_dir=data_dir,
                    window=21,
                )
            with self.assertRaises(ValueError):
                build_relative_momentum_baseline(
                    as_of,
                    pool,
                    data_dir=data_dir,
                    top_k=5,
                )

    def test_cli_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _, as_of = _write_fixture(data_dir)
            output = io.StringIO()
            with redirect_stdout(output):
                runner.main(
                    [
                        "--as-of",
                        as_of,
                        "--data-dir",
                        str(data_dir),
                    ]
                )
            result = json.loads(output.getvalue())
            self.assertEqual(result["as_of"], as_of)
            self.assertEqual(len(result["weights"]), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
