#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""qlib_skill CLI — Microsoft Qlib wrapped as a standardized skill.

Subcommands:
    convert   Tushare CSV cache -> qlib bin provider dir (no qlib needed)
    train     LGBM + Alpha158 on config segments; saves model/pred/label
    predict   score an arbitrary window with a trained model
    signal    daily cross-sectional IC / RankIC / ICIR from pred vs label
    backtest  TopkDropoutStrategy portfolio backtest vs benchmark
    report    aggregate meta/signal/backtest into outputs/<run>/report.md

`convert`, `signal` and `report` run on plain numpy/pandas; `train`,
`predict` and `backtest` require the Qlib conda env
(deploy/conda/setup_qlib.sh -> /home/msj_team/.conda/envs/Qlib).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SKILL_DIR))

# qlib's model.fit logs metrics through mlflow; mlflow >= 3.14 refuses the
# file backend unless explicitly allowed. We don't use the tracking UI, so
# opt in and keep the store under outputs/ (gitignored).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from schema import (  # noqa: E402
    BacktestMetrics,
    SignalMetrics,
    envelope,
    label_expression,
    normalize_date,
    validate_run_name,
)


def load_config() -> dict[str, Any]:
    with open(SKILL_DIR / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]
    for key in ("csv_cache_dir", "provider_uri"):
        data[key] = str((SKILL_DIR / data[key]).resolve())
    return cfg


def run_dir(cfg: dict, run_name: str) -> Path:
    d = SKILL_DIR / cfg.get("output_dir", "outputs") / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def require_qlib():
    try:
        import qlib  # noqa: F401

        return qlib
    except ImportError:
        sys.exit(
            "pyqlib is not importable in this interpreter. Use the Qlib conda env:\n"
            "  /home/msj_team/.conda/envs/Qlib/bin/python run.py ...\n"
            "(create it with deploy/conda/setup_qlib.sh)"
        )


_QLIB_INITIALIZED = False


def init_qlib(cfg: dict) -> None:
    """Idempotent qlib.init (re-init in one process raises RecorderInitializationError)."""
    global _QLIB_INITIALIZED
    if _QLIB_INITIALIZED:
        return
    qlib = require_qlib()
    mlruns = SKILL_DIR / cfg.get("output_dir", "outputs") / "mlruns"
    mlruns.mkdir(parents=True, exist_ok=True)
    qlib.init(
        provider_uri=cfg["data"]["provider_uri"],
        region="cn",
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {"uri": mlruns.as_uri(), "default_exp_name": "Experiment"},
        },
    )
    _QLIB_INITIALIZED = True


def print_envelope(body: dict[str, Any]) -> None:
    print(json.dumps(body, ensure_ascii=False, indent=2, default=str))


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


def cmd_convert(args: argparse.Namespace, cfg: dict) -> None:
    from qlib_dump import convert

    summary = convert(cfg["data"]["csv_cache_dir"], cfg["data"]["provider_uri"])
    print_envelope(envelope("convert", {"src": cfg["data"]["csv_cache_dir"]}, summary))


# ---------------------------------------------------------------------------
# train / predict
# ---------------------------------------------------------------------------


def _build_dataset(cfg: dict, segments: dict[str, tuple[str, str]]):
    """Alpha158 handler + DatasetH with the given segments."""
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH

    exp = cfg["experiment"]
    train_seg = exp["segments"]["train"]
    handler = Alpha158(
        instruments="all",
        start_time=exp["handler_start"],
        end_time=exp["handler_end"],
        fit_start_time=train_seg[0],
        fit_end_time=train_seg[1],
        label=([label_expression(int(exp["label_horizon"]))], ["LABEL0"]),
    )
    return DatasetH(handler, segments=segments)


def cmd_train(args: argparse.Namespace, cfg: dict) -> None:
    init_qlib(cfg)
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset.handler import DataHandlerLP

    exp = cfg["experiment"]
    segments = {k: tuple(v) for k, v in exp["segments"].items()}
    dataset = _build_dataset(cfg, segments)

    lgbm = exp["lgbm"]
    model = LGBModel(
        loss=lgbm.get("loss", "mse"),
        num_boost_round=int(lgbm.get("num_boost_round", 200)),
        early_stopping_rounds=int(lgbm.get("early_stopping_rounds", 30)),
        learning_rate=float(lgbm.get("learning_rate", 0.05)),
        max_depth=int(lgbm.get("max_depth", 6)),
        num_leaves=int(lgbm.get("num_leaves", 31)),
        seed=int(lgbm.get("seed", 42)),
    )
    model.fit(dataset)

    pred = model.predict(dataset, segment="test").rename("score")
    label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R)

    out = run_dir(cfg, args.run_name)
    with open(out / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    pred.to_pickle(out / "pred.pkl")
    label.to_pickle(out / "label.pkl")

    meta = {
        "run_name": args.run_name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "label_horizon": exp["label_horizon"],
        "label_expression": label_expression(int(exp["label_horizon"])),
        "segments": exp["segments"],
        "lgbm": lgbm,
        "pred_rows": int(len(pred)),
        "pred_days": int(pred.index.get_level_values(0).nunique()),
        "instruments": int(pred.index.get_level_values(1).nunique()),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print_envelope(envelope("train", {"run_name": args.run_name}, meta, out=str(out)))


def cmd_predict(args: argparse.Namespace, cfg: dict) -> None:
    init_qlib(cfg)
    out = run_dir(cfg, args.run_name)
    model_path = out / "model.pkl"
    if not model_path.exists():
        sys.exit(f"model not found: {model_path} (run `train --run-name {args.run_name}` first)")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    window = (args.start, args.end)
    dataset = _build_dataset(cfg, {"test": window})
    pred = model.predict(dataset, segment="test").rename("score")

    csv_path = out / f"pred_{args.start}_{args.end}.csv"
    pred.reset_index().to_csv(csv_path, index=False)
    payload = {
        "rows": int(len(pred)),
        "days": int(pred.index.get_level_values(0).nunique()) if len(pred) else 0,
    }
    print_envelope(
        envelope("predict", {"run_name": args.run_name, "start": args.start, "end": args.end},
                 payload, out=str(csv_path))
    )


# ---------------------------------------------------------------------------
# walkforward: monthly point-in-time scores for the agent benchmark
# ---------------------------------------------------------------------------


def _from_qlib_code(code: str) -> str:
    """SH600418 -> 600418.SH."""
    return f"{code[2:]}.{code[:2]}"


def cmd_walkforward(args: argparse.Namespace, cfg: dict) -> None:
    """One LGBM+Alpha158 model per monthly decision date.

    Each month's model trains ONLY on rows whose 20-trading-day labels are
    fully settled before that decision date, then scores the pool at the
    decision date. Output feeds the `ashare_ml_signal` MCP tool, giving the
    agent a leakage-free cross-sectional ML signal per task.
    """
    init_qlib(cfg)
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH

    exp = cfg["experiment"]
    horizon = int(exp["label_horizon"])
    train_start = exp["segments"]["train"][0]
    cache = Path(cfg["data"]["csv_cache_dir"])

    cal = pd.read_csv(cache / "trade_cal.csv", dtype={"cal_date": str})
    days = cal[cal["is_open"].astype(int) == 1]["cal_date"].sort_values().tolist()

    def iso(d: str) -> str:
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"

    lo = args.start_month.replace("-", "")
    hi = args.end_month.replace("-", "")
    firsts: dict[str, str] = {}
    for d in days:
        firsts.setdefault(d[:6], d)
    rebalances = [d for m, d in sorted(firsts.items()) if lo <= m <= hi]
    if not rebalances:
        sys.exit(f"no monthly decision dates in {args.start_month}..{args.end_month}")

    lgbm = exp["lgbm"]
    train_start_compact = train_start.replace("-", "")
    train_start_i = next(i for i, d in enumerate(days) if d >= train_start_compact)
    rows: list[dict] = []
    for day in rebalances:
        di = days.index(day)
        # Last row whose label close[T+h+1] settles strictly before `day`.
        label_end_i = di - (horizon + 1)
        valid_split_i = label_end_i - 60
        if valid_split_i <= train_start_i + 80:
            print(f"skip {day}: not enough history for train/valid")
            continue
        segments = {
            "train": (train_start, iso(days[valid_split_i])),
            "valid": (iso(days[valid_split_i + 1]), iso(days[label_end_i])),
            "test": (iso(day), iso(day)),
        }
        handler = Alpha158(
            instruments="all",
            start_time=train_start,
            end_time=iso(day),  # never sees data past the decision date
            fit_start_time=segments["train"][0],
            fit_end_time=segments["train"][1],
            label=([label_expression(horizon)], ["LABEL0"]),
        )
        dataset = DatasetH(handler, segments=segments)
        model = LGBModel(
            loss=lgbm.get("loss", "mse"),
            num_boost_round=int(lgbm.get("num_boost_round", 200)),
            early_stopping_rounds=int(lgbm.get("early_stopping_rounds", 30)),
            learning_rate=float(lgbm.get("learning_rate", 0.05)),
            max_depth=int(lgbm.get("max_depth", 6)),
            num_leaves=int(lgbm.get("num_leaves", 31)),
            seed=int(lgbm.get("seed", 42)),
        )
        model.fit(dataset)
        pred = model.predict(dataset, segment="test").rename("score").dropna()
        if pred.empty:
            print(f"skip {day}: empty prediction")
            continue
        day_scores = pred.xs(pred.index.get_level_values(0)[0], level=0)
        ranked = day_scores.sort_values(ascending=False)
        n = len(ranked)
        for rank, (inst, score) in enumerate(ranked.items(), start=1):
            rows.append(
                {
                    "entry_date": day,
                    "month": f"{day[:4]}-{day[4:6]}",
                    "ts_code": _from_qlib_code(str(inst)),
                    "qlib_code": str(inst),
                    "score": round(float(score), 6),
                    "rank": rank,
                    "n_stocks": n,
                }
            )
        print(f"  {day}: scored {n} stocks (train <= {segments['valid'][1]})")

    out_path = Path(args.out) if args.out else cache / "qlib_signal.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print_envelope(
        envelope(
            "walkforward",
            {"start_month": args.start_month, "end_month": args.end_month},
            {"months": len(rebalances), "rows": len(rows)},
            out=str(out_path),
        )
    )


# ---------------------------------------------------------------------------
# signal (plain pandas; no qlib import needed)
# ---------------------------------------------------------------------------


def cmd_signal(args: argparse.Namespace, cfg: dict) -> None:
    out = run_dir(cfg, args.run_name)
    try:
        pred = pd.read_pickle(out / "pred.pkl")
        label = pd.read_pickle(out / "label.pkl")
    except FileNotFoundError as exc:
        sys.exit(f"{exc} (run `train --run-name {args.run_name}` first)")

    df = pd.concat([pred.rename("score"), label.iloc[:, 0].rename("label")], axis=1).dropna()
    by_day = df.groupby(level=0)
    # Days with a degenerate cross-section (fewer than 3 names) yield NaN corr.
    ic = by_day.apply(lambda g: g["score"].corr(g["label"])).dropna()
    ric = by_day.apply(lambda g: g["score"].corr(g["label"], method="spearman")).dropna()

    if ic.empty:
        sys.exit("no valid cross-sectional days — check pred/label overlap")

    metrics = SignalMetrics(
        n_days=int(len(ric)),
        ic_mean=float(ic.mean()),
        ic_std=float(ic.std()),
        icir=float(ic.mean() / ic.std()) if ic.std() > 0 else float("nan"),
        rank_ic_mean=float(ric.mean()),
        rank_ic_std=float(ric.std()),
        rank_icir=float(ric.mean() / ric.std()) if ric.std() > 0 else float("nan"),
        rank_ic_positive_rate=float((ric > 0).mean()),
    )
    (out / "signal.json").write_text(json.dumps(metrics.to_dict(), indent=2))
    print_envelope(envelope("signal", {"run_name": args.run_name}, metrics.to_dict(),
                            out=str(out / "signal.json")))


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------


def cmd_backtest(args: argparse.Namespace, cfg: dict) -> None:
    init_qlib(cfg)
    from qlib.backtest import backtest as qlib_backtest
    from qlib.contrib.evaluate import risk_analysis
    from qlib.contrib.strategy import TopkDropoutStrategy

    out = run_dir(cfg, args.run_name)
    try:
        pred = pd.read_pickle(out / "pred.pkl")
    except FileNotFoundError as exc:
        sys.exit(f"{exc} (run `train --run-name {args.run_name}` first)")

    test_seg = cfg["experiment"]["segments"]["test"]
    start = args.start or test_seg[0]
    end = args.end or test_seg[1]
    bt = cfg["backtest"]
    benchmark = cfg["data"]["benchmark"]

    strategy = TopkDropoutStrategy(signal=pred, topk=int(bt["topk"]), n_drop=int(bt["n_drop"]))
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    exchange_kwargs = {
        "freq": "day",
        "deal_price": bt.get("deal_price", "close"),
        "open_cost": float(bt.get("open_cost", 0.0005)),
        "close_cost": float(bt.get("close_cost", 0.0015)),
        "min_cost": float(bt.get("min_cost", 5)),
    }
    portfolio_metric_dict, _ = qlib_backtest(
        start_time=start,
        end_time=end,
        strategy=strategy,
        executor=executor_config,
        account=float(bt.get("account", 1_000_000)),
        benchmark=benchmark,
        exchange_kwargs=exchange_kwargs,
    )
    report, _positions = portfolio_metric_dict["1day"]
    report = report.dropna(subset=["return"])

    strat_net = report["return"] - report["cost"]
    excess_net = report["return"] - report["bench"] - report["cost"]
    strat_risk = risk_analysis(strat_net, freq="day")["risk"]
    excess_risk = risk_analysis(excess_net, freq="day")["risk"]

    metrics = BacktestMetrics(
        start=str(report.index.min().date()),
        end=str(report.index.max().date()),
        annualized_return=float(strat_risk["annualized_return"]),
        max_drawdown=float(strat_risk["max_drawdown"]),
        information_ratio=float(strat_risk["information_ratio"]),
        excess_annualized_return=float(excess_risk["annualized_return"]),
        excess_information_ratio=float(excess_risk["information_ratio"]),
        excess_max_drawdown=float(excess_risk["max_drawdown"]),
        turnover_mean=float(report["turnover"].mean()),
        benchmark=benchmark,
    )
    (out / "backtest.json").write_text(json.dumps(metrics.to_dict(), indent=2))
    report.to_csv(out / "backtest_daily.csv")
    print_envelope(envelope("backtest", {"run_name": args.run_name, "start": start, "end": end},
                            metrics.to_dict(), out=str(out / "backtest.json")))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _fmt_pct(x: float) -> str:
    return f"{100 * x:+.2f}%"


def cmd_report(args: argparse.Namespace, cfg: dict) -> None:
    out = run_dir(cfg, args.run_name)

    def read_json(name: str) -> Optional[dict]:
        p = out / name
        return json.loads(p.read_text()) if p.exists() else None

    meta, signal, bt = read_json("meta.json"), read_json("signal.json"), read_json("backtest.json")
    if not meta:
        sys.exit(f"meta.json not found in {out} — run train first")

    lines = [
        f"# qlib_skill 实验报告：{args.run_name}",
        "",
        f"- 训练时间: {meta['trained_at']}",
        f"- 模型: LightGBM + Alpha158（label = {meta['label_expression']}，horizon {meta['label_horizon']} 交易日）",
        f"- 段切分: train {meta['segments']['train']} / valid {meta['segments']['valid']} / test {meta['segments']['test']}",
        f"- 测试段预测: {meta['pred_rows']} 行 / {meta['pred_days']} 天 / {meta['instruments']} 只股票",
        "",
    ]
    if signal:
        lines += [
            "## 信号质量（测试段，逐日横截面）",
            "",
            "| 指标 | IC | RankIC |",
            "|------|-----|--------|",
            f"| mean | {signal['ic_mean']:.4f} | {signal['rank_ic_mean']:.4f} |",
            f"| std | {signal['ic_std']:.4f} | {signal['rank_ic_std']:.4f} |",
            f"| IR | {signal['icir']:.3f} | {signal['rank_icir']:.3f} |",
            f"| 正率 | — | {100 * signal['rank_ic_positive_rate']:.1f}% |",
            "",
            f"（有效天数 {signal['n_days']}；小股票池横截面噪声大，以整段均值为准）",
            "",
        ]
    if bt:
        lines += [
            "## 组合回测（TopkDropout，含费率，vs " + bt["benchmark"] + "）",
            "",
            f"- 区间: {bt['start']} ~ {bt['end']}",
            f"- 策略年化(扣费): {_fmt_pct(bt['annualized_return'])} | IR {bt['information_ratio']:.2f} | 最大回撤 {_fmt_pct(bt['max_drawdown'])}",
            f"- 超额年化(扣费): {_fmt_pct(bt['excess_annualized_return'])} | 超额IR {bt['excess_information_ratio']:.2f} | 超额最大回撤 {_fmt_pct(bt['excess_max_drawdown'])}",
            f"- 日均换手: {100 * bt['turnover_mean']:.2f}%",
            "",
        ]
    lines += [
        "## 口径说明",
        "",
        "- qlib 为批式流程（一次训练、整段测试），与 agent 逐题 as_of 点时预测的口径不同，对比需注明；",
        "- 特征 Alpha158 只用截至 T 的数据，label 前瞻仅存在于训练目标，test 段与 train/valid 无重叠。",
    ]
    report_path = out / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    payload = {"sections": {"meta": True, "signal": bool(signal), "backtest": bool(bt)}}
    print_envelope(envelope("report", {"run_name": args.run_name}, payload, out=str(report_path)))


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qlib_skill",
        description="Microsoft Qlib wrapped as a standardized skill (see SKILL.md)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="Tushare CSV cache -> qlib bin data")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("train", help="train LGBM+Alpha158, save model/pred/label")
    p.add_argument("--run-name", default="default", help="output subdir under outputs/")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="score an arbitrary window with a trained model")
    p.add_argument("--run-name", default="default")
    p.add_argument("--start", required=True, help="window start YYYY-MM-DD")
    p.add_argument("--end", required=True, help="window end YYYY-MM-DD")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser(
        "walkforward",
        help="monthly point-in-time scores -> qlib_signal.csv (ashare_ml_signal tool)",
    )
    p.add_argument("--start-month", default="2024-07", help="first decision month YYYY-MM")
    p.add_argument("--end-month", default="2025-06", help="last decision month YYYY-MM")
    p.add_argument("--out", default=None, help="output CSV (default: <csv_cache_dir>/qlib_signal.csv)")
    p.set_defaults(func=cmd_walkforward)

    p = sub.add_parser("signal", help="IC / RankIC / ICIR from saved pred+label")
    p.add_argument("--run-name", default="default")
    p.set_defaults(func=cmd_signal)

    p = sub.add_parser("backtest", help="TopkDropout portfolio backtest vs benchmark")
    p.add_argument("--run-name", default="default")
    p.add_argument("--start", help="override backtest start (default: test segment)")
    p.add_argument("--end", help="override backtest end (default: test segment)")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("report", help="aggregate meta/signal/backtest into report.md")
    p.add_argument("--run-name", default="default")
    p.set_defaults(func=cmd_report)

    return parser


def normalize_args(args: argparse.Namespace) -> None:
    if getattr(args, "run_name", None):
        args.run_name = validate_run_name(args.run_name)
    for attr in ("start", "end"):
        if getattr(args, attr, None):
            setattr(args, attr, normalize_date(getattr(args, attr), attr))


def main(argv: Optional[list[str]] = None) -> None:
    cfg = load_config()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        normalize_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.func(args, cfg)


if __name__ == "__main__":
    main()
