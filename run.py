#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sm702 command line entry point.

Examples
--------
python run.py --mode all --rebuild
python run.py --mode quick
python run.py --mode audit
python run.py --mode predict
"""
from __future__ import annotations

import argparse
import json
import os
import time

from audit import run_audit
from backtest import run_backtest
from dataset import build_dataset, log
from predict_latest import predict_latest
from walkforward import run_walkforward

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["all", "build", "train", "backtest", "audit",
                             "predict", "quick"])
    ap.add_argument("--db", default=os.path.abspath(
        os.path.join(HERE, "..", "sm701", "stock_cache.db")))
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--data-start", default="2016-01-01")
    ap.add_argument("--sample-start", default="2017-01-01")
    ap.add_argument("--data-end", default="2026-09-30")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--dd-threshold", type=float, default=0.08)
    ap.add_argument("--newhigh-threshold", type=float, default=0.03)
    ap.add_argument("--chunk-size", type=int, default=300)
    ap.add_argument("--limit-codes", type=int, default=0)
    ap.add_argument("--test-start-year", type=int, default=2020)
    ap.add_argument("--test-end-year", type=int, default=2026)
    ap.add_argument("--train-years", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=350)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--min-data", type=int, default=500)
    ap.add_argument("--early-stop", type=int, default=40)
    ap.add_argument("--threshold-mode", default="fixed",
                    choices=["fixed", "f0.5", "f1", "f2", "precision",
                             "utility"])
    ap.add_argument("--precision-target", type=float, default=0.30)
    ap.add_argument("--fixed-threshold", type=float, default=0.25)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cost-buy", type=float, default=0.001)
    ap.add_argument("--cost-sell", type=float, default=0.001)
    ap.add_argument("--cooldown", type=int, default=10)
    ap.add_argument("--reentry", default="clear",
                    choices=["clear", "cooldown", "never"])
    ap.add_argument("--universe-top", type=int, default=1500)
    ap.add_argument("--min-price", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--market", default="cn", choices=["cn", "crypto"],
                    help="cn: A 股; crypto: sm701 crypto_daily USDT 合约")
    ap.add_argument("--min-train", type=int, default=50000,
                    help="walk-forward 训练集最小样本数（crypto 可调小）")
    ap.add_argument("--rebuild", action="store_true")
    return ap.parse_args()


def _dataset_ok(args) -> bool:
    meta_path = os.path.join(args.out, "meta.npz")
    feat_path = os.path.join(args.out, "features.npy")
    cfg_path = os.path.join(args.out, "build_config.json")
    if not (os.path.exists(meta_path) and os.path.exists(feat_path)
            and os.path.exists(cfg_path)):
        return False
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    return (cfg.get("start") == args.data_start
            and cfg.get("end") == args.data_end
            and cfg.get("horizon") == args.horizon
            and cfg.get("dd_threshold") == args.dd_threshold
            and cfg.get("newhigh_threshold") == args.newhigh_threshold
            and cfg.get("limit_codes") == args.limit_codes
            and cfg.get("market") == args.market)


def main():
    args = parse_args()
    if args.mode == "quick":
        args.out = os.path.join(HERE, "outputs_quick")
        args.limit_codes = 300
        args.test_start_year = 2025
        args.test_end_year = 2025
        args.train_years = 3
        args.rounds = 200
        args.universe_top = 300
        args.rebuild = True
    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, "meta.npz")
    need_build = args.rebuild or not _dataset_ok(args)
    if args.mode in ("all", "build", "quick", "train", "backtest") and need_build:
        build_dataset(
            args.out, args.db, args.data_start, args.data_end,
            args.horizon, args.dd_threshold, args.newhigh_threshold,
            chunk_size=args.chunk_size, limit_codes=args.limit_codes,
            seed=args.seed, market=args.market)
    if args.mode == "build":
        return
    if args.mode in ("all", "quick"):
        run_audit(args.db, args.out, args.horizon, market=args.market)
    if args.mode == "audit":
        run_audit(args.db, args.out if os.path.exists(meta_path) else None,
                  args.horizon, market=args.market)
        return
    if args.mode in ("all", "quick", "train"):
        t0 = time.time()
        run_walkforward(
            args.out, args.db, args.sample_start, args.test_start_year,
            args.test_end_year, args.train_years, args.horizon, args.rounds,
            args.lr, args.leaves, args.min_data, args.early_stop,
            args.threshold_mode, args.precision_target,
            args.fixed_threshold, args.threads, seed=args.seed,
            market=args.market, min_train=args.min_train)
        log(f"walk-forward 完成，用时 {(time.time() - t0) / 60:.1f} min")
    if args.mode in ("all", "quick", "backtest"):
        run_backtest(args.out, args.test_start_year, args.test_end_year,
                     args.cost_buy, args.cost_sell, args.cooldown,
                     args.reentry, args.universe_top, args.min_price,
                     market=args.market)
    if args.mode in ("all", "quick", "predict"):
        predict_latest(args.out, args.db, market=args.market,
                       min_price=args.min_price)
    with open(os.path.join(args.out, "run_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=1)
    log("完成")


if __name__ == "__main__":
    main()
