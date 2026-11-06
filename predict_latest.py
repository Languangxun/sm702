#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate today's stop-loss flags using the latest walk-forward model."""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import lightgbm as lgb
import pandas as pd

from data import connect, load_bars, load_stock_names
from features import compute_features


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def predict_latest(out_dir: str, db_path: str, asof: str | None = None,
                   lookback_days: int = 900, min_hist: int = 250,
                   min_price: float = 2.0, market: str = "cn"
                   ) -> pd.DataFrame:
    con = connect(db_path)
    if asof is None:
        if market == "crypto":
            asof = con.execute(
                "SELECT MAX(date) FROM crypto_daily").fetchone()[0]
        else:
            asof = con.execute(
                "SELECT MAX(date) FROM daily_bars").fetchone()[0]
    start = (pd.Timestamp(asof) - pd.Timedelta(days=lookback_days)).strftime(
        "%Y-%m-%d")
    log(f"加载 {start} ~ {asof} 行情 ...")
    bars = load_bars(con, start, asof, market=market)
    if market != "crypto":
        names = load_stock_names(con)
    con.close()
    if bars.empty:
        raise RuntimeError("no bars")
    bars["hist_len"] = bars.groupby("code", sort=False).cumcount() + 1
    feat = compute_features(bars)
    last_date = bars["date"].max()
    mask = bars["date"].to_numpy() == last_date
    cur = bars.loc[mask, ["code", "close", "hist_len"]].copy()
    fcur = feat.loc[mask].reset_index(drop=True)
    cur = cur.reset_index(drop=True)
    cur = cur[(cur["hist_len"] >= min_hist) & (cur["close"] >= min_price)]
    fcur = fcur.loc[cur.index]
    with open(os.path.join(out_dir, "feature_names.json"), encoding="utf-8") as f:
        feature_names = json.load(f)
    fcur = fcur.reindex(columns=feature_names)
    fold_path = os.path.join(out_dir, "fold_metrics.csv")
    model_path = None
    threshold = 0.5
    if os.path.exists(fold_path):
        folds = pd.read_csv(fold_path).sort_values("year")
        if not folds.empty:
            threshold = float(folds.iloc[-1]["threshold"])
            model_path = os.path.join(out_dir, "models",
                                      f"lgbm_{int(folds.iloc[-1]['year'])}.txt")
    if model_path is None or not os.path.exists(model_path):
        cands = sorted(glob.glob(os.path.join(out_dir, "models", "lgbm_*.txt")))
        if not cands:
            raise FileNotFoundError("no model files")
        model_path = cands[-1]
    model = lgb.Booster(model_file=model_path)
    log(f"模型 {os.path.basename(model_path)}, 阈值 {threshold:.4f}, "
        f"股票 {len(cur):,}, 日期 {last_date}")
    prob = model.predict(fcur)
    out = cur.copy()
    out["date"] = last_date
    out["prob"] = prob
    out["threshold"] = threshold
    out["stop_loss"] = (out["prob"] >= threshold).astype(int)
    for t in (0.20, 0.25, 0.30, 0.35):
        out[f"stop_loss_{int(round(t * 100)):03d}"] = (
            out["prob"] >= t).astype(int)
    out = out.copy()
    if market == "crypto":
        out["name"] = ""
        out["industry"] = ""
    else:
        out = out.merge(names, on="code", how="left")
    out = out.sort_values(["stop_loss", "prob"], ascending=[False, False])
    out.to_csv(os.path.join(out_dir, "latest_signals.csv"), index=False)
    stop = out[out["stop_loss"] == 1]
    log(f"今日止损信号: {len(stop):,}/{len(out):,} 只 -> latest_signals.csv")
    if len(stop):
        print(stop[["code", "name", "prob", "close"]].head(30).to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--db", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "sm701",
                     "stock_cache.db")))
    ap.add_argument("--asof", default=None)
    ap.add_argument("--lookback-days", type=int, default=900)
    ap.add_argument("--min-price", type=float, default=2.0)
    ap.add_argument("--market", default="cn", choices=["cn", "crypto"])
    args = ap.parse_args()
    predict_latest(args.out, args.db, args.asof, args.lookback_days,
                   min_price=args.min_price, market=args.market)
