#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stop-loss overlay backtest.

The model is evaluated as a risk overlay on an equal-weight, buy-and-hold
universe.  A stop signal at t close sells at the next open (t+1), then the
position can re-enter after a cooldown.  Orders that cannot be filled at the
open (limit-up for buys, limit-down for sells) stay pending until the next
tradable open.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from walkforward import load_dataset, log


def _metrics(eq: np.ndarray) -> dict:
    if len(eq) == 0:
        return {}
    full = np.r_[1.0, eq]
    rets = full[1:] / (full[:-1] + 1e-15) - 1.0
    total = float(full[-1] - 1.0)
    n = len(rets)
    ann = float((full[-1]) ** (252.0 / n) - 1.0) if n else 0.0
    sd = float(np.std(rets))
    sharpe = float(np.mean(rets) / (sd + 1e-12) * np.sqrt(252.0))
    mdd = float((full / np.maximum.accumulate(full) - 1.0).min())
    win = float((rets > 0).mean())
    calmar = float(ann / (abs(mdd) + 1e-12))
    return {"total_return": total, "ann_return": ann, "sharpe": sharpe,
            "mdd": mdd, "calmar": calmar, "win_daily": win,
            "days": int(n)}


def _pre_universe(meta, te_start: int, u_candidates: np.ndarray,
                  min_hist: int, min_price: float, top_n: int):
    didx = meta["date_idx"]
    cidx = meta["code_idx"]
    m = ((didx < te_start) & (didx >= te_start - 400)
         & np.isin(cidx, u_candidates))
    idx = np.flatnonzero(m)
    if len(idx) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    sub = pd.DataFrame({
        "code_idx": cidx[idx].astype(np.int64),
        "date_idx": didx[idx].astype(np.int32),
        "close": meta["close"][idx],
        "hist_len": meta["hist_len"][idx],
        "amt60": meta["avg_amount60"][idx],
    })
    last = sub.sort_values(["code_idx", "date_idx"]).groupby(
        "code_idx", sort=False).tail(1)
    last = last[(last["hist_len"] >= min_hist)
                & (last["close"] >= min_price)
                & np.isfinite(last["amt60"]) & (last["amt60"] > 0)]
    last = last.sort_values("amt60", ascending=False)
    if top_n and top_n > 0:
        last = last.head(top_n)
    codes = last["code_idx"].to_numpy(np.int32)
    return codes, last["close"].to_numpy(np.float32)


def simulate_year(meta, dates, year: int, preds: pd.DataFrame,
                  cost_buy: float, cost_sell: float, cooldown: int,
                  reentry: str, universe_top: int, min_price: float,
                  stop_enabled: bool = True) -> tuple[pd.DataFrame, dict, list]:
    year_pred = preds[preds["year"] == year].copy()
    if year_pred.empty:
        return pd.DataFrame(), {}, []
    year_pred["date_idx"] = year_pred["date"].map(
        {d: i for i, d in enumerate(dates)})
    ti0 = int(year_pred["date_idx"].min())
    ti1 = int(year_pred["date_idx"].max())
    day_ids = np.sort(year_pred["date_idx"].unique())
    te_mask = (meta["date_idx"] >= ti0) & (meta["date_idx"] <= ti1)
    te_codes = np.unique(meta["code_idx"][te_mask])
    u_codes, last_close_pre = _pre_universe(
        meta, ti0, te_codes, min_hist=250, min_price=min_price,
        top_n=universe_top)
    if len(u_codes) == 0:
        return pd.DataFrame(), {}, []
    order = np.argsort(u_codes)
    u_codes = u_codes[order]
    last_close_pre = last_close_pre[order]
    u_pos = {int(c): i for i, c in enumerate(u_codes)}
    U = len(u_codes)
    last_close = np.array(last_close_pre, dtype=np.float64)
    cash = np.full(U, 1.0 / U, dtype=np.float64)
    shares = np.zeros(U, dtype=np.float64)
    holding = np.zeros(U, dtype=bool)
    pending_buy = np.ones(U, dtype=bool)
    pending_sell = np.zeros(U, dtype=bool)
    pending_sell_p = np.full(U, np.nan)
    pending_sell_fwd = np.full(U, np.nan)
    pending_sell_dd = np.full(U, np.nan)
    cool = np.zeros(U, dtype=np.int32)
    entry_date = np.full(U, -1, dtype=np.int64)
    entry_price = np.full(U, np.nan)
    entry_p = np.full(U, np.nan)

    meta_idx = np.flatnonzero(te_mask & np.isin(meta["code_idx"], u_codes))
    sim = pd.DataFrame({
        "date_idx": meta["date_idx"][meta_idx].astype(np.int32),
        "code_idx": meta["code_idx"][meta_idx].astype(np.int32),
        "open": meta["open"][meta_idx].astype(np.float64),
        "close": meta["close"][meta_idx].astype(np.float64),
        "prev_close": meta["prev_close"][meta_idx].astype(np.float64),
        "lim": meta["lim"][meta_idx].astype(np.float64),
    })
    sim["pos"] = sim["code_idx"].map(u_pos).astype(np.int32)
    yp = year_pred[["date_idx", "code_idx", "prob", "signal", "threshold",
                    "fwd_ret_h", "fwd_maxdd"]].copy()
    yp["date_idx"] = yp["date_idx"].astype(np.int32)
    yp["code_idx"] = yp["code_idx"].astype(np.int32)
    sim = sim.merge(yp, on=["date_idx", "code_idx"], how="left")
    sim = sim.sort_values(["date_idx", "pos"])
    groups = {int(d): g for d, g in sim.groupby("date_idx", sort=False)}

    eq = []
    trade_rows = []
    n_buys = 0
    n_stops = 0
    exposure = []
    for d in day_ids:
        g = groups.get(int(d))
        if g is None:
            eq.append(float(cash.sum() + (shares * last_close).sum()))
            exposure.append(0.0)
            continue
        pos = g["pos"].to_numpy(np.int32)
        op = g["open"].to_numpy(np.float64)
        cl = g["close"].to_numpy(np.float64)
        pc = g["prev_close"].to_numpy(np.float64)
        lim = g["lim"].to_numpy(np.float64)
        buy_ok = op < pc * (1.0 + lim - 0.005)
        sell_ok = op > pc * (1.0 - lim + 0.005)
        prob = g["prob"].to_numpy(np.float64)
        sig = g["signal"].to_numpy(np.float64)
        thr = g["threshold"].to_numpy(np.float64)
        fwd = g["fwd_ret_h"].to_numpy(np.float64)
        fdd = g["fwd_maxdd"].to_numpy(np.float64)

        for j, i in enumerate(pos):
            if pending_sell[i]:
                if sell_ok[j]:
                    cash[i] = shares[i] * op[j] * (1.0 - cost_sell)
                    hold_days = int(d - entry_date[i])
                    trade_rows.append({
                        "year": year,
                        "code_idx": int(u_codes[i]),
                        "entry_date_idx": int(entry_date[i]),
                        "exit_date_idx": int(d),
                        "entry_price": float(entry_price[i]),
                        "exit_price": float(op[j]),
                        "holding_days": hold_days,
                        "ret": float(op[j] / entry_price[i] - 1.0),
                        "entry_p": float(entry_p[i]),
                        "signal_p": float(pending_sell_p[i]),
                        "fwd_ret_h": float(pending_sell_fwd[i]),
                        "fwd_maxdd": float(pending_sell_dd[i]),
                    })
                    shares[i] = 0.0
                    holding[i] = False
                    pending_sell[i] = False
                    pending_sell_p[i] = np.nan
                    pending_sell_fwd[i] = np.nan
                    pending_sell_dd[i] = np.nan
                    cool[i] = cooldown
                    n_stops += 1
            elif pending_buy[i] and not holding[i]:
                if buy_ok[j]:
                    shares[i] = cash[i] * (1.0 - cost_buy) / op[j]
                    cash[i] = 0.0
                    holding[i] = True
                    pending_buy[i] = False
                    entry_date[i] = d
                    entry_price[i] = op[j]
                    entry_p[i] = prob[j] if np.isfinite(prob[j]) else np.nan
                    n_buys += 1

        last_close[pos] = cl
        eq.append(float(cash.sum() + (shares * last_close).sum()))
        exposure.append(float(holding.sum()) / U)

        if stop_enabled:
            for j, i in enumerate(pos):
                p = prob[j]
                if not np.isfinite(p):
                    continue
                if holding[i]:
                    if sig[j] > 0 and not pending_sell[i]:
                        pending_sell[i] = True
                        pending_sell_p[i] = p
                        pending_sell_fwd[i] = fwd[j]
                        pending_sell_dd[i] = fdd[j]
                else:
                    if pending_buy[i] and sig[j] > 0:
                        pending_buy[i] = False
                    if cool[i] <= 0 and not pending_buy[i]:
                        if reentry == "never":
                            ok = False
                        elif reentry == "cooldown":
                            ok = True
                        else:
                            t = thr[j] if np.isfinite(thr[j]) else 0.5
                            ok = p < t
                        if ok:
                            pending_buy[i] = True
        cool[~holding] = np.maximum(cool[~holding] - 1, 0)

    final = float(cash.sum() + (shares * last_close * (1.0 - cost_sell)).sum())
    if len(eq) > 0:
        eq[-1] = final
    daily = pd.DataFrame({"date_idx": day_ids, "year": year, "equity": eq})
    m = _metrics(np.asarray(eq, dtype=np.float64))
    m.update({"year": year, "universe": int(U), "n_buys": int(n_buys),
              "n_stops": int(n_stops),
              "avg_exposure": float(np.mean(exposure)) if exposure else 0.0,
              "final_equity": final})
    tr = pd.DataFrame(trade_rows)
    return daily, m, tr


def run_backtest(out_dir: str, test_start_year: int, test_end_year: int,
                 cost_buy: float = 0.001, cost_sell: float = 0.001,
                 cooldown: int = 10, reentry: str = "clear",
                 universe_top: int = 1500, min_price: float = 2.0) -> None:
    meta, X, codes, dates, feature_names = load_dataset(out_dir)
    dates = [str(x) for x in dates]
    preds = pd.read_csv(os.path.join(out_dir, "predictions_wf.csv.gz"))
    # fix code strings that pandas may parse as e.g. sz000001 (no issue) and
    # attach integer code/date positions
    code_to_i = {str(c): i for i, c in enumerate(codes)}
    date_to_i = {d: i for i, d in enumerate(dates)}
    preds["code_idx"] = preds["code"].astype(str).map(code_to_i)
    preds["date_idx"] = preds["date"].astype(str).map(date_to_i)
    if preds["code_idx"].isna().any() or preds["date_idx"].isna().any():
        raise RuntimeError("predictions contain unknown code/date")
    preds["code_idx"] = preds["code_idx"].astype(np.int32)
    preds["date_idx"] = preds["date_idx"].astype(np.int32)
    years = sorted(preds["year"].unique())
    daily_stop_all = []
    daily_bench_all = []
    rows = []
    trades_all = []
    for year in years:
        if year < test_start_year or year > test_end_year:
            continue
        d_stop, m_stop, tr = simulate_year(
            meta, dates, int(year), preds, cost_buy, cost_sell, cooldown,
            reentry, universe_top, min_price, stop_enabled=True)
        d_bench, m_bench, _ = simulate_year(
            meta, dates, int(year), preds, cost_buy, cost_sell, cooldown,
            reentry, universe_top, min_price, stop_enabled=False)
        if d_stop.empty:
            continue
        row = {"year": int(year), "universe": int(m_stop["universe"])}
        for k in ("total_return", "ann_return", "sharpe", "mdd", "calmar",
                  "win_daily", "avg_exposure"):
            row[f"stop_{k}"] = m_stop.get(k)
        row["n_stops"] = int(m_stop["n_stops"])
        row["n_buys"] = int(m_stop["n_buys"])
        for k in ("total_return", "ann_return", "sharpe", "mdd", "calmar",
                  "win_daily", "avg_exposure"):
            row[f"bench_{k}"] = m_bench.get(k)
        row["excess_return"] = row["stop_total_return"] - row["bench_total_return"]
        row["mdd_improvement"] = row["bench_mdd"] - row["stop_mdd"]
        rows.append(row)
        daily_stop_all.append(d_stop)
        daily_bench_all.append(d_bench)
        if not tr.empty:
            trades_all.extend(tr.to_dict("records"))
        log(f"[{year}] stop ann={row['stop_ann_return']:.2%} "
            f"mdd={row['stop_mdd']:.2%} | bench ann="
            f"{row['bench_ann_return']:.2%} mdd={row['bench_mdd']:.2%} | "
            f"U={row['universe']} stops={row['n_stops']}")
    if not daily_stop_all:
        raise RuntimeError("no backtest years")
    yearly = pd.DataFrame(rows).sort_values("year")
    yearly.to_csv(os.path.join(out_dir, "backtest_yearly.csv"), index=False)
    daily = pd.concat(daily_stop_all, ignore_index=True)
    bench = pd.concat(daily_bench_all, ignore_index=True)
    daily = daily.merge(bench[["year", "date_idx", "equity"]].rename(
        columns={"equity": "equity_bench"}), on=["year", "date_idx"], how="left")
    daily = daily.sort_values(["year", "date_idx"]).reset_index(drop=True)
    cum_stop = []
    cum_bench = []
    cs = cb = 1.0
    for y, g in daily.groupby("year", sort=True):
        g = g.sort_values("date_idx")
        eqs = g["equity"].to_numpy()
        eqb = g["equity_bench"].to_numpy()
        cs_arr = cs * eqs
        cb_arr = cb * eqb
        cum_stop.extend(cs_arr.tolist())
        cum_bench.extend(cb_arr.tolist())
        cs = cs_arr[-1]
        cb = cb_arr[-1]
    daily["equity_stop_cum"] = cum_stop
    daily["equity_bench_cum"] = cum_bench
    daily["date"] = [dates[int(i)] for i in daily["date_idx"]]
    daily.to_csv(os.path.join(out_dir, "equity_curve.csv"), index=False)

    stop_m = _metrics(daily["equity_stop_cum"].to_numpy())
    bench_m = _metrics(daily["equity_bench_cum"].to_numpy())
    summary = pd.DataFrame([
        {"strategy": "stop_overlay", **stop_m,
         "avg_exposure": float(yearly["stop_avg_exposure"].mean()),
         "total_stops": int(yearly["n_stops"].sum())},
        {"strategy": "buy_hold", **bench_m,
         "avg_exposure": float(yearly["bench_avg_exposure"].mean()),
         "total_stops": 0},
    ])
    summary.to_csv(os.path.join(out_dir, "backtest_summary.csv"), index=False)
    if trades_all:
        tr = pd.DataFrame(trades_all)
        tr["code"] = [str(codes[int(i)]) for i in tr["code_idx"]]
        tr["entry_date"] = [dates[int(i)] for i in tr["entry_date_idx"]]
        tr["exit_date"] = [dates[int(i)] for i in tr["exit_date_idx"]]
        tr.to_csv(os.path.join(out_dir, "trades_stop.csv.gz"), index=False,
                  compression="gzip")
        fwd_ok = tr["fwd_ret_h"].dropna()
        hit = (fwd_ok < 0).mean() if len(fwd_ok) else float("nan")
        log(f"止损交易 {len(tr):,} 笔, 信号后H日下跌占比 {hit:.1%}, "
            f"平均持有 {tr['holding_days'].mean():.1f} 天")
    with open(os.path.join(out_dir, "backtest_config.json"), "w",
              encoding="utf-8") as f:
        json.dump({"cost_buy": cost_buy, "cost_sell": cost_sell,
                   "cooldown": cooldown, "reentry": reentry,
                   "universe_top": universe_top, "min_price": min_price,
                   "test_years": [int(x) for x in years]},
                  f, ensure_ascii=False, indent=1)
    log("回测完成")
