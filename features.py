#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Point-in-time features and forward top labels.

Every feature at date t is a function of bars with date <= t only.  Labels use
bars after t and are never fed back into the feature matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def _codes(df: pd.DataFrame) -> np.ndarray:
    if isinstance(df["code"].dtype, pd.CategoricalDtype):
        return df["code"].cat.codes.to_numpy()
    return pd.factorize(df["code"], sort=False)[0]


def _aligned(s: pd.Series) -> pd.Series:
    if isinstance(s.index, pd.MultiIndex):
        s = s.reset_index(level=0, drop=True)
    return s.sort_index()


def _roll(s: pd.Series, codes: np.ndarray, w: int, how: str) -> pd.Series:
    r = getattr(s.groupby(codes, sort=False).rolling(w), how)()
    return _aligned(r)


def _ewm(s: pd.Series, codes: np.ndarray, span: int | None = None,
         com: float | None = None) -> pd.Series:
    if span is not None:
        return s.groupby(codes, sort=False).transform(
            lambda x: x.ewm(span=span, adjust=False).mean())
    return s.groupby(codes, sort=False).transform(
        lambda x: x.ewm(com=com, adjust=False).mean())


def _shift(s: pd.Series, codes: np.ndarray, n: int) -> pd.Series:
    return s.groupby(codes, sort=False).shift(n)


def _days_since(flag: pd.Series, codes: np.ndarray) -> pd.Series:
    n = len(flag)
    idx = np.arange(n, dtype=np.float64)
    last = pd.Series(np.where(flag.to_numpy(bool), idx, np.nan))
    last = last.groupby(codes, sort=False).ffill()
    return pd.Series(idx - last.to_numpy(), index=flag.index, dtype=np.float64)


def _streak(flag: pd.Series, codes: np.ndarray) -> pd.Series:
    arr = flag.to_numpy(np.int8)
    codes_arr = np.asarray(codes)
    n = len(arr)
    boundary = np.empty(n, dtype=bool)
    boundary[0] = True
    if n > 1:
        boundary[1:] = (arr[1:] != arr[:-1]) | (codes_arr[1:] != codes_arr[:-1])
    run = np.cumsum(boundary)
    out = pd.Series(arr).groupby(run).cumsum()
    return pd.Series(out.to_numpy(dtype=np.float64), index=flag.index)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    codes = _codes(df)
    o, h, l, c, v = (df[x] for x in ("open", "high", "low", "close", "vol"))
    pc = _shift(c, codes, 1)
    po = _shift(o, codes, 1)
    ret1 = c / (pc + EPS) - 1.0
    F: dict[str, pd.Series] = {}

    def put(name: str, x) -> None:
        if isinstance(x, pd.Series):
            F[name] = x.astype(np.float32)
        else:
            F[name] = pd.Series(np.asarray(x, dtype=np.float32), index=df.index)

    for n in (1, 3, 5, 10, 20, 60, 120):
        put(f"ret{n}", (c / c.groupby(codes, sort=False).shift(n) - 1.0)
            if n > 1 else ret1)

    ma = {n: _roll(c, codes, n, "mean") for n in (5, 10, 20, 60, 120)}
    for n, s in ma.items():
        put(f"ma{n}r", c / (s + EPS) - 1.0)
    put("ma5_20", ma[5] / (ma[20] + EPS) - 1.0)
    put("ma20_60", ma[20] / (ma[60] + EPS) - 1.0)
    put("ma60_120", ma[60] / (ma[120] + EPS) - 1.0)
    put("ma20_slope5", ma[20] / (_shift(ma[20], codes, 5) + EPS) - 1.0)
    put("ma60_slope20", ma[60] / (_shift(ma[60], codes, 20) + EPS) - 1.0)
    above = None
    for n in (5, 20, 60, 120):
        a = (c > ma[n]).astype(np.float32)
        above = a if above is None else above + a
    put("ma_above_cnt", above / 4.0)

    hh = {n: _roll(h, codes, n, "max") for n in (20, 60, 120, 250)}
    ll = {n: _roll(l, codes, n, "min") for n in (20, 60, 120, 250)}
    for n in (20, 60, 120, 250):
        put(f"dist_high{n}", c / (hh[n] + EPS) - 1.0)
        put(f"dist_low{n}", c / (ll[n] + EPS) - 1.0)
    put("range_pos20", (c - ll[20]) / (hh[20] - ll[20] + EPS))
    put("range_pos250", (c - ll[250]) / (hh[250] - ll[250] + EPS))
    prev_hh20 = _shift(hh[20], codes, 1)
    prev_hh60 = _shift(hh[60], codes, 1)
    put("breakout20", c / (prev_hh20 + EPS) - 1.0)
    put("breakout60", c / (prev_hh60 + EPS) - 1.0)
    put("days_since_high20", _days_since(c >= hh[20] - 1e-8, codes))
    put("days_since_high60", _days_since(c >= hh[60] - 1e-8, codes))
    put("days_since_high250", _days_since(c >= hh[250] - 1e-8, codes))
    put("days_since_low20", _days_since(c <= ll[20] + 1e-8, codes))

    delta = c - pc
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    for n in (6, 14, 24):
        fu = _roll(up, codes, n, "mean")
        fd = _roll(dn, codes, n, "mean")
        put(f"rsi{n}", 100.0 * fu / (fu + fd + EPS))

    ema12 = _ewm(c, codes, span=12)
    ema26 = _ewm(c, codes, span=26)
    dif = ema12 - ema26
    dea = _ewm(dif, codes, span=9)
    hist = (dif - dea) * 2.0
    put("macd_dif", dif / (c + EPS))
    put("macd_dea", dea / (c + EPS))
    put("macd_hist", hist / (c + EPS))
    put("macd_hist_chg", hist / (c + EPS)
        - _shift(hist / (c + EPS), codes, 1))

    llv9 = _roll(l, codes, 9, "min")
    hhv9 = _roll(h, codes, 9, "max")
    rsv = (c - llv9) / (hhv9 - llv9 + EPS) * 100.0
    k = _ewm(rsv, codes, com=2)
    d = _ewm(k, codes, com=2)
    put("kdj_k", k)
    put("kdj_d", d)
    put("kdj_j", 3.0 * k - 2.0 * d)
    put("willr14", -100.0 * (_roll(h, codes, 14, "max") - c)
        / (_roll(h, codes, 14, "max") - _roll(l, codes, 14, "min") + EPS))

    tp = (h + l + c) / 3.0
    ma_tp20 = _roll(tp, codes, 20, "mean")
    sd_tp20 = _roll(tp, codes, 20, "std")
    put("cci20", (tp - ma_tp20) / (0.015 * sd_tp20 + EPS))
    put("mom_acc5_20", F["ret5"] - F["ret20"] / 4.0)
    put("mom_acc20_60", F["ret20"] - F["ret60"] / 3.0)
    put("psy20", _roll((ret1 > 0).astype(np.float32), codes, 20, "mean"))

    std20 = _roll(ret1, codes, 20, "std")
    std60 = _roll(ret1, codes, 60, "std")
    std120 = _roll(ret1, codes, 120, "std")
    put("std20", std20)
    put("std60", std60)
    put("std120", std120)
    put("vol_ratio20_60", std20 / (std60 + EPS))
    put("vol_ratio60_120", std60 / (std120 + EPS))
    put("ret5_z20", F["ret5"] / (std20 * np.sqrt(5.0) + EPS))
    put("ret20_z60", F["ret20"] / (std60 * np.sqrt(20.0) + EPS))

    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()],
                   axis=1).max(axis=1)
    atr14 = _roll(tr, codes, 14, "mean")
    atr20 = _roll(tr, codes, 20, "mean")
    atr60 = _roll(tr, codes, 60, "mean")
    put("atr14", atr14 / (c + EPS))
    put("atr20", atr20 / (c + EPS))
    put("atr_ratio14_60", atr14 / (atr60 + EPS))
    put("amp", (h - l) / (pc + EPS))
    put("amp_ma20", _roll((h - l) / (pc + EPS), codes, 20, "mean"))
    negret = ret1.where(ret1 < 0, 0.0)
    put("downside_std20", _roll(negret, codes, 20, "std"))
    put("downside_std60", _roll(negret, codes, 60, "std"))
    put("skew20", _roll(ret1, codes, 20, "skew"))
    put("max_ret20", _roll(ret1, codes, 20, "max"))
    put("min_ret20", _roll(ret1, codes, 20, "min"))

    mv = {n: _roll(v, codes, n, "mean") for n in (5, 20, 60)}
    for n, s in mv.items():
        put(f"vr{n}", v / (s + EPS))
    put("logvol", np.log1p(v))
    sv20 = _roll(v, codes, 20, "std")
    sv60 = _roll(v, codes, 60, "std")
    put("vol_z20", (v - mv[20]) / (sv20 + EPS))
    put("vol_z60", (v - mv[60]) / (sv60 + EPS))
    put("vma5_20", mv[5] / (mv[20] + EPS) - 1.0)
    put("vma20_60", mv[20] / (mv[60] + EPS) - 1.0)
    sgn = np.sign(c - pc).fillna(0.0)
    sgnv = sgn * v
    put("obv_ratio20", _roll(sgnv, codes, 20, "sum")
        / (_roll(v, codes, 20, "sum") + EPS))
    put("obv_ratio60", _roll(sgnv, codes, 60, "sum")
        / (_roll(v, codes, 60, "sum") + EPS))
    mf = tp * v
    posmf = mf.where(tp > _shift(tp, codes, 1), 0.0)
    negmf = mf.where(tp < _shift(tp, codes, 1), 0.0)
    mr = _roll(posmf, codes, 14, "sum") / (
        _roll(negmf, codes, 14, "sum") + EPS)
    put("mfi14", 100.0 - 100.0 / (1.0 + mr))
    upv = v.where(ret1 > 0, 0.0)
    put("upvol_ratio20", _roll(upv, codes, 20, "sum")
        / (_roll(v, codes, 20, "sum") + EPS))
    put("vspike_cnt20", _roll((v > 2.0 * mv[20]).astype(np.float32),
                              codes, 20, "sum"))
    dist_v = (ret1 < -0.02) & (v > 1.5 * mv[20])
    put("distribution_cnt20", _roll(dist_v.astype(np.float32),
                                    codes, 20, "sum"))
    climactic = (v > 3.0 * mv[60]) & (F["dist_high60"] > -0.03)
    put("climactic_cnt60", _roll(climactic.astype(np.float32), codes, 60, "sum"))

    bollpos = (c - ma[20]) / (2.0 * std20 + EPS)
    put("bollpos", bollpos)
    put("boll_width", 4.0 * std20 / (ma[20] + EPS))
    put("above_upper", (c > ma[20] + 2.0 * std20).astype(np.float32))
    overbought = (F["rsi14"] > 70) | (F["kdj_j"] > 100) | (bollpos > 1.0)
    put("overbought_cnt10", _roll(overbought.astype(np.float32),
                                  codes, 10, "sum"))
    put("newhigh20_cnt10", _roll((c >= hh[20] - 1e-8).astype(np.float32),
                                 codes, 10, "sum"))
    put("newhigh60_cnt20", _roll((c >= hh[60] - 1e-8).astype(np.float32),
                                 codes, 20, "sum"))
    gap = o / (pc + EPS) - 1.0
    put("gap_up_cnt20", _roll((gap > 0.02).astype(np.float32), codes, 20, "sum"))
    put("gap_dn_cnt20", _roll((gap < -0.02).astype(np.float32), codes, 20, "sum"))

    rng = h - l
    body = c - o
    body_abs = body.abs()
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    put("body_pct", body / (pc + EPS))
    put("upper_sh", upper / (pc + EPS))
    put("lower_sh", lower / (pc + EPS))
    put("body_pos", body / (rng + EPS))
    doji = body_abs <= 0.1 * rng
    hammer = (lower >= 2.0 * body_abs) & (upper <= body_abs)
    shooting = (upper >= 2.0 * body_abs) & (lower <= body_abs)
    bull_engulf = (c > o) & (pc < po) & (c >= po) & (o <= pc)
    bear_engulf = (c < o) & (pc > po) & (c <= po) & (o >= pc)
    dark_cloud = (pc > po) & (c < o) & (o > pc) & (c < (po + pc) / 2.0)
    bear_harami = ((pc > po) & (c < o)
                   & (np.maximum(o, c) < np.maximum(po, pc))
                   & (np.minimum(o, c) > np.minimum(po, pc)))
    c2 = _shift(c, codes, 2)
    o2 = _shift(o, codes, 2)
    b2 = c2 - o2
    c1 = _shift(c, codes, 1)
    o1 = _shift(o, codes, 1)
    b1 = c1 - o1
    evening_star = ((c2 > o2) & (b2 > 0.02 * _shift(pc, codes, 2))
                    & (b1.abs() < 0.4 * b2.abs())
                    & (np.maximum(o1, c1) > np.maximum(o2, c2))
                    & (c < o) & (c < (o2 + c2) / 2.0))
    hanging_man = hammer & (F["ret5"] > 0.03)
    gravestone = ((upper >= 2.0 * body_abs) & (lower <= 0.5 * body_abs)
                  & (body_abs <= 0.1 * rng))
    dn = c < o
    three_black = (_roll(dn.astype(np.float32), codes, 3, "sum") >= 3) & (
        c < _shift(c, codes, 3))
    put("doji", doji.astype(np.float32))
    put("hammer", hammer.astype(np.float32))
    put("shooting", shooting.astype(np.float32))
    put("bull_engulf", bull_engulf.astype(np.float32))
    put("bear_engulf", bear_engulf.astype(np.float32))
    put("dark_cloud", dark_cloud.astype(np.float32))
    put("bear_harami", bear_harami.astype(np.float32))
    put("evening_star", evening_star.astype(np.float32))
    put("hanging_man", hanging_man.astype(np.float32))
    put("gravestone", gravestone.astype(np.float32))
    put("three_black", three_black.astype(np.float32))
    put("three_up", (_roll((c > o).astype(np.float32), codes, 3, "sum")
                     >= 3).astype(np.float32))
    put("three_dn", (_roll(dn.astype(np.float32), codes, 3, "sum")
                     >= 3).astype(np.float32))
    put("up_streak", _streak(c > o, codes))
    put("dn_streak", _streak(dn, codes))
    put("yang_ratio20", _roll((c > o).astype(np.float32), codes, 20, "mean"))
    lim = df["lim"].to_numpy(np.float64)
    put("limit_up_cnt20", _roll((ret1 >= lim - 0.005).astype(np.float32),
                                codes, 20, "sum"))
    put("limit_dn_cnt20", _roll((ret1 <= -lim + 0.005).astype(np.float32),
                                codes, 20, "sum"))
    bearish = (shooting | bear_engulf | dark_cloud | bear_harami
               | evening_star | hanging_man | gravestone | three_black)
    put("bear_pattern_cnt5", _roll(bearish.astype(np.float32), codes, 5, "sum"))
    put("bear_pattern_cnt10", _roll(bearish.astype(np.float32), codes, 10, "sum"))

    for name, s in (("rsi14", F["rsi14"]), ("vr20", F["vr20"]),
                    ("atr14", F["atr14"]), ("ret20", F["ret20"]),
                    ("close", c), ("amp", F["amp"])):
        mu = _roll(s, codes, 250, "mean")
        sd = _roll(s, codes, 250, "std")
        put(f"{name}_z250", (s - mu) / (sd + EPS))
    lo = _roll(F["rsi14"], codes, 250, "min")
    hi = _roll(F["rsi14"], codes, 250, "max")
    put("rsi14_pos250", (F["rsi14"] - lo) / (hi - lo + EPS))

    out = pd.DataFrame(F, index=df.index)
    return out.astype(np.float32)


def compute_labels(
    df: pd.DataFrame,
    horizon: int = 10,
    dd_threshold: float = 0.08,
    newhigh_threshold: float = 0.03,
    max_gap: int = 5,
) -> pd.DataFrame:
    """Forward labels: a top is a >=dd_threshold fall within `horizon` days
    from the next open before any rise of newhigh_threshold."""
    df = df.sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    codes = _codes(df)
    c = df["close"]
    o = df["open"]
    v = df["vol"]
    entry = _shift(o, codes, -1)
    date_idx = pd.factorize(df["date"], sort=True)[0]
    dt = pd.Series(date_idx, index=df.index)
    gap_ok = pd.Series(True, index=df.index)
    prev_dt = dt
    R = np.empty((len(df), horizon), dtype=np.float64)
    for k in range(1, horizon + 1):
        fc = _shift(c, codes, -k)
        R[:, k - 1] = (fc / (entry + EPS) - 1.0).to_numpy()
        cur_dt = _shift(dt, codes, -k)
        gap_ok &= (cur_dt - prev_dt <= max_gap)
        prev_dt = cur_dt
    valid = (np.isfinite(R).all(axis=1) & entry.notna().to_numpy()
             & gap_ok.to_numpy())
    y = np.zeros(len(df), dtype=np.int8)
    hit_any = np.zeros(len(df), dtype=bool)
    pre_max = np.full(len(df), -np.inf)
    for k in range(horizon):
        hit = R[:, k] <= -dd_threshold
        y |= (hit & ~hit_any & (pre_max < newhigh_threshold))
        hit_any |= hit
        pre_max = np.maximum(pre_max, R[:, k])
    fwd_ret = R[:, horizon - 1]
    fwd_maxdd = R.min(axis=1)
    fwd_maxup = R.max(axis=1)
    y = y.astype(np.float32)
    y[~valid] = np.nan
    hist_len = df.groupby("code", sort=False).cumcount().to_numpy() + 1
    amt = c * v
    amt60 = _roll(amt, codes, 60, "mean").to_numpy()
    out = pd.DataFrame({
        "label_top": y,
        "label_valid": valid,
        "fwd_ret_h": np.where(valid, fwd_ret, np.nan).astype(np.float32),
        "fwd_maxdd": np.where(valid, fwd_maxdd, np.nan).astype(np.float32),
        "fwd_maxup": np.where(valid, fwd_maxup, np.nan).astype(np.float32),
        "y_dd": np.where(valid, fwd_maxdd <= -dd_threshold, np.nan).astype(np.float32),
        "y_ret": np.where(valid, fwd_ret <= -dd_threshold, np.nan).astype(np.float32),
        "next_open": entry.astype(np.float32),
        "prev_close": _shift(c, codes, 1).astype(np.float32),
        "hist_len": hist_len.astype(np.int32),
        "avg_amount60": amt60.astype(np.float32),
    }, index=df.index)
    return out
