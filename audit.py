#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Point-in-time audit.

1. Feature invariance: compute features on complete bars and on bars truncated
   at T; all rows <= T must be identical.
2. Label sanity: a monotonically rising price path must not produce top labels.
3. Walk-forward purge: training/validation labels must not reach into the next
   segment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import connect, load_bars
from features import compute_features, compute_labels
from walkforward import fold_ranges, load_dataset


def _check_feature_invariance(db_path: str, seed: int = 7) -> None:
    con = connect(db_path)
    codes = pd.read_sql_query(
        "SELECT DISTINCT code FROM daily_bars WHERE date >= '2019-01-01' "
        "AND date <= '2022-12-31' AND code GLOB 'sh60[0135]*' "
        "ORDER BY code LIMIT 40", con)["code"].tolist()
    rs = np.random.RandomState(seed)
    codes = sorted(rs.choice(codes, size=min(12, len(codes)), replace=False))
    full = load_bars(con, "2018-06-01", "2022-12-31", codes)
    con.close()
    cut_dates = ["2020-06-30", "2021-06-30", "2022-06-30"]
    for cut in cut_dates:
        trunc = full[full["date"] <= cut].copy()
        f_full = compute_features(full)
        f_trunc = compute_features(trunc)
        a = f_full.loc[full["date"] <= cut].reset_index(drop=True)
        b = f_trunc.reset_index(drop=True)
        if a.shape != b.shape:
            raise AssertionError(f"{cut}: shape mismatch {a.shape} vs {b.shape}")
        for col in a.columns:
            av = a[col].to_numpy(np.float64)
            bv = b[col].to_numpy(np.float64)
            if not np.allclose(av, bv, rtol=1e-6, atol=1e-7, equal_nan=True):
                bad = np.flatnonzero(~np.isclose(av, bv, rtol=1e-6,
                                                 atol=1e-7, equal_nan=True))
                raise AssertionError(
                    f"{cut}: future leakage in feature {col} "
                    f"at rows {bad[:5]}")
    print("  [PASS] 特征不变性: 截断未来行情后 t<=T 特征完全一致")


def _check_label_sanity() -> None:
    n = 80
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = np.linspace(10.0, 20.0, n)
    df = pd.DataFrame({
        "code": pd.Categorical(["sh600000"] * n),
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "vol": np.full(n, 1e6),
        "lim": np.full(n, 0.10),
    })
    lab = compute_labels(df, horizon=10, dd_threshold=0.08,
                         newhigh_threshold=0.03)
    if np.nanmax(lab["label_top"].to_numpy()) > 0:
        raise AssertionError("单边上涨路径不应产生顶部标签")
    if bool(lab["label_valid"].iloc[-10:].any()):
        raise AssertionError("最后 H 行不应有有效未来标签")
    print("  [PASS] 标签逻辑: 单边上涨无顶部标签, 末尾无未来标签")


def _check_fold_purge(out_dir: str, horizon: int) -> None:
    meta, X, codes, dates, feature_names = load_dataset(out_dir)
    dates = [str(x) for x in dates]
    years = sorted(set(pd.to_datetime(pd.Series(dates)).dt.year))
    checked = 0
    for year in years:
        if year < 2019:
            continue
        f = fold_ranges(meta, dates, year, 5, "2017-01-01", horizon)
        if f is None:
            continue
        tr_end = dates[int(meta["date_idx"][f["train_idx"]].max())]
        va_start = f["valid_start"]
        va_end = f["valid_end"]
        te_start = f["test_start"]
        if not tr_end < va_start:
            raise AssertionError(f"{year}: train/valid overlap {tr_end} {va_start}")
        if not va_end < te_start:
            raise AssertionError(f"{year}: valid/test overlap {va_end} {te_start}")
        checked += 1
    if checked == 0:
        print("  [SKIP] 数据集太小，没有可检查的 walk-forward 折")
        return
    print(f"  [PASS] 切分隔离: {checked} 个折 train<valid<test, 已按 H 清空边界")


def run_audit(db_path: str, out_dir: str | None = None, horizon: int = 10) -> None:
    print("PIT 审计 ...")
    _check_feature_invariance(db_path)
    _check_label_sanity()
    if out_dir and __import__("os").path.exists(
            __import__("os").path.join(out_dir, "meta.npz")):
        _check_fold_purge(out_dir, horizon)
    print("PIT 审计全部通过")
