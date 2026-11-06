#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a memory-mapped point-in-time feature matrix plus label metadata."""
from __future__ import annotations

import gc
import json
import os
import time

import numpy as np
import pandas as pd

from data import (connect, load_bars, row_counts_by_code,
                  trading_calendar)
from features import compute_features, compute_labels


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_dataset(
    out_dir: str,
    db_path: str,
    start: str,
    end: str,
    horizon: int,
    dd_threshold: float,
    newhigh_threshold: float,
    chunk_size: int = 300,
    limit_codes: int = 0,
    seed: int = 42,
    max_gap: int = 5,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    con = connect(db_path)
    counts = row_counts_by_code(con, start, end)
    codes = counts["code"].tolist()
    if limit_codes and limit_codes < len(codes):
        rs = np.random.RandomState(seed)
        codes = sorted(rs.choice(codes, size=limit_codes, replace=False).tolist())
        counts = counts[counts["code"].isin(codes)].copy()
    dates = trading_calendar(con, start, end)
    date_map = {d: i for i, d in enumerate(dates)}
    code_map = {c: i for i, c in enumerate(codes)}
    counts = counts.set_index("code").loc[codes]
    n_rows = int(counts["n"].sum())
    offsets = np.concatenate([[0], np.cumsum(counts["n"].to_numpy())])
    log(f"数据集: {n_rows:,} 行, {len(codes)} 只股票, {len(dates)} 个交易日")

    meta_keys = ["label_top", "label_valid", "fwd_ret_h", "fwd_maxdd",
                 "fwd_maxup", "y_dd", "y_ret", "next_open", "prev_close",
                 "hist_len", "avg_amount60", "open", "close", "lim"]
    meta = {k: np.empty(n_rows, dtype=np.float32) for k in meta_keys}
    meta["label_valid"] = np.empty(n_rows, dtype=bool)
    meta["hist_len"] = np.empty(n_rows, dtype=np.int32)
    code_idx = np.empty(n_rows, dtype=np.int32)
    date_idx = np.empty(n_rows, dtype=np.int32)

    feature_names: list[str] | None = None
    X = None
    t0 = time.time()
    for ci, start_i in enumerate(range(0, len(codes), chunk_size)):
        chunk = codes[start_i:start_i + chunk_size]
        df = load_bars(con, start, end, chunk)
        if df.empty:
            raise RuntimeError(f"empty chunk {chunk[:2]}")
        lo, hi = int(offsets[start_i]), int(offsets[start_i + len(chunk)])
        if len(df) != hi - lo:
            raise RuntimeError(
                f"row count mismatch chunk={ci}: bars={len(df)} meta={hi - lo}")
        feat = compute_features(df)
        lab = compute_labels(df, horizon=horizon, dd_threshold=dd_threshold,
                             newhigh_threshold=newhigh_threshold,
                             max_gap=max_gap)
        if feature_names is None:
            feature_names = list(feat.columns)
            X = np.lib.format.open_memmap(
                os.path.join(out_dir, "features.npy"), mode="w+",
                dtype=np.float32, shape=(n_rows, len(feature_names)))
            log(f"特征数: {len(feature_names)}")
        else:
            feat = feat.reindex(columns=feature_names)
        X[lo:hi, :] = feat.to_numpy(np.float32)
        for k in meta_keys:
            if k == "open":
                meta[k][lo:hi] = df["open"].to_numpy(np.float32)
            elif k == "close":
                meta[k][lo:hi] = df["close"].to_numpy(np.float32)
            elif k == "lim":
                meta[k][lo:hi] = df["lim"].to_numpy(np.float32)
            elif k == "label_valid":
                meta[k][lo:hi] = lab[k].to_numpy(bool)
            elif k == "hist_len":
                meta[k][lo:hi] = lab[k].to_numpy(np.int32)
            else:
                meta[k][lo:hi] = lab[k].to_numpy(np.float32)
        code_idx[lo:hi] = df["code"].map(code_map).to_numpy(np.int32)
        date_idx[lo:hi] = df["date"].map(date_map).to_numpy(np.int32)
        del df, feat, lab
        gc.collect()
        if (ci + 1) % 2 == 0 or start_i + chunk_size >= len(codes):
            done = offsets[min(start_i + chunk_size, len(codes))]
            log(f"  进度 {done:,}/{n_rows:,} ({done / n_rows:.1%}), "
                f"{time.time() - t0:.0f}s")
    X.flush()
    del X
    con.close()

    meta["code_idx"] = code_idx
    meta["date_idx"] = date_idx
    np.savez(os.path.join(out_dir, "meta.npz"), **meta)
    np.save(os.path.join(out_dir, "codes.npy"), np.array(codes, dtype=object),
            allow_pickle=True)
    np.save(os.path.join(out_dir, "dates.npy"), np.array(dates, dtype=object),
            allow_pickle=True)
    with open(os.path.join(out_dir, "feature_names.json"), "w",
              encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "build_config.json"), "w",
              encoding="utf-8") as f:
        json.dump({"db": db_path, "start": start, "end": end,
                   "horizon": horizon, "dd_threshold": dd_threshold,
                   "newhigh_threshold": newhigh_threshold,
                   "max_gap": max_gap, "n_rows": n_rows,
                   "n_codes": len(codes), "n_dates": len(dates),
                   "n_features": len(feature_names),
                   "limit_codes": limit_codes, "seed": seed},
                  f, ensure_ascii=False, indent=1)
    log(f"数据集完成: {out_dir}, 用时 {(time.time() - t0) / 60:.1f} min")
