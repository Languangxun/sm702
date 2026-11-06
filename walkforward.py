#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Walk-forward training, threshold selection and out-of-sample prediction."""
from __future__ import annotations

import gc
import json
import os
import time
from bisect import bisect_left

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

from data import connect, load_stock_names


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_dataset(out_dir: str):
    meta = dict(np.load(os.path.join(out_dir, "meta.npz")))
    X = np.load(os.path.join(out_dir, "features.npy"), mmap_mode="r")
    codes = np.load(os.path.join(out_dir, "codes.npy"), allow_pickle=True)
    dates = np.load(os.path.join(out_dir, "dates.npy"), allow_pickle=True)
    with open(os.path.join(out_dir, "feature_names.json"), encoding="utf-8") as f:
        feature_names = json.load(f)
    return meta, X, codes, dates, feature_names


def _date_pos(dates: list[str], target: str, side: str = "left") -> int:
    if side == "left":
        i = bisect_left(dates, target)
        return min(i, len(dates) - 1)
    i = bisect_left(dates, target)
    return max(0, i - 1)


def fold_ranges(meta, dates, year: int, train_years: int,
                sample_start: str, horizon: int, min_train: int = 50000,
                min_valid: int = 2000) -> dict | None:
    test_start = f"{year}-01-01"
    test_end = f"{year}-12-31"
    val_start = f"{year - 1}-01-01"
    if train_years > 0:
        train_start = max(sample_start, f"{year - train_years}-01-01")
    else:
        train_start = sample_start
    ti0 = _date_pos(dates, test_start)
    ti1 = _date_pos(dates, test_end, "right")
    if dates[ti0] > test_end:
        return None
    vi0 = _date_pos(dates, val_start)
    tr0 = _date_pos(dates, train_start)
    tr1 = vi0 - horizon - 2
    va1 = ti0 - horizon - 2
    if tr1 <= tr0 or va1 < vi0:
        return None
    if va1 < vi0 + 60:
        vi0 = max(tr0 + 120, ti0 - 250)
        va1 = ti0 - horizon - 2
        if va1 < vi0:
            return None
    didx = meta["date_idx"]
    label_valid = meta["label_valid"]
    hist = meta["hist_len"]
    tr_mask = ((didx >= tr0) & (didx <= tr1) & label_valid
               & (hist >= 120))
    va_mask = ((didx >= vi0) & (didx <= va1) & label_valid & (hist >= 120))
    te_mask = (didx >= ti0) & (didx <= ti1)
    if tr_mask.sum() < min_train or va_mask.sum() < min_valid:
        return None
    return {
        "year": year,
        "train_start": dates[tr0], "train_end": dates[tr1],
        "valid_start": dates[vi0], "valid_end": dates[va1],
        "test_start": dates[ti0], "test_end": dates[ti1],
        "train_idx": np.flatnonzero(tr_mask),
        "valid_idx": np.flatnonzero(va_mask),
        "test_idx": np.flatnonzero(te_mask),
    }


def select_threshold(p: np.ndarray, y: np.ndarray, fwd_ret: np.ndarray,
                     mode: str = "fixed", min_cov: float = 0.002,
                     precision_target: float = 0.28,
                     fixed_threshold: float = 0.30) -> dict:
    ok = np.isfinite(p) & np.isfinite(y)
    p, y, fwd_ret = p[ok], y[ok], fwd_ret[ok]
    if len(p) == 0 or np.nanstd(p) == 0:
        return {"threshold": fixed_threshold if mode == "fixed" else 0.5,
                "f1": np.nan, "precision": np.nan,
                "recall": np.nan, "coverage": np.nan, "mode": mode}
    if mode == "fixed":
        pred = p >= fixed_threshold
        n = int(pred.sum())
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        return {"threshold": float(fixed_threshold), "f1": float(f1),
                "precision": float(precision), "recall": float(recall),
                "coverage": float(n / len(p)), "signals": n,
                "mode": mode, "score": float(f1),
                "utility": float((-fwd_ret[pred]).mean()) if n else np.nan}
    grid = np.unique(np.concatenate([
        np.quantile(p, np.linspace(0.50, 0.999, 400)),
        np.linspace(0.20, 0.90, 36),
    ]))
    min_n = max(50, int(min_cov * len(p)))
    best = None
    for thr in grid:
        pred = p >= thr
        n = int(pred.sum())
        if n < min_n:
            continue
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        if mode == "precision":
            score = recall if precision >= precision_target else -1.0
        elif mode == "utility":
            benefit = -fwd_ret[pred]
            score = float((benefit - 0.002).mean()) if n else -9.0
        elif mode == "f0.5":
            beta2 = 0.25
            score = ((1 + beta2) * precision * recall
                     / (beta2 * precision + recall + 1e-12))
        elif mode == "f2":
            beta2 = 4.0
            score = ((1 + beta2) * precision * recall
                     / (beta2 * precision + recall + 1e-12))
        else:
            score = f1
        key = (score, precision, thr)
        if best is None or key > best["key"]:
            best = {"key": key, "threshold": float(thr),
                    "f1": float(f1), "precision": float(precision),
                    "recall": float(recall), "coverage": float(n / len(p)),
                    "signals": n, "mode": mode,
                    "score": float(score),
                    "utility": float((-fwd_ret[pred]).mean())}
    if best is None:
        return {"threshold": 0.5, "f1": np.nan, "precision": np.nan,
                "recall": np.nan, "coverage": np.nan, "mode": mode}
    best.pop("key", None)
    return best


def _eval(y: np.ndarray, p: np.ndarray) -> dict:
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"n": len(y), "auc": np.nan, "ap": np.nan,
                "brier": np.nan, "logloss": np.nan, "base_rate": np.nan}
    return {"n": int(len(y)),
            "auc": float(roc_auc_score(y, p)),
            "ap": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            "base_rate": float(y.mean())}


def run_walkforward(out_dir: str, db_path: str, sample_start: str,
                    test_start_year: int, test_end_year: int,
                    train_years: int, horizon: int, rounds: int,
                    learning_rate: float, leaves: int, min_data: int,
                    early_stop: int, threshold_mode: str,
                    precision_target: float = 0.28,
                    fixed_threshold: float = 0.30,
                    num_threads: int = 4,
                    seed: int = 42,
                    market: str = "cn",
                    min_train: int = 50000) -> pd.DataFrame:
    meta, X, codes, dates, feature_names = load_dataset(out_dir)
    dates_np = np.asarray([str(x) for x in dates], dtype=object)
    dates = dates_np.tolist()
    codes = np.asarray(codes, dtype=object)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    row_dates = pd.to_datetime(
        pd.Series(dates_np[meta["date_idx"].astype(int)])).to_numpy()
    all_preds = []
    fold_rows = []
    gains = np.zeros(len(feature_names), dtype=np.float64)
    n_gain = 0
    for year in range(test_start_year, test_end_year + 1):
        fold = fold_ranges(meta, dates, year, train_years, sample_start,
                           horizon, min_train=min_train)
        if fold is None:
            log(f"[{year}] 跳过: 训练/验证样本不足")
            continue
        tr_idx, va_idx, te_idx = (fold["train_idx"], fold["valid_idx"],
                                  fold["test_idx"])
        log(f"[{year}] train {fold['train_start']}~{fold['train_end']} "
            f"{len(tr_idx):,} | valid {fold['valid_start']}~{fold['valid_end']} "
            f"{len(va_idx):,} | test {fold['test_start']}~{fold['test_end']} "
            f"{len(te_idx):,}")
        ytr = meta["label_top"][tr_idx].astype(np.float32)
        yva = meta["label_top"][va_idx].astype(np.float32)
        keep = np.isfinite(ytr)
        tr_idx = tr_idx[keep]
        ytr = ytr[keep]
        keep = np.isfinite(yva)
        va_idx = va_idx[keep]
        yva = yva[keep]
        age_days = ((row_dates[tr_idx].max() - row_dates[tr_idx])
                    / np.timedelta64(1, "D")).astype(np.float64)
        wtr = (0.5 ** (age_days / 756.0)).astype(np.float32)
        Xtr = np.asarray(X[tr_idx], dtype=np.float32)
        params = {
            "objective": "binary",
            "metric": ["auc"],
            "learning_rate": learning_rate,
            "num_leaves": leaves,
            "min_data_in_leaf": min_data,
            "feature_fraction": 0.75,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l2": 10.0,
            "max_bin": 127,
            "num_threads": num_threads,
            "verbosity": -1,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "force_col_wise": True,
        }
        dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr,
                          feature_name=feature_names, free_raw_data=True,
                          params=params)
        dtr.construct()
        del Xtr
        gc.collect()
        dva = lgb.Dataset(np.asarray(X[va_idx], dtype=np.float32), label=yva,
                          reference=dtr, free_raw_data=True, params=params)
        dva.construct()
        model = lgb.train(
            params, dtr, num_boost_round=rounds, valid_sets=[dva],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(early_stop, verbose=False),
                       lgb.log_evaluation(100)])
        best_iter = model.best_iteration or rounds
        pva = model.predict(np.asarray(X[va_idx], dtype=np.float32),
                            num_iteration=best_iter)
        sel = select_threshold(pva, yva, meta["fwd_ret_h"][va_idx],
                               mode=threshold_mode,
                               precision_target=precision_target,
                               fixed_threshold=fixed_threshold)
        thr = sel["threshold"]
        te_pred = np.empty(len(te_idx), dtype=np.float32)
        step = 500_000
        for s in range(0, len(te_idx), step):
            chunk = te_idx[s:s + step]
            te_pred[s:s + step] = model.predict(
                np.asarray(X[chunk], dtype=np.float32),
                num_iteration=best_iter)
        yte = meta["label_top"][te_idx]
        met = _eval(yte, te_pred)
        pred = pd.DataFrame({
            "date": dates_np[meta["date_idx"][te_idx].astype(int)],
            "code": codes[meta["code_idx"][te_idx].astype(int)],
            "year": year,
            "prob": te_pred,
            "threshold": thr,
            "signal": (te_pred >= thr).astype(np.int8),
            "label_top": yte,
            "label_valid": meta["label_valid"][te_idx],
            "fwd_ret_h": meta["fwd_ret_h"][te_idx],
            "fwd_maxdd": meta["fwd_maxdd"][te_idx],
            "fwd_maxup": meta["fwd_maxup"][te_idx],
        })
        all_preds.append(pred)
        model.save_model(os.path.join(out_dir, "models", f"lgbm_{year}.txt"),
                         num_iteration=best_iter)
        gain = model.feature_importance("gain").astype(np.float64)
        gains += gain
        n_gain += 1
        fold_rows.append({
            "year": year,
            "train_start": fold["train_start"], "train_end": fold["train_end"],
            "valid_start": fold["valid_start"], "valid_end": fold["valid_end"],
            "test_start": fold["test_start"], "test_end": fold["test_end"],
            "n_train": int(len(tr_idx)), "n_valid": int(len(va_idx)),
            "n_test": int(len(te_idx)), "best_iter": int(best_iter),
            "threshold": float(thr), "threshold_mode": threshold_mode,
            "valid_auc": _eval(yva, pva)["auc"],
            "test_auc": met["auc"], "test_ap": met["ap"],
            "test_brier": met["brier"], "test_logloss": met["logloss"],
            "test_base_rate": met["base_rate"],
            "val_precision": sel.get("precision"), "val_recall": sel.get("recall"),
            "val_f1": sel.get("f1"), "val_coverage": sel.get("coverage"),
            "val_score": sel.get("score"),
        })
        log(f"[{year}] best_iter={best_iter} thr={thr:.4f} "
            f"test_auc={met['auc']:.4f} ap={met['ap']:.4f} "
            f"base={met['base_rate']:.3f}")
        del dtr, dva, model, pva, te_pred
        gc.collect()
    if not all_preds:
        raise RuntimeError("没有可运行的 walk-forward 折")
    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_csv(os.path.join(out_dir, "predictions_wf.csv.gz"), index=False,
                 compression="gzip")
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(os.path.join(out_dir, "fold_metrics.csv"), index=False)
    imp = pd.DataFrame({"feature": feature_names,
                        "gain": gains / max(n_gain, 1)}).sort_values(
        "gain", ascending=False)
    imp.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

    # yearly aggregate metrics
    rows = []
    for y, g in preds.groupby("year"):
        ok = g["label_valid"].to_numpy(bool) & np.isfinite(g["prob"])
        yv = g.loc[ok, "label_top"].to_numpy(np.float32)
        pv = g.loc[ok, "prob"].to_numpy(np.float32)
        m = _eval(yv, pv)
        m["year"] = int(y)
        rows.append(m)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "yearly_signal_metrics.csv"),
                              index=False)

    # latest-date stop flags, using the most recent walk-forward model
    last_date = preds["date"].max()
    latest = preds[preds["date"] == last_date].copy()
    for t in (0.20, 0.25, 0.30, 0.35):
        latest[f"stop_loss_{int(round(t * 100)):03d}"] = (
            latest["prob"] >= t).astype(int)
    con = connect(db_path)
    if market == "crypto":
        latest["name"] = ""
        latest["industry"] = ""
    else:
        names = load_stock_names(con)
        latest = latest.merge(names, on="code", how="left")
    con.close()
    latest = latest.sort_values(["signal", "prob"], ascending=[False, False])
    latest.to_csv(os.path.join(out_dir, "latest_signals.csv"), index=False)
    log(f"最新交易日 {last_date}: {int(latest['signal'].sum())}/{len(latest)} "
        f"只触发止损信号 -> latest_signals.csv")
    return preds
