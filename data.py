#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data access for sm702 stop-loss model.

All reads are from the sm701 SQLite cache in read-only mode.  The cache stores
forward-adjusted (hfq) daily bars; every model feature is a price/volume ratio
or a within-stock rolling statistic, so the hfq scale cancels out.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Iterable, Optional

import numpy as np
import pandas as pd

DEFAULT_DB = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "sm701", "stock_cache.db")
)

STOCK_COND = (
    "(code GLOB 'sh60[0135]*' OR code GLOB 'sh68[89]*' "
    "OR code GLOB 'sz00[0123]*' OR code GLOB 'sz30[01]*' "
    "OR code GLOB 'bj[489]*')"
)

BAR_COLS = ["open", "high", "low", "close", "vol"]


def connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.execute("PRAGMA query_only=TRUE")
    return con


def trading_calendar(con: sqlite3.Connection, start: str, end: str) -> list[str]:
    sql = (
        "SELECT DISTINCT date FROM daily_bars "
        f"WHERE date >= ? AND date <= ? AND {STOCK_COND} ORDER BY date"
    )
    return pd.read_sql_query(sql, con, params=[start, end])["date"].tolist()


def list_codes(con: sqlite3.Connection, start: str, end: str) -> list[str]:
    sql = (
        "SELECT DISTINCT code FROM daily_bars "
        f"WHERE date >= ? AND date <= ? AND {STOCK_COND} ORDER BY code"
    )
    return pd.read_sql_query(sql, con, params=[start, end])["code"].tolist()


def load_stock_names(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT code, name, industry FROM stocks", con)


def calc_limit(df: pd.DataFrame) -> np.ndarray:
    """Board price-limit fraction for every row.

    Historical ST status is not available in the cache.  ST names therefore use
    the board limit here; the backtest additionally blocks fills at moves close
    to the board limit, which is conservative for the normal-limit boards.
    """
    codes = df["code"].astype(str)
    dates = df["date"].astype(str).to_numpy()
    n = len(df)
    lim = np.full(n, 0.10, dtype=np.float32)
    star = codes.str.startswith(("sh688", "sh689")).to_numpy()
    bj = codes.str.startswith("bj").to_numpy()
    cyb = codes.str.startswith(("sz300", "sz301")).to_numpy()
    lim[star] = 0.20
    lim[bj] = 0.30
    if cyb.any():
        lim[cyb] = np.where(dates[cyb] >= "2020-08-24", 0.20, 0.10)
    return lim


def load_bars(
    con: sqlite3.Connection,
    start: str,
    end: str,
    codes: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    sql = [
        "SELECT code, date, open, high, low, close, vol FROM daily_bars",
        f"WHERE date >= ? AND date <= ? AND {STOCK_COND}",
    ]
    params: list = [start, end]
    if codes is not None:
        codes = list(codes)
        if not codes:
            return pd.DataFrame(columns=["code", "date"] + BAR_COLS)
        sql.append("AND code IN (%s)" % ",".join(["?"] * len(codes)))
        params.extend(codes)
    sql.append("ORDER BY code, date")
    df = pd.read_sql_query(" ".join(sql), con, params=params)
    if df.empty:
        return df
    df = df[df["close"] > 0].copy()
    for c in BAR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df = df[df[BAR_COLS].notna().all(axis=1)]
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
    df = df[df["high"] >= df["low"]]
    df["code"] = pd.Categorical(df["code"], categories=sorted(df["code"].unique()))
    df = df.sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    df["lim"] = calc_limit(df)
    return df


def row_counts_by_code(con: sqlite3.Connection, start: str, end: str) -> pd.DataFrame:
    sql = (
        "SELECT code, COUNT(*) AS n FROM daily_bars "
        f"WHERE date >= ? AND date <= ? AND {STOCK_COND} "
        "GROUP BY code ORDER BY code"
    )
    return pd.read_sql_query(sql, con, params=[start, end])
