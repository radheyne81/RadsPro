"""
datahub.py - one place that gets market data, with an explicit fallback chain and provenance.

    mode "auto"   : Broker API (if configured) -> Yahoo Finance -> NSE bhavcopy archive (free, official, daily) -> Demo
    mode "broker" : Broker only (no silent fallback; missing series are reported)
    mode "yahoo"  : Yahoo -> Demo
    mode "nse"    : NSE bhavcopy -> Demo      (daily only; also supplies DELIVERY %)
    mode "demo"   : synthetic

Every returned series carries its source name so the UI can show exactly where each number came from.
"""
from __future__ import annotations

import datetime as dt
import io
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

import brokers as B
import engine as E

INDEX_YF = dict(E.INDICES, **{"SENSEX": "^BSESN"})
NSE_IDX_NAME = {"NIFTY 50": "NIFTY 50", "BANK NIFTY": "NIFTY BANK", "NIFTY IT": "NIFTY IT", "NIFTY AUTO": "NIFTY AUTO",
                "NIFTY FMCG": "NIFTY FMCG", "NIFTY METAL": "NIFTY METAL", "NIFTY PHARMA": "NIFTY PHARMA",
                "NIFTY REALTY": "NIFTY REALTY", "NIFTY ENERGY": "NIFTY ENERGY", "NIFTY PSU BANK": "NIFTY PSU BANK",
                "NIFTY MEDIA": "NIFTY MEDIA", "NIFTY INFRA": "NIFTY INFRASTRUCTURE"}


@dataclass(frozen=True)
class Cfg:
    mode: str = "auto"          # auto | broker | yahoo | nse | demo
    broker: str = ""            # "" | Dhan | Angel One | ICICI Breeze
    creds: tuple = ()           # sorted ((key, value), ...)  - hashable for st.cache_data

    @property
    def cred_dict(self):
        return dict(self.creds)


def _clean(df):
    return E._clean(df)


def _broker(cfg: Cfg):
    if cfg.mode in ("auto", "broker") and cfg.broker:
        try:
            return B.get_provider(cfg.broker, cfg.cred_dict), None
        except Exception as ex:
            return None, f"{cfg.broker}: {ex}"
    return None, None


# --------------------------------------------------------------------------- NSE bhavcopy (free, official, daily + delivery %)
EQ_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ds}.csv"
IDX_URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{ds}.csv"


def _nse_day(kind: str, d: pd.Timestamp):
    path = os.path.join(B.CACHE, f"{kind}_{d:%Y%m%d}.csv.gz"); miss = path + ".none"
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    if os.path.exists(miss):
        return None
    body = E.nse_get((EQ_URL if kind == "eq" else IDX_URL).format(ds=d.strftime("%d%m%Y")), 20)
    if not body:
        return None
    try:
        df = pd.read_csv(io.BytesIO(body), skipinitialspace=True); df.columns = df.columns.str.strip()
        if kind == "eq":
            df["SERIES"] = df["SERIES"].astype(str).str.strip(); df = df[df["SERIES"].isin(["EQ", "BE"])]
            df = df.rename(columns={"SYMBOL": "Symbol", "PREV_CLOSE": "Prev", "OPEN_PRICE": "Open", "HIGH_PRICE": "High", "LOW_PRICE": "Low",
                                    "CLOSE_PRICE": "Close", "TTL_TRD_QNTY": "Volume", "DELIV_PER": "DelPct"})
            df = df[["Symbol", "Prev", "Open", "High", "Low", "Close", "Volume", "DelPct"]]
            df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        else:
            df = df.rename(columns={"Index Name": "Symbol", "Open Index Value": "Open", "High Index Value": "High", "Low Index Value": "Low",
                                    "Closing Index Value": "Close", "Volume": "Volume"})
            df = df[[c for c in ["Symbol", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]]
            df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper(); df["Prev"] = np.nan; df["DelPct"] = np.nan
            if "Volume" not in df.columns:
                df["Volume"] = 0.0
        for c in df.columns:
            if c != "Symbol":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df.to_csv(path, index=False, compression="gzip")
        return df
    except Exception:
        return None


def _adjust(g: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust for splits/bonus using NSE's own PREV_CLOSE (same rule as hilega_milega_pc.py)."""
    if "Prev" not in g or g["Prev"].isna().all() or len(g) < 3:
        return g
    prev, close = g["Prev"].values.astype(float), g["Close"].values.astype(float)
    ratio = prev[1:] / close[:-1]
    ok = np.isfinite(ratio) & (np.abs(ratio - 1) > 0.03) & (ratio > 0.02) & (ratio < 50)
    f = np.ones(len(g)); f[1:] = np.where(ok, ratio, 1.0)
    adj = np.ones(len(g)); adj[:-1] = np.cumprod(f[::-1])[::-1][1:]
    g = g.copy()
    for c in ("Open", "High", "Low", "Close"):
        g[c] = g[c] * adj
    g["Volume"] = g["Volume"] / adj
    return g


def nse_daily_bulk(symbols: list[str], kind: str, days: int = 420) -> dict[str, pd.DataFrame]:
    k = "eq" if kind == "stock" else "idx"
    dates = list(pd.bdate_range(pd.Timestamp.today().normalize() - pd.Timedelta(days=days), pd.Timestamp.today().normalize()))
    with ThreadPoolExecutor(max_workers=8) as ex:
        frames = list(ex.map(lambda d: _nse_day(k, d), dates))
    parts = []
    for d, f in zip(dates, frames):
        if f is not None and len(f):
            f = f.copy(); f["Date"] = d; parts.append(f)
    if not parts:
        return {}
    big = pd.concat(parts, ignore_index=True); out = {}
    for s in symbols:
        key = (s if kind == "stock" else NSE_IDX_NAME.get(s, s)).upper()
        g = big[big["Symbol"] == key].sort_values("Date")
        if len(g) > 60:
            g = _adjust(g)
            df = g.set_index("Date")[["Open", "High", "Low", "Close", "Volume"] + (["DelPct"] if kind == "stock" else [])]
            out[s] = _clean(df).join(df[["DelPct"]]) if "DelPct" in df else _clean(df)
    return out


# --------------------------------------------------------------------------- Yahoo
def _yahoo_tk(sym, kind):
    return sym + ".NS" if kind == "stock" else INDEX_YF.get(sym, sym)


def yahoo_batch(symbols, kind, period="5y", interval="1d") -> dict[str, pd.DataFrame]:
    import yfinance as yf
    tks = [_yahoo_tk(s, kind) for s in symbols]; out = {}
    try:
        raw = yf.download(tks, period=period, interval=interval, group_by="ticker", auto_adjust=False, progress=False, threads=True)
    except Exception:
        return out
    for s, t in zip(symbols, tks):
        try:
            d = _clean(raw[t] if len(tks) > 1 else raw)
            if len(d) > (40 if interval == "1d" else 30):
                out[s] = d
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- DEMO intraday
def demo_intraday(sym: str, days: int = 40) -> pd.DataFrame:
    daily = E.demo_series(sym, 400).tail(days)
    rng = np.random.default_rng(zlib.crc32(sym.encode()) + 1); rows = []
    for dte, r in daily.iterrows():
        n = 25; step = rng.standard_normal(n) * (r["High"] - r["Low"]) / 6
        path = np.cumsum(step); path = path - np.linspace(0, path[-1], n) + np.linspace(0, r["Close"] - r["Open"], n)
        px = np.clip(r["Open"] + path, r["Low"], r["High"]); op = np.r_[r["Open"], px[:-1]]
        vv = rng.lognormal(0, .4, n) * r["Volume"] / n * (1 + np.r_[1.5, 0.6, np.zeros(n - 4), 0.4, 0.8][:n])
        ts = pd.date_range(dte + pd.Timedelta(hours=9, minutes=15), periods=n, freq="15min")
        rows.append(pd.DataFrame({"Open": op, "High": np.maximum(op, px) * 1.0008, "Low": np.minimum(op, px) * .9992, "Close": px, "Volume": vv}, index=ts))
    return pd.concat(rows)


def resample_intraday(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = []
    for _, g in df.groupby(df.index.date):
        r = g.resample(rule, origin=g.index[0].normalize() + pd.Timedelta(hours=9, minutes=15)).agg(agg).dropna(subset=["Close"])
        out.append(r)
    return pd.concat(out) if out else df


# --------------------------------------------------------------------------- unified entry points
def get_daily(symbols: list[str], kind: str, cfg: Cfg, years: int = 5) -> tuple[dict, dict, list]:
    """-> (data, source_by_symbol, messages)"""
    data, src, msgs = {}, {}, []
    need = list(symbols)
    if cfg.mode == "demo":
        return {s: E.demo_series(s) for s in symbols}, {s: "DEMO" for s in symbols}, ["Demo mode: synthetic data"]
    prov, err = _broker(cfg)
    if err:
        msgs.append("Broker login failed -> " + err)
    if prov:
        end = dt.datetime.now(); start = end - dt.timedelta(days=int(365 * years) + 10)
        for s in list(need):
            try:
                d = _clean(prov.history(s, kind, "1d", start, end))
                if len(d) > 60:
                    data[s], src[s] = d, prov.name; need.remove(s)
            except Exception as ex:
                msgs.append(f"{prov.name} {s}: {str(ex)[:120]}")
    if cfg.mode == "broker":
        for s in need:
            msgs.append(f"{s}: not available from broker (broker-only mode)")
        return data, src, msgs
    if need and cfg.mode in ("auto", "yahoo"):
        got = yahoo_batch(need, kind, f"{years}y")
        for s, d in got.items():
            data[s], src[s] = d, "Yahoo"
        need = [s for s in need if s not in got]
    if need and cfg.mode in ("auto", "nse"):
        got = nse_daily_bulk(need, kind, int(min(years, 2) * 380))
        for s, d in got.items():
            data[s], src[s] = d, "NSE bhavcopy"
        need = [s for s in need if s not in got]
    for s in need:
        data[s], src[s] = E.demo_series(s), "DEMO"
    if need:
        msgs.append(f"{len(need)} series fell back to DEMO data: {', '.join(need[:8])}{'…' if len(need) > 8 else ''}")
    return data, src, msgs


def get_intraday(symbols: list[str], kind: str, cfg: Cfg, interval: str = "15m", days: int = 55) -> tuple[dict, dict, list]:
    data, src, msgs = {}, {}, []
    need = list(symbols)
    if cfg.mode == "demo":
        return {s: demo_intraday(s) for s in symbols}, {s: "DEMO" for s in symbols}, ["Demo mode: synthetic intraday"]
    prov, err = _broker(cfg)
    if err:
        msgs.append("Broker login failed -> " + err)
    if prov:
        end = dt.datetime.now(); start = end - dt.timedelta(days=days)
        for s in list(need):
            try:
                iv = interval
                d = prov.history(s, kind, iv, start, end)
                if not len(d) and interval == "15m":                 # e.g. Breeze has no 15m -> build from 5m
                    d5 = prov.history(s, kind, "5m", start, end)
                    d = resample_intraday(d5, "15min") if len(d5) else d5
                d = _clean(d)
                if len(d) > 100:
                    data[s], src[s] = d, prov.name; need.remove(s)
            except Exception as ex:
                msgs.append(f"{prov.name} intraday {s}: {str(ex)[:120]}")
    if cfg.mode == "broker":
        for s in need:
            msgs.append(f"{s}: no intraday from broker")
        return data, src, msgs
    if need and cfg.mode in ("auto", "yahoo"):
        got = yahoo_batch(need, kind, "59d" if interval in ("5m", "15m") else "1mo", interval)
        for s, d in got.items():
            data[s], src[s] = d, "Yahoo (≤60d)"
        need = [s for s in need if s not in got]
    for s in need:
        data[s], src[s] = demo_intraday(s), "DEMO"
    if need:
        msgs.append(f"{len(need)} intraday series fell back to DEMO data")
    return data, src, msgs


def get_option_chain(sym: str, cfg: Cfg, spot: float | None) -> tuple[dict, str]:
    prov, err = _broker(cfg)
    if prov:
        try:
            oc = prov.option_chain(sym)
            if oc and oc.get("ce"):
                return oc, prov.name
        except Exception:
            pass
    if cfg.mode != "demo":
        oc = E.fetch_option_chain(sym if sym in ("NIFTY", "BANKNIFTY") else "NIFTY")
        if oc:
            return oc, "NSE live"
    return E.demo_option_chain(sym, spot), "DEMO"


def test_connection(cfg: Cfg) -> list[tuple[str, bool, str]]:
    """Step-by-step diagnostics for the Data Sources page."""
    out = []
    if cfg.broker:
        try:
            prov = B.get_provider(cfg.broker, cfg.cred_dict); out.append((f"{cfg.broker} login", True, "OK"))
            end = dt.datetime.now(); start = end - dt.timedelta(days=12)
            for lab, kind, sym, iv in [("Daily NIFTY 50", "index", "NIFTY 50", "1d"), ("15m NIFTY 50", "index", "NIFTY 50", "15m"), ("Daily RELIANCE", "stock", "RELIANCE", "1d")]:
                try:
                    d = prov.history(sym, kind, iv, start, end)
                    out.append((f"{cfg.broker} {lab}", len(d) > 0, f"{len(d)} bars, last {d.index[-1]}" if len(d) else "empty response"))
                except Exception as ex:
                    out.append((f"{cfg.broker} {lab}", False, str(ex)[:160]))
        except Exception as ex:
            out.append((f"{cfg.broker} login", False, str(ex)[:200]))
    for lab, fn in [("Yahoo daily ^NSEI", lambda: yahoo_batch(["NIFTY 50"], "index", "1mo")), ("Yahoo 15m ^NSEI", lambda: yahoo_batch(["NIFTY 50"], "index", "5d", "15m")),
                    ("NSE bhavcopy (any of last 6 days)", lambda: [x for x in (_nse_day("eq", pd.Timestamp.today().normalize() - pd.Timedelta(days=i)) for i in range(1, 7)) if x is not None])]:
        try:
            r = fn(); out.append((lab, bool(r), "OK" if r else "no data (blocked / offline?)"))
        except Exception as ex:
            out.append((lab, False, str(ex)[:160]))
    return out
