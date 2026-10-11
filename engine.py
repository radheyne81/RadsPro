"""
NSE AI SMART TERMINAL - engine.py
=================================
One shared engine that re-implements (and fixes) the logic of the five uploaded scripts:

  app.py                -> global cues, breadth, Heikin-Ashi screener, correlation
  hilega_milega_pc.py   -> Hilega-Milega (RSI9 / EMA3 / WMA21 vs 50 + SMA50), MTF, weekly breakout, ATR/ADX
  Market_Decoder.py     -> FII/DII/Pro/Client participant OI, option-chain PCR / walls / max-pain, VIX
  nifty_screener_pro.py -> NR7 squeeze, volume burst, EMA20 dip-buy / bear-flag, VWAP distance
  main.py               -> (placeholder only)

What is NEW vs the originals
  * every signal is BACK-TESTED on history (forward returns + ATR-bracket trade simulation)
  * out-of-sample split (first 60% train / last 40% test) so we can see if an edge survives
  * probability = Bayesian-shrunk win-rate (never raw), with Wilson lower bound
  * confluence engine that scores each live setup using all modules together
  * offline demo data so the app always renders (clearly flagged as DEMO)

Nothing here is investment advice. Probabilities are statistical estimates from past data.
"""
from __future__ import annotations

import datetime as dt
import io
import logging
import math
import os
import time
import warnings
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ----------------------------------------------------------------------------
# UNIVERSES
# ----------------------------------------------------------------------------
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV",
    "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC", "JIOFIN",
    "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE",
    "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT",
    "ULTRACEMCO", "WIPRO",
]
BANKNIFTY = ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "BANKBARODA", "PNB",
             "AUBANK", "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK"]

INDICES = {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "NIFTY IT": "^CNXIT", "NIFTY AUTO": "^CNXAUTO",
           "NIFTY FMCG": "^CNXFMCG", "NIFTY METAL": "^CNXMETAL", "NIFTY PHARMA": "^CNXPHARMA",
           "NIFTY REALTY": "^CNXREALTY", "NIFTY ENERGY": "^CNXENERGY", "NIFTY PSU BANK": "^CNXPSUBANK",
           "NIFTY MEDIA": "^CNXMEDIA", "NIFTY INFRA": "^CNXINFRA"}

GLOBAL = {"S&P 500 Fut": "ES=F", "Nasdaq Fut": "NQ=F", "Dow": "^DJI", "Nikkei": "^N225", "Hang Seng": "^HSI",
          "India VIX": "^INDIAVIX", "Dollar Index": "DX-Y.NYB", "USD/INR": "INR=X", "Brent": "BZ=F",
          "Gold": "GC=F", "US 10Y": "^TNX", "Bitcoin": "BTC-USD"}
# +1 = risk-on for India when it rises, -1 = risk-off when it rises
GLOBAL_SIGN = {"S&P 500 Fut": 1, "Nasdaq Fut": 1, "Dow": 1, "Nikkei": 1, "Hang Seng": 1, "India VIX": -1,
               "Dollar Index": -1, "USD/INR": -1, "Brent": -1, "Gold": 0, "US 10Y": -1, "Bitcoin": 0}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",
           "Accept": "*/*", "Referer": "https://www.nseindia.com/"}


# ----------------------------------------------------------------------------
# DATA LAYER  (live, with deterministic DEMO fallback)
# ----------------------------------------------------------------------------
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """OHLC consistency fix (from nifty_screener_pro) + de-dup + sort."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce").dropna(subset=["Open", "High", "Low", "Close"])
    df = df[df["Close"] > 0]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[~df.index.duplicated()].sort_index()


def demo_series(symbol: str, n: int = 1300, seed: int | None = None) -> pd.DataFrame:
    """Deterministic synthetic OHLCV (regime-switching GBM). ONLY for offline demo."""
    rng = np.random.default_rng(seed if seed is not None else zlib.crc32(symbol.encode()))
    reg = rng.choice([0.0006, -0.0004, 0.0001], size=n // 60 + 2, p=[.45, .25, .30])
    drift = np.repeat(reg, 60)[:n]
    vol = rng.choice([0.008, 0.013, 0.02], size=n // 80 + 2, p=[.5, .35, .15])
    vol = np.repeat(vol, 80)[:n]
    r = drift + vol * rng.standard_normal(n)
    close = 100 * np.exp(np.cumsum(r)) * (1 + (zlib.crc32(symbol.encode()) % 50))
    op = np.r_[close[0], close[:-1]] * (1 + 0.003 * rng.standard_normal(n))
    hi = np.maximum(op, close) * (1 + np.abs(0.005 * rng.standard_normal(n)))
    lo = np.minimum(op, close) * (1 - np.abs(0.005 * rng.standard_normal(n)))
    volu = rng.lognormal(13, 0.4, n) * (1 + 2 * np.abs(r) / 0.01)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close, "Volume": volu}, index=idx)


def fetch_history(symbol: str, period: str = "5y", interval: str = "1d", demo: bool = False) -> tuple[pd.DataFrame, bool]:
    """Returns (df, is_demo). Falls back to demo data if the network / Yahoo fails."""
    if not demo:
        try:
            import yfinance as yf
            df = _clean(yf.download(symbol, period=period, interval=interval, auto_adjust=False,
                                    progress=False, threads=False))
            if len(df) > 60:
                return df, False
        except Exception:
            pass
    return demo_series(symbol), True


def fetch_many(symbols: list[str], period: str = "5y", demo: bool = False) -> tuple[dict, bool]:
    """Batch download. Returns ({symbol: df}, any_demo)."""
    out, any_demo = {}, False
    if not demo:
        try:
            import yfinance as yf
            raw = yf.download(symbols, period=period, interval="1d", group_by="ticker", auto_adjust=False,
                              progress=False, threads=True)
            for s in symbols:
                try:
                    d = _clean(raw[s] if len(symbols) > 1 else raw)
                    if len(d) > 120:
                        out[s] = d
                except Exception:
                    pass
        except Exception:
            pass
    for s in symbols:
        if s not in out:
            out[s] = demo_series(s)
            any_demo = True
    return out, any_demo


def nse_get(url: str, timeout: int = 15):
    import requests
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:
        pass
    return None


def fetch_participant_oi(n_days: int = 5) -> pd.DataFrame:
    """FII / DII / Pro / Client index-derivative positions from NSE archives (Market_Decoder logic)."""
    rows, d, tried = [], dt.date.today(), 0
    while len({r["Date"] for r in rows}) < n_days and tried < 25:
        tried += 1
        if d.weekday() < 5:
            body = nse_get(f"https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{d:%d%m%Y}.csv", 10)
            if body and b"Future Index Long" in body:
                df = pd.read_csv(io.BytesIO(body), skiprows=1)
                df.columns = [c.strip() for c in df.columns]
                for who in ["FII", "DII", "Pro", "Client"]:
                    m = df[df["Client Type"].astype(str).str.strip().str.upper() == who.upper()]
                    if m.empty:
                        continue
                    r = m.iloc[0]
                    fl, fs = int(r["Future Index Long"]), int(r["Future Index Short"])
                    cl, cs = int(r["Option Index Call Long"]), int(r["Option Index Call Short"])
                    pl, ps = int(r["Option Index Put Long"]), int(r["Option Index Put Short"])
                    rows.append({"Date": pd.Timestamp(d), "Participant": who.upper(), "Fut Long": fl, "Fut Short": fs,
                                 "Net Futures": fl - fs, "Long %": round(fl / (fl + fs) * 100, 2) if fl + fs else 0,
                                 "Call Long": cl, "Call Short": cs, "Net Calls": cl - cs,
                                 "Put Long": pl, "Put Short": ps, "Net Puts": pl - ps})
        d -= dt.timedelta(days=1)
    return pd.DataFrame(rows).sort_values(["Date", "Participant"]).reset_index(drop=True) if rows else pd.DataFrame()


def demo_participant_oi(n_days: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    base = {"FII": -60000, "DII": 22000, "PRO": 15000, "CLIENT": 23000}
    for d in days:
        for who, b in base.items():
            fl = int(150000 + rng.normal(0, 8000) + max(b, 0)); fs = int(150000 + rng.normal(0, 8000) + max(-b, 0))
            cl, cs = int(rng.integers(80000, 160000)), int(rng.integers(80000, 160000))
            pl, ps = int(rng.integers(80000, 160000)), int(rng.integers(80000, 160000))
            rows.append({"Date": d, "Participant": who, "Fut Long": fl, "Fut Short": fs, "Net Futures": fl - fs,
                         "Long %": round(fl / (fl + fs) * 100, 2), "Call Long": cl, "Call Short": cs,
                         "Net Calls": cl - cs, "Put Long": pl, "Put Short": ps, "Net Puts": pl - ps})
    return pd.DataFrame(rows)


def fetch_option_chain(symbol: str = "NIFTY") -> dict | None:
    """Live NSE option chain -> OI by strike. Needs cookies; returns None on failure."""
    import requests
    s = requests.Session(); s.headers.update(HEADERS)
    try:
        for u in ("https://www.nseindia.com", "https://www.nseindia.com/option-chain"):
            s.get(u, timeout=10); time.sleep(0.4)
        r = s.get(f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}", timeout=15)
        if r.status_code != 200:
            return None
        j = r.json()["records"]
        exp = j["expiryDates"][0]
        rows = [x for x in j["data"] if x.get("expiryDate") == exp]
        ce = {x["strikePrice"]: (x["CE"].get("openInterest", 0), x["CE"].get("changeinOpenInterest", 0)) for x in rows if x.get("CE")}
        pe = {x["strikePrice"]: (x["PE"].get("openInterest", 0), x["PE"].get("changeinOpenInterest", 0)) for x in rows if x.get("PE")}
        return {"symbol": symbol, "expiry": exp, "spot": j.get("underlyingValue"), "ce": ce, "pe": pe}
    except Exception:
        return None


def demo_option_chain(symbol: str = "NIFTY", spot: float | None = None) -> dict:
    step = 50 if symbol == "NIFTY" else 100
    spot = spot or (25000 if symbol == "NIFTY" else 55000)
    atm = round(spot / step) * step
    rng = np.random.default_rng(11)
    ks = [atm + i * step for i in range(-15, 16)]
    ce = {k: (int(2e6 * math.exp(-abs(k - (atm + 4 * step)) / (5 * step)) * (1 + .2 * rng.random())), int(rng.normal(0, 2e5))) for k in ks}
    pe = {k: (int(2.2e6 * math.exp(-abs(k - (atm - 4 * step)) / (5 * step)) * (1 + .2 * rng.random())), int(rng.normal(0, 2e5))) for k in ks}
    return {"symbol": symbol, "expiry": "DEMO", "spot": spot, "ce": ce, "pe": pe}


def chain_stats(oc: dict) -> dict:
    ce = {k: v[0] for k, v in oc["ce"].items()}; pe = {k: v[0] for k, v in oc["pe"].items()}
    tc, tp = sum(ce.values()), sum(pe.values())
    strikes = sorted(set(ce) | set(pe))
    # max pain = strike where total option-writer payout is minimal
    pain = []
    for k in strikes:
        loss = sum(max(0, k - s) * ce.get(s, 0) for s in strikes) + sum(max(0, s - k) * pe.get(s, 0) for s in strikes)
        pain.append(loss)
    max_pain = strikes[int(np.argmin(pain))] if strikes else None
    cd = sum(v[1] for v in oc["ce"].values()); pd_ = sum(v[1] for v in oc["pe"].values())
    top = lambda d: [k for k, _ in sorted(d.items(), key=lambda x: -x[1])[:3]]
    return {"pcr": tp / tc if tc else np.nan, "res": top(ce)[0] if ce else None, "sup": top(pe)[0] if pe else None,
            "top_ce": top(ce), "top_pe": top(pe), "max_pain": max_pain, "ce_oi": tc, "pe_oi": tp,
            "ce_chg": cd, "pe_chg": pd_, "spot": oc.get("spot"), "expiry": oc.get("expiry")}


# ----------------------------------------------------------------------------
# INDICATORS
# ----------------------------------------------------------------------------
def rsi_wilder(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff(); up, dn = d.clip(lower=0), (-d).clip(lower=0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + au / ad.replace(0, np.nan))


def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["Close"].shift()
    return pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up, dn = df["High"].diff(), -df["Low"].diff()
    pdm = up.where((up > dn) & (up > 0), 0.0); ndm = dn.where((dn > up) & (dn > 0), 0.0)
    a = atr(df, n)
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    ndi = 100 * ndm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def macd(c: pd.Series, f=12, s=26, sg=9):
    m = c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()
    sig = m.ewm(span=sg, adjust=False).mean()
    return m, sig, m - sig


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised-ish HA (the recursion on HA_Open is inherent)."""
    o, h, l, c = (df[x].values for x in ("Open", "High", "Low", "Close"))
    hc = (o + h + l + c) / 4
    ho = np.empty(len(df)); ho[0] = (o[0] + c[0]) / 2
    for i in range(1, len(df)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    out = pd.DataFrame({"HA_Open": ho, "HA_Close": hc}, index=df.index)
    out["HA_High"] = np.maximum.reduce([h, ho, hc]); out["HA_Low"] = np.minimum.reduce([l, ho, hc])
    out["HA_Green"] = out["HA_Close"] > out["HA_Open"]
    return out


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample("W-FRI").agg(agg).dropna(subset=["Close"])


def hilega_milega(df: pd.DataFrame, rsi_len=9, wma_len=21, ema_len=3, sma_len=50, level=50) -> pd.DataFrame:
    """Hilega-Milega state per bar (exactly the rule in hilega_milega_pc.py)."""
    c = df["Close"]; r = rsi_wilder(c, rsi_len); w = wma(r, wma_len); e = r.ewm(span=ema_len, adjust=False).mean()
    sma = c.rolling(sma_len).mean()
    bull = (r > level) & (e >= level) & (w <= e) & (c >= sma)
    bear = (r < level) & (e <= level) & (w >= e) & (c <= sma)
    out = pd.DataFrame({"RSI9": r, "WMA21": w, "EMA3": e, "SMA50": sma, "HM_BULL": bull, "HM_BEAR": bear}, index=df.index)
    # streak of consecutive same-state bars
    st = np.where(bull, 1, np.where(bear, -1, 0))
    streak = np.zeros(len(st), int)
    for i in range(len(st)):
        streak[i] = (streak[i - 1] + 1 if i and st[i] == st[i - 1] and st[i] != 0 else (1 if st[i] != 0 else 0))
    out["HM_STREAK"] = streak
    out["HM_STATE"] = np.where(bull, "BULL", np.where(bear, "BEAR", "NEUTRAL"))
    return out


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Every indicator used by the 5 scripts, one frame."""
    d = df.copy(); c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"].fillna(0)
    d = d.join(hilega_milega(d))
    d["RSI14"] = rsi_wilder(c, 14); d["ATR"] = atr(d); d["ATR%"] = d["ATR"] / c * 100; d["ADX"] = adx(d)
    d["EMA5"] = c.ewm(span=5, adjust=False).mean(); d["EMA20"] = c.ewm(span=20, adjust=False).mean()
    d["EMA50"] = c.ewm(span=50, adjust=False).mean()
    d["SMA20"] = c.rolling(20).mean(); d["SMA50_"] = c.rolling(50).mean(); d["SMA200"] = c.rolling(200).mean()
    d["MACD"], d["MACD_SIG"], d["MACD_H"] = macd(c)
    d["VolX"] = v / v.rolling(20).mean().shift(1).replace(0, np.nan)
    rng = h - l
    d["NR7"] = rng <= rng.rolling(7).min()
    d["INSIDE"] = (h <= h.shift()) & (l >= l.shift()) & ((h < h.shift()) | (l > l.shift()))
    tp = (h + l + c) / 3
    d["VWAP20"] = (tp * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    d["VWAPd%"] = (c / d["VWAP20"] - 1) * 100
    d["Hi20"] = h.rolling(20).max().shift(1); d["Lo20"] = l.rolling(20).min().shift(1)
    d["Ret1"] = c.pct_change() * 100; d["Ret5"] = c.pct_change(5) * 100; d["Ret20"] = c.pct_change(20) * 100
    d["Ret63"] = c.pct_change(63) * 100
    d["Hi52"] = h.rolling(252, min_periods=60).max(); d["Dist52H"] = (c / d["Hi52"] - 1) * 100
    d["RangePos"] = ((c - l) / (h - l).replace(0, np.nan)).fillna(0.5)
    ha = heikin_ashi(d); d["HA_Green"] = ha["HA_Green"]
    # weekly-high breakout (5dayweekBullBO): close > previous complete week's high
    w = to_weekly(d[["Open", "High", "Low", "Close", "Volume"]])
    wk_hi = w["High"].shift(1).reindex(d.index, method="ffill"); wk_lo = w["Low"].shift(1).reindex(d.index, method="ffill")
    d["WK_BO_UP"] = c > wk_hi; d["WK_BO_DN"] = c < wk_lo
    return d


def weekly_hm_state(df: pd.DataFrame) -> pd.Series:
    """Weekly HM state, forward-filled to daily WITHOUT look-ahead (uses last COMPLETED week)."""
    w = to_weekly(df)
    s = hilega_milega(w)["HM_STATE"].shift(1)  # shift: week only known after it closes
    return s.reindex(df.index, method="ffill").fillna("NEUTRAL")


# ----------------------------------------------------------------------------
# STRATEGY LIBRARY  (each returns boolean Series for LONG and SHORT entries on bar close)
# ----------------------------------------------------------------------------
@dataclass
class Strategy:
    key: str
    name: str
    source: str
    logic: str
    why: str
    long_fn: callable
    short_fn: callable | None = None


def _s_hm(d):  return d["HM_BULL"], d["HM_BEAR"]
def _s_hm_fresh(d):
    return d["HM_BULL"] & (d["HM_STREAK"] <= 2), d["HM_BEAR"] & (d["HM_STREAK"] <= 2)
def _s_hm_adx(d):
    return d["HM_BULL"] & (d["ADX"] >= 25) & (d["Close"] > d["SMA200"].fillna(0)), \
           d["HM_BEAR"] & (d["ADX"] >= 25) & (d["Close"] < d["SMA200"].fillna(1e18))
def _s_ha_rebound(d):
    return d["HA_Green"] & (~d["HA_Green"].shift(1, fill_value=True)) & (d["RSI14"] < 40), \
           (~d["HA_Green"]) & d["HA_Green"].shift(1, fill_value=False) & (d["RSI14"] > 60)
def _s_ema_dip(d):
    up = (d["SMA50_"] > d["SMA200"]) & d["RSI14"].between(40, 55) & (d["Close"] > d["EMA20"]) & (d["Close"].shift() < d["EMA20"].shift())
    dn = (d["SMA50_"] < d["SMA200"]) & d["RSI14"].between(45, 60) & (d["Close"] < d["EMA20"]) & (d["Close"].shift() > d["EMA20"].shift())
    return up, dn
def _s_nr7(d):
    # NR7 day then break of its high/low next day -> signal is evaluated on the BREAK bar
    nr = d["NR7"].shift(1, fill_value=False) & (d["ADX"] > 20)
    return nr & (d["Close"] > d["High"].shift(1)), nr & (d["Close"] < d["Low"].shift(1))
def _s_vol_burst(d):
    b = (d["VolX"] > 1.5) & (d["ADX"] > 25)
    return b & (d["Close"] > d["SMA50_"]) & (d["Close"] > d["Open"]) & (d["RangePos"] > .6), \
           b & (d["Close"] < d["SMA50_"]) & (d["Close"] < d["Open"]) & (d["RangePos"] < .4)
def _s_weekbo(d):
    return d["WK_BO_UP"] & ~d["WK_BO_UP"].shift(1, fill_value=False) & (d["Close"] > d["SMA50_"]), \
           d["WK_BO_DN"] & ~d["WK_BO_DN"].shift(1, fill_value=False) & (d["Close"] < d["SMA50_"])
def _s_brk20(d):
    return (d["Close"] > d["Hi20"]) & (d["VolX"] > 1.2), (d["Close"] < d["Lo20"]) & (d["VolX"] > 1.2)
def _s_confluence(d):
    wk = d["WK_HM"]
    lg = d["HM_BULL"] & (wk == "BULL") & (d["ADX"] >= 20) & (d["Close"] > d["SMA200"].fillna(0)) & (d["MACD_H"] > 0)
    sh = d["HM_BEAR"] & (wk == "BEAR") & (d["ADX"] >= 20) & (d["Close"] < d["SMA200"].fillna(1e18)) & (d["MACD_H"] < 0)
    return lg, sh


def _mk(key, name, source, logic, why, fn):
    return Strategy(key, name, source, logic, why, lambda d, f=fn: f(d)[0], lambda d, f=fn: f(d)[1])


STRATEGIES: dict[str, Strategy] = {s.key: s for s in [
    _mk("HM", "Hilega-Milega (daily)", "hilega_milega_pc.py",
             "RSI9>50, EMA3(RSI)>=50, WMA21(RSI)<=EMA3, Close>=SMA50 (bear = mirror)",
             "Momentum has turned up while price sits above its 50-DMA: trend + momentum agree.", _s_hm),
    _mk("HM_FRESH", "HM fresh signal (<=2 bars)", "hilega_milega_pc.py",
             "HM state that started within the last 2 bars",
             "Early entries avoid chasing a move already extended.", _s_hm_fresh),
    _mk("HM_ADX", "HM + ADX25 + 200-DMA", "hilega_milega_pc.py + nifty_screener_pro.py",
             "HM state AND ADX>=25 AND price on the right side of 200-DMA",
             "Adds trend-strength and long-term trend filters to cut chop signals.", _s_hm_adx),
    _mk("HA_REBOUND", "Heikin-Ashi colour flip + RSI", "app.py",
             "HA candle flips green with RSI14<40 (short: flips red with RSI>60)",
             "Mean-reversion: a colour change after an oversold push often marks a swing low.", _s_ha_rebound),
    _mk("EMA_DIP", "EMA20 dip-buy / bear-flag", "nifty_screener_pro.py",
             "50>200 SMA, RSI 40-55, close crosses back above EMA20 (short = mirror)",
             "Buying shallow pullbacks inside an established trend.", _s_ema_dip),
    _mk("NR7", "NR7 squeeze breakout", "nifty_screener_pro.py",
             "Yesterday was narrowest range of 7 days, today breaks its high/low, ADX>20",
             "Volatility contraction usually precedes expansion.", _s_nr7),
    _mk("VOL_BURST", "Volume burst momentum", "nifty_screener_pro.py",
             "Volume >1.5x 20D avg, ADX>25, strong close on trend side of SMA50",
             "Abnormal volume with directional close = institutional participation.", _s_vol_burst),
    _mk("WEEK_BO", "Weekly range breakout", "hilega_milega_pc.py",
             "Close beyond the last completed week's high/low (first day only), with SMA50 trend",
             "Breaking prior-week range signals a swing-timeframe expansion.", _s_weekbo),
    _mk("BRK20", "20-day breakout + volume", "hilega_milega_pc.py",
             "Close > 20D high with volume >1.2x avg",
             "Classic Donchian breakout with participation filter.", _s_brk20),
    _mk("CONFLUENCE", "HM Daily+Weekly + ADX + MACD + 200DMA", "ALL (combined)",
             "Daily HM and Weekly HM agree, ADX>=20, MACD histogram agrees, 200-DMA side agrees",
             "Multi-timeframe + multi-indicator agreement: fewer but higher-quality trades.", _s_confluence),
]}


def prep(df: pd.DataFrame) -> pd.DataFrame:
    d = enrich(df); d["WK_HM"] = weekly_hm_state(df); return d


# ----------------------------------------------------------------------------
# BACK-TEST ENGINE
# ----------------------------------------------------------------------------
@dataclass
class BTParams:
    atr_sl: float = 1.5      # stop = atr_sl * ATR
    rr: float = 2.0          # target = rr * risk
    max_hold: int = 10       # bars
    cost_pct: float = 0.05   # per side, % of price (brokerage+slippage proxy)
    cooldown: int = 3        # bars to wait after a signal (no overlapping spam)


def simulate_trades(d: pd.DataFrame, signal: pd.Series, side: str, p: BTParams, sym: str = "") -> pd.DataFrame:
    """Enter NEXT bar open after signal close. Stop/target bracket using ATR at signal bar.
    If stop and target both inside one bar -> stop assumed hit first (conservative)."""
    o, h, l, c, a = (d[x].values for x in ("Open", "High", "Low", "Close", "ATR"))
    idx = d.index; n = len(d); sig = signal.fillna(False).values
    L = side == "LONG"; trades = []; i = 0
    while i < n - 2:
        if sig[i] and np.isfinite(a[i]) and a[i] > 0:
            e = o[i + 1]; risk = p.atr_sl * a[i]
            if risk <= 0 or not np.isfinite(e):
                i += 1; continue
            sl = e - risk if L else e + risk; tg = e + p.rr * risk if L else e - p.rr * risk
            exit_px, why, j_exit = None, "time", min(i + 1 + p.max_hold, n - 1)
            for j in range(i + 1, min(i + 1 + p.max_hold, n - 1) + 1):
                hit_sl = l[j] <= sl if L else h[j] >= sl
                hit_tg = h[j] >= tg if L else l[j] <= tg
                if j == i + 1 and (o[j] <= sl if L else o[j] >= sl):   # gap through stop
                    exit_px, why, j_exit = o[j], "stop(gap)", j; break
                if hit_sl:
                    exit_px, why, j_exit = sl, "stop", j; break
                if hit_tg:
                    exit_px, why, j_exit = tg, "target", j; break
            if exit_px is None:
                exit_px = c[j_exit]
            gross = (exit_px - e) / e * 100 * (1 if L else -1)
            net = gross - 2 * p.cost_pct
            r_mult = net / (risk / e * 100)
            trades.append({"Symbol": sym, "Side": side, "Signal": idx[i], "Entry": idx[i + 1], "Exit": idx[j_exit],
                           "EntryPx": e, "ExitPx": exit_px, "Net %": net, "R": r_mult, "Bars": j_exit - i, "Exit Why": why})
            i = j_exit + p.cooldown
        else:
            i += 1
    return pd.DataFrame(trades)


def baseline_trades(d: pd.DataFrame, side: str, p: BTParams, sym: str = "", step: int = 4) -> pd.DataFrame:
    """Random-entry benchmark: same bracket, entry every `step` bars. Tells us if a signal beats chance."""
    s = pd.Series(False, index=d.index); s.iloc[::step] = True
    return simulate_trades(d, s, side, BTParams(p.atr_sl, p.rr, p.max_hold, p.cost_pct, 0), sym)


def wilson_lb(w: int, n: int, z: float = 1.645) -> float:
    if n == 0:
        return 0.0
    ph = w / n
    return float((ph + z * z / (2 * n) - z * math.sqrt((ph * (1 - ph) + z * z / (4 * n)) / n)) / (1 + z * z / n))


def shrunk_prob(w: int, n: int, base: float, k: int = 25) -> float:
    """Bayesian shrinkage toward baseline win-rate: small samples can't claim big edges."""
    return (w + k * base) / (n + k)


def stats(tr: pd.DataFrame, base_wr: float | None = None) -> dict:
    if tr is None or tr.empty:
        return {"n": 0, "win%": np.nan, "avgR": np.nan, "PF": np.nan, "avg%": np.nan, "maxDD%": np.nan, "prob%": np.nan,
                "wilson%": np.nan, "edge_pp": np.nan, "expect%": np.nan, "avg_bars": np.nan}
    w = int((tr["Net %"] > 0).sum()); n = len(tr)
    gp, gl = tr.loc[tr["Net %"] > 0, "Net %"].sum(), -tr.loc[tr["Net %"] <= 0, "Net %"].sum()
    eq = tr.sort_values("Exit")["Net %"].cumsum(); dd = (eq - eq.cummax()).min()
    base = base_wr if base_wr is not None and np.isfinite(base_wr) else 0.45
    return {"n": n, "win%": w / n * 100, "avgR": tr["R"].mean(), "PF": gp / gl if gl > 0 else np.inf,
            "avg%": tr["Net %"].mean(), "expect%": tr["Net %"].mean(), "maxDD%": dd,
            "prob%": shrunk_prob(w, n, base) * 100, "wilson%": wilson_lb(w, n) * 100,
            "edge_pp": (w / n - base) * 100, "avg_bars": tr["Bars"].mean()}


def forward_stats(d: pd.DataFrame, signal: pd.Series, side: str) -> dict:
    """Simple forward-return study (1/5/10 bar) - no bracket, entry at signal close."""
    c = d["Close"]; sg = 1 if side == "LONG" else -1; sig = signal.fillna(False)
    out = {}
    for hz in (1, 5, 10):
        f = (c.shift(-hz) / c - 1) * 100 * sg
        s, b = f[sig].dropna(), f.dropna()
        out[hz] = {"n": len(s), "win%": (s > 0).mean() * 100 if len(s) else np.nan, "avg%": s.mean() if len(s) else np.nan,
                   "base_win%": (b > 0).mean() * 100, "base_avg%": b.mean()}
    return out


def run_strategy(data: dict[str, pd.DataFrame], key: str, side: str, p: BTParams, split: float = 0.6) -> dict:
    """Pooled back-test of strategy `key` across the whole universe + in/out-of-sample split + baseline."""
    st = STRATEGIES[key]; trs, bls = [], []
    for sym, d in data.items():
        sig = st.long_fn(d) if side == "LONG" else (st.short_fn(d) if st.short_fn else None)
        if sig is None:
            continue
        trs.append(simulate_trades(d, sig, side, p, sym)); bls.append(baseline_trades(d, side, p, sym))
    tr = pd.concat([t for t in trs if len(t)], ignore_index=True) if any(len(t) for t in trs) else pd.DataFrame()
    bl = pd.concat([t for t in bls if len(t)], ignore_index=True) if any(len(t) for t in bls) else pd.DataFrame()
    base_wr = (bl["Net %"] > 0).mean() if len(bl) else np.nan
    res = {"key": key, "side": side, "trades": tr, "baseline": bl, "all": stats(tr, base_wr),
           "base": stats(bl, base_wr), "base_wr": base_wr}
    if len(tr):
        cut = tr["Signal"].quantile(split)
        res["is"] = stats(tr[tr["Signal"] <= cut], base_wr); res["oos"] = stats(tr[tr["Signal"] > cut], base_wr)
        res["cut"] = cut
        # monthly return series for robustness view
        res["equity"] = tr.sort_values("Exit").assign(Cum=lambda x: x["Net %"].cumsum())[["Exit", "Cum"]]
    else:
        res["is"] = res["oos"] = stats(None); res["cut"] = None; res["equity"] = pd.DataFrame()
    return res


def verdict(res: dict) -> tuple[str, str]:
    """Plain-language grade of a back-test. Strict on purpose."""
    a, o = res["all"], res["oos"]
    if a["n"] < 30:
        return "INSUFFICIENT DATA", "Fewer than 30 trades - any edge is statistically unproven."
    ok_all = a["avgR"] > 0.1 and a["PF"] > 1.2 and a["edge_pp"] > 2
    ok_oos = o["n"] >= 15 and o["avgR"] > 0 and o["PF"] > 1.0
    if ok_all and ok_oos and a["wilson%"] > res["base"]["win%"]:
        return "ROBUST EDGE", "Positive in-sample AND out-of-sample, beats random-entry baseline, win-rate lower bound above baseline."
    if ok_all and ok_oos:
        return "PROBABLE EDGE", "Positive both periods and beats baseline, but not yet statistically conclusive."
    if ok_all and not ok_oos:
        return "DECAYING / OVERFIT", "Worked in the past sample but failed out-of-sample - treat with suspicion."
    if a["avgR"] > 0:
        return "MARGINAL", "Slightly positive but not clearly better than random entries."
    return "NO EDGE", "Does not beat random entries after costs."


# ----------------------------------------------------------------------------
# LIVE SCANNING + CONFLUENCE SCORING
# ----------------------------------------------------------------------------
def latest_signals(d: pd.DataFrame) -> dict:
    """Which strategies fire on the LAST bar?"""
    out = {}
    for k, st in STRATEGIES.items():
        l = bool(st.long_fn(d).iloc[-1]); s = bool(st.short_fn(d).iloc[-1]) if st.short_fn else False
        if l: out[k] = "LONG"
        if s: out[k] = "SHORT" if k not in out else out[k]
    return out


def market_context(global_df: pd.DataFrame | None, part: pd.DataFrame | None, chain: dict | None, nifty: pd.DataFrame | None) -> dict:
    """Combine macro (global cues), flows (FII/Client), derivatives (PCR / walls) and Nifty trend -> -10..+10 score."""
    score, notes = 0.0, []
    if global_df is not None and len(global_df):
        g = 0.0
        for _, r in global_df.iterrows():
            sgn = GLOBAL_SIGN.get(r["Asset"], 0)
            if sgn and abs(r["Chg%"]) > 0.4:
                g += sgn * np.sign(r["Chg%"])
        g = float(np.clip(g, -3, 3)); score += g
        notes.append(f"Global cues {g:+.0f}/3")
    if part is not None and len(part):
        last = part[part["Date"] == part["Date"].max()].set_index("Participant")
        if "FII" in last.index:
            f = last.loc["FII"]; fs = 0
            if f["Long %"] < 40: fs -= 2
            elif f["Long %"] > 60: fs += 2
            if f["Net Futures"] < -50000: fs -= 1
            elif f["Net Futures"] > 50000: fs += 1
            if "CLIENT" in last.index:
                cl = last.loc["CLIENT"]
                if f["Net Futures"] < -50000 and cl["Net Futures"] > 50000: fs -= 1; notes.append("Retail-trap: retail long vs FII short")
                if f["Net Futures"] > 50000 and cl["Net Futures"] < -50000: fs += 1; notes.append("Squeeze setup: retail short vs FII long")
            score += fs; notes.append(f"FII positioning {fs:+d}")
    if chain:
        p = chain["pcr"]; cs = 0
        if p > 1.3: cs += 1
        elif p < 0.7: cs -= 1
        if chain.get("spot") and chain.get("sup") and chain.get("res"):
            if chain["spot"] < chain["sup"]: cs -= 1
            if chain["spot"] > chain["res"]: cs += 1
        score += cs; notes.append(f"Options (PCR {p:.2f}) {cs:+d}")
    if nifty is not None and len(nifty) > 210:
        n = nifty.iloc[-1]; ts = 0
        ts += 1 if n["Close"] > n["SMA200"] else -1
        ts += 1 if n["Close"] > n["SMA20"] else -1
        ts += {"BULL": 1, "BEAR": -1}.get(n["HM_STATE"], 0) + {"BULL": 1, "BEAR": -1}.get(n["WK_HM"], 0)
        score += ts; notes.append(f"Nifty trend {ts:+d}/4")
    score = float(np.clip(score, -10, 10))
    label = ("STRONG BULLISH" if score >= 6 else "BULLISH" if score >= 3 else "MILD BULLISH" if score >= 1 else
             "SIDEWAYS" if score > -1 else "MILD BEARISH" if score > -3 else "BEARISH" if score > -6 else "STRONG BEARISH")
    return {"score": score, "label": label, "notes": notes}


def confluence_row(sym: str, d: pd.DataFrame, bt_cache: dict, ctx: dict, side_filter: str | None = None) -> list[dict]:
    """Build setup rows for the last bar: every firing strategy, its historical edge, and a combined probability."""
    last = d.iloc[-1]; rows = []
    fired = latest_signals(d)
    if not fired:
        return rows
    for key, side in fired.items():
        if side_filter and side != side_filter:
            continue
        res = bt_cache.get((key, side))
        if not res or res["all"]["n"] < 20:
            continue
        a, o = res["all"], res["oos"]
        v, _ = verdict(res)
        L = side == "LONG"
        # --- confluence points (0..100) ---------------------------------------------------
        pts, why = 0.0, []
        base_p = a["prob%"]                                  # historical shrunk win-rate of this strategy
        if (L and last["HM_STATE"] == "BULL") or ((not L) and last["HM_STATE"] == "BEAR"): pts += 12; why.append("Daily HM aligned")
        if (L and last["WK_HM"] == "BULL") or ((not L) and last["WK_HM"] == "BEAR"): pts += 14; why.append("Weekly HM aligned")
        elif (L and last["WK_HM"] == "BEAR") or ((not L) and last["WK_HM"] == "BULL"): pts -= 10; why.append("Weekly HM opposes")
        if (last["Close"] > last["SMA200"]) == L and np.isfinite(last["SMA200"]): pts += 8; why.append("200-DMA side OK")
        if last["ADX"] >= 25: pts += 8; why.append(f"ADX {last['ADX']:.0f} trending")
        elif last["ADX"] < 15: pts -= 6; why.append("ADX weak")
        if (last["MACD_H"] > 0) == L: pts += 5; why.append("MACD agrees")
        if last["VolX"] >= 1.3 and ((last["Close"] > last["Open"]) == L): pts += 7; why.append(f"Volume {last['VolX']:.1f}x")
        if L and last["Dist52H"] > -5: pts += 4; why.append("Near 52W high")
        if (not L) and last["Dist52H"] < -25: pts += 3; why.append("Deep below 52W high")
        if L and last["RSI14"] > 78: pts -= 6; why.append("Overbought - chase risk")
        if (not L) and last["RSI14"] < 22: pts -= 6; why.append("Oversold - squeeze risk")
        # strategy's own edge
        pts += float(np.clip(a["edge_pp"], -10, 15)) * 1.2
        pts += 6 if v == "ROBUST EDGE" else 3 if v == "PROBABLE EDGE" else -8 if v in ("NO EDGE", "DECAYING / OVERFIT") else 0
        # market regime
        reg = ctx["score"] if L else -ctx["score"]
        pts += float(np.clip(reg, -6, 6)) * 1.5
        if reg <= -3: why.append(f"Market regime ({ctx['label']}) against trade")
        elif reg >= 3: why.append(f"Market regime ({ctx['label']}) supports trade")
        score = float(np.clip(30 + pts, 0, 100))
        # --- probability: start from strategy's shrunk win-rate, tilt by confluence score ------
        tilt = (score - 50) / 100 * 0.25                    # +-12.5pp max
        prob = float(np.clip(base_p / 100 + tilt, 0.25, 0.80)) * 100
        atr_v = last["ATR"]; e = last["Close"]
        sl_d = BTParams().atr_sl * atr_v
        rows.append({"Symbol": sym, "Side": side, "Strategy": STRATEGIES[key].name, "Key": key, "Close": last["Close"],
                     "Entry~": e, "Stop": e - sl_d if L else e + sl_d, "Target": e + BTParams().rr * sl_d if L else e - BTParams().rr * sl_d,
                     "Risk%": sl_d / e * 100, "Score": score, "Prob%": prob, "Hist Win%": a["win%"], "Edge pp": a["edge_pp"],
                     "Hist AvgR": a["avgR"], "PF": a["PF"], "Trades(n)": a["n"], "OOS AvgR": o["avgR"], "Verdict": v,
                     "Why": " | ".join(why), "ADX": last["ADX"], "RSI14": last["RSI14"], "VolX": last["VolX"],
                     "HM D": last["HM_STATE"], "HM W": last["WK_HM"]})
    return rows


def grade(score: float, prob: float, verdict_: str) -> str:
    if score >= 75 and prob >= 58 and verdict_ in ("ROBUST EDGE", "PROBABLE EDGE"): return "A+"
    if score >= 62 and prob >= 54: return "A"
    if score >= 50 and prob >= 50: return "B"
    if score >= 40: return "C"
    return "D"


def pivots(h: float, l: float, c: float) -> dict:
    pp = (h + l + c) / 3
    return {"PP": pp, "R1": 2 * pp - l, "S1": 2 * pp - h, "R2": pp + (h - l), "S2": pp - (h - l),
            "BC": (h + l) / 2, "TC": 2 * pp - (h + l) / 2}


def global_snapshot(demo: bool = False) -> tuple[pd.DataFrame, bool]:
    rows, any_demo = [], False
    for name, tk in GLOBAL.items():
        d, is_demo = fetch_history(tk, "1mo", demo=demo)
        any_demo |= is_demo
        if len(d) >= 2:
            ch = (d["Close"].iloc[-1] / d["Close"].iloc[-2] - 1) * 100
            sgn = GLOBAL_SIGN.get(name, 0)
            impact = "Neutral" if abs(ch) < 0.4 or sgn == 0 else ("Bullish" if sgn * ch > 0 else "Bearish")
            rows.append({"Asset": name, "Last": float(d["Close"].iloc[-1]), "Chg%": float(ch), "Impact on India": impact})
    return pd.DataFrame(rows), any_demo
