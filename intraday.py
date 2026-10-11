"""
intraday.py - intraday factors, intraday strategy back-tests, and the next-session intraday plan.

Input is a 15-minute OHLCV frame (IST wall-clock, tz-naive). Sessions are 09:15-15:30 (25 bars).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine as E

OR_BARS = 2          # opening range = first 30 minutes (2 x 15m)


def sessions(df: pd.DataFrame) -> dict:
    return {d: g for d, g in df.groupby(df.index.normalize()) if len(g) >= 8}


def add_session_vwap(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy(); tp = (g["High"] + g["Low"] + g["Close"]) / 3; v = g["Volume"].replace(0, np.nan).fillna(1)
    g["VWAP"] = (tp * v).cumsum() / v.cumsum()
    return g


def _poc(g: pd.DataFrame, bins: int = 20) -> float:
    lo, hi = g["Low"].min(), g["High"].max()
    if hi <= lo:
        return float(g["Close"].iloc[-1])
    edges = np.linspace(lo, hi, bins + 1); mid = (g["High"] + g["Low"] + g["Close"]) / 3
    idx = np.clip(np.digitize(mid, edges) - 1, 0, bins - 1)
    vol = np.bincount(idx, weights=g["Volume"].values, minlength=bins)
    return float((edges[vol.argmax()] + edges[vol.argmax() + 1]) / 2)


def intraday_factors(df15: pd.DataFrame, daily: pd.DataFrame | None = None) -> dict | None:
    """Every intraday factor for the LATEST session in df15 + a bullish/bearish intraday score (-100..+100)."""
    S = sessions(df15)
    if len(S) < 3:
        return None
    days = sorted(S); today, prev = days[-1], days[-2]
    g = add_session_vwap(S[today]); p = S[prev]; last = g.iloc[-1]
    close, op = float(last["Close"]), float(g["Open"].iloc[0])
    pc, pdh, pdl = float(p["Close"].iloc[-1]), float(p["High"].max()), float(p["Low"].min())
    orh, orl = float(g["High"].iloc[:OR_BARS].max()), float(g["Low"].iloc[:OR_BARS].min())
    n = len(g); vwap = float(last["VWAP"])
    # relative volume: cumulative volume so far vs same-bar average of previous 10 sessions
    hist = [S[d]["Volume"].iloc[:n].sum() for d in days[-11:-1] if len(S[d]) >= n]
    rvol = float(g["Volume"].sum() / np.mean(hist)) if hist and np.mean(hist) > 0 else np.nan
    c = g["Close"]; rsi = E.rsi_wilder(df15["Close"], 14).reindex(g.index)
    ema9, ema21 = c.ewm(span=9, adjust=False).mean(), c.ewm(span=21, adjust=False).mean()
    d_hi, d_lo = float(g["High"].max()), float(g["Low"].min())
    rng_pos = (close - d_lo) / (d_hi - d_lo) if d_hi > d_lo else .5
    cross = int(((c > g["VWAP"]) != (c.shift() > g["VWAP"].shift())).iloc[1:].sum())
    atr_d = float(daily["ATR"].iloc[-1]) if daily is not None and "ATR" in daily and len(daily) else np.nan
    # 75-minute trend (3 x 15m closes per 45m is not standard; use 5 bars = 75m)
    c75 = c.iloc[4::5]
    t75 = (1 if len(c75) >= 2 and c75.iloc[-1] > c75.iloc[-2] else -1) if len(c75) >= 2 else 0
    # CPR from previous session
    pv = E.pivots(pdh, pdl, pc); cpr_w = abs(pv["TC"] - pv["BC"]) / pv["PP"] * 100
    first_hour = (float(g["Close"].iloc[min(3, n - 1)]) / op - 1) * 100
    f = {"date": today, "bars": n, "open": op, "close": close, "high": d_hi, "low": d_lo, "prev_close": pc, "PDH": pdh, "PDL": pdl,
         "gap%": (op / pc - 1) * 100, "OR_high": orh, "OR_low": orl, "VWAP": vwap, "VWAP_dist%": (close / vwap - 1) * 100,
         "ORB": "ABOVE OR HIGH" if close > orh else "BELOW OR LOW" if close < orl else "INSIDE OR", "rvol": rvol,
         "RSI15": float(rsi.iloc[-1]), "EMA9>21": bool(ema9.iloc[-1] > ema21.iloc[-1]), "trend75": t75, "range_pos": rng_pos,
         "vwap_cross": cross, "POC": _poc(g), "first_hour%": first_hour, "day_chg%": (close / pc - 1) * 100,
         "range_vs_ATR": (d_hi - d_lo) / atr_d if atr_d == atr_d and atr_d > 0 else np.nan, "CPR": (pv["BC"], pv["TC"]), "CPR_w%": cpr_w,
         "vs_PDH": "ABOVE PDH" if close > pdh else "BELOW PDL" if close < pdl else "INSIDE PD RANGE", "pivots": pv, "frame": g}
    # ---- score & logic
    pts, why = 0.0, []
    def add(v, txt): nonlocal pts; pts += v; why.append((v, txt))
    add(+14 if close > vwap else -14, f"Price {'above' if close > vwap else 'below'} VWAP ({f['VWAP_dist%']:+.2f}%)")
    if close > orh: add(+14, f"Broke above 30-min opening range high {orh:,.1f}")
    elif close < orl: add(-14, f"Broke below 30-min opening range low {orl:,.1f}")
    else: add(0, "Still inside the opening range (no breakout yet)")
    add(+8 if f["EMA9>21"] else -8, f"15m EMA9 {'>' if f['EMA9>21'] else '<'} EMA21")
    add(+6 * t75, f"75-min trend {'up' if t75 > 0 else 'down' if t75 < 0 else 'flat'}")
    if close > pdh: add(+10, "Trading above previous-day high")
    elif close < pdl: add(-10, "Trading below previous-day low")
    gp = f["gap%"]
    if abs(gp) >= 0.3:
        add(+6 if gp > 0 else -6, f"Gap {'up' if gp > 0 else 'down'} {gp:+.2f}%")
    rp = (rng_pos - .5) * 2; add(10 * rp, f"Close at {rng_pos*100:.0f}% of day's range")
    if rvol == rvol:
        sgn = 1 if close > op else -1
        if rvol >= 1.3: add(8 * sgn, f"Relative volume {rvol:.1f}x confirms the {'up' if sgn > 0 else 'down'} move")
        elif rvol < 0.7: add(0, f"Weak participation (rvol {rvol:.1f}x) - moves are less reliable")
    r = f["RSI15"]
    if r > 70: add(-4, f"15m RSI {r:.0f} stretched")
    elif r < 30: add(+4, f"15m RSI {r:.0f} oversold")
    if cross >= 4: add(0, f"VWAP crossed {cross}x - choppy / mean-reverting session")
    f["score"] = float(np.clip(pts * 1.15, -100, 100)); f["why"] = why
    f["bias"] = ("STRONG BULL" if f["score"] >= 45 else "BULL" if f["score"] >= 15 else "STRONG BEAR" if f["score"] <= -45 else "BEAR" if f["score"] <= -15 else "NEUTRAL")
    f["choppy"] = cross >= 4 or (f["range_vs_ATR"] == f["range_vs_ATR"] and f["range_vs_ATR"] < 0.4)
    return f


# --------------------------------------------------------------------------- intraday back-test
def _sim_day(g: pd.DataFrame, side: str, entry_i: int, stop: float, rr: float, cost_pct: float):
    """Enter at close of bar entry_i; exit at stop / rr*risk target / last bar close. Stop assumed first if both in one bar."""
    L = side == "LONG"; e = float(g["Close"].iloc[entry_i]); risk = abs(e - stop)
    if risk <= 0 or risk / e > 0.03:
        return None
    tg = e + rr * risk if L else e - rr * risk; px, why, j = float(g["Close"].iloc[-1]), "eod", len(g) - 1
    for j in range(entry_i + 1, len(g)):
        lo, hi = float(g["Low"].iloc[j]), float(g["High"].iloc[j])
        if (lo <= stop if L else hi >= stop): px, why = stop, "stop"; break
        if (hi >= tg if L else lo <= tg): px, why = tg, "target"; break
    gross = (px - e) / e * 100 * (1 if L else -1); net = gross - 2 * cost_pct
    return {"Net %": net, "R": net / (risk / e * 100), "Exit Why": why, "Bars": j - entry_i}


def orb_trades(df15: pd.DataFrame, sym: str, variant: str = "ORB30", rr: float = 1.5, cost_pct: float = 0.03) -> pd.DataFrame:
    """variants: ORB30 (plain) | ORB30_VWAP (breakout must be on the VWAP side AND gap not against) | VWAP_TREND (VWAP reclaim + EMA trend)"""
    S = sessions(df15); days = sorted(S); out = []
    for k, d in enumerate(days):
        g = add_session_vwap(S[d])
        if len(g) <= OR_BARS + 2:
            continue
        orh, orl = g["High"].iloc[:OR_BARS].max(), g["Low"].iloc[:OR_BARS].min()
        gap = (g["Open"].iloc[0] / S[days[k - 1]]["Close"].iloc[-1] - 1) * 100 if k else 0.0
        ema9, ema21 = g["Close"].ewm(span=9, adjust=False).mean(), g["Close"].ewm(span=21, adjust=False).mean()
        taken = False
        for i in range(OR_BARS, len(g) - 2):           # no new entries in the last 2 bars
            c, v = g["Close"].iloc[i], g["VWAP"].iloc[i]; side = None
            if variant in ("ORB30", "ORB30_VWAP"):
                if c > orh: side = "LONG"
                elif c < orl: side = "SHORT"
                if variant == "ORB30_VWAP" and side:
                    if (side == "LONG" and (c < v or gap < -0.4)) or (side == "SHORT" and (c > v or gap > 0.4)):
                        side = None
                stop = orl if side == "LONG" else orh
            else:
                up = c > v and g["Close"].iloc[i - 1] <= g["VWAP"].iloc[i - 1] and ema9.iloc[i] > ema21.iloc[i]
                dn = c < v and g["Close"].iloc[i - 1] >= g["VWAP"].iloc[i - 1] and ema9.iloc[i] < ema21.iloc[i]
                side = "LONG" if up else "SHORT" if dn else None
                if side:
                    rng = (g["High"].iloc[:i + 1] - g["Low"].iloc[:i + 1]).mean()
                    stop = v - 0.8 * rng if side == "LONG" else v + 0.8 * rng
            if side:
                r = _sim_day(g, side, i, float(stop), rr, cost_pct)
                if r:
                    out.append({"Symbol": sym, "Date": d, "Exit": d, "Side": side, "Variant": variant, **r})
                taken = True
            if taken:
                break
    return pd.DataFrame(out)


INTRADAY_VARIANTS = {"ORB30": "Opening-range breakout (30 min)", "ORB30_VWAP": "ORB30 + VWAP side + gap not against", "VWAP_TREND": "VWAP reclaim + EMA9/21 trend"}


def intraday_backtest(data15: dict[str, pd.DataFrame], rr: float = 1.5, cost_pct: float = 0.03) -> dict:
    res = {}
    for var in INTRADAY_VARIANTS:
        tr = pd.concat([orb_trades(df, s, var, rr, cost_pct) for s, df in data15.items()], ignore_index=True) if data15 else pd.DataFrame()
        tr = tr if len(tr) else pd.DataFrame(columns=["Symbol", "Date", "Exit", "Side", "Variant", "Net %", "R", "Exit Why", "Bars"])
        out = {"trades": tr}
        for side in ("LONG", "SHORT", "ALL"):
            t = tr if side == "ALL" else tr[tr["Side"] == side]
            out[side] = E.stats(t, 0.5) if len(t) else E.stats(None)
        # out-of-sample: last 40% of sessions
        if len(tr):
            cut = tr["Date"].quantile(.6); out["oos"] = E.stats(tr[tr["Date"] > cut], 0.5); out["is"] = E.stats(tr[tr["Date"] <= cut], 0.5)
        else:
            out["oos"] = out["is"] = E.stats(None)
        res[var] = out
    return res


# --------------------------------------------------------------------------- next-session intraday plan (from daily data)
def next_session_plan(d: pd.DataFrame, p_up: float | None = None, turnover_cr: float | None = None) -> dict:
    """Levels + conditional playbook for the NEXT session from the last daily bar."""
    x = d.iloc[-1]; pv = E.pivots(x["High"], x["Low"], x["Close"]); atr = float(x["ATR"]); c = float(x["Close"])
    R = float(np.clip(0.6 * atr, 0.004 * c, 0.015 * c))
    cpr_hist = ((d["High"] + d["Low"]) / 2 - (2 * (d["High"] + d["Low"] + d["Close"]) / 3 - (d["High"] + d["Low"]) / 2)).abs() / ((d["High"] + d["Low"] + d["Close"]) / 3) * 100
    narrow = bool((cpr_hist.tail(60) <= cpr_hist.iloc[-1]).mean() <= 0.25)
    comp = bool(x["NR7"]) or bool(x["INSIDE"])
    long_trig, short_trig = x["High"] * 1.0005, x["Low"] * 0.9995
    tags = []
    if narrow: tags.append("Narrow CPR (trend-day odds up)")
    if comp: tags.append("NR7 / inside day (expansion due)")
    if x["VolX"] >= 1.5: tags.append(f"Volume {x['VolX']:.1f}x yesterday")
    return {"pivots": pv, "R": R, "long_trigger": long_trig, "short_trigger": short_trig, "long_sl": long_trig - R, "short_sl": short_trig + R,
            "long_t1": long_trig + R, "long_t2": long_trig + 2 * R, "short_t1": short_trig - R, "short_t2": short_trig - 2 * R,
            "exp_range": (c - .6 * atr, c + .6 * atr), "narrow_cpr": narrow, "compression": comp, "tags": tags, "atr%": float(x["ATR%"])}
