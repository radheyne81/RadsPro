"""
predict.py - next-trading-day direction model for indices, sectors and stocks.

Design (deliberately conservative):
  * Target: does the NEXT close finish above today's close?  (binary)
  * Features: only information known at today's close (no look-ahead) - trend, momentum, Hilega-Milega D/W,
    ADX, MACD, volume, range position, 52W distance, compression, market (Nifty) context.
  * Model: L2-regularised logistic regression (numpy, Newton/IRLS) - transparent, hard to overfit.
  * Validation: WALK-FORWARD. Expanding window, refit every `step` days, predict the following `step` days.
    All accuracy / calibration numbers reported are strictly out-of-sample.
  * Reliability: predictions are bucketed by confidence; the measured out-of-sample hit-rate of each bucket
    is shown next to the live probability, so you can see how much trust that confidence level has earned.

Daily direction is close to a coin-flip in liquid markets. Expect edges of a few percentage points, not certainty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine as E

FEATURES = ["r1", "r5", "r20", "rsi14", "rsi9", "hm_d", "hm_w", "hm_streak", "adx", "macd_h", "d20", "d50", "d200",
            "volx", "rpos", "dist52h", "nr7", "atrp", "mkt_r1", "mkt_hm"]
FEATURE_TXT = {"r1": "1-day return", "r5": "5-day return", "r20": "20-day return", "rsi14": "RSI14", "rsi9": "RSI9",
               "hm_d": "Hilega-Milega daily", "hm_w": "Hilega-Milega weekly", "hm_streak": "HM streak length", "adx": "ADX trend strength",
               "macd_h": "MACD histogram", "d20": "distance from 20-EMA", "d50": "distance from 50-SMA", "d200": "distance from 200-SMA",
               "volx": "volume vs 20D avg", "rpos": "close position in day's range", "dist52h": "distance from 52W high",
               "nr7": "NR7 compression", "atrp": "volatility (ATR%)", "mkt_r1": "Nifty 1-day return", "mkt_hm": "Nifty HM state"}


def _num(s):
    return s.map({"BULL": 1.0, "BEAR": -1.0, "NEUTRAL": 0.0}).fillna(0.0)


def feature_frame(d: pd.DataFrame, mkt: pd.DataFrame | None = None) -> pd.DataFrame:
    atrp = d["ATR%"].replace(0, np.nan)
    f = pd.DataFrame(index=d.index)
    f["r1"] = d["Ret1"] / atrp; f["r5"] = d["Ret5"] / atrp / np.sqrt(5); f["r20"] = d["Ret20"] / atrp / np.sqrt(20)
    f["rsi14"] = (d["RSI14"] - 50) / 25; f["rsi9"] = (d["RSI9"] - 50) / 25
    f["hm_d"] = _num(d["HM_STATE"]); f["hm_w"] = _num(d["WK_HM"]); f["hm_streak"] = np.log1p(d["HM_STREAK"].clip(upper=30)) * f["hm_d"]
    f["adx"] = (d["ADX"] - 20) / 15
    f["macd_h"] = d["MACD_H"] / d["Close"] * 100 / atrp
    f["d20"] = (d["Close"] / d["EMA20"] - 1) * 100 / atrp; f["d50"] = (d["Close"] / d["SMA50_"] - 1) * 100 / atrp
    f["d200"] = ((d["Close"] / d["SMA200"] - 1) * 100 / atrp).clip(-15, 15) / 3
    f["volx"] = np.log(d["VolX"].clip(0.2, 5)).fillna(0) * np.sign(d["Ret1"].fillna(0))
    f["rpos"] = (d["RangePos"] - .5) * 2; f["dist52h"] = (d["Dist52H"] / 20).clip(-3, 0)
    f["nr7"] = d["NR7"].astype(float); f["atrp"] = (atrp - 1.5) / 1.0
    if mkt is not None and len(mkt):
        m = mkt.reindex(d.index, method="ffill")
        f["mkt_r1"] = (m["Ret1"] / m["ATR%"].replace(0, np.nan)); f["mkt_hm"] = _num(m["HM_STATE"])
    else:
        f["mkt_r1"] = 0.0; f["mkt_hm"] = 0.0
    return f.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- numpy logistic regression
def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logit(X: np.ndarray, y: np.ndarray, l2: float = 30.0, iters: int = 25):
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]; w = np.zeros(Z.shape[1]); reg = np.eye(Z.shape[1]) * l2; reg[0, 0] = 0
    for _ in range(iters):
        p = _sigmoid(Z @ w); g = Z.T @ (p - y) + reg @ w
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + reg + 1e-6 * np.eye(len(w))
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-6:
            break
    return {"w": w, "mu": mu, "sd": sd}


def predict_logit(m, X: np.ndarray):
    return _sigmoid(np.c_[np.ones(len(X)), (X - m["mu"]) / m["sd"]] @ m["w"])


def contributions(m, x: np.ndarray) -> list[tuple[str, float]]:
    z = (x - m["mu"]) / m["sd"] * m["w"][1:]
    return sorted(zip(FEATURES, z), key=lambda t: -abs(t[1]))


# --------------------------------------------------------------------------- panel assembly + walk-forward
def build_panel(data: dict[str, pd.DataFrame], mkt: pd.DataFrame | None) -> pd.DataFrame:
    parts = []
    for s, d in data.items():
        if len(d) < 320 or "HM_STATE" not in d:
            continue
        f = feature_frame(d, mkt if s != "__mkt__" else None)
        f["y"] = np.where(d["Close"].shift(-1).notna(), (d["Close"].shift(-1) > d["Close"]).astype(float), np.nan)
        f["fwd"] = (d["Close"].shift(-1) / d["Close"] - 1) * 100
        f["sym"] = s; f["date"] = d.index; parts.append(f)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def walk_forward(panel: pd.DataFrame, min_train: int = 750, step: int = 42, l2: float = 30.0) -> pd.DataFrame:
    """Expanding-window walk-forward. Returns panel rows (labelled ones) with out-of-sample p_up."""
    lab = panel.dropna(subset=FEATURES + ["y"]).copy()
    if lab.empty:
        return lab.assign(p_up=np.nan)
    dates = np.sort(lab["date"].unique()); lab["p_up"] = np.nan
    if len(dates) < 120:
        return lab
    first_cut = max(int(len(dates) * 0.45), 1)
    for i in range(first_cut, len(dates), step):
        tr = lab[lab["date"] < dates[i]]
        if len(tr) < min_train:
            continue
        te_idx = lab.index[(lab["date"] >= dates[i]) & (lab["date"] < (dates[min(i + step, len(dates) - 1)] if i + step < len(dates) else pd.Timestamp.max))]
        m = fit_logit(tr[FEATURES].values, tr["y"].values, l2)
        lab.loc[te_idx, "p_up"] = predict_logit(m, lab.loc[te_idx, FEATURES].values)
    return lab


def evaluate(oos: pd.DataFrame) -> dict:
    """Out-of-sample quality: accuracy, Brier, base rate, and hit-rate by confidence bucket."""
    o = oos.dropna(subset=["p_up"]).copy()
    if len(o) < 200:
        return {"n": len(o)}
    o["pred_up"] = o["p_up"] > 0.5; o["hit"] = (o["pred_up"] == (o["y"] == 1))
    base = o["y"].mean(); brier = float(((o["p_up"] - o["y"]) ** 2).mean()); brier_base = float(((base - o["y"]) ** 2).mean())
    o["conf"] = (o["p_up"] - 0.5).abs()
    bk = []
    for lo, hi, name in [(0, .02, "<52%"), (.02, .05, "52-55%"), (.05, .08, "55-58%"), (.08, .12, "58-62%"), (.12, 1, ">62%")]:
        g = o[(o["conf"] >= lo) & (o["conf"] < hi)]
        if len(g) >= 20:
            bk.append({"Confidence": name, "n": len(g), "Hit-rate %": g["hit"].mean() * 100,
                       "Avg dir. move %": (g["fwd"] * np.where(g["pred_up"], 1, -1)).mean()})
    # long/short spread: top vs bottom decile of p_up each day
    o["rk"] = o.groupby("date")["p_up"].rank(pct=True)
    top, bot = o[o["rk"] >= .9], o[o["rk"] <= .1]
    spread = (top["fwd"].mean() - bot["fwd"].mean()) if len(top) and len(bot) else np.nan
    # significance of overall accuracy vs base-rate guess (normal approx, one-sided)
    acc = o["hit"].mean(); naive = max(base, 1 - base); se = np.sqrt(naive * (1 - naive) / len(o)); z = (acc - naive) / se
    return {"n": len(o), "acc%": acc * 100, "naive%": naive * 100, "base_up%": base * 100, "brier": brier, "brier_base": brier_base,
            "z_vs_naive": float(z), "buckets": pd.DataFrame(bk), "top_decile_fwd%": float(top["fwd"].mean()) if len(top) else np.nan,
            "bottom_decile_fwd%": float(bot["fwd"].mean()) if len(bot) else np.nan, "spread%": float(spread)}


def reliability_lookup(ev: dict, p: float) -> tuple[float, int]:
    """Measured out-of-sample hit-rate for predictions of this confidence level."""
    b = ev.get("buckets")
    if b is None or not len(b):
        return np.nan, 0
    c = abs(p - .5); name = "<52%" if c < .02 else "52-55%" if c < .05 else "55-58%" if c < .08 else "58-62%" if c < .12 else ">62%"
    r = b[b["Confidence"] == name]
    return (float(r["Hit-rate %"].iloc[0]), int(r["n"].iloc[0])) if len(r) else (np.nan, 0)


def _expl(m, x_last: np.ndarray, top: int = 4) -> str:
    cs = contributions(m, x_last)[:top]
    return "; ".join(f"{FEATURE_TXT[k]} {'↑' if v > 0 else '↓'}" for k, v in cs)


def next_day_table(data: dict[str, pd.DataFrame], mkt: pd.DataFrame | None, group: str, l2: float = 30.0,
                   p_extra: dict | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """-> (table of next-day predictions, evaluation dict, final model). `group` is a label like 'Stocks'."""
    panel = build_panel(data, mkt)
    if panel.empty:
        return pd.DataFrame(), {"n": 0}, None
    oos = walk_forward(panel, l2=l2); ev = evaluate(oos)
    lab = panel.dropna(subset=FEATURES + ["y"])
    model = fit_logit(lab[FEATURES].values, lab["y"].values, l2)
    rows = []
    for s, d in data.items():
        f = feature_frame(d, mkt).iloc[[-1]]
        if f[FEATURES].isna().any(axis=1).iloc[0]:
            continue
        x = f[FEATURES].values[0]; p = float(predict_logit(model, x[None, :])[0]); x_ = d.iloc[-1]
        rel, nrel = reliability_lookup(ev, p)
        atr = float(x_["ATR"]); c = float(x_["Close"]); pv = E.pivots(x_["High"], x_["Low"], x_["Close"])
        rows.append({"Symbol": s, "Group": group, "Close": c, "1D %": x_["Ret1"], "P(up) %": p * 100, "Direction": "BULL" if p > .5 else "BEAR",
                     "Confidence pp": abs(p - .5) * 100, "Measured hit-rate %": rel, "Bucket n": nrel,
                     "Exp. range low": c - .6 * atr, "Exp. range high": c + .6 * atr, "Exp. move %": .6 * x_["ATR%"],
                     "Support": pv["S1"], "Pivot": pv["PP"], "Resistance": pv["R1"], "HM D": x_["HM_STATE"], "HM W": x_["WK_HM"],
                     "ADX": x_["ADX"], "RSI14": x_["RSI14"], "VolX": x_["VolX"], "Drivers": _expl(model, x),
                     "Playbook": ("Buy above " + f"{x_['High']:,.2f}" + f" · SL {x_['High'] - 1.5 * atr:,.2f}") if p > .5 else
                                 ("Sell below " + f"{x_['Low']:,.2f}" + f" · SL {x_['Low'] + 1.5 * atr:,.2f}")})
    df = pd.DataFrame(rows)
    if len(df):
        df["Rank"] = df["P(up) %"].rank(ascending=False, method="first").astype(int)
        # conviction grade requires BOTH a decent probability and a bucket that has actually earned it out-of-sample
        def gr(r):
            ok = r["Measured hit-rate %"] == r["Measured hit-rate %"] and r["Measured hit-rate %"] >= 53 and r["Bucket n"] >= 100
            if ok and r["Confidence pp"] >= 8: return "A"
            if ok and r["Confidence pp"] >= 5: return "B"
            if r["Confidence pp"] >= 3: return "C"
            return "D"
        df["Grade"] = df.apply(gr, axis=1)
    return df.sort_values("P(up) %", ascending=False).reset_index(drop=True), ev, model
